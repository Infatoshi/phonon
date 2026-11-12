//! The server side of a WebSocket (RFC 6455), as much of it as the WebSocket API
//! (`remote.rs`) needs: the opening handshake, reading whole messages, writing frames.
//! No TLS and no extensions; a tailnet or `tailscale serve` supplies the encryption.
//!
//! Written here rather than taken from a crate so the engine keeps its four
//! dependencies, and so a connection can be read on one thread and written on another.

use anyhow::{bail, Context, Result};
use std::io::{BufRead, Read};

/// A request line plus headers larger than this is refused.
const MAX_HANDSHAKE_BYTES: usize = 8 * 1024;
const WEBSOCKET_GUID: &str = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11";

pub const OP_CONTINUATION: u8 = 0x0;
pub const OP_TEXT: u8 = 0x1;
pub const OP_BINARY: u8 = 0x2;
pub const OP_CLOSE: u8 = 0x8;
pub const OP_PING: u8 = 0x9;
pub const OP_PONG: u8 = 0xA;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Message {
    Text(String),
    Binary(Vec<u8>),
    Ping(Vec<u8>),
    Pong,
    Close,
}

/// SHA-1, for the handshake's accept key only (not a security use).
pub fn sha1(data: &[u8]) -> [u8; 20] {
    let mut state: [u32; 5] = [0x67452301, 0xEFCDAB89, 0x98BADCFE, 0x10325476, 0xC3D2E1F0];
    let mut message = data.to_vec();
    let bit_len = (data.len() as u64).wrapping_mul(8);
    message.push(0x80);
    while message.len() % 64 != 56 {
        message.push(0);
    }
    message.extend_from_slice(&bit_len.to_be_bytes());
    for block in message.as_chunks::<64>().0 {
        let mut w = [0u32; 80];
        for (i, word) in block.as_chunks::<4>().0.iter().enumerate() {
            w[i] = u32::from_be_bytes(*word);
        }
        for i in 16..80 {
            w[i] = (w[i - 3] ^ w[i - 8] ^ w[i - 14] ^ w[i - 16]).rotate_left(1);
        }
        let [mut a, mut b, mut c, mut d, mut e] = state;
        for (i, word) in w.iter().enumerate() {
            let (f, k) = match i {
                0..=19 => ((b & c) | (!b & d), 0x5A827999),
                20..=39 => (b ^ c ^ d, 0x6ED9EBA1),
                40..=59 => ((b & c) | (b & d) | (c & d), 0x8F1BBCDC),
                _ => (b ^ c ^ d, 0xCA62C1D6),
            };
            let next = a
                .rotate_left(5)
                .wrapping_add(f)
                .wrapping_add(e)
                .wrapping_add(k)
                .wrapping_add(*word);
            e = d;
            d = c;
            c = b.rotate_left(30);
            b = a;
            a = next;
        }
        for (slot, value) in state.iter_mut().zip([a, b, c, d, e]) {
            *slot = slot.wrapping_add(value);
        }
    }
    let mut out = [0u8; 20];
    for (chunk, word) in out.as_chunks_mut::<4>().0.iter_mut().zip(state) {
        *chunk = word.to_be_bytes();
    }
    out
}

/// Standard base64 with padding.
pub fn base64(data: &[u8]) -> String {
    const ALPHABET: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut out = String::with_capacity(data.len().div_ceil(3) * 4);
    for chunk in data.chunks(3) {
        let b = [
            chunk[0],
            chunk.get(1).copied().unwrap_or(0),
            chunk.get(2).copied().unwrap_or(0),
        ];
        let n = (u32::from(b[0]) << 16) | (u32::from(b[1]) << 8) | u32::from(b[2]);
        out.push(ALPHABET[(n >> 18) as usize & 63] as char);
        out.push(ALPHABET[(n >> 12) as usize & 63] as char);
        out.push(if chunk.len() > 1 {
            ALPHABET[(n >> 6) as usize & 63] as char
        } else {
            '='
        });
        out.push(if chunk.len() > 2 {
            ALPHABET[n as usize & 63] as char
        } else {
            '='
        });
    }
    out
}

/// `Sec-WebSocket-Accept` for a client's `Sec-WebSocket-Key`.
pub fn accept_key(key: &str) -> String {
    base64(&sha1(format!("{key}{WEBSOCKET_GUID}").as_bytes()))
}

/// Read the client's opening request and return its `Sec-WebSocket-Key`.
pub fn read_handshake(reader: &mut impl BufRead) -> Result<String> {
    let mut total = 0usize;
    let mut line = String::new();
    let mut read_line = |reader: &mut dyn BufRead, line: &mut String| -> Result<bool> {
        line.clear();
        let mut limited = reader.take((MAX_HANDSHAKE_BYTES - total) as u64 + 1);
        let read = limited.read_line(line).context("read handshake")?;
        total += read;
        if total > MAX_HANDSHAKE_BYTES {
            bail!("handshake too large");
        }
        if read == 0 || !line.ends_with('\n') {
            bail!("handshake ended early");
        }
        Ok(!line.trim_end_matches(['\r', '\n']).is_empty())
    };
    if !read_line(reader, &mut line)? {
        bail!("empty request line");
    }
    let mut parts = line.split_whitespace();
    if parts.next() != Some("GET") || parts.next().is_none() {
        bail!("not a GET request");
    }
    let (mut key, mut upgrade, mut version) = (None, false, false);
    while read_line(reader, &mut line)? {
        let Some((name, value)) = line.split_once(':') else {
            bail!("malformed header");
        };
        let value = value.trim();
        match name.trim().to_ascii_lowercase().as_str() {
            "sec-websocket-key" => key = Some(value.to_string()),
            "upgrade" => upgrade = value.eq_ignore_ascii_case("websocket"),
            "sec-websocket-version" => version = value == "13",
            _ => {}
        }
    }
    if !upgrade {
        bail!("not a WebSocket upgrade");
    }
    if !version {
        bail!("unsupported WebSocket version");
    }
    key.filter(|key| !key.is_empty())
        .context("no Sec-WebSocket-Key")
}

pub fn handshake_response(key: &str) -> String {
    format!(
        "HTTP/1.1 101 Switching Protocols\r\nUpgrade: websocket\r\nConnection: Upgrade\r\n\
         Sec-WebSocket-Accept: {}\r\n\r\n",
        accept_key(key)
    )
}

pub fn http_error(status: &str) -> String {
    format!("HTTP/1.1 {status}\r\nConnection: close\r\nContent-Length: 0\r\n\r\n")
}

/// One unmasked frame, as a server sends it.
pub fn encode(opcode: u8, payload: &[u8]) -> Vec<u8> {
    let mut frame = Vec::with_capacity(payload.len() + 10);
    frame.push(0x80 | opcode);
    match payload.len() {
        len if len < 126 => frame.push(len as u8),
        len if len <= u16::MAX as usize => {
            frame.push(126);
            frame.extend_from_slice(&(len as u16).to_be_bytes());
        }
        len => {
            frame.push(127);
            frame.extend_from_slice(&(len as u64).to_be_bytes());
        }
    }
    frame.extend_from_slice(payload);
    frame
}

pub fn encode_text(text: &str) -> Vec<u8> {
    encode(OP_TEXT, text.as_bytes())
}

/// Reads whole messages from a client: fragments are joined, control frames between
/// them are returned as they come. `max_message` bounds a message's total size.
pub struct MessageReader<R> {
    inner: R,
    max_message: usize,
    /// The opcode and bytes so far of a fragmented message.
    partial: Option<(u8, Vec<u8>)>,
}

impl<R: Read> MessageReader<R> {
    pub fn new(inner: R, max_message: usize) -> Self {
        Self {
            inner,
            max_message,
            partial: None,
        }
    }

    pub fn set_max_message(&mut self, max_message: usize) {
        self.max_message = max_message;
    }

    pub fn get_ref(&self) -> &R {
        &self.inner
    }

    pub fn read(&mut self) -> Result<Message> {
        loop {
            let mut head = [0u8; 2];
            self.inner.read_exact(&mut head).context("read frame")?;
            let fin = head[0] & 0x80 != 0;
            if head[0] & 0x70 != 0 {
                bail!("reserved bits set (no extension was negotiated)");
            }
            let opcode = head[0] & 0x0F;
            if head[1] & 0x80 == 0 {
                bail!("client frame is not masked");
            }
            let len = match head[1] & 0x7F {
                126 => {
                    let mut bytes = [0u8; 2];
                    self.inner.read_exact(&mut bytes)?;
                    u64::from(u16::from_be_bytes(bytes))
                }
                127 => {
                    let mut bytes = [0u8; 8];
                    self.inner.read_exact(&mut bytes)?;
                    u64::from_be_bytes(bytes)
                }
                short => u64::from(short),
            };
            let control = opcode & 0x8 != 0;
            if control && (len > 125 || !fin) {
                bail!("malformed control frame");
            }
            let so_far = self.partial.as_ref().map_or(0, |(_, bytes)| bytes.len());
            if len > (self.max_message.saturating_sub(so_far)) as u64 {
                bail!("message larger than {} bytes", self.max_message);
            }
            let mut mask = [0u8; 4];
            self.inner.read_exact(&mut mask)?;
            let mut payload = vec![0u8; len as usize];
            self.inner.read_exact(&mut payload)?;
            for (i, byte) in payload.iter_mut().enumerate() {
                *byte ^= mask[i % 4];
            }
            match opcode {
                OP_CLOSE => return Ok(Message::Close),
                OP_PING => return Ok(Message::Ping(payload)),
                OP_PONG => return Ok(Message::Pong),
                OP_TEXT | OP_BINARY => {
                    if self.partial.is_some() {
                        bail!("new message inside a fragmented one");
                    }
                    if fin {
                        return finish(opcode, payload);
                    }
                    self.partial = Some((opcode, payload));
                }
                OP_CONTINUATION => {
                    let Some((first, mut bytes)) = self.partial.take() else {
                        bail!("continuation without a first fragment");
                    };
                    bytes.extend_from_slice(&payload);
                    if fin {
                        return finish(first, bytes);
                    }
                    self.partial = Some((first, bytes));
                }
                other => bail!("unknown opcode {other}"),
            }
        }
    }
}

fn finish(opcode: u8, payload: Vec<u8>) -> Result<Message> {
    if opcode == OP_TEXT {
        Ok(Message::Text(
            String::from_utf8(payload).context("text message is not UTF-8")?,
        ))
    } else {
        Ok(Message::Binary(payload))
    }
}

/// A masked frame, as a client sends it (the tests' client and `phonon ws-check`).
pub fn encode_masked(opcode: u8, fin: bool, payload: &[u8], mask: [u8; 4]) -> Vec<u8> {
    let mut frame = encode(opcode, payload);
    if !fin {
        frame[0] &= 0x7F;
    }
    frame[1] |= 0x80;
    let header = frame.len() - payload.len();
    frame.splice(header..header, mask);
    for (i, byte) in frame[header + 4..].iter_mut().enumerate() {
        *byte ^= mask[i % 4];
    }
    frame
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    fn hex(bytes: &[u8]) -> String {
        bytes.iter().map(|b| format!("{b:02x}")).collect()
    }

    #[test]
    fn sha1_matches_the_published_vectors() {
        assert_eq!(hex(&sha1(b"")), "da39a3ee5e6b4b0d3255bfef95601890afd80709");
        assert_eq!(
            hex(&sha1(b"abc")),
            "a9993e364706816aba3e25717850c26c9cd0d89d"
        );
        assert_eq!(
            hex(&sha1(
                b"abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq"
            )),
            "84983e441c3bd26ebaae4aa1f95129e5e54670f1"
        );
        // 64 bytes: the padding spills into a second block
        assert_eq!(
            hex(&sha1(&[b'a'; 64])),
            "0098ba824b5c16427bd7a1122a5a442a25ec644d"
        );
    }

    #[test]
    fn base64_pads_each_remainder() {
        assert_eq!(base64(b""), "");
        assert_eq!(base64(b"f"), "Zg==");
        assert_eq!(base64(b"fo"), "Zm8=");
        assert_eq!(base64(b"foo"), "Zm9v");
        assert_eq!(base64(b"foobar"), "Zm9vYmFy");
        assert_eq!(base64(&[0xff, 0xfe, 0xfd]), "//79");
    }

    #[test]
    fn accept_key_is_the_rfc_example() {
        assert_eq!(
            accept_key("dGhlIHNhbXBsZSBub25jZQ=="),
            "s3pPLMBiTxaQ9kYGzzhZRbK+xOo="
        );
    }

    fn request(extra: &str) -> String {
        format!(
            "GET /v1 HTTP/1.1\r\nHost: mac:8377\r\nUpgrade: WebSocket\r\nConnection: Upgrade\r\n\
             Sec-WebSocket-Key: dGhlIHNhbXBsZSBub25jZQ==\r\nSec-WebSocket-Version: 13\r\n{extra}\r\n"
        )
    }

    #[test]
    fn handshake_returns_the_key_and_leaves_the_rest_unread() {
        let mut reader = Cursor::new(format!("{}after", request("")).into_bytes());
        assert_eq!(
            read_handshake(&mut reader).unwrap(),
            "dGhlIHNhbXBsZSBub25jZQ=="
        );
        let mut rest = String::new();
        reader.read_to_string(&mut rest).unwrap();
        assert_eq!(rest, "after");
    }

    #[test]
    fn handshake_refuses_what_is_not_a_websocket_upgrade() {
        let plain = "GET / HTTP/1.1\r\nHost: mac\r\n\r\n";
        assert!(read_handshake(&mut Cursor::new(plain.as_bytes())).is_err());
        let post = request("").replacen("GET", "POST", 1);
        assert!(read_handshake(&mut Cursor::new(post.as_bytes())).is_err());
        let old = request("").replace("Version: 13", "Version: 8");
        assert!(read_handshake(&mut Cursor::new(old.as_bytes())).is_err());
        let cut = &request("")[..40];
        assert!(read_handshake(&mut Cursor::new(cut.as_bytes())).is_err());
        let huge = request(&format!("X-Fill: {}\r\n", "a".repeat(MAX_HANDSHAKE_BYTES)));
        assert!(read_handshake(&mut Cursor::new(huge.as_bytes())).is_err());
    }

    #[test]
    fn frames_round_trip_at_each_length_encoding() {
        for len in [0usize, 5, 125, 126, 65_535, 65_536] {
            let payload: Vec<u8> = (0..len).map(|i| i as u8).collect();
            let frame = encode_masked(OP_BINARY, true, &payload, [1, 2, 3, 4]);
            let mut reader = MessageReader::new(Cursor::new(frame), 1 << 20);
            assert_eq!(reader.read().unwrap(), Message::Binary(payload));
        }
    }

    #[test]
    fn server_frames_are_unmasked() {
        assert_eq!(encode_text("hi"), vec![0x81, 2, b'h', b'i']);
        assert_eq!(encode(OP_PONG, &[]), vec![0x8A, 0]);
        assert_eq!(&encode(OP_BINARY, &[0; 300])[..4], &[0x82, 126, 1, 44]);
    }

    #[test]
    fn fragments_join_and_a_ping_between_them_comes_first() {
        let mut bytes = encode_masked(OP_TEXT, false, b"hel", [9, 8, 7, 6]);
        bytes.extend(encode_masked(OP_PING, true, b"p", [1, 1, 1, 1]));
        bytes.extend(encode_masked(OP_CONTINUATION, true, b"lo", [5, 5, 5, 5]));
        let mut reader = MessageReader::new(Cursor::new(bytes), 1024);
        assert_eq!(reader.read().unwrap(), Message::Ping(b"p".to_vec()));
        assert_eq!(reader.read().unwrap(), Message::Text("hello".into()));
    }

    #[test]
    fn bad_frames_are_errors() {
        let read = |bytes: Vec<u8>, max: usize| MessageReader::new(Cursor::new(bytes), max).read();
        // unmasked, as a server would send
        assert!(read(encode_text("hi"), 1024).is_err());
        // over the limit, alone and across fragments
        assert!(read(encode_masked(OP_BINARY, true, &[0; 2000], [0; 4]), 1024).is_err());
        let mut split = encode_masked(OP_BINARY, false, &[0; 600], [0; 4]);
        split.extend(encode_masked(OP_CONTINUATION, true, &[0; 600], [0; 4]));
        assert!(read(split, 1024).is_err());
        // a declared length far beyond the limit is refused before any allocation
        let mut giant = vec![0x82, 0xFF];
        giant.extend_from_slice(&u64::MAX.to_be_bytes());
        giant.extend_from_slice(&[0; 4]);
        assert!(read(giant, 1024).is_err());
        // text that is not UTF-8, a stray continuation, a reserved bit, a cut frame
        assert!(read(encode_masked(OP_TEXT, true, &[0xff, 0xfe], [0; 4]), 1024).is_err());
        assert!(read(encode_masked(OP_CONTINUATION, true, b"x", [0; 4]), 1024).is_err());
        let mut reserved = encode_masked(OP_TEXT, true, b"x", [0; 4]);
        reserved[0] |= 0x40;
        assert!(read(reserved, 1024).is_err());
        let whole = encode_masked(OP_TEXT, true, b"hello", [0; 4]);
        assert!(read(whole[..whole.len() - 2].to_vec(), 1024).is_err());
    }

    #[test]
    fn close_and_pong_are_reported() {
        let mut bytes = encode_masked(OP_PONG, true, b"", [0; 4]);
        bytes.extend(encode_masked(OP_CLOSE, true, &[0x03, 0xE8], [0; 4]));
        let mut reader = MessageReader::new(Cursor::new(bytes), 1024);
        assert_eq!(reader.read().unwrap(), Message::Pong);
        assert_eq!(reader.read().unwrap(), Message::Close);
    }
}
