"""Build the shortlist-head experiment GGUF from a shipped refiner GGUF (tied output head).

Adds an untied output.weight: the token_embd rows reordered so the K static shortlist rows come first, then every
other row in id order (Q4_K bytes copied unchanged, so every row is the same data as the tied head). Adds
output.slx: SLX_ROWS rows of zeros that llama_phonon_sl_set_extras fills per dictation with the 8-row groups holding
the dictation's own token ids. Metadata: phonon.sl.k (K) and phonon.sl.order (slot -> token id, n_vocab int32).
Every other tensor and key is copied unchanged. Needs ios/llama-b12132-shortlist.patch to load.

usage: uv run --no-project --with numpy python -I shortlist_gguf.py GGUF_PY_DIR IN.gguf LISTS.json OUT.gguf [K]
  GGUF_PY_DIR: gguf-py from the llama.cpp b12132 tree; LISTS.json: {"32000": [ids...]} (static list incl. forced rows)
"""
import json
import sys

sys.path.insert(0, sys.argv[1])
import numpy as np  # noqa: E402
import gguf  # noqa: E402

SLX_ROWS = 2048

src, lists_path, dst = sys.argv[2], sys.argv[3], sys.argv[4]
K = int(sys.argv[5]) if len(sys.argv) > 5 else 32000
r = gguf.GGUFReader(src)
tensors = list(r.tensors)
assert not any(t.name == "output.weight" for t in tensors), "input already has an untied output head"
emb = next(t for t in tensors if t.name == "token_embd.weight")
n_vocab = emb.data.shape[0]
static = json.load(open(lists_path))[str(K)]
assert len(static) == K == len(set(static)) and K % 8 == 0
in_static = np.zeros(n_vocab, dtype=bool)
in_static[static] = True
order = np.concatenate([np.array(static, dtype=np.int32), np.nonzero(~in_static)[0].astype(np.int32)])
assert len(order) == n_vocab and len(np.unique(order)) == n_vocab
head = np.ascontiguousarray(emb.data[order])  # rows of raw Q4_K bytes, reordered
slx = np.zeros((SLX_ROWS, emb.data.shape[1]), dtype=emb.data.dtype)

arch = r.fields[gguf.Keys.General.ARCHITECTURE].contents()
w = gguf.GGUFWriter(dst, arch=arch, endianess=r.endianess)
for f in r.fields.values():
    if f.name == gguf.Keys.General.ARCHITECTURE or f.name.startswith("GGUF."):
        continue
    vt = f.types[0]
    w.add_key_value(f.name, f.contents(), vt, sub_type=f.types[-1] if vt == gguf.GGUFValueType.ARRAY else None)
w.add_uint32("phonon.sl.k", K)
w.add_array("phonon.sl.order", [int(x) for x in order])
for t in tensors:
    w.add_tensor_info(t.name, t.data.shape, t.data.dtype, t.data.nbytes, t.tensor_type)
w.add_tensor_info("output.weight", head.shape, head.dtype, head.nbytes, emb.tensor_type)
w.add_tensor_info("output.slx", slx.shape, slx.dtype, slx.nbytes, emb.tensor_type)
w.write_header_to_file()
w.write_kv_data_to_file()
w.write_ti_data_to_file()
for t in tensors:
    w.write_tensor_data(t.data, tensor_endianess=r.endianess)
w.write_tensor_data(head, tensor_endianess=r.endianess)
w.write_tensor_data(slx, tensor_endianess=r.endianess)
w.close()
print(f"K {K}, n_vocab {n_vocab}, output.weight {head.nbytes} B, output.slx {slx.nbytes} B, "
      f"static ids in first 8 slots {order[:8].tolist()}")
