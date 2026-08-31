//! One HTTP client for the whole program.
//!
//! Transport layer security comes from the operating system: SChannel on
//! Windows, Security.framework on macOS. Nothing here compiles C, and a machine
//! whose certificate store an administrator manages is respected.

use std::sync::OnceLock;
use std::time::Duration;

/// The shared agent. Connections to Hugging Face and to the local correction
/// server are pooled through it.
pub fn agent() -> &'static ureq::Agent {
    static AGENT: OnceLock<ureq::Agent> = OnceLock::new();
    AGENT.get_or_init(|| {
        let mut builder = ureq::AgentBuilder::new()
            // A stalled connection must not hang a dictation forever.
            .timeout_connect(Duration::from_secs(30))
            .user_agent(concat!("phonon-win/", env!("CARGO_PKG_VERSION")));
        match native_tls::TlsConnector::new() {
            Ok(connector) => builder = builder.tls_connector(std::sync::Arc::new(connector)),
            Err(error) => eprintln!("phonon: system TLS is unavailable: {error}"),
        }
        builder.build()
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Building the agent must not panic on a machine with an unusual store.
    #[test]
    fn the_agent_builds_once() {
        let first = agent() as *const ureq::Agent;
        let second = agent() as *const ureq::Agent;
        assert_eq!(first, second);
    }
}
