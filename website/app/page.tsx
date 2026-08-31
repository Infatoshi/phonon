import Image from "next/image";
import InstallCommand from "./InstallCommand";

const features = [
  {
    title: "Local by design",
    detail: "Audio, transcripts, dictionary terms, and screen context stay on your Mac.",
  },
  {
    title: "Ready before you speak",
    detail: "Speech and language models warm at launch, so the first real dictation is fast.",
  },
  {
    title: "Learns your vocabulary",
    detail: "A personal dictionary repairs technical names without flattening the rest of your sentence.",
  },
];

export default function Home() {
  return (
    <main>
      <nav className="nav shell" aria-label="Primary navigation">
        <a className="brand" href="#top" aria-label="Phonon home">
          <Image src="/phonon-icon.png" alt="" width={34} height={34} priority />
          <span>Phonon</span>
        </a>
        <div className="nav-links">
          <a className="nav-link" href="https://github.com/Infatoshi/phonon">GitHub</a>
          <a className="nav-link" href="#release">macOS release</a>
          <a className="nav-link" href="#windows">Windows beta</a>
        </div>
      </nav>

      <section className="hero shell" id="top">
        <h1>Fast. Local.<br />Sovereignty.</h1>
        <p className="hero-copy">
          Voice typing that runs on your Mac, understands your vocabulary, and stays out of your way.
        </p>
        <InstallCommand />
        <div className="hero-actions">
          <a
            className="button primary"
            href="https://github.com/Infatoshi/phonon/releases/latest/download/Phonon.dmg"
          >
            Get Phonon for macOS
          </a>
          <span className="platform">Apple silicon · macOS 14+</span>
        </div>
      </section>

      <section className="feature-grid shell" aria-label="Features">
        {features.map((feature, index) => (
          <article className="feature" key={feature.title}>
            <span className="feature-number">0{index + 1}</span>
            <h2>{feature.title}</h2>
            <p>{feature.detail}</p>
          </article>
        ))}
      </section>

      <section className="flow shell">
        <div>
          <p className="section-label">The whole loop</p>
          <h2>Your voice in.<br />Clean text out.</h2>
        </div>
        <div className="pipeline" aria-label="Phonon processing pipeline">
          <span>Microphone</span><i>→</i><span>Local speech</span><i>→</i><span>Personal polish</span><i>→</i><span>Your text</span>
        </div>
      </section>

      <section className="release shell" id="release">
        <div>
          <p className="section-label">macOS release</p>
          <h2>Available now.</h2>
          <p>
            Install the signed and notarized macOS app with Homebrew or download the DMG directly. Phonon downloads
            its open model weights on first launch and keeps dictation data on your Mac.
          </p>
          <InstallCommand release />
        </div>
        <div className="release-state" aria-label="Release status">
          <span className="status-dot" />
          <div>
            <strong>Public release</strong>
            <span>No cloud account required</span>
          </div>
        </div>
      </section>

      <section className="release shell" id="windows">
        <div>
          <p className="section-label">Windows beta</p>
          <h2>Beta. Untested on real hardware.</h2>
          <p>
            This build is compiled and tested end to end on Windows in continuous integration.
            Nobody has run it on a real Windows machine yet. Phonon is looking for one tester.
          </p>
          <div className="hero-actions">
            <a
              className="button primary"
              href="https://github.com/Infatoshi/phonon/releases/download/v0.2.0-win-beta.1/phonon-win.exe"
            >
              Download the Windows beta
            </a>
            <span className="platform">Windows 10 or 11 · x64 · CPU inference · 16 GB RAM recommended · 3.8 GB downloaded on first run</span>
          </div>
          <p className="smartscreen">
            The build is not code signed, so Windows SmartScreen shows &quot;Windows protected your PC&quot;.
            Choose &quot;More info&quot;, then &quot;Run anyway&quot;.
          </p>
          <ul className="windows-facts">
            <li>Hold Right Ctrl to dictate.</li>
            <li>Double-tap to latch.</li>
            <li>Speech recognition and correction both run on the CPU with no account and no cloud.</li>
          </ul>
        </div>
        <div className="release-state" aria-label="Windows release status">
          <span className="status-dot" />
          <div>
            <strong>Beta build</strong>
            <span>Seeking one tester</span>
          </div>
        </div>
      </section>

      <footer className="footer shell">
        <a className="brand" href="#top">
          <Image src="/phonon-icon.png" alt="" width={28} height={28} />
          <span>Phonon</span>
        </a>
        <p>Open-source voice typing for macOS.</p>
      </footer>
    </main>
  );
}
