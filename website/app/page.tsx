import Image from "next/image";

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
        </div>
      </nav>

      <section className="hero shell" id="top">
        <h1>Fast. Local.<br />Sovereignty.</h1>
        <p className="hero-copy">
          Voice typing that runs on your Mac, understands your vocabulary, and stays out of your way.
        </p>
        <div className="hero-actions">
          <a className="button primary" href="#release">Get Phonon for macOS</a>
          <span className="platform">Apple silicon · macOS 14+</span>
        </div>
      </section>

      <section className="product shell" aria-label="Phonon app preview">
        <div className="window-glow" />
        <Image
          className="app-shot"
          src="/phonon-app.png"
          alt="Phonon showing voice typing speed, microphone priority, permissions, and activity"
          width={1600}
          height={1162}
          priority
        />
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
          <h2>The public build is being prepared.</h2>
          <p>
            Phonon is open source and in alpha while its local model runtime is packaged and the app is notarized.
            The first download will be a signed macOS installer, followed by a Homebrew cask.
          </p>
        </div>
        <div className="release-state" aria-label="Release status">
          <span className="status-dot" />
          <div>
            <strong>Release in preparation</strong>
            <span>No cloud account required</span>
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
