"use client";

import { useState } from "react";

const command = "brew install infatoshi/phonon/phonon";

export default function InstallCommand({ release = false }: { release?: boolean }) {
  const [copied, setCopied] = useState(false);

  async function copyCommand() {
    await navigator.clipboard.writeText(command);
    setCopied(true);
    window.setTimeout(() => setCopied(false), 1600);
  }

  return (
    <div className={`install-command${release ? " release-command" : ""}`}>
      <code>{command}</code>
      <button type="button" onClick={copyCommand} aria-label="Copy Homebrew install command">
        {copied ? "Copied" : "Copy"}
      </button>
    </div>
  );
}
