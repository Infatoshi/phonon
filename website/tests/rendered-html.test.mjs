import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

test("renders the Phonon landing page", async () => {
  const html = await readFile(new URL("../.next/server/app/index.html", import.meta.url), "utf8");

  assert.match(html, /<title>Phonon — Fast, local voice typing<\/title>/i);
  assert.match(html, /Fast\. Local\.<br\/>Sovereignty\./);
  assert.match(html, /Open-source voice typing/);
  assert.doesNotMatch(html, /Open-source voice typing for Mac/);
  assert.match(html, /Local by design/);
  assert.match(html, /Available now\./);
  assert.match(html, /brew install infatoshi\/phonon\/phonon/);
  assert.match(html, /Copy Homebrew install command/);
  assert.ok(
    html.indexOf("brew install infatoshi/phonon/phonon") <
      html.indexOf("Get Phonon for macOS"),
    "install command should appear above the macOS download row",
  );
  assert.doesNotMatch(html, /being prepared|Release in preparation/);
  assert.match(html, /phonon-app\.png/);
  assert.doesNotMatch(html, /codex-preview|Your site is taking shape|react-loading-skeleton/i);
});
