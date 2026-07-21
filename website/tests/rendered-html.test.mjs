import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

test("renders the Phonon landing page", async () => {
  const html = await readFile(new URL("../.next/server/app/index.html", import.meta.url), "utf8");

  assert.match(html, /<title>Phonon — Open-source voice typing for Mac<\/title>/i);
  assert.match(html, /Fast\. Local\./);
  assert.match(html, /Open-source voice typing/);
  assert.match(html, /Local by design/);
  assert.match(html, /Release in preparation/);
  assert.match(html, /phonon-app\.png/);
  assert.doesNotMatch(html, /codex-preview|Your site is taking shape|react-loading-skeleton/i);
});
