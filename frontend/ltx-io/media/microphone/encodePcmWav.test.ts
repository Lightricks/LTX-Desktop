import assert from "node:assert/strict";
import { describe, it } from "node:test";

import { encodePcmWav } from "./encodePcmWav.ts";

describe("encodePcmWav", () => {
  it("writes a 44-byte RIFF header and PCM samples", () => {
    const left = new Float32Array([0, 1, -1]);
    const encoded = encodePcmWav([left], 48_000);
    const view = new DataView(encoded);
    const ascii = (offset: number, length: number) =>
      String.fromCharCode(...new Uint8Array(encoded.slice(offset, offset + length)));

    assert.equal(ascii(0, 4), "RIFF");
    assert.equal(ascii(8, 4), "WAVE");
    assert.equal(ascii(12, 4), "fmt ");
    assert.equal(ascii(36, 4), "data");
    assert.equal(view.getUint16(22, true), 1);
    assert.equal(view.getUint32(24, true), 48_000);
    assert.equal(view.getUint32(40, true), 6);
    assert.equal(encoded.byteLength, 50);
    assert.equal(view.getInt16(44, true), 0);
    assert.equal(view.getInt16(46, true), 0x7fff);
    assert.equal(view.getInt16(48, true), -0x8000);
  });
});
