/**
 * ZipFileSystem / ZipReadFileSystem zip64 support.
 *
 * Writing 4 GiB per test is too slow (the CRC runs in JS), so the zip64 cases
 * lower ZipFileSystem.zip64Limit: fields at or above it take the zip64 form
 * exactly as they would past 4 GiB. The 65535-entry limit is exercised for
 * real.
 */

import assert from 'node:assert';
import { describe, it } from 'node:test';

import { MemoryFileSystem, MemoryReadFileSystem, ZipFileSystem, ZipReadFileSystem } from '../src/lib/index.js';

const encoder = new TextEncoder();

// entries: [name, parts[]] - each part is a separate write() into the entry
const writeZip = async (entries, zip64Limit = ZipFileSystem.zip64Limit) => {
    const memFs = new MemoryFileSystem();
    const writer = await memFs.createWriter('out.zip');
    const defaultLimit = ZipFileSystem.zip64Limit;
    ZipFileSystem.zip64Limit = zip64Limit;
    let zipFs;
    try {
        zipFs = new ZipFileSystem(writer);
    } finally {
        ZipFileSystem.zip64Limit = defaultLimit;
    }
    for (const [name, parts] of entries) {
        const entryWriter = await zipFs.createWriter(name);
        for (const part of parts) {
            await entryWriter.write(part);
        }
        await entryWriter.close();
    }
    await zipFs.close();
    return memFs.results.get('out.zip');
};

const readZip = async (bytes) => {
    const rfs = new MemoryReadFileSystem();
    rfs.set('out.zip', bytes);
    return new ZipReadFileSystem(await rfs.createSource('out.zip'));
};

const readEntry = async (zip, name) => {
    const source = await zip.createSource(name);
    const stream = source.read();
    const data = await stream.readAll();
    stream.close();
    return data;
};

const concat = (parts) => {
    const out = new Uint8Array(parts.reduce((n, p) => n + p.length, 0));
    let o = 0;
    for (const p of parts) {
        out.set(p, o);
        o += p.length;
    }
    return out;
};

const bytesOf = (n, seed) => Uint8Array.from({ length: n }, (_, i) => (i * 31 + seed) & 0xff);

const tail = (bytes) => {
    const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
    const eocd = bytes.length - 22;
    return {
        view,
        entryCount: view.getUint16(eocd + 10, true),
        cdSize: view.getUint32(eocd + 12, true),
        cdOffset: view.getUint32(eocd + 16, true),
        hasZip64Locator: bytes.length >= 42 && view.getUint32(eocd - 20, true) === 0x07064b50
    };
};

describe('ZipFileSystem zip64', () => {
    it('keeps the classic layout below the limits', async () => {
        const entries = [['meta.json', [encoder.encode('{"a":1}')]], ['data.bin', [bytesOf(1000, 1), bytesOf(24, 2)]]];
        const bytes = await writeZip(entries);

        // local header + data + 16-byte descriptor, 46-byte CD records, 22-byte EOCD
        const nameBytes = entries.reduce((n, [name]) => n + name.length, 0);
        const dataBytes = entries.reduce((n, [, parts]) => n + concat(parts).length, 0);
        assert.strictEqual(bytes.length, entries.length * (30 + 16 + 46) + 2 * nameBytes + dataBytes + 22);

        const { view, entryCount, cdSize, cdOffset, hasZip64Locator } = tail(bytes);
        assert.strictEqual(hasZip64Locator, false);
        assert.strictEqual(entryCount, entries.length);
        assert.strictEqual(cdOffset + cdSize, bytes.length - 22);
        assert.strictEqual(view.getUint16(cdOffset + 6, true), 20, 'version needed stays 2.0');

        const zip = await readZip(bytes);
        for (const [name, parts] of entries) {
            assert.deepStrictEqual(await readEntry(zip, name), concat(parts));
        }
        zip.close();
    });

    it('writes zip64 sizes, offsets and end record only where fields overflow', async () => {
        // limit 100: 'a' stays classic, 'b' overflows its size (header at 97),
        // 'c' overflows its header offset, and the central directory's offset
        // and size both overflow
        const a = bytesOf(50, 3);
        const b = [bytesOf(90, 4), bytesOf(60, 5)];
        const c = bytesOf(10, 6);
        const bytes = await writeZip([['a', [a]], ['b', b], ['c', [c]]], 100);

        const { view, entryCount, cdSize, cdOffset, hasZip64Locator } = tail(bytes);
        assert.strictEqual(hasZip64Locator, true);
        assert.strictEqual(entryCount, 3);
        assert.strictEqual(cdSize, 0xffffffff);
        assert.strictEqual(cdOffset, 0xffffffff);

        // 'b' ends in the 24-byte zip64 data descriptor, then 'c' begins
        const bDescriptor = 97 + 31 + 150;
        assert.strictEqual(view.getUint32(bDescriptor, true), 0x08074b50);
        assert.strictEqual(Number(view.getBigUint64(bDescriptor + 8, true)), 150);
        assert.strictEqual(view.getUint32(bDescriptor + 24, true), 0x04034b50);

        const zip = await readZip(bytes);
        assert.deepStrictEqual(await zip.list(), ['a', 'b', 'c']);
        const expected = { a: [0, 50], b: [97, 150], c: [302, 10] };
        for (const [name, [offset, size]] of Object.entries(expected)) {
            const entry = await zip.getEntry(name);
            assert.strictEqual(entry.offset, offset, `${name} header offset`);
            assert.strictEqual(entry.uncompressedSize, size, `${name} size`);
            assert.strictEqual(entry.compressedSize, size, `${name} compressed size`);
        }
        assert.deepStrictEqual(await readEntry(zip, 'a'), a);
        assert.deepStrictEqual(await readEntry(zip, 'b'), concat(b));
        assert.deepStrictEqual(await readEntry(zip, 'c'), c);
        zip.close();
    });

    it('writes the zip64 end record past 65534 entries', async () => {
        const entries = Array.from({ length: 0xffff }, (_, i) => [`f${i}`, [bytesOf(i % 7, i)]]);
        const bytes = await writeZip(entries);

        const { entryCount, cdOffset, hasZip64Locator } = tail(bytes);
        assert.strictEqual(hasZip64Locator, true);
        assert.strictEqual(entryCount, 0xffff);
        assert.notStrictEqual(cdOffset, 0xffffffff, 'offset still fits, so stays classic');

        const zip = await readZip(bytes);
        const names = await zip.list();
        assert.strictEqual(names.length, 0xffff);
        assert.strictEqual(names[0xfffe], 'f65534');
        assert.deepStrictEqual(await readEntry(zip, 'f65534'), bytesOf(65534 % 7, 65534));
        zip.close();
    });
});
