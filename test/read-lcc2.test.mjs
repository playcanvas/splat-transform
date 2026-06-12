/**
 * Unit tests for the LCC2 reader.
 *
 * Covers the pure helpers (meta parsing for new/legacy protocols, octree LOD
 * collection with per-chunk counts, LOD selection, chunk-count fallbacks) and
 * the main readLcc2 flow (deterministic task ordering, the output lod column,
 * environment-chunk handling and failure modes), exercised against in-memory
 * .spz / .sog chunks built with writeSpz / writeSog.
 */

import assert from 'node:assert';
import { dirname, join } from 'node:path';
import { describe, it } from 'node:test';
import { fileURLToPath } from 'node:url';

import { createTestDataTable } from './helpers/test-utils.mjs';
import {
    MemoryReadFileSystem,
    MemoryFileSystem,
    WebPCodec,
    writeSog,
    writeSpz
} from '../src/lib/index.js';
import {
    stripTrailingCommas,
    parseLcc2Meta,
    getChildren,
    collectChunksByLevel,
    resolveLodSelection,
    isMissingError,
    openChunkSource,
    parseSpzNumPoints,
    readChunkCount,
    readLcc2,
    LOAD_CONCURRENCY
} from '../src/lib/readers/read-lcc2.js';

const __dirname = dirname(fileURLToPath(import.meta.url));
WebPCodec.wasmUrl = join(__dirname, '..', 'lib', 'webp.wasm');


describe('stripTrailingCommas', () => {
    it('strips trailing commas before } and ]', () => {
        const text = '{ "a": [1, 2, ], "b": { "c": 3, }, }';
        assert.deepStrictEqual(
            JSON.parse(stripTrailingCommas(text)),
            { a: [1, 2], b: { c: 3 } }
        );
    });

    it('preserves string contents containing ",}" and ",]"', () => {
        const text = '{ "name": "weird,}.sog", "arr": ["x,]" ], }';
        const parsed = JSON.parse(stripTrailingCommas(text));
        assert.strictEqual(parsed.name, 'weird,}.sog');
        assert.deepStrictEqual(parsed.arr, ['x,]']);
    });

    it('handles escaped quotes inside strings', () => {
        const text = '{ "a": "q\\",}", }';
        assert.strictEqual(JSON.parse(stripTrailingCommas(text)).a, 'q",}');
    });

    it('leaves valid JSON unchanged', () => {
        const text = JSON.stringify({ a: [1, 2], b: 'x,}' });
        assert.strictEqual(stripTrailingCommas(text), text);
    });
});

describe('parseLcc2Meta', () => {
    it('parses the new protocol verbatim', () => {
        const meta = parseLcc2Meta(
            JSON.stringify({
                totalSplats: 100,
                lodSplats: [60, 40],
                totalLevels: 2,
                splatType: '.sog',
                root: { splatFiles: ['a.sog', 'b.sog'], data: { env: { name: 1 } } }
            }),
            'meta.lcc2'
        );

        assert.strictEqual(meta.totalSplats, 100);
        assert.deepStrictEqual(meta.lodSplats, [60, 40]);
        assert.strictEqual(meta.totalLevels, 2);
        assert.strictEqual(meta.splatType, '.sog');
        assert.deepStrictEqual(meta.splatFiles, ['a.sog', 'b.sog']);
        assert.strictEqual(meta.envFileIndex, 1);
    });

    it('maps legacy protocol fields and normalizes file names', () => {
        const meta = parseLcc2Meta(
            JSON.stringify({
                total_splats: 50,
                lod_3dgs_info: [10, 40],
                lod_level: 2,
                root: {
                    files: ['/a', '/b.sog'],
                    child_num: 1,
                    child: [{ child_num: 0, datatype: 'x' }]
                }
            }),
            'meta.lcc2'
        );

        assert.strictEqual(meta.totalSplats, 50);
        // lod_3dgs_info is reversed
        assert.deepStrictEqual(meta.lodSplats, [40, 10]);
        assert.strictEqual(meta.totalLevels, 2);
        // leading '/' dropped, '.sog' appended when missing
        assert.deepStrictEqual(meta.splatFiles, ['a.sog', 'b.sog']);
        // child_num -> childNum recursively; datatype -> dataType
        assert.strictEqual(meta.tree.childNum, 1);
        assert.strictEqual(meta.tree.child[0].childNum, 0);
        assert.strictEqual(meta.tree.child[0].dataType, 'x');
    });

    it('detects the legacy protocol even when total_splats is 0', () => {
        // Presence (not truthiness) must drive legacy detection so a genuine
        // zero count still maps the legacy fields.
        const meta = parseLcc2Meta(
            JSON.stringify({
                total_splats: 0,
                lod_3dgs_info: [0],
                lod_level: 1,
                root: { files: ['/a'] }
            }),
            'meta.lcc2'
        );
        assert.strictEqual(meta.totalSplats, 0);
        assert.deepStrictEqual(meta.lodSplats, [0]);
        assert.strictEqual(meta.totalLevels, 1);
        assert.deepStrictEqual(meta.splatFiles, ['a.sog']);
    });

    it('defaults splatType to .sog when absent', () => {
        const meta = parseLcc2Meta(
            JSON.stringify({
                totalSplats: 1,
                lodSplats: [1],
                totalLevels: 1,
                root: { splatFiles: ['a.sog'] }
            }),
            'meta.lcc2'
        );
        assert.strictEqual(meta.splatType, '.sog');
        assert.strictEqual(meta.envFileIndex, undefined);
    });

    it('tolerates trailing commas before braces', () => {
        const meta = parseLcc2Meta(
            '{ "totalSplats": 1, "lodSplats": [1], "totalLevels": 1, "root": { "splatFiles": ["a.sog"], } , }',
            'meta.lcc2'
        );
        assert.strictEqual(meta.totalLevels, 1);
        assert.deepStrictEqual(meta.splatFiles, ['a.sog']);
    });

    it('does not corrupt string values containing ",}"', () => {
        // Valid JSON is parsed strictly, so a file name containing ",}" must
        // survive (the old regex-based cleanup rewrote it to "}").
        const meta = parseLcc2Meta(
            JSON.stringify({
                totalSplats: 1,
                lodSplats: [1],
                totalLevels: 1,
                root: { splatFiles: ['weird,}.sog'] }
            }),
            'meta.lcc2'
        );
        assert.deepStrictEqual(meta.splatFiles, ['weird,}.sog']);
    });

    it('throws a descriptive error on invalid JSON', () => {
        assert.throws(
            () => parseLcc2Meta('{ not json', 'bad.lcc2'),
            /Failed to parse meta\.lcc2 as JSON: bad\.lcc2/
        );
    });

    it('throws when totalLevels is missing', () => {
        assert.throws(
            () => parseLcc2Meta(
                JSON.stringify({ totalSplats: 1, lodSplats: [1], root: { splatFiles: ['a.sog'] } }),
                'meta.lcc2'
            ),
            /missing or non-numeric totalLevels/
        );
    });

    it('throws when root.splatFiles is missing', () => {
        assert.throws(
            () => parseLcc2Meta(
                JSON.stringify({ totalSplats: 1, lodSplats: [1], totalLevels: 1, root: {} }),
                'meta.lcc2'
            ),
            /missing root\.splatFiles/
        );
    });

    it('throws when totalSplats is missing', () => {
        assert.throws(
            () => parseLcc2Meta(
                JSON.stringify({ lodSplats: [1], totalLevels: 1, root: { splatFiles: ['a.sog'] } }),
                'meta.lcc2'
            ),
            /missing or non-numeric totalSplats/
        );
    });

    it('throws when lodSplats is missing', () => {
        assert.throws(
            () => parseLcc2Meta(
                JSON.stringify({ totalSplats: 1, totalLevels: 1, root: { splatFiles: ['a.sog'] } }),
                'meta.lcc2'
            ),
            /missing lodSplats/
        );
    });
});

describe('getChildren', () => {
    it('returns [] when child is absent', () => {
        assert.deepStrictEqual(getChildren({}), []);
    });

    it('returns array children as-is', () => {
        const a = { id: 0 };
        const b = { id: 1 };
        assert.deepStrictEqual(getChildren({ child: [a, b] }), [a, b]);
    });

    it('returns values of an object-map child', () => {
        const a = { id: 0 };
        const b = { id: 1 };
        assert.deepStrictEqual(getChildren({ child: { 0: a, 1: b } }), [a, b]);
    });
});

describe('collectChunksByLevel', () => {
    // depth counts from 1 at the root's children; level = totalLevels - depth.
    // totalLevels = 2: depth 1 -> level 1, depth 2 -> level 0 (finest).
    const tree = {
        child: [
            {
                data: { '3dgs': { name: 0 } },
                child: [
                    { data: { '3dgs': { name: 1 } } },
                    { data: { '3dgs': { name: 2 } } }
                ]
            }
        ]
    };

    it('collects every level in a single traversal', () => {
        const byLevel = collectChunksByLevel(tree, 2, undefined);
        assert.deepStrictEqual(
            [...byLevel.get(0).keys()].sort((a, b) => a - b),
            [1, 2]
        );
        assert.deepStrictEqual([...byLevel.get(1).keys()], [0]);
    });

    it('skips the environment file index', () => {
        const byLevel = collectChunksByLevel(tree, 2, 2);
        assert.deepStrictEqual([...byLevel.get(0).keys()], [1]);
    });

    it('handles object-map children', () => {
        const mapTree = { child: { 0: { data: { '3dgs': { name: 7 } } } } };
        const byLevel = collectChunksByLevel(mapTree, 1, undefined);
        assert.deepStrictEqual([...byLevel.get(0).keys()], [7]);
    });

    it('has no entry for levels without chunks', () => {
        const byLevel = collectChunksByLevel(tree, 2, undefined);
        assert.strictEqual(byLevel.get(5), undefined);
    });

    it('sums per-file counts across nodes sharing a chunk file', () => {
        // Real LCC2 metas share one chunk file between several same-level
        // nodes, each owning a [start, start + count) row range.
        const shared = {
            child: [
                { data: { '3dgs': { name: 0, start: 0, count: 5 } } },
                { data: { '3dgs': { name: 0, start: 5, count: 7 } } },
                { data: { '3dgs': { name: 1, count: 3 } } }
            ]
        };
        const byLevel = collectChunksByLevel(shared, 1, undefined);
        assert.strictEqual(byLevel.get(0).get(0), 12);
        assert.strictEqual(byLevel.get(0).get(1), 3);
    });

    it('marks a file count unknown when any node lacks a valid count', () => {
        const partial = {
            child: [
                { data: { '3dgs': { name: 0, count: 5 } } },
                { data: { '3dgs': { name: 0 } } }, // no count -> file 0 unknown
                { data: { '3dgs': { name: 1, count: -2 } } } // invalid count
            ]
        };
        const byLevel = collectChunksByLevel(partial, 1, undefined);
        assert.strictEqual(byLevel.get(0).has(0), true);
        assert.strictEqual(byLevel.get(0).get(0), undefined);
        assert.strictEqual(byLevel.get(0).get(1), undefined);

        // An uncounted node arriving before a counted one must also poison.
        const reversed = {
            child: [
                { data: { '3dgs': { name: 0 } } },
                { data: { '3dgs': { name: 0, count: 5 } } }
            ]
        };
        assert.strictEqual(collectChunksByLevel(reversed, 1, undefined).get(0).get(0), undefined);
    });
});

describe('resolveLodSelection', () => {
    it('returns all levels for an empty selection', () => {
        assert.deepStrictEqual(resolveLodSelection([], 3), [0, 1, 2]);
    });

    it('maps negative indices from the end', () => {
        assert.deepStrictEqual(resolveLodSelection([-1], 3), [2]);
    });

    it('filters out-of-range indices', () => {
        assert.deepStrictEqual(resolveLodSelection([0, 3, -4], 3), [0]);
    });

    it('preserves order and duplicates of the selection', () => {
        assert.deepStrictEqual(resolveLodSelection([2, 0], 3), [2, 0]);
    });
});

// --- Main readLcc2 flow -----------------------------------------------------

// Build an in-memory .spz chunk with the given number of splats. The default
// version (4) emits a plaintext NGSP header; version 3 emits the legacy
// gzip-wrapped container.
const makeSpzBytes = async (count, { version, includeSH } = {}) => {
    const writeFs = new MemoryFileSystem();
    await writeSpz(
        {
            filename: 'chunk.spz',
            dataTable: createTestDataTable(count, includeSH ? { includeSH, shBands: 3 } : {}),
            ...(version ? { version } : {})
        },
        writeFs
    );
    return writeFs.results.get('chunk.spz');
};

// Build an in-memory bundled .sog chunk with the given number of splats.
const makeSogBytes = async (count) => {
    const writeFs = new MemoryFileSystem();
    await writeSog(
        { filename: 'chunk.sog', dataTable: createTestDataTable(count), bundle: true, iterations: 5 },
        writeFs
    );
    return writeFs.results.get('chunk.sog');
};

// Default options for readLcc2 (only lodSelect matters here).
const opts = (lodSelect = []) => ({ lodSelect });

describe('readLcc2 (main flow)', () => {
    it('orders chunks by LOD then ascending file index and tags the lod column', async () => {
        // totalLevels = 2. LOD 0 (finest) = depth-2 nodes (files 2, 3);
        // LOD 1 = depth-1 node (file 1). Distinct splat counts let us verify
        // both ordering and the per-chunk lod tag.
        const fs = new MemoryReadFileSystem();
        fs.set('chunk1.spz', await makeSpzBytes(5)); // LOD 1
        fs.set('chunk2.spz', await makeSpzBytes(3)); // LOD 0
        fs.set('chunk3.spz', await makeSpzBytes(4)); // LOD 0

        const meta = {
            totalSplats: 12,
            lodSplats: [7, 5],
            totalLevels: 2,
            splatType: '.spz',
            root: {
                splatFiles: ['', 'chunk1.spz', 'chunk2.spz', 'chunk3.spz'],
                child: [
                    {
                        data: { '3dgs': { name: 1 } },
                        child: [
                            { data: { '3dgs': { name: 2 } } },
                            { data: { '3dgs': { name: 3 } } }
                        ]
                    }
                ]
            }
        };
        fs.set('meta.lcc2', new TextEncoder().encode(JSON.stringify(meta)));

        const result = await readLcc2(fs, 'meta.lcc2', opts());
        assert.strictEqual(result.length, 1);

        const merged = result[0];
        // 3 + 4 (LOD 0) + 5 (LOD 1) = 12 splats.
        assert.strictEqual(merged.numRows, 12);

        // Output order: LOD 0 first (files 2 then 3), then LOD 1 (file 1).
        const lod = merged.getColumnByName('lod').data;
        assert.deepStrictEqual(
            Array.from(lod),
            [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
        );
    });

    it('returns the environment chunk as a second table tagged lod = -1', async () => {
        const fs = new MemoryReadFileSystem();
        fs.set('main.spz', await makeSpzBytes(4));
        fs.set('env.spz', await makeSpzBytes(2));

        const meta = {
            totalSplats: 4,
            lodSplats: [4],
            totalLevels: 1,
            splatType: '.spz',
            root: {
                splatFiles: ['main.spz', 'env.spz'],
                data: { env: { name: 1 } },
                child: [{ data: { '3dgs': { name: 0 } } }]
            }
        };
        fs.set('meta.lcc2', new TextEncoder().encode(JSON.stringify(meta)));

        const result = await readLcc2(fs, 'meta.lcc2', opts());
        assert.strictEqual(result.length, 2);

        const env = result[1];
        assert.strictEqual(env.numRows, 2);
        assert.ok(Array.from(env.getColumnByName('lod').data).every(v => v === -1));
    });

    it('ignores a missing environment chunk without throwing', async () => {
        const fs = new MemoryReadFileSystem();
        fs.set('main.spz', await makeSpzBytes(4));
        // env.spz declared in meta but intentionally not provided.

        const meta = {
            totalSplats: 4,
            lodSplats: [4],
            totalLevels: 1,
            splatType: '.spz',
            root: {
                splatFiles: ['main.spz', 'env.spz'],
                data: { env: { name: 1 } },
                child: [{ data: { '3dgs': { name: 0 } } }]
            }
        };
        fs.set('meta.lcc2', new TextEncoder().encode(JSON.stringify(meta)));

        const result = await readLcc2(fs, 'meta.lcc2', opts());
        // Only the main table; the absent env chunk is treated as normal.
        assert.strictEqual(result.length, 1);
        assert.strictEqual(result[0].numRows, 4);
    });

    it('throws when the selected LODs contain no decodable chunks', async () => {
        const fs = new MemoryReadFileSystem();
        // Root has no '3dgs' nodes at all, so no chunk is collected.
        const meta = {
            totalSplats: 0,
            lodSplats: [0],
            totalLevels: 1,
            splatType: '.spz',
            root: { splatFiles: ['main.spz'], child: [] }
        };
        fs.set('meta.lcc2', new TextEncoder().encode(JSON.stringify(meta)));

        await assert.rejects(
            () => readLcc2(fs, 'meta.lcc2', opts()),
            /No chunks found for selected LODs/
        );
    });

    it('propagates a decode failure for a missing main chunk', async () => {
        const fs = new MemoryReadFileSystem();
        // meta references main.spz but the file is absent: a main-chunk read
        // failure must reject (unlike the optional environment chunk).
        const meta = {
            totalSplats: 4,
            lodSplats: [4],
            totalLevels: 1,
            splatType: '.spz',
            root: {
                splatFiles: ['main.spz'],
                child: [{ data: { '3dgs': { name: 0 } } }]
            }
        };
        fs.set('meta.lcc2', new TextEncoder().encode(JSON.stringify(meta)));

        await assert.rejects(() => readLcc2(fs, 'meta.lcc2', opts()));
    });
});

describe('parseSpzNumPoints', () => {
    it('parses numPoints from a v4 SPZ header', async () => {
        const bytes = await makeSpzBytes(7);
        assert.strictEqual(parseSpzNumPoints(bytes.subarray(0, 16), 'chunk.spz'), 7);
    });

    it('throws on garbage or truncated headers', () => {
        assert.throws(() => parseSpzNumPoints(new Uint8Array(16), 'x.spz'), /Invalid SPZ chunk header/);
        assert.throws(() => parseSpzNumPoints(new Uint8Array(4), 'x.spz'), /Invalid SPZ chunk header/);
    });
});

describe('readChunkCount', () => {
    it('reads the count from a v4 (plaintext header) SPZ chunk', async () => {
        const fs = new MemoryReadFileSystem();
        fs.set('chunk.spz', await makeSpzBytes(9));
        assert.strictEqual(await readChunkCount(fs, '.spz', 'chunk.spz'), 9);
    });

    it('reads the count from a v3 (gzip-wrapped) SPZ chunk', async () => {
        const fs = new MemoryReadFileSystem();
        const bytes = await makeSpzBytes(9, { version: 3 });
        assert.strictEqual(bytes[0], 0x1f); // sanity: legacy container is gzip
        fs.set('chunk.spz', bytes);
        assert.strictEqual(await readChunkCount(fs, '.spz', 'chunk.spz'), 9);
    });

    it('reads the count from a SOG chunk meta.json without decoding textures', async () => {
        const fs = new MemoryReadFileSystem();
        fs.set('chunk.sog', await makeSogBytes(9));
        assert.strictEqual(await readChunkCount(fs, '.sog', 'chunk.sog'), 9);
    });

    it('throws on an unsupported splatType', async () => {
        const fs = new MemoryReadFileSystem();
        fs.set('x.ply', new Uint8Array(4));
        await assert.rejects(() => readChunkCount(fs, '.ply', 'x.ply'), /Unsupported LCC2 splatType/);
    });
});

// Wraps a MemoryReadFileSystem, recording every createSource path. Used to
// prove the meta-count path opens each chunk exactly once (no fallback scan).
class RecordingFileSystem {
    constructor(inner) {
        this.inner = inner;
        this.attempts = [];
    }

    createSource(filename) {
        this.attempts.push(filename);
        return this.inner.createSource(filename);
    }
}

describe('readLcc2 (meta-provided counts)', () => {
    it('uses meta counts without extra chunk reads', async () => {
        const inner = new MemoryReadFileSystem();
        inner.set('chunk1.spz', await makeSpzBytes(5));
        inner.set('chunk2.spz', await makeSpzBytes(3));

        const meta = {
            totalSplats: 8,
            lodSplats: [8],
            totalLevels: 1,
            splatType: '.spz',
            root: {
                splatFiles: ['chunk1.spz', 'chunk2.spz'],
                child: [
                    { data: { '3dgs': { name: 0, start: 0, count: 5 } } },
                    { data: { '3dgs': { name: 1, start: 0, count: 3 } } }
                ]
            }
        };
        inner.set('meta.lcc2', new TextEncoder().encode(JSON.stringify(meta)));

        const fs = new RecordingFileSystem(inner);
        const result = await readLcc2(fs, 'meta.lcc2', opts());
        assert.strictEqual(result[0].numRows, 8);
        assert.deepStrictEqual(Array.from(result[0].getColumnByName('lod').data), [0, 0, 0, 0, 0, 0, 0, 0]);

        // Each chunk opened exactly once (decode only - no scanning pass).
        const chunkOpens = fs.attempts.filter(p => p.endsWith('.spz'));
        assert.deepStrictEqual(chunkOpens.sort(), ['chunk1.spz', 'chunk2.spz']);
    });

    it('rejects when a meta count disagrees with the decoded chunk', async () => {
        const fs = new MemoryReadFileSystem();
        fs.set('chunk1.spz', await makeSpzBytes(5));

        const meta = {
            totalSplats: 4,
            lodSplats: [4],
            totalLevels: 1,
            splatType: '.spz',
            root: {
                splatFiles: ['chunk1.spz'],
                child: [{ data: { '3dgs': { name: 0, start: 0, count: 4 } } }] // actual chunk has 5
            }
        };
        fs.set('meta.lcc2', new TextEncoder().encode(JSON.stringify(meta)));

        await assert.rejects(
            () => readLcc2(fs, 'meta.lcc2', opts()),
            /splat count mismatch .* expected 4, decoded 5/
        );
    });
});

describe('readLcc2 (chunk format coverage)', () => {
    it('decodes SOG chunks end-to-end', async () => {
        const fs = new MemoryReadFileSystem();
        fs.set('chunk1.sog', await makeSogBytes(5));
        fs.set('chunk2.sog', await makeSogBytes(3));

        const meta = {
            totalSplats: 8,
            lodSplats: [8],
            totalLevels: 1,
            splatType: '.sog',
            root: {
                splatFiles: ['chunk1.sog', 'chunk2.sog'],
                child: [
                    { data: { '3dgs': { name: 0 } } },
                    { data: { '3dgs': { name: 1 } } }
                ]
            }
        };
        fs.set('meta.lcc2', new TextEncoder().encode(JSON.stringify(meta)));

        const result = await readLcc2(fs, 'meta.lcc2', opts());
        const merged = result[0];
        assert.strictEqual(merged.numRows, 8);
        assert.ok(merged.getColumnByName('x'));
        assert.ok(Array.from(merged.getColumnByName('lod').data).every(v => v === 0));
    });

    it('decodes v3 (gzip-wrapped) SPZ chunks end-to-end', async () => {
        // No meta counts, so the scanning fallback exercises the gzip header
        // path on every chunk before decoding.
        const fs = new MemoryReadFileSystem();
        fs.set('chunk1.spz', await makeSpzBytes(5, { version: 3 }));
        fs.set('chunk2.spz', await makeSpzBytes(3, { version: 3 }));

        const meta = {
            totalSplats: 8,
            lodSplats: [8],
            totalLevels: 1,
            splatType: '.spz',
            root: {
                splatFiles: ['chunk1.spz', 'chunk2.spz'],
                child: [
                    { data: { '3dgs': { name: 0 } } },
                    { data: { '3dgs': { name: 1 } } }
                ]
            }
        };
        fs.set('meta.lcc2', new TextEncoder().encode(JSON.stringify(meta)));

        const result = await readLcc2(fs, 'meta.lcc2', opts());
        assert.strictEqual(result[0].numRows, 8);
    });

    it('zero-fills columns missing from some chunks (heterogeneous SH)', async () => {
        const fs = new MemoryReadFileSystem();
        fs.set('plain.spz', await makeSpzBytes(3));
        fs.set('sh.spz', await makeSpzBytes(4, { includeSH: true }));

        const meta = {
            totalSplats: 7,
            lodSplats: [7],
            totalLevels: 1,
            splatType: '.spz',
            root: {
                splatFiles: ['plain.spz', 'sh.spz'],
                child: [
                    { data: { '3dgs': { name: 0 } } },
                    { data: { '3dgs': { name: 1 } } }
                ]
            }
        };
        fs.set('meta.lcc2', new TextEncoder().encode(JSON.stringify(meta)));

        const result = await readLcc2(fs, 'meta.lcc2', opts());
        const merged = result[0];
        assert.strictEqual(merged.numRows, 7);

        const fRest = merged.getColumnByName('f_rest_0');
        assert.ok(fRest, 'merged table has SH columns');
        // Rows of the SH-less chunk (task order: plain.spz first) stay zero.
        assert.deepStrictEqual(Array.from(fRest.data.slice(0, 3)), [0, 0, 0]);
        // The SH chunk's region carries actual (quantized, nonzero) SH data
        // in at least one coefficient.
        let nonzero = false;
        for (let c = 0; c < 45 && !nonzero; c++) {
            const col = merged.getColumnByName(`f_rest_${c}`);
            nonzero = Array.from(col.data.slice(3)).some(v => v !== 0);
        }
        assert.ok(nonzero, 'SH chunk region contains nonzero SH data');
    });
});

describe('readLcc2 (file index validation)', () => {
    it('rejects an out-of-range chunk file index', async () => {
        const fs = new MemoryReadFileSystem();
        const meta = {
            totalSplats: 1,
            lodSplats: [1],
            totalLevels: 1,
            splatType: '.spz',
            root: {
                splatFiles: ['a.spz'],
                child: [{ data: { '3dgs': { name: 5 } } }]
            }
        };
        fs.set('meta.lcc2', new TextEncoder().encode(JSON.stringify(meta)));

        await assert.rejects(
            () => readLcc2(fs, 'meta.lcc2', opts()),
            /Invalid chunk file index 5/
        );
    });

    it('rejects an empty chunk file path', async () => {
        const fs = new MemoryReadFileSystem();
        const meta = {
            totalSplats: 1,
            lodSplats: [1],
            totalLevels: 1,
            splatType: '.spz',
            root: {
                splatFiles: [''],
                child: [{ data: { '3dgs': { name: 0 } } }]
            }
        };
        fs.set('meta.lcc2', new TextEncoder().encode(JSON.stringify(meta)));

        await assert.rejects(
            () => readLcc2(fs, 'meta.lcc2', opts()),
            /Invalid chunk file index 0/
        );
    });
});

// A ReadFileSystem stub whose createSource fails with a controllable error,
// recording which paths were attempted. Lets us verify the fallback policy
// without real files.
class FailingReadFileSystem {
    constructor(error, openable = new Map()) {
        this.error = error;
        this.openable = openable; // path -> Uint8Array that opens successfully
        this.attempts = [];
    }

    createSource(filename) {
        this.attempts.push(filename);
        const data = this.openable.get(filename);
        if (data) {
            return Promise.resolve({
                size: data.length,
                seekable: true,
                read: () => {
                    throw new Error('not used');
                },
                close: () => {}
            });
        }
        return Promise.reject(this.error);
    }
}

describe('isMissingError', () => {
    it('recognizes ENOENT, Entry not found and HTTP 404', () => {
        assert.strictEqual(isMissingError({ code: 'ENOENT' }), true);
        assert.strictEqual(isMissingError(new Error('Entry not found: a.spz')), true);
        assert.strictEqual(isMissingError(new Error('HTTP error 404')), true);
    });

    it('treats other errors as non-missing', () => {
        assert.strictEqual(isMissingError(new Error('EACCES: permission denied')), false);
        assert.strictEqual(isMissingError(new Error('HTTP error 500')), false);
        assert.strictEqual(isMissingError(undefined), false);
    });
});

describe('openChunkSource', () => {
    it('does not fall back to basename on a non-missing error', async () => {
        const err = new Error('EACCES: permission denied');
        const fs = new FailingReadFileSystem(err);

        await assert.rejects(() => openChunkSource(fs, 'dir/chunk.spz'), /EACCES/);
        // Only the full path is attempted; no basename retry on a real error.
        assert.deepStrictEqual(fs.attempts, ['dir/chunk.spz']);
    });

    it('falls back to basename only when the full path is missing', async () => {
        const data = new Uint8Array([1, 2, 3]);
        const fs = new FailingReadFileSystem(
            new Error('Entry not found: dir/chunk.spz'),
            new Map([['chunk.spz', data]])
        );

        const source = await openChunkSource(fs, 'dir/chunk.spz');
        assert.strictEqual(source.size, 3);
        // Full path tried first, then the bare filename.
        assert.deepStrictEqual(fs.attempts, ['dir/chunk.spz', 'chunk.spz']);
    });

    it('does not retry when the path has no directory component', async () => {
        const fs = new FailingReadFileSystem(new Error('Entry not found: chunk.spz'));

        await assert.rejects(() => openChunkSource(fs, 'chunk.spz'), /Entry not found/);
        // basename === fullPath, so no second attempt.
        assert.deepStrictEqual(fs.attempts, ['chunk.spz']);
    });
});

// Wraps a MemoryReadFileSystem and tracks how many createSource calls are
// in flight at once, so we can assert the worker pool honours its bound.
class ConcurrencyTrackingFileSystem {
    constructor(inner) {
        this.inner = inner;
        this.inFlight = 0;
        this.peak = 0;
    }

    async createSource(filename) {
        this.inFlight++;
        this.peak = Math.max(this.peak, this.inFlight);
        try {
            // Yield to the event loop so overlapping workers accumulate before
            // any single open resolves; this makes the peak observable.
            await new Promise((resolve) => {
                setTimeout(resolve, 5);
            });
            return await this.inner.createSource(filename);
        } finally {
            this.inFlight--;
        }
    }
}

describe('readLcc2 (bounded concurrency)', () => {
    it('never decodes more than LOAD_CONCURRENCY chunks at once', async () => {
        const inner = new MemoryReadFileSystem();
        // More chunks than the concurrency bound so the limit is exercised.
        const chunkCount = LOAD_CONCURRENCY * 2 + 1;
        const splatFiles = [];
        const child = [];
        // Build all chunk bytes in parallel to avoid awaiting inside the loop.
        const allBytes = await Promise.all(
            Array.from({ length: chunkCount }, () => makeSpzBytes(1))
        );
        for (let i = 0; i < chunkCount; i++) {
            const name = `chunk${i}.spz`;
            inner.set(name, allBytes[i]);
            splatFiles.push(name);
            child.push({ data: { '3dgs': { name: i } } });
        }

        const meta = {
            totalSplats: chunkCount,
            lodSplats: [chunkCount],
            totalLevels: 1,
            splatType: '.spz',
            root: { splatFiles, child }
        };
        inner.set('meta.lcc2', new TextEncoder().encode(JSON.stringify(meta)));

        const fs = new ConcurrencyTrackingFileSystem(inner);
        const result = await readLcc2(fs, 'meta.lcc2', opts());

        // All chunks decoded into a single combined table...
        assert.strictEqual(result[0].numRows, chunkCount);
        // ...but never more than the bound ran concurrently, and the pool did
        // actually run in parallel (peak > 1) rather than serially.
        assert.ok(
            fs.peak <= LOAD_CONCURRENCY,
            `peak concurrency ${fs.peak} exceeded bound ${LOAD_CONCURRENCY}`
        );
        assert.ok(fs.peak > 1, `expected parallel decoding, peak was ${fs.peak}`);
    });
});
