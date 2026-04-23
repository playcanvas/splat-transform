/**
 * Tests for reading splat files via http(s):// URLs through UrlReadFileSystem.
 *
 * Spins up an in-process http server serving fixture files in two modes:
 *  - Range-supporting (responds to Range headers with 206 Partial Content)
 *  - Range-ignoring (always returns the whole body with 200 OK)
 *
 * This validates both code paths in UrlReadFileSystem (streaming + memory fallback).
 */

import { describe, it, before } from 'node:test';
import assert from 'node:assert';
import http from 'node:http';
import { readFile as fsReadFile } from 'node:fs/promises';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

import {
    getInputFormat,
    readFile,
    UrlReadFileSystem
} from '../src/lib/index.js';

const __dirname = dirname(fileURLToPath(import.meta.url));
const fixturesDir = join(__dirname, 'fixtures', 'splat');

/**
 * Start an http server serving a fixed map of `pathname -> body`.
 *
 * Any request whose pathname is not present in `routes` returns 404. This
 * means tests fail loudly if URL resolution (baseUrl + leaf joining,
 * querystring stripping, sibling lookup, ...) targets the wrong path
 * rather than silently succeeding.
 *
 * @param {Record<string, Uint8Array>} routes - map of pathname (e.g. `/minimal.splat`) to body
 * @param {boolean} supportRange - if true, respect Range headers (206); else always 200
 * @returns {Promise<{ url: string, requests: string[], close: () => Promise<void> }>}
 */
const startServer = (routes, supportRange) => {
    return new Promise((resolve) => {
        const requests = [];
        const server = http.createServer((req, res) => {
            // Track the full request line so tests can assert on querystrings too.
            requests.push(req.url);

            // Match on pathname only; querystring/fragment are ignored for routing
            // (but the server records them via `requests` for assertions).
            const pathname = new URL(req.url, 'http://localhost').pathname;
            const body = routes[pathname];
            if (!body) {
                res.writeHead(404, { 'Content-Type': 'text/plain' });
                res.end(`Not Found: ${pathname}`);
                return;
            }

            const range = req.headers.range;
            if (supportRange && range) {
                const match = /^bytes=(\d+)-(\d*)$/.exec(range);
                if (match) {
                    const start = parseInt(match[1], 10);
                    const end = match[2] ? parseInt(match[2], 10) : body.length - 1;
                    const slice = body.subarray(start, end + 1);
                    res.writeHead(206, {
                        'Content-Type': 'application/octet-stream',
                        'Content-Length': String(slice.length),
                        'Content-Range': `bytes ${start}-${end}/${body.length}`,
                        'Accept-Ranges': 'bytes'
                    });
                    res.end(slice);
                    return;
                }
            }

            // Plain 200 OK with the entire body (no Range support).
            res.writeHead(200, {
                'Content-Type': 'application/octet-stream',
                'Content-Length': String(body.length)
            });
            res.end(body);
        });
        server.listen(0, '127.0.0.1', () => {
            const { port } = server.address();
            resolve({
                url: `http://127.0.0.1:${port}`,
                requests,
                close: () => new Promise(r => server.close(() => r()))
            });
        });
    });
};

describe('UrlReadFileSystem (CLI integration)', () => {
    let splatBody;
    let ksplatBody;

    before(async () => {
        splatBody = await fsReadFile(join(fixturesDir, 'minimal.splat'));
        ksplatBody = await fsReadFile(join(fixturesDir, 'minimal.ksplat'));
    });

    it('reads a .splat file via URL with Range support (streaming path)', async () => {
        const server = await startServer({ '/minimal.splat': splatBody }, true);
        try {
            const url = `${server.url}/minimal.splat`;
            const fileSystem = new UrlReadFileSystem();
            const tables = await readFile({
                filename: url,
                inputFormat: getInputFormat(url),
                options: {},
                params: [],
                fileSystem
            });
            assert.strictEqual(tables.length, 1);
            assert.strictEqual(tables[0].numRows, 4);
        } finally {
            await server.close();
        }
    });

    it('reads a .ksplat file via URL with no Range support (memory fallback)', async () => {
        const server = await startServer({ '/minimal.ksplat': ksplatBody }, false);
        try {
            const url = `${server.url}/minimal.ksplat`;
            const fileSystem = new UrlReadFileSystem();
            const tables = await readFile({
                filename: url,
                inputFormat: getInputFormat(url),
                options: {},
                params: [],
                fileSystem
            });
            assert.strictEqual(tables.length, 1);
            assert.strictEqual(tables[0].numRows, 4);
        } finally {
            await server.close();
        }
    });

    it('resolves leaf and sibling fetches via baseUrl', async () => {
        // Models the CLI's baseUrl + leaf-filename split used for multi-file
        // formats (SOG meta.json / LCC). Serves a primary leaf plus a sibling
        // file at distinct paths and asserts both resolve correctly via
        // `new URL(name, baseUrl)`.
        const siblingBody = new Uint8Array([1, 2, 3, 4, 5]);
        const server = await startServer({
            '/scene/minimal.splat': splatBody,
            '/scene/sibling.bin': siblingBody
        }, true);
        try {
            const baseUrl = `${server.url}/scene/`;
            const fileSystem = new UrlReadFileSystem(baseUrl);

            // Primary leaf read goes through readFile().
            const tables = await readFile({
                filename: 'minimal.splat',
                inputFormat: 'splat',
                options: {},
                params: [],
                fileSystem
            });
            assert.strictEqual(tables[0].numRows, 4);

            // Explicit sibling fetch via createSource() to prove relative
            // resolution (e.g. SOG fetching `means_l.webp` next to
            // `meta.json`, or LCC fetching chunk files).
            const siblingSource = await fileSystem.createSource('sibling.bin');
            try {
                const bytes = await siblingSource.read().readAll();
                assert.deepStrictEqual(Array.from(bytes), Array.from(siblingBody));
            } finally {
                siblingSource.close();
            }

            // The sibling request must have hit `/scene/sibling.bin`, not
            // `/sibling.bin` (which would 404). The 404 path would already
            // have failed the assertion above; this is a belt-and-braces
            // check that the server actually saw both expected pathnames.
            const pathnames = server.requests.map(u => new URL(u, 'http://localhost').pathname);
            assert.ok(pathnames.includes('/scene/minimal.splat'), `expected /scene/minimal.splat in ${JSON.stringify(pathnames)}`);
            assert.ok(pathnames.includes('/scene/sibling.bin'), `expected /scene/sibling.bin in ${JSON.stringify(pathnames)}`);
        } finally {
            await server.close();
        }
    });

    it('preserves URL querystring on the initial fetch (presigned URL use case)', async () => {
        // CLI splits `https://host/scene.splat?token=abc` into
        // baseUrl=`https://host/` and filename=`scene.splat?token=abc`. The
        // initial fetch must carry the token; the in-process server records
        // the raw request line so we can assert on it.
        const server = await startServer({ '/scene.splat': splatBody }, true);
        try {
            const baseUrl = `${server.url}/`;
            const fileSystem = new UrlReadFileSystem(baseUrl);
            const tables = await readFile({
                filename: 'scene.splat?token=abc',
                inputFormat: 'splat',
                options: {},
                params: [],
                fileSystem
            });
            assert.strictEqual(tables[0].numRows, 4);

            assert.ok(
                server.requests.some(u => u.includes('token=abc')),
                `expected querystring on at least one request, got ${JSON.stringify(server.requests)}`
            );
        } finally {
            await server.close();
        }
    });
});
