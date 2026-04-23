/**
 * Tests for reading splat files via http(s):// URLs through UrlReadFileSystem.
 *
 * Spins up an in-process http server serving fixture files in two modes:
 *  - Range-supporting (responds to Range headers with 206 Partial Content)
 *  - Range-ignoring (always returns the whole body with 200 OK)
 *
 * This validates both code paths in UrlReadFileSystem (streaming + memory fallback).
 */

import { describe, it, before, after } from 'node:test';
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
 * Start an http server serving a single fixture file at /fixture.<ext>.
 * @param {Uint8Array} body - file body to serve
 * @param {boolean} supportRange - if true, respect Range headers (206); else always 200
 * @returns {Promise<{ url: string, close: () => Promise<void> }>}
 */
const startServer = (body, supportRange) => {
    return new Promise((resolve) => {
        const server = http.createServer((req, res) => {
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
        const server = await startServer(splatBody, true);
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
        const server = await startServer(ksplatBody, false);
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

    it('resolves sibling fetches via baseUrl for multi-file scenarios', async () => {
        // Use a baseUrl + leaf-filename split (the same pattern used by the CLI
        // for multi-file formats like SOG meta.json / LCC).
        const server = await startServer(splatBody, true);
        try {
            const baseUrl = `${server.url}/`;
            const fileSystem = new UrlReadFileSystem(baseUrl);
            const tables = await readFile({
                filename: 'minimal.splat',
                inputFormat: 'splat',
                options: {},
                params: [],
                fileSystem
            });
            assert.strictEqual(tables[0].numRows, 4);
        } finally {
            await server.close();
        }
    });
});
