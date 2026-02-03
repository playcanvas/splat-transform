/**
 * Integration tests for progress callbacks in file system implementations.
 */

import { describe, it } from 'node:test';
import assert from 'node:assert';

import { MemoryReadFileSystem } from '../dist/index.mjs';

describe('Progress Callbacks', () => {

    describe('MemoryReadFileSystem', () => {
        it('should report complete progress immediately on createSource', async () => {
            const testData = new Uint8Array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
            const fs = new MemoryReadFileSystem();
            fs.set('test.bin', testData);

            const progressCalls = [];
            const progress = (loaded, total) => {
                progressCalls.push({ loaded, total });
            };

            await fs.createSource('test.bin', progress);

            // Should report 100% immediately since data is in memory
            assert.strictEqual(progressCalls.length, 1, 'Should have exactly one progress call');
            assert.strictEqual(progressCalls[0].loaded, testData.length, 'Should report full size loaded');
            assert.strictEqual(progressCalls[0].total, testData.length, 'Should report correct total');
        });

        it('should work without progress callback', async () => {
            const testData = new Uint8Array([1, 2, 3, 4, 5]);
            const fs = new MemoryReadFileSystem();
            fs.set('test.bin', testData);

            // Should not throw when no progress callback provided
            const source = await fs.createSource('test.bin');
            assert.strictEqual(source.size, testData.length);
            source.close();
        });
    });
});
