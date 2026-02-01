/**
 * Tests for BufferedReadStream.
 */

import { describe, it } from 'node:test';
import assert from 'node:assert';

import { ReadStream, BufferedReadStream } from '../dist/index.mjs';

/**
 * Mock ReadStream that reads from a buffer and tracks pull() calls.
 */
class MockReadStream extends ReadStream {
    constructor(data) {
        super(data.length);
        this.data = data;
        this.offset = 0;
        this.pullCount = 0;
        this.pullSizes = [];
    }

    async pull(target) {
        this.pullCount++;
        this.pullSizes.push(target.length);

        const remaining = this.data.length - this.offset;
        if (remaining <= 0) return 0;

        const bytesToCopy = Math.min(target.length, remaining);
        target.set(this.data.subarray(this.offset, this.offset + bytesToCopy));
        this.offset += bytesToCopy;
        this.bytesRead += bytesToCopy;
        return bytesToCopy;
    }
}

describe('BufferedReadStream', () => {
    it('should read all data correctly', async () => {
        const testData = new Uint8Array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
        const inner = new MockReadStream(testData);
        const buffered = new BufferedReadStream(inner, 4);

        const result = await buffered.readAll();

        assert.deepStrictEqual(result, testData);
        assert.strictEqual(buffered.bytesRead, testData.length);
    });

    it('should buffer data and reduce pull() calls', async () => {
        // 100 bytes of data
        const testData = new Uint8Array(100);
        for (let i = 0; i < 100; i++) testData[i] = i;

        const inner = new MockReadStream(testData);
        // Use 50 byte chunks
        const buffered = new BufferedReadStream(inner, 50);

        // Read 10 bytes at a time (10 reads)
        const results = [];
        for (let i = 0; i < 10; i++) {
            const buf = new Uint8Array(10);
            const n = await buffered.pull(buf);
            assert.strictEqual(n, 10);
            results.push(...buf);
        }

        // Verify all data was read correctly
        assert.deepStrictEqual(new Uint8Array(results), testData);

        // Inner stream should have been called only twice (2 x 50 bytes)
        // not 10 times (10 x 10 bytes)
        assert.strictEqual(inner.pullCount, 2, 'Should buffer reads to reduce pull() calls');
    });

    it('should request at least chunkSize bytes from inner stream', async () => {
        const testData = new Uint8Array(100);
        const inner = new MockReadStream(testData);
        const chunkSize = 32;
        const buffered = new BufferedReadStream(inner, chunkSize);

        // Request only 4 bytes
        const buf = new Uint8Array(4);
        await buffered.pull(buf);

        // Inner stream should have been asked for at least chunkSize bytes
        assert.strictEqual(inner.pullSizes[0], chunkSize);
    });

    it('should request larger than chunkSize if target is larger', async () => {
        const testData = new Uint8Array(100);
        const inner = new MockReadStream(testData);
        const chunkSize = 16;
        const buffered = new BufferedReadStream(inner, chunkSize);

        // Request 50 bytes (larger than chunkSize)
        const buf = new Uint8Array(50);
        await buffered.pull(buf);

        // Inner stream should have been asked for 50 bytes (the larger value)
        assert.strictEqual(inner.pullSizes[0], 50);
    });

    it('should handle empty stream', async () => {
        const inner = new MockReadStream(new Uint8Array(0));
        const buffered = new BufferedReadStream(inner, 16);

        const buf = new Uint8Array(10);
        const n = await buffered.pull(buf);

        assert.strictEqual(n, 0);
        assert.strictEqual(buffered.bytesRead, 0);
    });

    it('should handle stream smaller than chunk size', async () => {
        const testData = new Uint8Array([1, 2, 3]);
        const inner = new MockReadStream(testData);
        const buffered = new BufferedReadStream(inner, 100); // chunkSize > data

        const result = await buffered.readAll();

        assert.deepStrictEqual(result, testData);
        // Two calls: one to get data, one to confirm EOF (readAll grows buffer and retries)
        assert.strictEqual(inner.pullCount, 2);
    });

    it('should serve subsequent reads from buffer', async () => {
        const testData = new Uint8Array([1, 2, 3, 4, 5, 6, 7, 8]);
        const inner = new MockReadStream(testData);
        const buffered = new BufferedReadStream(inner, 8);

        // First read: 2 bytes
        const buf1 = new Uint8Array(2);
        const n1 = await buffered.pull(buf1);
        assert.strictEqual(n1, 2);
        assert.deepStrictEqual(buf1, new Uint8Array([1, 2]));

        // Second read: 2 bytes (should come from buffer)
        const buf2 = new Uint8Array(2);
        const n2 = await buffered.pull(buf2);
        assert.strictEqual(n2, 2);
        assert.deepStrictEqual(buf2, new Uint8Array([3, 4]));

        // Only one pull to inner stream
        assert.strictEqual(inner.pullCount, 1);
    });

    it('should handle read spanning buffer boundary', async () => {
        const testData = new Uint8Array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
        const inner = new MockReadStream(testData);
        const buffered = new BufferedReadStream(inner, 4);

        // Read 6 bytes (spans first chunk of 4)
        const buf = new Uint8Array(6);
        const n = await buffered.pull(buf);

        // Should get first 4 from first chunk, then request more
        assert.strictEqual(n, 6);
        assert.deepStrictEqual(buf, new Uint8Array([1, 2, 3, 4, 5, 6]));
    });

    it('should track bytesRead correctly', async () => {
        const testData = new Uint8Array(50);
        const inner = new MockReadStream(testData);
        const buffered = new BufferedReadStream(inner, 20);

        const buf = new Uint8Array(10);
        await buffered.pull(buf);
        assert.strictEqual(buffered.bytesRead, 10);

        await buffered.pull(buf);
        assert.strictEqual(buffered.bytesRead, 20);

        await buffered.pull(buf);
        assert.strictEqual(buffered.bytesRead, 30);
    });

    it('should close inner stream on close()', async () => {
        let innerClosed = false;
        const inner = new MockReadStream(new Uint8Array(10));
        inner.close = () => { innerClosed = true; };

        const buffered = new BufferedReadStream(inner, 4);
        buffered.close();

        assert.strictEqual(innerClosed, true);
    });

    it('should pass through expectedSize', () => {
        const inner = new MockReadStream(new Uint8Array(42));
        const buffered = new BufferedReadStream(inner, 16);

        assert.strictEqual(buffered.expectedSize, 42);
    });

    it('should use default chunkSize of 64KB', async () => {
        const testData = new Uint8Array(1000);
        const inner = new MockReadStream(testData);
        const buffered = new BufferedReadStream(inner); // No chunkSize specified

        const buf = new Uint8Array(10);
        await buffered.pull(buf);

        // Default is 64KB, but we only have 1000 bytes so it should request 64KB
        // but only get 1000 bytes back
        assert.strictEqual(inner.pullSizes[0], 65536);
    });
});
