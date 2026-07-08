import { type ChunkSource, type ChunkSourceMetadata, type ReadRequest } from '../chunk';
import { type ReadSource, type ReadStream } from '../io/read';

/**
 * Read exactly `length` bytes from `stream` into `buffer` at `offset`. A stream
 * may satisfy a read in several pulls; returns the number of bytes actually read
 * (less than `length` only at EOF).
 *
 * @param stream - The stream to read from.
 * @param buffer - Destination buffer.
 * @param offset - Byte offset in `buffer` to write the first byte.
 * @param length - Number of bytes to read.
 * @returns The number of bytes read.
 */
const readExact = async (stream: ReadStream, buffer: Uint8Array, offset: number, length: number): Promise<number> => {
    let total = 0;
    while (total < length) {
        const n = await stream.pull(buffer.subarray(offset + total, offset + length));
        if (n === 0) break;
        total += n;
    }
    return total;
};

/** Gather coalescing: merge records separated by less than this many bytes. */
const GATHER_MERGE_GAP = 64 * 1024;
/** Gather coalescing: cap any one merged read (and the caller's scratch). */
const GATHER_MAX_WINDOW = 8 * 1024 * 1024;

/** One coalesced range read of a gather plan. */
type GatherRun = {
    /** Sorted-slot range [j0, j1) served by this window. */
    j0: number;
    j1: number;
    /** Byte offset of the window's first record in the primary file. */
    firstByte: number;
    /** Window length in whole records, gap records included. */
    recordCount: number;
};

/**
 * Sort gather output slots into source order so reads run strictly forward.
 *
 * @param count - Number of output slots.
 * @param keyOf - Maps an output slot to its source position (row or byte).
 * @returns Slot indices ordered by ascending `keyOf`.
 */
const sortGatherSlots = (count: number, keyOf: (slot: number) => number): Uint32Array => {
    const slot = new Uint32Array(count);
    for (let j = 0; j < count; j++) slot[j] = j;
    slot.sort((a, b) => keyOf(a) - keyOf(b));
    return slot;
};

/**
 * Plan coalesced range reads over a sorted gather. A positioned read costs
 * ~a syscall + await round-trip regardless of size, so records separated by
 * less than {@link GATHER_MERGE_GAP} bytes of unwanted data are cheaper to
 * read-and-skip than to fetch separately. The {@link GATHER_MAX_WINDOW} cap
 * bounds the caller's scratch and the worst-case skip waste, and splits
 * fully-consecutive runs that would otherwise grow the scratch without bound.
 *
 * A record joins a window only at a whole multiple of `recordBytes` from the
 * window start, so callers can locate record `t` in the window at integer
 * record index `(byteAt(t) - firstByte) / recordBytes` (readers whose records
 * live at per-unit file offsets merge across units only when those offsets
 * stay record-aligned).
 *
 * @param count - Number of records to gather.
 * @param byteAt - Byte offset of sorted record `t` in the primary file; must
 * be non-decreasing in `t`.
 * @param recordBytes - Record size in the primary file.
 * @param costBytes - Per-record byte cost for the gap/window accounting; the
 * sum of parallel record sizes when one plan drives range reads in several
 * files (e.g. data + SH), so the combined scratch stays bounded.
 * @yields One {@link GatherRun} per coalesced range read.
 */
function *gatherRuns(
    count: number,
    byteAt: (t: number) => number,
    recordBytes: number,
    costBytes = recordBytes
): Generator<GatherRun> {
    let j = 0;
    while (j < count) {
        const firstByte = byteAt(j);
        let lastByte = firstByte;
        let k = j + 1;
        while (k < count) {
            const nextByte = byteAt(k);
            if ((nextByte - lastByte) / recordBytes * costBytes > GATHER_MERGE_GAP) break;
            if (((nextByte - firstByte) / recordBytes + 1) * costBytes > GATHER_MAX_WINDOW) break;
            if ((nextByte - firstByte) % recordBytes !== 0) break;
            lastByte = nextByte;
            k++;
        }
        yield { j0: j, j1: k, firstByte, recordCount: (lastByte - firstByte) / recordBytes + 1 };
        j = k;
    }
}

/**
 * Build a {@link ChunkSource} backed by a {@link ReadSource}. `close()` releases
 * the source, so every file-backed reader owns its source's lifetime uniformly
 * (a reader can't forget to close it). Used by the lazy PLY/splat/SPZ readers.
 *
 * @param source - The read source whose lifetime this `ChunkSource` owns.
 * @param meta - The source metadata.
 * @param read - The read implementation (serves both chunk and gather requests).
 * @returns A `ChunkSource` whose `close()` closes `source`.
 */
const fileChunkSource = (
    source: ReadSource,
    meta: ChunkSourceMetadata,
    read: (request: ReadRequest) => Promise<void>
): ChunkSource => ({
    meta,
    read,
    close: () => {
        source.close();
        return Promise.resolve();
    }
});

export { readExact, fileChunkSource, sortGatherSlots, gatherRuns, type GatherRun };
