import { join } from 'pathe';

import { type BlockPlan } from './block-plan';
import { type ReadSource, type ReadStream } from '../io/read';
import { type FileSystem } from '../io/write';

const RECORD_BYTES = 12;
const CURSOR_RECORDS = 4096;

type PlanScratch = {
    writeFs: FileSystem;
    readFs: {
        createSource(filename: string): Promise<ReadSource>;
    };
    scratchDir: string;
    remove?: (path: string) => Promise<void>;
};

type StoredBlockPlan = {
    path: string;
    count: number;
    blockIndex: number;
};

type PlanRecord = {
    a: number;
    b: number;
    cost: number;
};

const encodePlan = (plan: BlockPlan): Uint8Array => {
    const count = plan.costs.length;
    const bytes = new Uint8Array(count * RECORD_BYTES);
    const view = new DataView(bytes.buffer);
    for (let i = 0; i < count; i++) {
        view.setUint32(i * RECORD_BYTES, plan.pairs[i * 2], true);
        view.setUint32(i * RECORD_BYTES + 4, plan.pairs[i * 2 + 1], true);
        view.setFloat32(i * RECORD_BYTES + 8, plan.costs[i], true);
    }
    return bytes;
};

/**
 * Persist one block-local plan, aborting without publishing on failure.
 *
 * @param scratch - Scratch filesystem pair.
 * @param generation - Generation number.
 * @param blockIndex - Block number.
 * @param plan - Local plan.
 * @returns Stored-plan descriptor.
 */
const storeBlockPlan = async (
    scratch: PlanScratch,
    generation: number,
    blockIndex: number,
    plan: BlockPlan
): Promise<StoredBlockPlan> => {
    const path = join(
        scratch.scratchDir,
        `.decimate-plan-g${generation}-b${blockIndex}-${Date.now().toString(36)}.tmp`
    );
    const writer = await scratch.writeFs.createWriter(path);
    try {
        await writer.write(encodePlan(plan));
        await writer.close();
    } catch (err) {
        await writer.abort();
        throw err;
    }
    return { path, count: plan.costs.length, blockIndex };
};

const pullExact = async (stream: ReadStream, target: Uint8Array): Promise<number> => {
    let read = 0;
    while (read < target.length) {
        const n = await stream.pull(target.subarray(read));
        if (n === 0) break;
        read += n;
    }
    return read;
};

class PlanCursor {
    private readonly source: ReadSource;
    private readonly stream: ReadStream;
    private readonly buffer = new Uint8Array(CURSOR_RECORDS * RECORD_BYTES);
    private buffered = 0;
    private offset = 0;
    private consumed = 0;

    constructor(source: ReadSource, readonly plan: StoredBlockPlan) {
        this.source = source;
        this.stream = source.read();
    }

    async next(): Promise<PlanRecord | null> {
        if (this.consumed === this.plan.count) return null;
        if (this.offset === this.buffered) {
            const remaining = (this.plan.count - this.consumed) * RECORD_BYTES;
            this.buffered = await pullExact(this.stream, this.buffer.subarray(0, Math.min(this.buffer.length, remaining)));
            this.offset = 0;
            if (this.buffered < RECORD_BYTES) throw new Error(`truncated decimation plan '${this.plan.path}'`);
        }
        const view = new DataView(this.buffer.buffer, this.buffer.byteOffset + this.offset, RECORD_BYTES);
        const record = {
            a: view.getUint32(0, true),
            b: view.getUint32(4, true),
            cost: view.getFloat32(8, true)
        };
        this.offset += RECORD_BYTES;
        this.consumed++;
        return record;
    }

    close(): void {
        this.stream.close();
        this.source.close();
    }
}

type HeapEntry = PlanRecord & {
    cursor: PlanCursor;
};

/**
 * Exact k-way allocation over block-plan prefixes. Only each block's next
 * commit is exposed, so non-monotonic continuation costs and prefix
 * dependencies are preserved.
 *
 * @param plans - Stored block plans.
 * @param scratch - Scratch read filesystem.
 * @param needed - Exact generation removal quota.
 * @param onSelect - Optional observation hook for proof/reference tests.
 * @returns Selected prefix per block and achieved removals.
 */
const allocatePlanPrefixes = async (
    plans: StoredBlockPlan[],
    scratch: PlanScratch,
    needed: number,
    onSelect?: (blockIndex: number, localIndex: number, record: PlanRecord) => void
): Promise<{ prefixes: Uint32Array; removed: number }> => {
    const prefixes = new Uint32Array(plans.length);
    const heap: HeapEntry[] = [];
    const cursors: PlanCursor[] = [];
    const less = (a: HeapEntry, b: HeapEntry): boolean => a.cost < b.cost ||
        (a.cost === b.cost && (a.cursor.plan.blockIndex < b.cursor.plan.blockIndex ||
            (a.cursor.plan.blockIndex === b.cursor.plan.blockIndex && (a.a < b.a || (a.a === b.a && a.b < b.b)))));
    const push = (entry: HeapEntry): void => {
        let i = heap.length;
        heap.push(entry);
        while (i > 0) {
            const p = (i - 1) >> 1;
            if (!less(heap[i], heap[p])) break;
            [heap[i], heap[p]] = [heap[p], heap[i]];
            i = p;
        }
    };
    const pop = (): HeapEntry => {
        const out = heap[0];
        const tail = heap.pop()!;
        if (heap.length > 0) {
            heap[0] = tail;
            let i = 0;
            for (;;) {
                const l = i * 2 + 1;
                const r = l + 1;
                let m = i;
                if (l < heap.length && less(heap[l], heap[m])) m = l;
                if (r < heap.length && less(heap[r], heap[m])) m = r;
                if (m === i) break;
                [heap[i], heap[m]] = [heap[m], heap[i]];
                i = m;
            }
        }
        return out;
    };

    try {
        for (const plan of plans) {
            if (plan.count === 0) continue;
            const source = await scratch.readFs.createSource(plan.path);
            const cursor = new PlanCursor(source, plan);
            cursors.push(cursor);
            const first = await cursor.next();
            if (first) push({ ...first, cursor });
        }

        let removed = 0;
        while (removed < needed && heap.length > 0) {
            const entry = pop();
            onSelect?.(
                entry.cursor.plan.blockIndex,
                prefixes[entry.cursor.plan.blockIndex],
                entry
            );
            prefixes[entry.cursor.plan.blockIndex]++;
            removed++;
            const next = await entry.cursor.next();
            if (next) push({ ...next, cursor: entry.cursor });
        }
        return { prefixes, removed };
    } finally {
        for (const cursor of cursors) cursor.close();
    }
};

/**
 * Read one selected block prefix for replay.
 *
 * @param stored - Stored plan.
 * @param scratch - Scratch read filesystem.
 * @param count - Selected prefix length.
 * @returns Compact plan prefix.
 */
const readBlockPlanPrefix = async (
    stored: StoredBlockPlan,
    scratch: PlanScratch,
    count: number
): Promise<BlockPlan> => {
    if (count < 0 || count > stored.count) throw new Error('invalid block plan prefix');
    const pairs = new Uint32Array(count * 2);
    const costs = new Float32Array(count);
    if (count === 0) return { pairs, costs, frozen: 0, unfrozen: 0 };
    const source = await scratch.readFs.createSource(stored.path);
    const stream = source.read(0, count * RECORD_BYTES);
    try {
        const bytes = new Uint8Array(count * RECORD_BYTES);
        if (await pullExact(stream, bytes) !== bytes.length) throw new Error(`truncated decimation plan '${stored.path}'`);
        const view = new DataView(bytes.buffer);
        for (let i = 0; i < count; i++) {
            pairs[i * 2] = view.getUint32(i * RECORD_BYTES, true);
            pairs[i * 2 + 1] = view.getUint32(i * RECORD_BYTES + 4, true);
            costs[i] = view.getFloat32(i * RECORD_BYTES + 8, true);
        }
        return { pairs, costs, frozen: 0, unfrozen: 0 };
    } finally {
        stream.close();
        source.close();
    }
};

export {
    allocatePlanPrefixes,
    readBlockPlanPrefix,
    storeBlockPlan,
    type PlanScratch,
    type StoredBlockPlan
};
