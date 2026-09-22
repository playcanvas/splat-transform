import { type BlockPlan } from './block-plan';

type PlanRecord = {
    a: number;
    b: number;
    cost: number;
};

type HeapEntry = PlanRecord & {
    block: number;
    index: number;
};

/**
 * Exact k-way allocation over block-plan prefixes. Only each block's next
 * commit is exposed, so non-monotonic continuation costs and prefix
 * dependencies are preserved.
 *
 * @param plans - Block-local plans, indexed by block.
 * @param needed - Exact generation removal quota.
 * @param onSelect - Optional observation hook for proof/reference tests.
 * @returns Selected prefix per block and achieved removals.
 */
const allocatePlanPrefixes = (
    plans: BlockPlan[],
    needed: number,
    onSelect?: (blockIndex: number, localIndex: number, record: PlanRecord) => void
): { prefixes: Uint32Array; removed: number } => {
    const prefixes = new Uint32Array(plans.length);
    const heap: HeapEntry[] = [];
    const less = (a: HeapEntry, b: HeapEntry): boolean => a.cost < b.cost ||
        (a.cost === b.cost && (a.block < b.block ||
            (a.block === b.block && (a.a < b.a || (a.a === b.a && a.b < b.b)))));
    const entryAt = (block: number, index: number): HeapEntry => {
        const plan = plans[block];
        return { block, index, a: plan.pairs[index * 2], b: plan.pairs[index * 2 + 1], cost: plan.costs[index] };
    };
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

    for (let block = 0; block < plans.length; block++) {
        if (plans[block].costs.length > 0) push(entryAt(block, 0));
    }

    let removed = 0;
    while (removed < needed && heap.length > 0) {
        const entry = pop();
        onSelect?.(entry.block, entry.index, entry);
        prefixes[entry.block]++;
        removed++;
        const next = entry.index + 1;
        if (next < plans[entry.block].costs.length) push(entryAt(entry.block, next));
    }
    return { prefixes, removed };
};

/**
 * One selected block prefix for replay.
 *
 * @param plan - Block-local plan.
 * @param count - Selected prefix length.
 * @returns The plan's leading `count` commits (views, not copies).
 */
const blockPlanPrefix = (plan: BlockPlan, count: number): BlockPlan => {
    if (count < 0 || count > plan.costs.length) throw new Error('invalid block plan prefix');
    return {
        pairs: plan.pairs.subarray(0, count * 2),
        costs: plan.costs.subarray(0, count),
        frozen: 0,
        unfrozen: 0
    };
};

export { allocatePlanPrefixes, blockPlanPrefix };
