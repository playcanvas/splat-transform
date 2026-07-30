import { readBlockPlanPrefix, type PlanScratch, type StoredBlockPlan } from './block-allocation';
import { replayBlockPlan } from './block-plan';
import { type DestBuffers } from './block-producer';
import { type ResidentPositions } from './partition';
import { gatherBlockView, type PriorityContext } from './priority';
import { WorkerQueue } from '../workers';

type BlockMergeStreamContext = Pick<PriorityContext, 'source' | 'pool' | 'pos' | 'order' | 'blocks'> & {
    plans: StoredBlockPlan[];
    prefixes: Uint32Array;
    scratch: PlanScratch;
    nextPositions?: ResidentPositions;
};

/**
 * Replay selected plans and emit one core block at a time. Selection arrays,
 * gathered fields, and moment-matching inputs are all block-local.
 *
 * @param ctx - Generation inputs and selected plan prefixes.
 * @param chunkSize - Destination chunk size.
 * @param tick - Progress callback.
 * @yields Filled destination row counts.
 */
async function *blockPlanMergeStream(
    ctx: BlockMergeStreamContext,
    chunkSize: number,
    tick?: (n: number) => void
): AsyncGenerator<number, void, DestBuffers> {
    const { source, pos, order, blocks, plans, prefixes, scratch, nextPositions } = ctx;
    const { layouts, availableLayers } = source.meta;
    const colorDim = layouts.color!.stride >> 2;
    const hasOther = availableLayers.has('other') && (layouts.other?.stride ?? 0) > 0;
    const otherDim = hasOther ? layouts.other!.stride >> 2 : 0;

    let rows = 0;
    let emitted = 0;
    let dest = yield 0;

    for (let bi = 0; bi < blocks.length; bi++) {
        const owned = order.subarray(blocks[bi].start, blocks[bi].end);
        const prefix = await readBlockPlanPrefix(plans[bi], scratch, prefixes[bi]);
        const selection = replayBlockPlan(owned.length, prefix);
        const { memberGroup, groupMin, groupOffsets, groupMembers } = selection;
        const { view, other } = await gatherBlockView(ctx, bi, new Uint32Array(0), hasOther);

        let mergedPos: Float32Array | null = null;
        let mergedGeo: Float32Array | null = null;
        let mergedColor: Float32Array | null = null;
        let mergedOther: Uint32Array | undefined;
        if (selection.mergedGroups > 0) {
            const totalMembers = groupMembers.length;
            const mPos = new Float32Array(totalMembers * 3);
            const mGeo = new Float32Array(totalMembers * 8);
            const mColor = new Float32Array(totalMembers * colorDim);
            const mOther = hasOther ? new Uint32Array(totalMembers * otherDim) : undefined;
            const sizes = new Uint32Array(selection.mergedGroups);
            let at = 0;
            for (let g = 0; g < selection.mergedGroups; g++) {
                sizes[g] = groupOffsets[g + 1] - groupOffsets[g];
                for (let m = groupOffsets[g]; m < groupOffsets[g + 1]; m++) {
                    const row = groupMembers[m];
                    mPos.set(view.pos.subarray(row * 3, row * 3 + 3), at * 3);
                    mGeo.set(view.geo.subarray(row * 8, row * 8 + 8), at * 8);
                    mColor.set(view.color.subarray(row * colorDim, (row + 1) * colorDim), at * colorDim);
                    if (mOther) mOther.set(other!.subarray(row * otherDim, (row + 1) * otherDim), at * otherDim);
                    at++;
                }
            }
            const transfer: ArrayBuffer[] = [
                mPos.buffer as ArrayBuffer,
                mGeo.buffer as ArrayBuffer,
                mColor.buffer as ArrayBuffer
            ];
            if (mOther) transfer.push(mOther.buffer as ArrayBuffer);
            const merged = await WorkerQueue.run('mergeGroups', {
                pos: mPos,
                geo: mGeo,
                color: mColor,
                sizes,
                colorDim,
                other: mOther,
                otherDim
            }, transfer);
            mergedPos = merged.pos;
            mergedGeo = merged.geo;
            mergedColor = merged.color;
            mergedOther = merged.other;
        }

        let nextMerged = 0;
        for (let i = 0; i < owned.length; i++) {
            const group = memberGroup[i];
            if (group !== -1 && groupMin[group] !== i) continue;

            let px: number, py: number, pz: number;
            if (group === -1) {
                px = pos.x[owned[i]];
                py = pos.y[owned[i]];
                pz = pos.z[owned[i]];
                if (dest.geometric) dest.geometric.set(view.geo.subarray(i * 8, i * 8 + 8), rows * 8);
                if (dest.color) dest.color.set(view.color.subarray(i * colorDim, (i + 1) * colorDim), rows * colorDim);
                if (dest.other) dest.other.set(other!.subarray(i * otherDim, (i + 1) * otherDim), rows * otherDim);
            } else {
                const mi = nextMerged++;
                px = mergedPos![mi * 3];
                py = mergedPos![mi * 3 + 1];
                pz = mergedPos![mi * 3 + 2];
                if (dest.geometric) dest.geometric.set(mergedGeo!.subarray(mi * 8, mi * 8 + 8), rows * 8);
                if (dest.color) dest.color.set(mergedColor!.subarray(mi * colorDim, (mi + 1) * colorDim), rows * colorDim);
                if (dest.other) dest.other.set(mergedOther!.subarray(mi * otherDim, (mi + 1) * otherDim), rows * otherDim);
            }
            if (dest.position) {
                dest.position[rows * 3] = px;
                dest.position[rows * 3 + 1] = py;
                dest.position[rows * 3 + 2] = pz;
            }
            if (nextPositions) {
                nextPositions.x[emitted] = px;
                nextPositions.y[emitted] = py;
                nextPositions.z[emitted] = pz;
            }
            rows++;
            emitted++;
            if (rows === chunkSize) {
                dest = yield rows;
                rows = 0;
            }
        }
        tick?.(owned.length);
    }
    if (rows > 0) yield rows;
}

export { blockPlanMergeStream, type BlockMergeStreamContext };
