import { type ChunkPayload } from './block-producer';
import { type ResidentPositions } from './partition';
import { gatherBlockView, indexOfSorted, type PriorityContext } from './priority';
import { type SelectionResult } from './select';
import { WorkerQueue } from '../workers';

/** Context for the merge stream: the priority context plus the selection. */
type MergeStreamContext = Pick<PriorityContext, 'source' | 'pool' | 'pos' | 'order' | 'blocks'> & {
    selection: SelectionResult;
    /** When provided (sized to the output count), filled with output positions in emission order — the next generation's resident positions. */
    nextPositions?: ResidentPositions;
};

/**
 * The merge stream (heavy read 2): walk blocks in partition order, gather
 * geometric/color/(other) for owned rows + out-of-block group members,
 * moment-match groups in workers, pass survivors through, and emit output
 * rows in block order as chunk payloads of `chunkSize` rows (last partial).
 *
 * A group is emitted exactly once, at its minimum member's position; other
 * members are consumed silently. Positions are never gathered — survivor
 * positions and merged means come from the resident arrays / the merge.
 *
 * @param ctx - The stream context.
 * @param chunkSize - Output rows per payload.
 * @param tick - Optional progress callback (owned gaussians processed).
 * @yields One {@link ChunkPayload} per output chunk, in order.
 */
async function *mergeStream(
    ctx: MergeStreamContext,
    chunkSize: number,
    tick?: (n: number) => void
): AsyncGenerator<ChunkPayload> {
    const { source, pos, order, blocks, selection, nextPositions } = ctx;
    const { memberGroup, groupMin, groupOffsets, groupMembers } = selection;
    const { layouts, availableLayers } = source.meta;

    const colorDim = layouts.color!.stride >> 2;
    const hasOther = availableLayers.has('other') && (layouts.other?.stride ?? 0) > 0;
    const otherDim = hasOther ? layouts.other!.stride >> 2 : 0;

    // Rolling output buffers (reused across payloads: the consumer copies
    // before pulling the next chunk).
    const outPos = new Float32Array(chunkSize * 3);
    const outGeo = new Float32Array(chunkSize * 8);
    const outColor = new Float32Array(chunkSize * colorDim);
    const outOther = hasOther ? new Uint32Array(chunkSize * otherDim) : undefined;
    let rows = 0;
    let emitted = 0;

    const payload = (): ChunkPayload => {
        const p: ChunkPayload = {
            count: rows,
            position: outPos.subarray(0, rows * 3),
            geometric: outGeo.subarray(0, rows * 8),
            color: outColor.subarray(0, rows * colorDim)
        };
        if (outOther) p.other = outOther.subarray(0, rows * otherDim);
        return p;
    };

    for (let bi = 0; bi < blocks.length; bi++) {
        const block = blocks[bi];
        const owned = order.subarray(block.start, block.end);
        const nOwned = owned.length;

        // This block's emitted groups (min member owned here), in owned order,
        // and the out-of-block members they pull in.
        const blockGroups: number[] = [];
        const extSet = new Map<number, number>();
        for (let i = 0; i < nOwned; i++) {
            const g = owned[i];
            const mg = memberGroup[g];
            if (mg === -1 || groupMin[mg] !== g) continue;
            blockGroups.push(mg);
            for (let m = groupOffsets[mg]; m < groupOffsets[mg + 1]; m++) {
                const member = groupMembers[m];
                if (indexOfSorted(owned, member) < 0 && !extSet.has(member)) extSet.set(member, 0);
            }
        }
        const extraGlobals = Uint32Array.from(extSet.keys()).sort();
        for (let i = 0; i < extraGlobals.length; i++) extSet.set(extraGlobals[i], nOwned + i);

        const { view, other } = await gatherBlockView(ctx, bi, extraGlobals, hasOther);

        // Merge this block's groups in a worker: pack member-major inputs.
        let mergedPos: Float32Array | null = null;
        let mergedGeo: Float32Array | null = null;
        let mergedColor: Float32Array | null = null;
        let mergedOther: Uint32Array | undefined;
        if (blockGroups.length > 0) {
            let totalMembers = 0;
            for (const mg of blockGroups) totalMembers += groupOffsets[mg + 1] - groupOffsets[mg];
            const mPos = new Float32Array(totalMembers * 3);
            const mGeo = new Float32Array(totalMembers * 8);
            const mColor = new Float32Array(totalMembers * colorDim);
            const mOther = hasOther ? new Uint32Array(totalMembers * otherDim) : undefined;
            const sizes = new Uint32Array(blockGroups.length);
            let mi = 0;
            for (let gi = 0; gi < blockGroups.length; gi++) {
                const mg = blockGroups[gi];
                sizes[gi] = groupOffsets[mg + 1] - groupOffsets[mg];
                for (let m = groupOffsets[mg]; m < groupOffsets[mg + 1]; m++) {
                    const member = groupMembers[m];
                    const oi = indexOfSorted(owned, member);
                    const row = oi >= 0 ? oi : extSet.get(member)!;
                    mPos[mi * 3] = view.pos[row * 3];
                    mPos[mi * 3 + 1] = view.pos[row * 3 + 1];
                    mPos[mi * 3 + 2] = view.pos[row * 3 + 2];
                    mGeo.set(view.geo.subarray(row * 8, row * 8 + 8), mi * 8);
                    mColor.set(view.color.subarray(row * colorDim, (row + 1) * colorDim), mi * colorDim);
                    if (mOther) mOther.set(other!.subarray(row * otherDim, (row + 1) * otherDim), mi * otherDim);
                    mi++;
                }
            }
            const transfer: ArrayBuffer[] = [mPos.buffer as ArrayBuffer, mGeo.buffer as ArrayBuffer, mColor.buffer as ArrayBuffer];
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

        // Emit rows in owned order.
        let nextMerged = 0;
        for (let i = 0; i < nOwned; i++) {
            const g = owned[i];
            const mg = memberGroup[g];
            if (mg !== -1 && groupMin[mg] !== g) continue;   // consumed member

            if (mg === -1) {
                // Survivor pass-through: position from resident arrays,
                // geometric/color/other block-copied from the view.
                outPos[rows * 3] = pos.x[g];
                outPos[rows * 3 + 1] = pos.y[g];
                outPos[rows * 3 + 2] = pos.z[g];
                outGeo.set(view.geo.subarray(i * 8, i * 8 + 8), rows * 8);
                outColor.set(view.color.subarray(i * colorDim, (i + 1) * colorDim), rows * colorDim);
                if (outOther) outOther.set(other!.subarray(i * otherDim, (i + 1) * otherDim), rows * otherDim);
            } else {
                const mi = nextMerged++;
                outPos.set(mergedPos!.subarray(mi * 3, mi * 3 + 3), rows * 3);
                outGeo.set(mergedGeo!.subarray(mi * 8, mi * 8 + 8), rows * 8);
                outColor.set(mergedColor!.subarray(mi * colorDim, (mi + 1) * colorDim), rows * colorDim);
                if (outOther) outOther.set(mergedOther!.subarray(mi * otherDim, (mi + 1) * otherDim), rows * otherDim);
            }

            if (nextPositions) {
                nextPositions.x[emitted] = outPos[rows * 3];
                nextPositions.y[emitted] = outPos[rows * 3 + 1];
                nextPositions.z[emitted] = outPos[rows * 3 + 2];
            }
            rows++;
            emitted++;
            if (rows === chunkSize) {
                yield payload();
                rows = 0;
            }
        }
        tick?.(nOwned);
    }

    if (rows > 0) {
        yield payload();
        rows = 0;
    }
}

export { mergeStream, type MergeStreamContext };
