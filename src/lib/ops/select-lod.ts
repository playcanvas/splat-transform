import {
    type ChunkSource,
    type ChunkSourceMetadata,
    type ReadRequest
} from '../chunk';

/**
 * View a single LOD level of a multi-LOD source as a single-LOD source — the
 * inverse of {@link stackLods}. Reads (chunk or gather) forward to the parent
 * with `lod: level`; metadata is narrowed to that level's counts.
 *
 * Several `selectLod` views typically share one parent (one per level, then
 * re-stacked), so `close()` is a **no-op**: the caller owns the parent's
 * lifetime and closes it once. (A view closing the shared parent would break the
 * other levels — e.g. a per-level pass that materializes one level.)
 *
 * @param src - The parent (multi-LOD) source.
 * @param level - The LOD level to expose (0-based).
 * @returns A single-LOD source over `src`'s level `level`.
 */
const selectLod = (src: ChunkSource, level: number): ChunkSource => {
    if (level < 0 || level >= src.meta.numLods) {
        throw new Error(`selectLod: level ${level} out of range (numLods ${src.meta.numLods})`);
    }
    const count = src.meta.lodCounts[level];
    const meta: ChunkSourceMetadata = {
        ...src.meta,
        numGaussians: count,
        numLods: 1,
        lodCounts: [count],
        numChunks: [src.meta.numChunks[level]]
    };
    return {
        meta,
        read: (request: ReadRequest) => src.read({ ...request, lod: level }),
        close: () => Promise.resolve()
    };
};

/**
 * Resolve a user `lodSelect` (from `--select-lod`) into concrete level indices
 * against a source's actual level count: negative indices count from the end,
 * out-of-range entries are dropped, and an empty selection means all levels.
 * Used by the CLI to drive {@link selectLod} nodes — LOD selection is a pipeline
 * step, not a reader parameter.
 *
 * @param lodSelect - Requested levels (may be empty or include negatives).
 * @param numLods - The source's level count.
 * @returns Ordered concrete level indices.
 */
const resolveLodLevels = (lodSelect: number[], numLods: number): number[] => {
    if (lodSelect.length > 0) {
        return lodSelect
        .map(lod => (lod < 0 ? numLods + lod : lod))
        .filter(lod => lod >= 0 && lod < numLods);
    }
    return Array.from({ length: numLods }, (_, i) => i);
};

export { selectLod, resolveLodLevels };
