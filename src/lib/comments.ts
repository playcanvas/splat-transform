import type { ChunkSource } from './chunk';
import { logger } from './utils';

/**
 * The same source with `meta.comments` replaced (or removed when `comments` is
 * undefined or empty). Reads are untouched.
 *
 * @param source - The source.
 * @param comments - The producer comments to attach.
 * @returns A source carrying `comments`.
 */
const withComments = (source: ChunkSource, comments: readonly string[] | undefined): ChunkSource => {
    if (source.meta.comments === comments) return source;
    const { comments: _, ...meta } = source.meta;
    // delegate rather than spread: sources may be class instances
    return {
        meta: comments?.length ? { ...meta, comments } : meta,
        read: (request) => source.read(request),
        close: () => source.close()
    };
};

/**
 * The comments of a combined scene: kept when every input carries the same
 * comments (or none), dropped with a warning when they disagree.
 *
 * @param sources - The sources being combined.
 * @returns The agreed comments, or undefined.
 */
const resolveComments = (sources: readonly ChunkSource[]): readonly string[] | undefined => {
    const keys = sources.map((s) => JSON.stringify(s.meta.comments ?? []));
    if (keys.every((k) => k === keys[0])) return sources[0]?.meta.comments;
    logger.warn('inputs carry different comments; the combined scene keeps none');
    return undefined;
};

export { withComments, resolveComments };
