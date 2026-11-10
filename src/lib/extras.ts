import type { ChunkSource } from './chunk';
import { logger } from './utils';

/**
 * The same source with `meta.extras` replaced (or removed when `extras` is
 * undefined). Reads are untouched.
 *
 * @param source - The source.
 * @param extras - The application-defined JSON to attach.
 * @returns A source carrying `extras`.
 */
const withExtras = (source: ChunkSource, extras: unknown): ChunkSource => {
    if (source.meta.extras === extras) return source;
    const { extras: _, ...meta } = source.meta;
    // delegate rather than spread: sources may be class instances
    return {
        meta: extras === undefined ? meta : { ...meta, extras },
        read: (request) => source.read(request),
        close: () => source.close()
    };
};

/**
 * The extras of a combined scene: kept when every input carries the same JSON
 * (or none), dropped with a warning when they disagree — no merge of arbitrary
 * JSON is right.
 *
 * @param sources - The sources being combined.
 * @returns The agreed extras, or undefined.
 */
const resolveExtras = (sources: readonly ChunkSource[]): unknown => {
    const keys = sources.map((s) => (s.meta.extras === undefined ? undefined : JSON.stringify(s.meta.extras)));
    if (keys.every((k) => k === keys[0])) return sources[0]?.meta.extras;
    logger.warn('inputs carry different extras; the combined scene keeps none');
    return undefined;
};

export { withExtras, resolveExtras };
