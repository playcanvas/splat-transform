import { type ProgressCallback } from './file-system';

/**
 * Aggregates per-source progress callbacks into a single combined progress
 * stream, using engine-style extrapolation when not all sources have started
 * reporting yet.
 *
 * Modelled on the `combineProgress` helper in the PlayCanvas engine's SOG
 * parser. Multi-source readers (e.g. SOG, LCC) use this to drive a single
 * progress UI that paces against the aggregate of all payload bytes.
 *
 * Usage:
 *
 * ```ts
 * const combine = new CombineProgress(payloadFiles.length, onProgress);
 * for (const name of payloadFiles) {
 *     await fileSystem.createSource(name, combine.track(name));
 * }
 * ```
 *
 * Bootstrapping reads (e.g. metadata) should be performed without a progress
 * callback so they don't contribute to the aggregate.
 */
class CombineProgress {
    private map = new Map<string, { loaded: number; total: number }>();
    private expectedCount: number;
    private emit: ProgressCallback;

    /**
     * @param expectedCount - Number of sources expected to report. Used to
     * extrapolate `total` when fewer sources have started reporting.
     * @param emit - Combined progress callback fired on every per-source update.
     */
    constructor(expectedCount: number, emit: ProgressCallback) {
        this.expectedCount = expectedCount;
        this.emit = emit;
    }

    /**
     * Returns a per-source ProgressCallback to pass into createSource().
     * @param id - Unique identifier for this source within the combined stream.
     * @returns A ProgressCallback that updates this source's contribution to
     * the aggregate and re-emits the combined progress.
     */
    track(id: string): ProgressCallback {
        return (loaded, total) => {
            this.map.set(id, { loaded, total: total ?? 0 });
            this.fire();
        };
    }

    private fire(): void {
        let loaded = 0;
        let total = 0;
        for (const v of this.map.values()) {
            loaded += v.loaded;
            total += v.total;
        }
        const reporting = this.map.size;
        if (reporting > 0 && reporting < this.expectedCount) {
            total = Math.ceil(total * this.expectedCount / reporting);
        }
        this.emit(loaded, total);
    }
}

export { CombineProgress };
