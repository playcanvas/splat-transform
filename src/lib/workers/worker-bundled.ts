/**
 * Build-time flag: false when running from source (tsx: dev, tests), flipped
 * to true by the library/CLI builds (see the markWorkerBundled plugin in
 * rollup.config.mjs). When false, WorkerQueue has no worker file to spawn and
 * runs every task inline on the calling thread instead.
 */
export const workerBundled = false;
