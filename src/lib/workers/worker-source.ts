/**
 * Source code of the bundled worker (worker-entry.ts plus its dependencies),
 * inlined at build time: rollup first builds the worker entry to
 * build/worker.mjs, then the library/CLI builds replace this module with the
 * generated text (see the inlineWorkerSource plugin in rollup.config.mjs).
 *
 * When running from source (tsx: dev, tests) the placeholder below survives
 * and WorkerQueue runs every task inline on the calling thread instead.
 */
export const workerSource: string | null = null;
