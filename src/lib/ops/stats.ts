import {
    type ChunkData,
    type ChunkDataPool,
    type ChunkLayer,
    type ChunkSource,
    type ChunkSourceMetadata,
    SH_REST_COUNTS
} from '../chunk';

/**
 * One-pass streaming statistics over a {@link ChunkSource}: exact
 * min/max/mean/stdDev/NaN/Inf per column, plus an approximate median and a
 * 16-bin histogram derived from a self-scaling fine histogram — no
 * materialization, constant memory per column. Stats are computed per LOD
 * (LODs are overlapping representations of one scene, never flattened).
 */

/** Number of output histogram bins. */
const NUM_BINS = 16;

/** Fine accumulation bins; the median error bound is ~span/FINE_BINS. */
const FINE_BINS = 1024;

const PRECISION = 6;

const round = (value: number): number => {
    if (!Number.isFinite(value)) return value;
    return Math.round(value * Math.pow(10, PRECISION)) / Math.pow(10, PRECISION);
};

/**
 * A LOD's measurements in columnar (struct-of-arrays) form: every field is an
 * array index-aligned with the owning {@link LodStats}'s `columns`.
 */
type LodStatsData = {
    /** Per-column minimum (excluding NaN/Inf). */
    min: number[];
    /** Per-column maximum (excluding NaN/Inf). */
    max: number[];
    /** Per-column approximate median (from the fine histogram; error ~(max-min)/1024). */
    median: number[];
    /** Per-column arithmetic mean. */
    mean: number[];
    /** Per-column population standard deviation. */
    stdDev: number[];
    /** Per-column NaN count. */
    nanCount: number[];
    /** Per-column Infinity count. */
    infCount: number[];
    /** Per-column histogram: {@link NUM_BINS} counts over `[min[i], max[i]]`, raw space. */
    histogram: number[][];
};

/**
 * Per-LOD column statistics: identity (`lod`, `numGaussians`), the column-name
 * axis (`columns`), and the aligned measurement arrays (`data`).
 * `JSON.stringify` of this is the stats JSON output shape (NaN fields — e.g.
 * an all-NaN column's min — serialize as `null`).
 */
type LodStats = {
    /** The LOD level these statistics describe. */
    lod: number;
    /** Gaussian count of this LOD. */
    numGaussians: number;
    /** Canonical column names; every array in `data` aligns with this order. */
    columns: string[];
    /** The column-aligned measurement arrays. */
    data: LodStatsData;
    /**
     * Fill (overdraw) ratio: total splat footprint area (1-sigma ellipse over
     * the two largest linear scales, summed) divided by the scene's robust
     * cross-section (mean face area of the p1-p99 extent box, floored at the
     * median splat footprint when the extents are degenerate). Approximates
     * the average number of splat layers a ray through the scene crosses:
     * healthy scenes land in the ones-to-hundreds; scenes that will overwhelm
     * a GPU with fill (zero-padded exports, garbage scales, deliberately
     * adversarial content) land orders of magnitude higher, so backends can
     * gate publishes on it. A `+Infinity` scale propagates to `Infinity`
     * (serialized as `null` in JSON) — reject those outright. Present when the
     * source has position and geometric data.
     */
    fillRatio?: number;
};

/**
 * Statistics for an entire source: one {@link LodStats} per LOD level.
 */
type SourceStats = {
    /** Per-LOD statistics, indexed by LOD level. */
    lods: LodStats[];
};

/**
 * Streaming per-column accumulator: exact running min/max, Welford
 * mean/variance, NaN/Inf counts, and a fine histogram with exact fixed-width
 * bins whose range doubles (merging bin pairs losslessly) whenever a value
 * falls outside — so a single pass needs no prior knowledge of the range.
 */
class StatsAccumulator {
    n = 0;
    nanCount = 0;
    infCount = 0;
    min = Infinity;
    max = -Infinity;
    private mean = 0;
    private m2 = 0;

    // Fine histogram, seeded lazily from the first two distinct values (a
    // constant column never allocates). `firstValue`/`firstCount` track the
    // pre-seed run of identical values.
    private firstValue = 0;
    private firstCount = 0;
    private lo = 0;
    private hi = 0;
    private bins: Uint32Array | null = null;

    add(v: number): void {
        if (Number.isNaN(v)) {
            this.nanCount++;
            return;
        }
        if (!Number.isFinite(v)) {
            this.infCount++;
            return;
        }

        this.n++;
        if (v < this.min) this.min = v;
        if (v > this.max) this.max = v;
        const d = v - this.mean;
        this.mean += d / this.n;
        this.m2 += d * (v - this.mean);

        if (this.bins === null) {
            if (this.firstCount === 0 || v === this.firstValue) {
                this.firstValue = v;
                this.firstCount++;
                return;
            }
            // Second distinct value: seed the histogram over [lo, hi] and
            // replay the identical-value run.
            this.lo = Math.min(this.firstValue, v);
            this.hi = Math.max(this.firstValue, v);
            this.bins = new Uint32Array(FINE_BINS);
            this.insert(this.firstValue, this.firstCount);
            this.insert(v, 1);
            return;
        }

        this.expandToFit(v);
        this.insert(v, 1);
    }

    // Double the range toward `v` until it fits, merging bin pairs. The merge
    // is lossless: doubling the span keeps every old bin edge on the new grid
    // (left: old bin i -> (i + B) >> 1; right: i -> i >> 1, for even B).
    private expandToFit(v: number): void {
        while (v < this.lo || v > this.hi) {
            const span = this.hi - this.lo;
            const left = v < this.lo;
            const merged = new Uint32Array(FINE_BINS);
            const bins = this.bins!;
            for (let i = 0; i < FINE_BINS; i++) {
                if (bins[i] !== 0) merged[left ? (i + FINE_BINS) >> 1 : i >> 1] += bins[i];
            }
            this.bins = merged;
            if (left) {
                this.lo -= span;
            } else {
                this.hi += span;
            }
        }
    }

    private insert(v: number, count: number): void {
        const bin = Math.min(FINE_BINS - 1, Math.floor((v - this.lo) * FINE_BINS / (this.hi - this.lo)));
        this.bins![bin] += count;
    }

    // Grouped quantile from the fine histogram: locate the bin where the
    // cumulative count crosses q*n and linearly interpolate within it. A
    // crossing exactly at a bin's end (the target count splits across a gap)
    // averages with the next populated bin's start — at q=0.5 this mirrors the
    // middle-pair averaging of an exact even-n median, and the same convention
    // applies at every q. Results are clamped to the exact [min, max]
    // (interpolation can otherwise drift past the true extremes by up to a bin
    // width). The median is this at q=0.5.
    quantile(q: number): number {
        if (this.n === 0) return NaN;
        if (this.bins === null) return this.firstValue; // all values identical
        const clamp = (v: number): number => Math.min(this.max, Math.max(this.min, v));
        const target = q * this.n;
        const width = (this.hi - this.lo) / FINE_BINS;
        let cum = 0;
        for (let b = 0; b < FINE_BINS; b++) {
            const c = this.bins[b];
            if (c === 0) continue;
            if (cum + c > target) {
                return clamp(this.lo + (b + (target - cum) / c) * width);
            }
            if (cum + c === target) {
                const end = this.lo + (b + 1) * width;
                for (let nb = b + 1; nb < FINE_BINS; nb++) {
                    if (this.bins[nb] > 0) return clamp((end + this.lo + nb * width) / 2);
                }
                return clamp(end);
            }
            cum += c;
        }
        return this.max; // unreachable (target <= n)
    }

    // Aggregate the fine bins into NUM_BINS output bins spanning exactly
    // [min, max]: each fine bin's count lands in the output bin containing its
    // midpoint. A constant column concentrates in the middle bin; an empty
    // column is all zeros.
    private computeHistogram(): number[] {
        const out = new Array<number>(NUM_BINS).fill(0);
        if (this.n === 0) return out;
        if (this.bins === null || this.min === this.max) {
            out[NUM_BINS >>> 1] = this.n;
            return out;
        }
        const range = this.max - this.min;
        const width = (this.hi - this.lo) / FINE_BINS;
        for (let b = 0; b < FINE_BINS; b++) {
            const c = this.bins[b];
            if (c === 0) continue;
            const mid = this.lo + (b + 0.5) * width;
            const idx = Math.min(NUM_BINS - 1, Math.max(0, Math.floor((mid - this.min) / range * NUM_BINS)));
            out[idx] += c;
        }
        return out;
    }

    finalize(): { min: number; max: number; median: number; mean: number; stdDev: number; nanCount: number; infCount: number; histogram: number[] } {
        const empty = this.n === 0;
        return {
            min: empty ? NaN : round(this.min),
            max: empty ? NaN : round(this.max),
            median: round(this.quantile(0.5)),
            mean: empty ? NaN : round(this.mean),
            stdDev: empty ? NaN : round(Math.sqrt(this.m2 / this.n)),
            nanCount: this.nanCount,
            infCount: this.infCount,
            histogram: this.computeHistogram()
        };
    }
}

// One accumulation site: a canonical column and where its values live within
// a layer's interleaved record.
type ColumnPlan = { name: string; layer: ChunkLayer; elem: number; uint: boolean };

// Enumerate the source's columns in canonical order (matching
// columnNamesFromMeta): position, geometric, color (+ f_rest), extras.
const buildColumnPlans = (meta: ChunkSourceMetadata): ColumnPlan[] => {
    const plans: ColumnPlan[] = [];
    if (meta.availableLayers.has('position')) {
        ['x', 'y', 'z'].forEach((name, elem) => plans.push({ name, layer: 'position', elem, uint: false }));
    }
    if (meta.availableLayers.has('geometric')) {
        ['rot_0', 'rot_1', 'rot_2', 'rot_3', 'scale_0', 'scale_1', 'scale_2', 'opacity']
        .forEach((name, elem) => plans.push({ name, layer: 'geometric', elem, uint: false }));
    }
    if (meta.availableLayers.has('color')) {
        ['f_dc_0', 'f_dc_1', 'f_dc_2'].forEach((name, elem) => plans.push({ name, layer: 'color', elem, uint: false }));
        for (let r = 0; r < SH_REST_COUNTS[meta.shBands]; r++) {
            plans.push({ name: `f_rest_${r}`, layer: 'color', elem: 3 + r, uint: false });
        }
    }
    if (meta.availableLayers.has('other')) {
        meta.extraColumns.forEach((e, elem) => plans.push({ name: e.name, layer: 'other', elem, uint: e.type === 'uint32' }));
    }
    return plans;
};

type MutableReadRequest = { chunkIndex: number; lod: number } & { [L in ChunkLayer]?: ChunkData };

/**
 * Compute per-LOD, per-column statistics for a source in a single streaming
 * pass: each LOD's chunks are read once and fed to that LOD's accumulators.
 * Values are the source's raw, unbaked values (any pending `meta.transform` is
 * not applied), matching the historical summary behavior.
 *
 * @param src - The source to analyze (left unchanged).
 * @param pool - Pool for the temporary layer read buffers.
 * @returns The source's {@link SourceStats}, one {@link LodStats} per LOD.
 */
const computeSourceStats = async (src: ChunkSource, pool: ChunkDataPool): Promise<SourceStats> => {
    const { meta } = src;
    const plans = buildColumnPlans(meta);
    const layers = [...new Set(plans.map(p => p.layer))];
    const lods: LodStats[] = [];

    const hasFill = meta.availableLayers.has('position') && meta.availableLayers.has('geometric');

    for (let lod = 0; lod < meta.numLods; lod++) {
        const accs = plans.map(() => new StatsAccumulator());
        const lodCount = meta.lodCounts[lod];
        const numChunks = meta.numChunks[lod] ?? 0;

        // Per-splat footprint areas stream through their own accumulator (for
        // the median floor); the total is an explicit sum so +Infinity areas
        // propagate into it rather than being counted out.
        const areaAcc = new StatsAccumulator();
        let totalArea = 0;

        for (let c = 0; c < numChunks; c++) {
            const count = Math.min(meta.chunkSize, lodCount - c * meta.chunkSize);

            const acquired: ChunkData[] = [];
            const buffers: Partial<Record<ChunkLayer, ArrayBuffer>> = {};
            const req: MutableReadRequest = { chunkIndex: c, lod };
            for (const layer of layers) {
                const cd = pool.acquire(layer, meta.layouts[layer]!, count);
                req[layer] = cd;
                buffers[layer] = cd.data;
                acquired.push(cd);
            }
            await src.read(req);

            for (let p = 0; p < plans.length; p++) {
                const { layer, elem, uint } = plans[p];
                const acc = accs[p];
                const stride32 = meta.layouts[layer]!.stride >>> 2;
                const view = uint ? new Uint32Array(buffers[layer]!) : new Float32Array(buffers[layer]!);
                for (let r = 0; r < count; r++) {
                    acc.add(view[r * stride32 + elem]);
                }
            }

            if (hasFill) {
                const geo = new Float32Array(buffers.geometric!);
                const stride32 = meta.layouts.geometric!.stride >>> 2;
                for (let r = 0; r < count; r++) {
                    const o = r * stride32;
                    const s0 = Math.exp(geo[o + 4]);
                    const s1 = Math.exp(geo[o + 5]);
                    const s2 = Math.exp(geo[o + 6]);
                    // 1-sigma ellipse area over the two largest linear scales.
                    // The smallest may be 0 (a -Infinity flat-splat scale), so
                    // pick the top two explicitly rather than dividing it out.
                    const lo = Math.min(s0, Math.min(s1, s2));
                    const top2 = s0 === lo ? s1 * s2 : (s1 === lo ? s0 * s2 : s0 * s1);
                    const area = Math.PI * top2;
                    areaAcc.add(area);
                    if (!Number.isNaN(area)) totalArea += area;
                }
            }

            for (const cd of acquired) cd.release();
        }

        const results = accs.map(a => a.finalize());

        let fillRatio: number | undefined;
        if (hasFill) {
            // Robust extents (p1-p99 per axis) so flyaway splats can't inflate
            // the cross-section and mask the fill.
            const axis = (name: string): StatsAccumulator => accs[plans.findIndex(p => p.name === name)];
            const [dx, dy, dz] = ['x', 'y', 'z'].map((name) => {
                const acc = axis(name);
                return Math.max(0, acc.quantile(0.99) - acc.quantile(0.01));
            });
            let crossSection = (dx * dy + dy * dz + dz * dx) / 3;
            if (!(crossSection > 0)) {
                // Degenerate extents (coincident splats, or a single splat):
                // measure fill against the median splat footprint instead, so
                // the ratio approximates the coincident layer count.
                const medianArea = areaAcc.quantile(0.5);
                crossSection = Number.isFinite(medianArea) ? medianArea : 0;
            }
            fillRatio = round(crossSection > 0 ? totalArea / crossSection : (totalArea > 0 ? Infinity : 0));
        }

        lods.push({
            lod,
            numGaussians: lodCount,
            columns: plans.map(p => p.name),
            data: {
                min: results.map(r => r.min),
                max: results.map(r => r.max),
                median: results.map(r => r.median),
                mean: results.map(r => r.mean),
                stdDev: results.map(r => r.stdDev),
                nanCount: results.map(r => r.nanCount),
                infCount: results.map(r => r.infCount),
                histogram: results.map(r => r.histogram)
            },
            ...(fillRatio !== undefined ? { fillRatio } : {})
        });
    }

    return { lods };
};

export { computeSourceStats, type LodStats, type LodStatsData, type SourceStats };
