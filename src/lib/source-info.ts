import { type ChunkSourceMetadata, hasGaussianLayers, orderedLayers } from './chunk';
import { type LodStats, type SourceStats } from './ops';
import { type InputFormat } from './read';
import { forwardTransforms } from './value-transforms';

/**
 * Shared rendering for the `info` and `stats` actions. Both actions report the
 * same structural block (the info fields); `stats` appends per-LOD, per-column
 * statistics — so the stats output is a strict superset of the info output in
 * both text and JSON form.
 */

type OutputFormat = 'text' | 'json';

// Pretty-print, but collapse innermost arrays (numbers, strings, null — no
// nested brackets) onto one line so the columnar stat arrays read as table
// rows instead of one value per line.
const stringifyCompact = (value: unknown): string => {
    return JSON.stringify(value, null, 4)
    .replace(/\[[^[\]{}]*\]/g, m => m.replace(/\s+/g, ' ').replace('[ ', '[').replace(' ]', ']'));
};

/**
 * Build the info object — the JSON form of a source's structural metadata and
 * the shared head of the stats JSON output. `extraColumns` lists only the
 * non-standard (`other`-layer) columns; the standard columns are fully implied
 * by `layers` + `shBands`.
 * @param meta - The source metadata.
 * @param format - Detected input format; included only when provided.
 * @returns The info fields.
 */
const buildSourceInfo = (meta: ChunkSourceMetadata, format?: InputFormat) => ({
    ...(format ? { format } : {}),
    gaussian: hasGaussianLayers(meta.availableLayers),
    numGaussians: meta.numGaussians,
    numLods: meta.numLods,
    lodCounts: [...meta.lodCounts],
    shBands: meta.shBands,
    layers: orderedLayers(meta.availableLayers),
    extraColumns: meta.extraColumns.map(e => ({ name: e.name, type: e.type }))
});

/**
 * The info text lines — the text form of {@link buildSourceInfo}, mirroring
 * its fields one-to-one with exact (unabbreviated) counts.
 * @param meta - The source metadata.
 * @param format - Detected input format; emitted only when provided.
 * @returns One `key: value` line per field.
 */
const sourceInfoLines = (meta: ChunkSourceMetadata, format?: InputFormat): string[] => {
    return [
        ...(format ? [`format: ${format}`] : []),
        `gaussian: ${hasGaussianLayers(meta.availableLayers) ? 'yes' : 'no'}`,
        `gaussians: ${meta.numGaussians}`,
        `lods: ${meta.numLods}`,
        `lod counts: ${meta.lodCounts.join(', ')}`,
        `sh bands: ${meta.shBands}`,
        `layers: ${orderedLayers(meta.availableLayers).join(', ')}`,
        `extra columns: ${meta.extraColumns.length > 0 ? meta.extraColumns.map(e => `${e.name} (${e.type})`).join(', ') : '(none)'}`
    ];
};

/**
 * Render a source's structural metadata for the `info` action — the detected
 * input format (when known), the gaussian verdict ({@link hasGaussianLayers}:
 * `false` for e.g. a plain point-cloud PLY), per-LOD counts, SH bands, available
 * layers, and the extra (non-standard `other`-layer) columns, all from `meta`
 * (no data read).
 * @param meta - The source metadata.
 * @param format - Output format. Default: 'text'
 * @param sourceFormat - Detected input format; reported when provided.
 * @returns A text or JSON block for `logger.output`.
 */
const formatSourceInfo = (meta: ChunkSourceMetadata, format: OutputFormat = 'text', sourceFormat?: InputFormat): string => {
    if (format === 'json') {
        return stringifyCompact(buildSourceInfo(meta, sourceFormat));
    }
    return sourceInfoLines(meta, sourceFormat).join('\n');
};

// Display transform: raw values map to user-friendly space for output
// (opacity -> sigmoid, scale -> linear, f_dc -> 0-1 color); untransformed
// values pass through as computed (already rounded to 6 decimals).
const displayValue = (column: string, v: number): number => {
    const fn = forwardTransforms[column];
    return fn ? +fn(v).toPrecision(6) : v;
};

/** Unicode block characters for histogram visualization (lowest to highest). */
const BARS = '▁▂▃▄▅▆▇█';

// 16-character sparkline from a histogram's bin counts (blank for empty bins,
// all-blank for an empty column).
const sparkline = (counts: number[]): string => {
    const maxBin = Math.max(...counts);
    return counts.map((c) => {
        if (c === 0) return ' ';
        return BARS[maxBin > 0 ? Math.floor(c / maxBin * (BARS.length - 1)) : 0];
    }).join('');
};

// Map a LOD's stats to display space for JSON output: value arrays through the
// per-column display transform, counts and histograms as computed.
const displayLodStats = (lod: LodStats): LodStats => {
    const mapped = (values: number[]): number[] => values.map((v, i) => displayValue(lod.columns[i], v));
    return {
        ...lod,
        data: {
            ...lod.data,
            min: mapped(lod.data.min),
            max: mapped(lod.data.max),
            median: mapped(lod.data.median),
            mean: mapped(lod.data.mean),
            stdDev: mapped(lod.data.stdDev)
        }
    };
};

// Render one LOD's stats as an aligned markdown-style table.
const statsTable = (lod: LodStats): string[] => {
    const { data } = lod;
    const headers = ['Column', 'min', 'max', 'median', 'mean', 'stdDev', 'nans', 'infs', 'histogram'];
    const rows = lod.columns.map((name, i) => [
        name,
        String(displayValue(name, data.min[i])),
        String(displayValue(name, data.max[i])),
        String(displayValue(name, data.median[i])),
        String(displayValue(name, data.mean[i])),
        String(displayValue(name, data.stdDev[i])),
        String(data.nanCount[i]),
        String(data.infCount[i]),
        sparkline(data.histogram[i])
    ]);

    const colWidths = headers.map((header, colIndex) => {
        const dataWidths = rows.map(row => row[colIndex].length);
        return Math.max(header.length, ...dataWidths);
    });

    const padRow = (cells: string[]) => `| ${cells.map((cell, i) => cell.padEnd(colWidths[i])).join(' | ')} |`;
    const separator = `|${colWidths.map(w => '-'.repeat(w + 2)).join('|')}|`;

    return [padRow(headers), separator, ...rows.map(padRow)];
};

/**
 * Render a source's statistics for the `stats` action: the info block followed
 * by one table per LOD (text), or the info object plus a per-LOD columnar
 * `stats` array (JSON — exactly the {@link LodStats} shape). Values are shown
 * in display space (see {@link forwardTransforms}); histogram bin edges span
 * `[min[i], max[i]]`.
 * @param meta - The source metadata.
 * @param stats - The computed per-LOD statistics.
 * @param format - Output format. Default: 'text'
 * @param sourceFormat - Detected input format; reported when provided.
 * @returns A text or JSON block for `logger.output`.
 */
const formatSourceStats = (meta: ChunkSourceMetadata, stats: SourceStats, format: OutputFormat = 'text', sourceFormat?: InputFormat): string => {
    if (format === 'json') {
        return stringifyCompact({
            ...buildSourceInfo(meta, sourceFormat),
            stats: stats.lods.map(displayLodStats)
        });
    }

    const lines = sourceInfoLines(meta, sourceFormat);
    for (const lod of stats.lods) {
        lines.push('');
        if (stats.lods.length > 1) {
            lines.push(`lod ${lod.lod}: ${lod.numGaussians} gaussians`);
        }
        lines.push(...statsTable(lod));
        if (lod.fillRatio !== undefined) {
            lines.push('');
            lines.push(`fill ratio: ${lod.fillRatio} (~avg overdraw layers)`);
        }
    }
    return lines.join('\n');
};

export { formatSourceInfo, formatSourceStats };
