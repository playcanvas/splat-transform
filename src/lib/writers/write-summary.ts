import { DataTable } from '../data-table/data-table';
import { type FileSystem } from '../io/write';

type ColumnStats = {
    min: number;
    max: number;
    median: number;
    mean: number;
    stdDev: number;
    nanCount: number;
    infCount: number;
};

type SummaryData = {
    version: number;
    rowCount: number;
    columns: Record<string, ColumnStats>;
};

type WriteSummaryOptions = {
    filename: string;
    dataTable: DataTable;
    format: 'json' | 'md';
};

const PRECISION = 6;

const round = (value: number): number => {
    if (!Number.isFinite(value)) return value;
    return Math.round(value * Math.pow(10, PRECISION)) / Math.pow(10, PRECISION);
};

const computeColumnStats = (data: ArrayLike<number>): ColumnStats => {
    const len = data.length;

    // Count NaN and Inf values
    let nanCount = 0;
    let infCount = 0;
    const validValues: number[] = [];

    for (let i = 0; i < len; i++) {
        const v = data[i];
        if (Number.isNaN(v)) {
            nanCount++;
        } else if (!Number.isFinite(v)) {
            infCount++;
        } else {
            validValues.push(v);
        }
    }

    // Handle case where all values are NaN/Inf
    if (validValues.length === 0) {
        return {
            min: NaN,
            max: NaN,
            median: NaN,
            mean: NaN,
            stdDev: NaN,
            nanCount,
            infCount
        };
    }

    // Compute min, max, mean
    let min = validValues[0];
    let max = validValues[0];
    let sum = 0;

    for (let i = 0; i < validValues.length; i++) {
        const v = validValues[i];
        if (v < min) min = v;
        if (v > max) max = v;
        sum += v;
    }

    const mean = sum / validValues.length;

    // Compute standard deviation
    let sumSquaredDiff = 0;
    for (let i = 0; i < validValues.length; i++) {
        const diff = validValues[i] - mean;
        sumSquaredDiff += diff * diff;
    }
    const stdDev = Math.sqrt(sumSquaredDiff / validValues.length);

    // Compute median (requires sorting)
    validValues.sort((a, b) => a - b);
    const mid = Math.floor(validValues.length / 2);
    const median = validValues.length % 2 === 0 ?
        (validValues[mid - 1] + validValues[mid]) / 2 :
        validValues[mid];

    return {
        min: round(min),
        max: round(max),
        median: round(median),
        mean: round(mean),
        stdDev: round(stdDev),
        nanCount,
        infCount
    };
};

const computeSummary = (dataTable: DataTable): SummaryData => {
    const columns: Record<string, ColumnStats> = {};

    for (const column of dataTable.columns) {
        columns[column.name] = computeColumnStats(column.data);
    }

    return {
        version: 1,
        rowCount: dataTable.numRows,
        columns
    };
};

const formatJson = (summary: SummaryData): string => {
    return JSON.stringify(summary, null, 2);
};

const formatMarkdown = (summary: SummaryData): string => {
    const lines: string[] = [];

    lines.push('# Summary');
    lines.push('');
    lines.push(`**Row Count:** ${summary.rowCount}`);
    lines.push('');

    // Table header
    lines.push('| Column | min | max | median | mean | stdDev | nanCount | infCount |');
    lines.push('|--------|-----|-----|--------|------|--------|----------|----------|');

    // Table rows
    for (const [name, stats] of Object.entries(summary.columns)) {
        const row = [
            name,
            stats.min,
            stats.max,
            stats.median,
            stats.mean,
            stats.stdDev,
            stats.nanCount,
            stats.infCount
        ];
        lines.push(`| ${row.join(' | ')} |`);
    }

    return lines.join('\n');
};

const writeSummary = async (options: WriteSummaryOptions, fs: FileSystem) => {
    const { filename, dataTable, format } = options;

    const summary = computeSummary(dataTable);

    const content = format === 'json' ?
        formatJson(summary) :
        formatMarkdown(summary);

    const textEncoder = new TextEncoder();
    const writer = await fs.createWriter(filename);
    await writer.write(textEncoder.encode(content));
    await writer.close();
};

export { writeSummary };
