/**
 * Format a duration in milliseconds as a human-readable string.
 *
 * - Sub-minute durations render in seconds (e.g. `1.234s`).
 * - Sub-hour durations render as `MmS.SSSs`.
 * - Otherwise as `HhMmS.SSSs`.
 *
 * @param ms - The duration in milliseconds.
 * @returns The formatted string.
 */
const fmtTime = (ms: number): string => {
    if (!Number.isFinite(ms) || ms < 0) return `${ms}ms`;
    if (ms < 60_000) return `${(ms / 1000).toFixed(3)}s`;

    const h = Math.floor(ms / 3_600_000);
    const m = Math.floor((ms % 3_600_000) / 60_000);
    const s = ((ms % 60_000) / 1000).toFixed(3);

    return h > 0 ? `${h}h${m}m${s}s` : `${m}m${s}s`;
};

/**
 * Format a byte count using binary (1024-based) units.
 *
 * @param n - The number of bytes.
 * @returns The formatted string (e.g. `1.5MB`).
 */
const fmtBytes = (n: number): string => {
    if (!Number.isFinite(n) || n < 0) return `${n}B`;
    if (n < 1024) return `${n}B`;
    if (n < 1024 * 1024) return `${(n / 1024).toFixed(1)}KB`;
    if (n < 1024 * 1024 * 1024) return `${(n / (1024 * 1024)).toFixed(1)}MB`;
    return `${(n / (1024 * 1024 * 1024)).toFixed(2)}GB`;
};

/**
 * Format a distance in metres as a human-readable string, picking the most
 * appropriate unit (mm/cm/m/km).
 *
 * @param m - The distance in metres.
 * @returns The formatted string.
 */
const fmtDistance = (m: number): string => {
    if (!Number.isFinite(m)) return `${m}m`;
    const abs = Math.abs(m);
    if (abs === 0) return '0m';
    if (abs < 0.01) return `${+(m * 1000).toPrecision(3)}mm`;
    if (abs < 1) return `${+(m * 100).toPrecision(3)}cm`;
    if (abs < 1000) return `${+m.toPrecision(3)}m`;
    return `${+(m / 1000).toPrecision(3)}km`;
};

/**
 * Format a count using SI suffixes (K/M/B/T) above 1000.
 *
 * @param n - The count to format.
 * @returns The formatted string.
 */
const fmtCount = (n: number): string => {
    if (!Number.isFinite(n)) return `${n}`;
    const abs = Math.abs(n);
    if (abs < 1000) return `${n}`;
    if (abs < 1e6) return `${+(n / 1e3).toPrecision(3)}K`;
    if (abs < 1e9) return `${+(n / 1e6).toPrecision(3)}M`;
    if (abs < 1e12) return `${+(n / 1e9).toPrecision(3)}B`;
    return `${+(n / 1e12).toPrecision(3)}T`;
};

export { fmtBytes, fmtCount, fmtDistance, fmtTime };
