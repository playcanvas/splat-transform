/**
 * Platform-agnostic path utilities for use in browser and Node.js.
 * Uses forward slashes as the path separator.
 */

/**
 * Get the directory portion of a path.
 * @param path - The path to get the directory from
 * @returns The directory portion, or empty string if no directory
 */
const dirname = (path: string): string => {
    // Handle empty path
    if (!path) return '';

    // Normalize backslashes to forward slashes
    const normalized = path.replace(/\\/g, '/');

    // Find last slash
    const lastSlash = normalized.lastIndexOf('/');
    if (lastSlash === -1) return '';

    // Return everything before the last slash
    return normalized.slice(0, lastSlash);
};

/**
 * Get the base name (file name) from a path.
 * @param path - The path to get the base name from
 * @param ext - Optional extension to remove from the result
 * @returns The base name portion of the path
 */
const basename = (path: string, ext?: string): string => {
    // Handle empty path
    if (!path) return '';

    // Normalize backslashes to forward slashes
    const normalized = path.replace(/\\/g, '/');

    // Remove trailing slashes
    const trimmed = normalized.replace(/\/+$/, '');

    // Find last slash
    const lastSlash = trimmed.lastIndexOf('/');
    const base = lastSlash === -1 ? trimmed : trimmed.slice(lastSlash + 1);

    // Remove extension if provided
    if (ext && base.endsWith(ext)) {
        return base.slice(0, -ext.length);
    }

    return base;
};

/**
 * Join path segments together.
 * @param segments - Path segments to join
 * @returns The joined path
 */
const join = (...segments: string[]): string => {
    // Filter out empty segments and join with /
    const parts: string[] = [];

    for (const segment of segments) {
        if (!segment) continue;

        // Normalize backslashes to forward slashes
        const normalized = segment.replace(/\\/g, '/');

        // Split by / and add non-empty parts
        for (const part of normalized.split('/')) {
            if (part && part !== '.') {
                if (part === '..') {
                    parts.pop();
                } else {
                    parts.push(part);
                }
            }
        }
    }

    // Preserve leading slash if first segment had one
    const first = segments[0] || '';
    const leadingSlash = first.startsWith('/') || first.startsWith('\\');

    return (leadingSlash ? '/' : '') + parts.join('/');
};

/**
 * Resolve path segments into an absolute path.
 * In a browser context without a true filesystem, this simply joins the segments.
 * @param segments - Path segments to resolve
 * @returns The resolved path
 */
const resolve = (...segments: string[]): string => {
    // Process segments from right to left, stopping at first absolute path
    const parts: string[] = [];
    let hasAbsolute = false;
    let pendingDotDot = 0;  // Track unresolved .. count

    for (let i = segments.length - 1; i >= 0 && !hasAbsolute; i--) {
        const segment = segments[i];
        if (!segment) continue;

        // Normalize backslashes to forward slashes
        const normalized = segment.replace(/\\/g, '/');

        // Check if this is an absolute path
        if (normalized.startsWith('/')) {
            hasAbsolute = true;
        }

        // Split and process parts in reverse
        const segParts = normalized.split('/').filter(p => p && p !== '.');
        for (let j = segParts.length - 1; j >= 0; j--) {
            const part = segParts[j];
            if (part === '..') {
                pendingDotDot++;
            } else if (pendingDotDot > 0) {
                pendingDotDot--;  // This segment is canceled by a pending ..
            } else {
                parts.push(part);
            }
        }
    }

    // Add any remaining .. that couldn't be resolved
    while (pendingDotDot > 0) {
        parts.push('..');
        pendingDotDot--;
    }

    // Reverse to get correct order
    parts.reverse();

    // Return with leading slash if absolute
    return (hasAbsolute ? '/' : '') + parts.join('/');
};

export { basename, dirname, join, resolve };
