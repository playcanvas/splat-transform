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

export { dirname, join };
