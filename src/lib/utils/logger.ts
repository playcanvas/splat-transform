/**
 * Logger interface for injectable logging implementation.
 */
interface Logger {
    /** Log normal messages. */
    log(...args: any[]): void;
    /** Log warning messages. */
    warn(...args: any[]): void;
    /** Log error messages. */
    error(...args: any[]): void;
    /** Log debug/verbose messages. */
    debug(...args: any[]): void;
    /** Output text without newline (for progress indicators). */
    progress(text: string): void;
}

/**
 * Default logger implementation (browser-safe).
 * Progress is a no-op since process.stdout is not available in browsers.
 */
const defaultLogger: Logger = {
    log: (...args) => console.log(...args),
    warn: (...args) => console.warn(...args),
    error: (...args) => console.error(...args),
    debug: (...args) => console.log(...args),
    progress: () => {}
};

let impl: Logger = defaultLogger;
let quiet = false;

/**
 * Global logger instance with injectable implementation.
 * Use setLogger() to provide a custom implementation (e.g., Node.js with process.stdout).
 * Use setQuiet() to suppress log/warn/progress output.
 */
const logger = {
    /**
     * Set a custom logger implementation.
     * @param l - The logger implementation to use.
     */
    setLogger(l: Logger) {
        impl = l;
    },

    /**
     * Set quiet mode. When quiet, log/warn/progress are suppressed. Errors always show.
     * @param q - Whether to enable quiet mode.
     */
    setQuiet(q: boolean) {
        quiet = q;
    },

    /**
     * Log normal messages. Suppressed in quiet mode.
     * @param args - The arguments to log.
     */
    log(...args: any[]) {
        if (!quiet) impl.log(...args);
    },

    /**
     * Log warning messages. Suppressed in quiet mode.
     * @param args - The arguments to log.
     */
    warn(...args: any[]) {
        if (!quiet) impl.warn(...args);
    },

    /**
     * Log error messages. Always shown, even in quiet mode.
     * @param args - The arguments to log.
     */
    error(...args: any[]) {
        impl.error(...args);
    },

    /**
     * Log debug/verbose messages. Suppressed in quiet mode.
     * @param args - The arguments to log.
     */
    debug(...args: any[]) {
        if (!quiet) impl.debug(...args);
    },

    /**
     * Output text without newline (for progress indicators). Suppressed in quiet mode.
     * @param text - The text to output.
     */
    progress(text: string) {
        if (!quiet) impl.progress(text);
    }
};

export { logger };
export type { Logger };
