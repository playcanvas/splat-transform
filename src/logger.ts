type LogLevel = 'silent' | 'normal';

/**
 * Simple logger utility. Currently supports quiet mode. Designed to be extended for multiple log
 * levels in the future.
 */
class Logger {
    private level: LogLevel = 'normal';

    setLevel(level: LogLevel) {
        this.level = level;
    }

    setQuiet(quiet: boolean) {
        this.level = quiet ? 'silent' : 'normal';
    }

    /**
     * Log informational messages (file operations, progress, etc.). Suppressed in quiet mode.
     */
    info(...args: any[]) {
        if (this.level !== 'silent') {
            console.log(...args);
        }
    }

    /**
     * Log warning messages. Suppressed in quiet mode.
     */
    warn(...args: any[]) {
        if (this.level !== 'silent') {
            console.warn(...args);
        }
    }

    /**
     * Log error messages. Always shown, even in quiet mode.
     */
    error(...args: any[]) {
        console.error(...args);
    }

    /**
     * Log debug/verbose messages. Currently treated the same as info, but can be filtered
     * separately in the future.
     */
    debug(...args: any[]) {
        if (this.level !== 'silent') {
            console.log(...args);
        }
    }

    /**
     * Write progress indicators directly to stdout (without newline). Suppressed in quiet mode.
     */
    progress(text: string) {
        if (this.level !== 'silent') {
            process.stdout.write(text);
        }
    }

    /**
     * Check if logger is in quiet/silent mode.
     */
    isQuiet(): boolean {
        return this.level === 'silent';
    }
}

// Export singleton instance
export const logger = new Logger();
