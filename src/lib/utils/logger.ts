/**
 * Progress node representing a step in a nested progress tree.
 * Walk up via `parent` to access enclosing steps.
 */
interface ProgressNode {
    /** Current step number at this level. */
    step: number;
    /** Total number of steps at this level. */
    totalSteps: number;
    /** Name of the current step (undefined for anonymous steps). */
    stepName?: string;
    /** Parent node (undefined for root level). */
    parent?: ProgressNode;
}

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
    /** Output data to stdout (for piping). */
    output(text: string): void;
    /** Called on progress step updates with the current node. */
    onProgress(node: ProgressNode): void;
}

/**
 * Default logger implementation (browser-safe).
 */
const defaultLogger: Logger = {
    log: (...args) => console.log(...args),
    warn: (...args) => console.warn(...args),
    error: (...args) => console.error(...args),
    debug: (...args) => console.log(...args),
    output: text => console.log(text),
    onProgress: (node) => {
        // step 0 is the begin notification - nothing to print
        if (node.step === 0) return;

        // Calculate depth by walking up
        let depth = 0;
        for (let n = node.parent; n; n = n.parent) depth++;

        const indent = '  '.repeat(depth);
        const name = node.stepName ?? '';
        console.log(`${indent}[${node.step}/${node.totalSteps}] ${name}`);
    }
};

let impl: Logger = defaultLogger;
let quiet = false;

/**
 * Progress tracking with nested step support.
 * Access via logger.progress.begin(), logger.progress.step()
 */
class Progress {
    private currentNode: ProgressNode | undefined;

    /**
     * Start a multi-step progress operation. Creates a new node with current as parent.
     * Calls onProgress with step: 0 to notify consumers of the new progress block.
     * @param totalSteps - Total number of steps in the operation.
     */
    begin(totalSteps: number) {
        this.currentNode = {
            step: 0,
            totalSteps,
            stepName: undefined,
            parent: this.currentNode
        };

        if (!quiet) impl.onProgress(this.currentNode);
    }

    /**
     * Advance to the next step. Auto-increments the step counter.
     * Auto-ends when all steps are complete.
     * @param name - Optional name of the step.
     */
    step(name?: string) {
        if (!this.currentNode) return;

        this.currentNode.step++;
        this.currentNode.stepName = name;

        if (!quiet) impl.onProgress(this.currentNode);

        // Auto-end when all steps complete
        if (this.currentNode.step === this.currentNode.totalSteps) {
            this.currentNode = this.currentNode.parent;
        }
    }
}

/**
 * Global logger instance with injectable implementation.
 * Use setLogger() to provide a custom implementation (e.g., Node.js with process.stdout).
 * Use setQuiet() to suppress log/warn/progress output.
 */
const logger = {
    /**
     * Progress tracking with nested step support.
     * Call begin(n) to start, then step() n times. Auto-ends when complete.
     */
    progress: new Progress(),

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
     * Output data to stdout (for piping). Always shown, even in quiet mode.
     * @param text - The text to output.
     */
    output(text: string) {
        impl.output(text);
    }
};

export { logger };
export type { Logger, ProgressNode };
