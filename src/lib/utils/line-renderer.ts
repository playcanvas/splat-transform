import { fmtBytes, fmtTime } from './fmt';
import type { LogEvent, MessageKind, Renderer, Verbosity } from './logger';

/**
 * Output streams and optional memory-usage probe for {@link LineRenderer}.
 */
interface LineRendererOptions {
    /**
     * Receives all status chunks (scopes, bars, messages). May contain
     * partial-line writes - hand this to a stream that flushes on partials
     * (e.g. `process.stderr.write.bind(process.stderr)` in Node).
     */
    write: (chunk: string) => void;
    /**
     * Receives raw `output` events. Hand this to the pipeable channel
     * (typically `process.stdout.write.bind(process.stdout)`). Defaults to
     * the same sink as `write` when omitted.
     */
    output?: (chunk: string) => void;
    /**
     * Optional memory probe. When provided, scope-end and bar-end lines
     * gain a `[rss: X, heap: X, ab: X]` overlay. Use `process.memoryUsage`
     * in Node, or omit for a clean view.
     */
    getMemoryUsage?: () => { rss: number; heapUsed: number; arrayBuffers: number };
}

const verbosityRank: Record<Verbosity, number> = {
    quiet: 0,
    normal: 1,
    verbose: 2
};

const messageMinVerbosity: Record<MessageKind, Verbosity> = {
    error: 'quiet',
    warn: 'quiet',
    info: 'normal',
    debug: 'verbose'
};

const messageVisible = (kind: MessageKind, v: Verbosity): boolean => {
    return verbosityRank[v] >= verbosityRank[messageMinVerbosity[kind]];
};

const taskVisible = (v: Verbosity): boolean => verbosityRank[v] >= verbosityRank.normal;

const indent = (depth: number): string => '  '.repeat(Math.max(0, depth));

const BAR_WIDTH = 20;

/**
 * Single line-based renderer. Emits one event per line - no carriage-return
 * rewriting, no TTY detection, no buffering. Scope starts and ends each
 * produce their own line, so even fast childless scopes render as a header /
 * footer pair. Bars render as `[#### ...... ] duration`, with `#` appended
 * incrementally on each `barTick` and the remainder padded with `.` on
 * `barEnd`.
 *
 * Sinks are injected (no `process` reference here) so the renderer works in
 * both Node CLI and browser/bundle contexts: the CLI passes
 * `process.stderr.write` for status and `process.stdout.write` for raw
 * output; library/browser consumers can pass a `console.log` line buffer.
 */
class LineRenderer implements Renderer {
    private verbosity: Verbosity = 'normal';

    private readonly write: (chunk: string) => void;

    private readonly output: (chunk: string) => void;

    private readonly getMemoryUsage?: () => { rss: number; heapUsed: number; arrayBuffers: number };

    /** True while a bar header has been written without its closing `\n`. */
    private lineDirty = false;

    /**
     * Hash count already written for the active bar. Bars are strictly LIFO
     * (the active-scope stack guarantees it), so a single counter suffices.
     */
    private barFilled = 0;

    constructor(options: LineRendererOptions) {
        this.write = options.write;
        this.output = options.output ?? options.write;
        this.getMemoryUsage = options.getMemoryUsage;
    }

    setVerbosity(v: Verbosity): void {
        this.verbosity = v;
    }

    handle(event: LogEvent): void {
        switch (event.kind) {
            case 'scopeStart': {
                this.commitDirty();
                if (!taskVisible(this.verbosity)) return;
                const numbered = event.index !== undefined && event.total !== undefined ?
                    `[${event.index}/${event.total}] ` : '';
                this.write(`${indent(event.depth)}\u25b8 ${numbered}${event.name}\n`);
                return;
            }
            case 'scopeEnd': {
                this.commitDirty();
                if (!taskVisible(this.verbosity)) return;
                const verb = event.failed ? 'failed in' : 'done in';
                this.write(`${indent(event.depth + 1)}${verb} ${fmtTime(event.durationMs)}${this.memSuffix()}\n`);
                return;
            }
            case 'barStart': {
                this.commitDirty();
                if (!taskVisible(this.verbosity)) return;
                this.write(`${indent(event.depth)}\u25b8 ${event.name} [`);
                this.lineDirty = true;
                this.barFilled = 0;
                return;
            }
            case 'barTick': {
                if (!taskVisible(this.verbosity)) return;
                if (!this.lineDirty) return;
                const target = event.total <= 0 ? 0 :
                    Math.min(BAR_WIDTH, Math.floor((event.current / event.total) * BAR_WIDTH));
                if (target > this.barFilled) {
                    this.write('#'.repeat(target - this.barFilled));
                    this.barFilled = target;
                }
                return;
            }
            case 'barEnd': {
                if (!taskVisible(this.verbosity)) return;
                const remaining = Math.max(0, BAR_WIDTH - this.barFilled);
                const tail = '.'.repeat(remaining);
                const suffix = event.failed ?
                    `] (failed) ${fmtTime(event.durationMs)}` :
                    `] ${fmtTime(event.durationMs)}`;
                if (this.lineDirty) {
                    this.write(`${tail}${suffix}${this.memSuffix()}\n`);
                    this.lineDirty = false;
                    this.barFilled = 0;
                } else {
                    // bar header was suppressed (e.g. quiet) but end is still
                    // inside taskVisible - emit a synthetic full line for
                    // consistency.
                    this.write(`${indent(event.depth)}\u25b8 ${event.name} [${'#'.repeat(BAR_WIDTH)}${suffix}${this.memSuffix()}\n`);
                }
                return;
            }
            case 'message': {
                if (!messageVisible(event.level, this.verbosity)) return;
                this.commitDirty();
                // info/debug get a `\u00b7` glyph only when nested under a
                // scope - at depth 0 they're framing lines (banners,
                // summaries) and read cleaner without decoration. warn/error
                // always carry their severity glyph regardless of depth.
                let prefix = '';
                if (event.level === 'error') prefix = '\u2717 ';
                else if (event.level === 'warn') prefix = '! ';
                else if (event.depth > 0) prefix = '\u00b7 ';
                this.write(`${indent(event.depth)}${prefix}${event.text}\n`);
                return;
            }
            case 'output': {
                this.commitDirty();
                this.output(`${event.text}\n`);
            }
        }
    }

    /**
     * Terminate any in-progress bar line so subsequent output starts on a
     * fresh line. The bar's `#` count is preserved on its own line; the
     * eventual `barEnd` will produce its own footer line if it fires later.
     */
    private commitDirty(): void {
        if (this.lineDirty) {
            this.write('\n');
            this.lineDirty = false;
        }
    }

    private memSuffix(): string {
        if (!this.getMemoryUsage) return '';
        const m = this.getMemoryUsage();
        return `  [rss: ${fmtBytes(m.rss)}, heap: ${fmtBytes(m.heapUsed)}, ab: ${fmtBytes(m.arrayBuffers)}]`;
    }
}

export { LineRenderer };
export type { LineRendererOptions };
