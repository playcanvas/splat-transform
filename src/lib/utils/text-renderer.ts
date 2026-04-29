import { fmtBytes, fmtTime } from './fmt';
import { logger, verbosityRank, type LogEvent, type Renderer } from './logger';

/**
 * Output streams and optional memory-usage probe for {@link TextRenderer}.
 */
interface TextRendererOptions {
    /**
     * Receives all status chunks (scopes, bars, messages). May contain
     * partial-line writes (e.g. progress-bar `#` ticks). For TTY output,
     * hand this to a stream that flushes on partials
     * (`process.stderr.write.bind(process.stderr)` in Node) so bars
     * render in place. For non-interactive output (CI logs, file
     * redirects), wrap in a line buffer that holds chunks until a `\n`
     * arrives - the bar's incremental writes then coalesce into a single
     * complete line per bar.
     */
    write: (chunk: string) => void;
    /**
     * Receives `output` events, one logical unit per call, each already
     * terminated with `\n` by the renderer. Hand this to the pipeable
     * channel (typically `process.stdout.write.bind(process.stdout)`).
     * Defaults to the same sink as `write` when omitted.
     */
    output?: (chunk: string) => void;
    /**
     * Optional memory probe. Used by the `[rss: X, heap: X, ab: X]`
     * overlay gated by the renderer's `mem` field. Use
     * `process.memoryUsage` in Node.
     */
    getMemoryUsage?: () => { rss: number; heapUsed: number; arrayBuffers: number };
}

const indent = (depth: number): string => '  '.repeat(Math.max(0, depth));

const BAR_WIDTH = 20;

/**
 * Default human-readable text renderer. Emits one event per line - no
 * carriage-return rewriting, no TTY detection, no buffering. Bars render
 * as `[#### ...... ] duration`, with `#` appended incrementally on each
 * `barTick` and the remainder padded with `.` on `barEnd`. `output`
 * events are treated as line-oriented: their text is written to the
 * pipeable sink with a trailing `\n` appended (callers should not include
 * one themselves).
 *
 * Verbosity is consulted directly from the shared {@link logger} on each
 * event, so this renderer alone decides what to display - the core
 * delivers every scope/bar lifecycle event so embedders consuming the
 * event stream see a faithful record. The display rules are:
 *
 * - `quiet` - suppresses every scope/bar lifecycle line (start, tick,
 *   end - including failed ends). Errors, warnings and `output` still
 *   show.
 * - `normal` (default) - shows scope/bar headers and bar progress;
 *   shows failed `scopeEnd` / `barEnd` footers (the "failed in ..."
 *   cascade from `logger.error` / `unwindAll(true)`); hides successful
 *   `scopeEnd` footers.
 * - `verbose` - shows everything, including successful `scopeEnd`
 *   footers ("done in ...").
 *
 * Sinks are injected (no `process` reference here) so the renderer works in
 * both Node CLI and browser/bundle contexts: the CLI passes
 * `process.stderr.write` for status and `process.stdout.write` for raw
 * output; library/browser consumers can pass a `console.log` line buffer.
 */
class TextRenderer implements Renderer {
    private readonly write: (chunk: string) => void;

    private readonly output: (chunk: string) => void;

    private readonly getMemoryUsage?: () => { rss: number; heapUsed: number; arrayBuffers: number };

    /**
     * When true, scope-end and bar-end lines gain a
     * `[rss: X, heap: X, ab: X]` suffix sourced from
     * {@link TextRendererOptions.getMemoryUsage}. No effect when the
     * probe is omitted. Defaults to `true` when `getMemoryUsage` is
     * provided so embedders that supply a probe see the overlay
     * automatically (matching the prior behavior). Mutable so the
     * host can toggle the overlay without re-installing the renderer.
     */
    mem: boolean;

    /** True while a bar header has been written without its closing `\n`. */
    private lineDirty = false;

    /**
     * Hash count already written for the active bar. Bars are strictly LIFO
     * (the active-scope stack guarantees it), so a single counter suffices.
     */
    private barFilled = 0;

    constructor(options: TextRendererOptions) {
        this.write = options.write;
        this.output = options.output ?? options.write;
        this.getMemoryUsage = options.getMemoryUsage;
        this.mem = options.getMemoryUsage !== undefined;
    }

    private rank(): number {
        return verbosityRank[logger.getVerbosity()];
    }

    handle(event: LogEvent): void {
        switch (event.kind) {
            case 'scopeStart': {
                if (this.rank() < verbosityRank.normal) return;
                this.commitDirty();
                const numbered = event.index !== undefined && event.total !== undefined ?
                    `[${event.index}/${event.total}] ` : '';
                this.write(`${indent(event.depth)}\u25b8 ${numbered}${event.name}\n`);
                return;
            }
            case 'scopeEnd': {
                const rank = this.rank();
                if (event.failed) {
                    if (rank < verbosityRank.normal) return;
                } else if (rank < verbosityRank.verbose) {
                    return;
                }
                this.commitDirty();
                const verb = event.failed ? 'failed in' : 'done in';
                this.write(`${indent(event.depth + 1)}${verb} ${fmtTime(event.durationMs)}${this.memSuffix()}\n`);
                return;
            }
            case 'barStart': {
                if (this.rank() < verbosityRank.normal) return;
                this.commitDirty();
                this.write(`${indent(event.depth)}\u25b8 ${event.name} [`);
                this.lineDirty = true;
                this.barFilled = 0;
                return;
            }
            case 'barTick': {
                if (!this.lineDirty) return;
                if (this.rank() < verbosityRank.normal) return;
                const target = event.total <= 0 ? 0 :
                    Math.min(BAR_WIDTH, Math.floor((event.current / event.total) * BAR_WIDTH));
                if (target > this.barFilled) {
                    this.write('#'.repeat(target - this.barFilled));
                    this.barFilled = target;
                }
                return;
            }
            case 'barEnd': {
                if (this.rank() < verbosityRank.normal) return;
                const suffix = event.failed ?
                    `] (failed) ${fmtTime(event.durationMs)}` :
                    `] ${fmtTime(event.durationMs)}`;
                if (this.lineDirty) {
                    const remaining = Math.max(0, BAR_WIDTH - this.barFilled);
                    this.write(`${'.'.repeat(remaining)}${suffix}${this.memSuffix()}\n`);
                    this.lineDirty = false;
                    this.barFilled = 0;
                } else {
                    // bar's inline line was committed early by a nested
                    // event (e.g. a child group/message). Emit a recap
                    // line whose fill reflects actual progress, so bars
                    // that ended early or failed don't read as complete.
                    const filled = event.total <= 0 ? 0 :
                        Math.min(BAR_WIDTH, Math.floor((event.current / event.total) * BAR_WIDTH));
                    const bar = `${'#'.repeat(filled)}${'.'.repeat(BAR_WIDTH - filled)}`;
                    this.write(`${indent(event.depth)}\u25b8 ${event.name} [${bar}${suffix}${this.memSuffix()}\n`);
                }
                return;
            }
            case 'message': {
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
        if (!this.mem || !this.getMemoryUsage) return '';
        const m = this.getMemoryUsage();
        return `  [rss: ${fmtBytes(m.rss)}, heap: ${fmtBytes(m.heapUsed)}, ab: ${fmtBytes(m.arrayBuffers)}]`;
    }
}

export { TextRenderer };
export type { TextRendererOptions };
