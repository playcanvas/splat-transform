import { fmtBytes, fmtTime } from './fmt';
import type { LogEvent, Renderer } from './logger';

/**
 * Output streams and optional memory-usage probe for {@link TextRenderer}.
 */
interface TextRendererOptions {
    /**
     * Receives all status chunks (scopes, bars, messages). May contain
     * partial-line writes - hand this to a stream that flushes on partials
     * (e.g. `process.stderr.write.bind(process.stderr)` in Node).
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
     * Optional memory probe. When provided, scope-end and bar-end lines
     * gain a `[rss: X, heap: X, ab: X]` overlay. Use `process.memoryUsage`
     * in Node, or omit for a clean view.
     */
    getMemoryUsage?: () => { rss: number; heapUsed: number; arrayBuffers: number };
}

const indent = (depth: number): string => '  '.repeat(Math.max(0, depth));

const BAR_WIDTH = 20;

/**
 * Default human-readable text renderer. Emits one event per line - no
 * carriage-return rewriting, no TTY detection, no buffering. Scope starts
 * always emit a header line; successful `scopeEnd` footers are filtered
 * out at `normal` verbosity by `LoggerCore` (kept at `verbose`, and always
 * shown when `failed`), so default-mode runs see headers without timing
 * footers and `--verbose` adds the matching `done in ...` lines. Bars
 * render as `[#### ...... ] duration`, with `#` appended incrementally on
 * each `barTick` and the remainder padded with `.` on `barEnd`. `output`
 * events are treated as line-oriented: their text is written to the
 * pipeable sink with a trailing `\n` appended (callers should not include
 * one themselves).
 *
 * Verbosity filtering is handled centrally by `LoggerCore` - this renderer
 * receives only events that have already passed the visibility gate, so it
 * is pure presentation.
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
    }

    handle(event: LogEvent): void {
        switch (event.kind) {
            case 'scopeStart': {
                this.commitDirty();
                const numbered = event.index !== undefined && event.total !== undefined ?
                    `[${event.index}/${event.total}] ` : '';
                this.write(`${indent(event.depth)}\u25b8 ${numbered}${event.name}\n`);
                return;
            }
            case 'scopeEnd': {
                this.commitDirty();
                const verb = event.failed ? 'failed in' : 'done in';
                this.write(`${indent(event.depth + 1)}${verb} ${fmtTime(event.durationMs)}${this.memSuffix()}\n`);
                return;
            }
            case 'barStart': {
                this.commitDirty();
                this.write(`${indent(event.depth)}\u25b8 ${event.name} [`);
                this.lineDirty = true;
                this.barFilled = 0;
                return;
            }
            case 'barTick': {
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
        if (!this.getMemoryUsage) return '';
        const m = this.getMemoryUsage();
        return `  [rss: ${fmtBytes(m.rss)}, heap: ${fmtBytes(m.heapUsed)}, ab: ${fmtBytes(m.arrayBuffers)}]`;
    }
}

export { TextRenderer };
export type { TextRendererOptions };
