import process from 'node:process';

import type { LogEvent, Renderer, Verbosity } from '../lib/index';
import {
    fmtTime,
    indent,
    isPhaseHeader,
    messageVisible,
    taskVisible
} from '../lib/utils/logger';

interface WriteStreamLike {
    write(chunk: string): unknown;
    isTTY?: boolean;
}

interface TtyRendererOptions {
    stream?: WriteStreamLike;
    showMem?: boolean;
}

const fmtMem = (): string => {
    const m = process.memoryUsage();
    const mb = (n: number) => `${(n / (1024 * 1024)).toFixed(0)}MB`;
    return `  [rss: ${mb(m.rss)}, heap: ${mb(m.heapUsed)}, ab: ${mb(m.arrayBuffers)}]`;
};

const BAR_WIDTH = 20;

const renderBar = (current: number, total: number): string => {
    const ratio = total <= 0 ? 0 : Math.min(1, Math.max(0, current / total));
    const filled = Math.round(ratio * BAR_WIDTH);
    return '\u2588'.repeat(filled) + '\u2591'.repeat(BAR_WIDTH - filled);
};

/**
 * TTY-aware renderer used by the CLI. Uses Unicode glyphs and animates
 * progress bars in place via carriage returns. Falls back to plain newline
 * output when stderr is not a TTY (e.g. when output is piped).
 *
 * Buffers `scopeStart` events so that scopes with no nested children
 * collapse to a single `\u2713 name X.XXXs` line, while scopes with children
 * print a header `\u25b8 name`, indented children, and an indented closer
 * `done in X.XXXs` (or `failed in X.XXXs`).
 *
 * Bars render as a single line `\u25b8 name [bar] %` ticking in place,
 * finalizing as `\u2713 name [bar] X.XXXs` (or `\u2717 name [bar] (failed) X.XXXs`).
 *
 * Numbered groups at depth 0 (phase headers) are not buffered - they print
 * immediately as `[N/T] name` flush left, and their closer renders like any
 * other group's (`done in X.XXXs` / `failed in X.XXXs` at content indent).
 */
class TtyRenderer implements Renderer {
    private verbosity: Verbosity = 'normal';

    private readonly stream: WriteStreamLike;

    private readonly isTty: boolean;

    private readonly showMem: boolean;

    /** True if we have written a partial line (no trailing newline). */
    private barOpen = false;

    /**
     * Queue of `scopeStart` events whose header has not yet been written,
     * ordered by depth ascending (shallowest first).
     */
    private pendingStarts: Array<Extract<LogEvent, { kind: 'scopeStart' }>> = [];

    /**
     * @param options - Configuration. Supports an optional output `stream` (defaults to
     * `process.stderr`) and a `showMem` flag to append a memory-usage overlay.
     */
    constructor(options: TtyRendererOptions = {}) {
        this.stream = options.stream ?? process.stderr;
        this.isTty = !!this.stream.isTTY;
        this.showMem = !!options.showMem;
    }

    setVerbosity(v: Verbosity): void {
        this.verbosity = v;
    }

    private write(text: string): void {
        this.stream.write(text);
    }

    private commitBar(): void {
        if (this.barOpen) {
            this.write('\n');
            this.barOpen = false;
        }
    }

    private memSuffix(): string {
        return this.showMem ? fmtMem() : '';
    }

    handle(event: LogEvent): void {
        switch (event.kind) {
            case 'scopeStart': {
                this.commitBar();
                if (isPhaseHeader(event)) {
                    this.flushPendingDownTo(0);
                    if (taskVisible(this.verbosity)) {
                        this.write(`\n[${event.index}/${event.total}] ${event.name}\n`);
                    }
                    return;
                }
                this.flushPendingDownTo(event.depth);
                this.pendingStarts.push(event);
                return;
            }
            case 'scopeEnd': {
                this.commitBar();
                const top = this.pendingStarts[this.pendingStarts.length - 1];
                if (top && top.depth === event.depth) {
                    this.pendingStarts.pop();
                    this.flushPendingDownTo(event.depth);
                    if (!taskVisible(this.verbosity)) return;
                    const glyph = event.failed ? '\u2717' : '\u2713';
                    const suffix = event.failed ? ` (failed) ${fmtTime(event.durationMs)}` : ` ${fmtTime(event.durationMs)}`;
                    this.write(`${indent(event.depth)}${glyph} ${event.name}${suffix}${this.memSuffix()}\n`);
                    return;
                }
                if (!taskVisible(this.verbosity)) return;
                const verb = event.failed ? 'failed in' : 'done in';
                this.write(`${indent(event.depth + 1)}${verb} ${fmtTime(event.durationMs)}${this.memSuffix()}\n`);
                return;
            }
            case 'barStart': {
                this.commitBar();
                this.flushPendingDownTo(event.depth);
                if (!taskVisible(this.verbosity)) return;
                if (this.isTty) {
                    this.write(`${indent(event.depth)}\u25b8 ${event.name}  [${renderBar(0, event.total)}]   0%`);
                    this.barOpen = true;
                }
                return;
            }
            case 'barTick': {
                if (!taskVisible(this.verbosity)) return;
                if (this.isTty && this.barOpen) {
                    const pct = event.total <= 0 ? 0 : Math.min(100, Math.round((event.current / event.total) * 100));
                    this.write(`\r${indent(event.depth)}\u25b8 ${event.name}  [${renderBar(event.current, event.total)}] ${String(pct).padStart(3, ' ')}%`);
                }
                return;
            }
            case 'barEnd': {
                if (!taskVisible(this.verbosity)) return;
                const glyph = event.failed ? '\u2717' : '\u2713';
                const suffix = event.failed ? ` (failed) ${fmtTime(event.durationMs)}` : ` ${fmtTime(event.durationMs)}`;
                const bar = renderBar(event.current, event.total);
                if (this.isTty && this.barOpen) {
                    this.write(`\r${indent(event.depth)}${glyph} ${event.name}  [${bar}]${suffix}${this.memSuffix()}\n`);
                    this.barOpen = false;
                } else {
                    this.write(`${indent(event.depth)}${glyph} ${event.name}  [${bar}]${suffix}${this.memSuffix()}\n`);
                }
                return;
            }
            case 'message': {
                if (!messageVisible(event.level, this.verbosity)) return;
                this.commitBar();
                this.flushPendingDownTo(event.depth);
                // info/debug get a `\u00b7` glyph only when nested under a scope -
                // at depth 0 they're framing lines (banners, summaries) and read
                // cleaner without decoration. warn/error always carry their
                // severity glyph regardless of depth.
                let prefix = '';
                if (event.level === 'error') prefix = '\u2717 ';
                else if (event.level === 'warn') prefix = '! ';
                else if (event.depth > 0) prefix = '\u00b7 ';
                this.write(`${indent(event.depth)}${prefix}${event.text}\n`);
                return;
            }
            case 'output': {
                this.commitBar();
                this.flushPendingDownTo(0);
                process.stdout.write(`${event.text}\n`);
            }
        }
    }

    /**
     * Flush any pending `scopeStart` headers whose depth is strictly less
     * than `depth`. Walks shallowest-first so parent headers print before
     * children. Flushed entries are removed from the queue; their presence
     * in the queue WAS the "is empty" signal, so removing them implicitly
     * marks them as non-empty.
     *
     * @param depth - The depth threshold.
     */
    private flushPendingDownTo(depth: number): void {
        while (this.pendingStarts.length > 0 && this.pendingStarts[0].depth < depth) {
            const entry = this.pendingStarts.shift()!;
            if (!taskVisible(this.verbosity)) continue;
            const numbered = entry.index !== undefined && entry.total !== undefined;
            const headerName = numbered ? `[${entry.index}/${entry.total}] ${entry.name}` : entry.name;
            this.write(`${indent(entry.depth)}\u25b8 ${headerName}\n`);
        }
    }
}

export { TtyRenderer };
