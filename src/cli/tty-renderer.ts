import process from 'node:process';

import type { LogEvent, Renderer, Verbosity } from '../lib/index';

interface WriteStreamLike {
    write(chunk: string): unknown;
    isTTY?: boolean;
}

interface TtyRendererOptions {
    stream?: WriteStreamLike;
    showMem?: boolean;
}

const verbosityRank: Record<Verbosity, number> = {
    quiet: 0,
    normal: 1,
    verbose: 2
};

const fmtTime = (ms: number): string => {
    if (ms >= 60_000) return `${(ms / 60_000).toFixed(2)}m`;
    return `${(ms / 1000).toFixed(3)}s`;
};

const fmtMem = (): string => {
    const m = process.memoryUsage();
    const mb = (n: number) => `${(n / (1024 * 1024)).toFixed(0)}MB`;
    return `  [rss: ${mb(m.rss)}, heap: ${mb(m.heapUsed)}, ab: ${mb(m.arrayBuffers)}]`;
};

const indent = (depth: number): string => '  '.repeat(Math.max(0, depth));

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
 * Numbered groups at depth 0 render in the legacy `[N/T] name` flush-left
 * style; nested numbered groups use indented `[N/T] name`; un-numbered groups
 * use `\u25b8` / `\u25c2` glyphs indented by depth.
 */
class TtyRenderer implements Renderer {
    private verbosity: Verbosity = 'normal';

    private readonly stream: WriteStreamLike;

    private readonly isTty: boolean;

    private readonly showMem: boolean;

    /** True if we have written a partial line (no trailing newline). */
    private barOpen = false;

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

    private taskVisible(): boolean {
        return verbosityRank[this.verbosity] >= verbosityRank.normal;
    }

    private messageVisible(level: 'error' | 'warn' | 'info' | 'debug'): boolean {
        if (level === 'error') return true;
        if (level === 'debug') return verbosityRank[this.verbosity] >= verbosityRank.verbose;
        return verbosityRank[this.verbosity] >= verbosityRank.normal;
    }

    handle(event: LogEvent): void {
        switch (event.kind) {
            case 'groupStart': {
                this.commitBar();
                if (!this.taskVisible()) break;
                const numbered = event.index !== undefined && event.total !== undefined;
                if (numbered && event.depth === 0) {
                    this.write(`\n[${event.index}/${event.total}] ${event.name}\n`);
                } else if (numbered) {
                    this.write(`${indent(event.depth)}[${event.index}/${event.total}] ${event.name}\n`);
                } else {
                    this.write(`${indent(event.depth)}\u25b8 ${event.name}\n`);
                }
                break;
            }
            case 'groupEnd': {
                this.commitBar();
                const numbered = event.index !== undefined && event.total !== undefined;
                if (numbered && event.depth === 0) {
                    if (!event.failed) break;
                    this.write(`${indent(event.depth)}\u2717 ${event.name} (failed) ${fmtTime(event.durationMs)}${this.memSuffix()}\n`);
                } else {
                    if (!this.taskVisible()) break;
                    const glyph = event.failed ? '\u2717' : '\u25c2';
                    this.write(`${indent(event.depth)}${glyph} ${event.name} ${fmtTime(event.durationMs)}${this.memSuffix()}\n`);
                }
                break;
            }
            case 'stepStart': {
                this.commitBar();
                if (!this.taskVisible()) break;
                this.write(`${indent(event.depth)}\u25b8 ${event.name}\n`);
                break;
            }
            case 'stepEnd': {
                this.commitBar();
                if (!this.taskVisible()) break;
                const glyph = event.failed ? '\u2717' : '\u2713';
                this.write(`${indent(event.depth)}${glyph} ${event.name} ${fmtTime(event.durationMs)}${this.memSuffix()}\n`);
                break;
            }
            case 'barStart': {
                this.commitBar();
                if (!this.taskVisible()) break;
                if (this.isTty) {
                    this.write(`${indent(event.depth)}[${renderBar(0, event.total)}]   0%`);
                    this.barOpen = true;
                }
                break;
            }
            case 'barTick': {
                if (!this.taskVisible()) break;
                if (this.isTty && this.barOpen) {
                    const pct = event.total <= 0 ? 0 : Math.min(100, Math.round((event.current / event.total) * 100));
                    this.write(`\r${indent(event.depth)}[${renderBar(event.current, event.total)}] ${String(pct).padStart(3, ' ')}%`);
                }
                break;
            }
            case 'barEnd': {
                if (!this.taskVisible()) break;
                if (this.isTty && this.barOpen) {
                    this.write(`\r${indent(event.depth)}[${renderBar(1, 1)}] ${fmtTime(event.durationMs)}${this.memSuffix()}\n`);
                    this.barOpen = false;
                } else {
                    this.write(`${indent(event.depth)}[${renderBar(1, 1)}] ${fmtTime(event.durationMs)}${this.memSuffix()}\n`);
                }
                break;
            }
            case 'message': {
                this.commitBar();
                if (!this.messageVisible(event.level)) break;
                // info/debug get a `\u00b7` glyph only when nested under a scope -
                // at depth 0 they're framing lines (banners, summaries) and read
                // cleaner without decoration. warn/error always carry their
                // severity glyph regardless of depth.
                let prefix = '';
                if (event.level === 'error') prefix = '\u2717 ';
                else if (event.level === 'warn') prefix = '! ';
                else if (event.depth > 0) prefix = '\u00b7 ';
                this.write(`${indent(event.depth)}${prefix}${event.text}\n`);
                break;
            }
            case 'output': {
                this.commitBar();
                process.stdout.write(`${event.text}\n`);
                break;
            }
        }
    }
}

export { TtyRenderer };
