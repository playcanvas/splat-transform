/**
 * Verbosity level controlling which messages reach the renderer.
 *
 * - `quiet`   - errors and warnings only.
 * - `normal`  - tasks, bars, info, warn, error (default).
 * - `verbose` - normal + debug messages.
 */
type Verbosity = 'quiet' | 'normal' | 'verbose';

/** Severity tag for free-form messages (ordered descending by severity). */
type MessageKind = 'error' | 'warn' | 'info' | 'debug';

/**
 * Semantic event delivered to a {@link Renderer}. Renderers can filter, format
 * and display these as they wish.
 *
 * `scopeStart` / `scopeEnd` represent the open/close of a {@link Group}.
 * They carry optional `index` / `total` fields when the scope is part of a
 * numbered series, which renderers can use to switch to a `[N/T] name` style.
 *
 * `barStart` / `barTick` / `barEnd` represent an indeterminate progress bar.
 * The bar's `name` is repeated on every event so the renderer can keep its
 * label stable across in-place tick updates.
 */
type LogEvent =
    | { kind: 'scopeStart'; depth: number; name: string; index?: number; total?: number }
    | { kind: 'scopeEnd'; depth: number; name: string; durationMs: number; failed?: boolean; index?: number; total?: number }
    | { kind: 'barStart'; depth: number; name: string; total: number }
    | { kind: 'barTick'; depth: number; name: string; current: number; total: number }
    | { kind: 'barEnd'; depth: number; name: string; durationMs: number; current: number; total: number; failed?: boolean }
    | { kind: 'message'; depth: number; level: MessageKind; text: string }
    | { kind: 'output'; text: string };

/**
 * Renderer interface. Receives semantic events and decides how to display them.
 */
interface Renderer {
    /**
     * Handle a log event.
     * @param event - The event to render.
     */
    handle(event: LogEvent): void;
    /**
     * Set the active verbosity level.
     * @param v - The new verbosity level.
     */
    setVerbosity(v: Verbosity): void;
}

/**
 * Determinate progress bar handle. Closed explicitly via `end()`, or
 * implicitly when an enclosing {@link Group}'s `end()` (or a
 * {@link Logger.unwindAll}) pops it as part of cleanup.
 */
interface Bar {
    /**
     * Advance the bar by `n` ticks.
     * @param n - Number of ticks to advance (default 1).
     */
    tick(n?: number): void;
    /**
     * Close the bar and emit final timing.
     */
    end(): void;
}

/**
 * Named, timed scope returned from {@link Logger.group}. Manages the scope's
 * lifecycle only - free-form messages, nested groups and bars are emitted via
 * the global `logger` (they auto-indent under whatever is on top of the
 * active-scope stack).
 *
 * Open scopes with `logger.group(name)` and close them with `sub.end()` after
 * the body. Embedders that catch their own exceptions (rather than letting
 * them propagate to a `logger.error()` call) should call
 * {@link Logger.unwindAll} from their catch to close any scopes/bars left
 * dangling on the stack.
 */
interface Group {
    /**
     * Close the group, popping anything still open above it on the stack
     * (defensively handles forgotten inner scopes) and emit the timing event.
     */
    end(): void;
}

/**
 * Internal node tracked on the active-scope stack. Both kinds carry their own
 * depth and start time; `group` may additionally carry `index` / `total` so a
 * numbered series can re-emit the same numbering on close.
 */
type Scope =
    | { kind: 'group'; name: string; depth: number; start: number; index?: number; total?: number }
    | { kind: 'bar'; name: string; depth: number; start: number; total: number; current: number };

const now = (): number => {
    if (typeof performance !== 'undefined' && typeof performance.now === 'function') {
        return performance.now();
    }
    return Date.now();
};

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

const fmtTime = (ms: number): string => {
    if (!Number.isFinite(ms) || ms < 0) return `${ms}ms`;
    if (ms < 60_000) return `${(ms / 1000).toFixed(3)}s`;

    const h = Math.floor(ms / 3_600_000);
    const m = Math.floor((ms % 3_600_000) / 60_000);
    const s = ((ms % 60_000) / 1000).toFixed(3);

    return h > 0 ? `${h}h${m}m${s}s` : `${m}m${s}s`;
};

const fmtBytes = (n: number): string => {
    if (!Number.isFinite(n) || n < 0) return `${n}B`;
    if (n < 1024) return `${n}B`;
    if (n < 1024 * 1024) return `${(n / 1024).toFixed(1)}KB`;
    if (n < 1024 * 1024 * 1024) return `${(n / (1024 * 1024)).toFixed(1)}MB`;
    return `${(n / (1024 * 1024 * 1024)).toFixed(2)}GB`;
};

const fmtArgs = (args: any[]): string => {
    return args.map((a) => {
        if (a instanceof Error) return a.stack ?? a.message;
        if (typeof a === 'string') return a;
        if (typeof a === 'number' || typeof a === 'boolean' || a == null) return String(a);
        try {
            return JSON.stringify(a);
        } catch {
            return String(a);
        }
    }).join(' ');
};

const PLAIN_BAR_WIDTH = 20;

const renderPlainBar = (current: number, total: number): string => {
    const ratio = total <= 0 ? 0 : Math.min(1, Math.max(0, current / total));
    const filled = Math.round(ratio * PLAIN_BAR_WIDTH);
    return '#'.repeat(filled) + '.'.repeat(PLAIN_BAR_WIDTH - filled);
};

const indent = (depth: number): string => '  '.repeat(Math.max(0, depth));

const isPhaseHeader = (event: { depth: number; index?: number; total?: number }): boolean => {
    return event.depth === 0 && event.index !== undefined && event.total !== undefined;
};

/**
 * Default browser-safe renderer. Uses only `console.log/warn/error`. Bars
 * are not animated - only the bar's final state is printed when it closes.
 *
 * Buffers `scopeStart` events so that scopes with no nested children
 * collapse to a single `\u2713 name X.XXXs` line, while scopes with children
 * print a header `\u25b8 name`, indented children, and an indented closer
 * `done in X.XXXs` (or `failed in X.XXXs`).
 *
 * Numbered groups at depth 0 (phase headers) are not buffered - they print
 * immediately as `[N/T] name` flush left, and their closer renders like any
 * other group's (`done in X.XXXs` / `failed in X.XXXs` at content indent).
 */
class PlainRenderer implements Renderer {
    private verbosity: Verbosity = 'normal';

    /**
     * Queue of `scopeStart` events whose header has not yet been written,
     * ordered by depth ascending (shallowest first). When a child event
     * arrives at a deeper depth, the entries strictly shallower than the
     * child are flushed (headers written, entries removed).
     */
    private pendingStarts: Array<Extract<LogEvent, { kind: 'scopeStart' }>> = [];

    setVerbosity(v: Verbosity): void {
        this.verbosity = v;
    }

    handle(event: LogEvent): void {
        switch (event.kind) {
            case 'scopeStart': {
                if (isPhaseHeader(event)) {
                    this.flushPendingDownTo(0);
                    if (taskVisible(this.verbosity)) {
                        console.log(`\n[${event.index}/${event.total}] ${event.name}`);
                    }
                    return;
                }
                this.flushPendingDownTo(event.depth);
                this.pendingStarts.push(event);
                return;
            }
            case 'scopeEnd': {
                const top = this.pendingStarts[this.pendingStarts.length - 1];
                if (top && top.depth === event.depth) {
                    this.pendingStarts.pop();
                    this.flushPendingDownTo(event.depth);
                    if (!taskVisible(this.verbosity)) return;
                    const glyph = event.failed ? '\u2717' : '\u2713';
                    const suffix = event.failed ? ` (failed) ${fmtTime(event.durationMs)}` : ` ${fmtTime(event.durationMs)}`;
                    console.log(`${indent(event.depth)}${glyph} ${event.name}${suffix}`);
                    return;
                }
                if (!taskVisible(this.verbosity)) return;
                const verb = event.failed ? 'failed in' : 'done in';
                console.log(`${indent(event.depth + 1)}${verb} ${fmtTime(event.durationMs)}`);
                return;
            }
            case 'barStart': {
                this.flushPendingDownTo(event.depth);
                return;
            }
            case 'barTick':
                return;
            case 'barEnd': {
                if (!taskVisible(this.verbosity)) return;
                const glyph = event.failed ? '\u2717' : '\u2713';
                const suffix = event.failed ? ` (failed) ${fmtTime(event.durationMs)}` : ` ${fmtTime(event.durationMs)}`;
                const bar = renderPlainBar(event.current, event.total);
                console.log(`${indent(event.depth)}${glyph} ${event.name} [${bar}]${suffix}`);
                return;
            }
            case 'message': {
                if (!messageVisible(event.level, this.verbosity)) return;
                this.flushPendingDownTo(event.depth);
                const prefix = event.level === 'warn' ? 'warn: ' : event.level === 'error' ? 'error: ' : '';
                const fn = event.level === 'error' ? console.error : event.level === 'warn' ? console.warn : console.log;
                fn(`${indent(event.depth)}${prefix}${event.text}`);
                return;
            }
            case 'output':
                this.flushPendingDownTo(0);
                console.log(event.text);
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
            console.log(`${indent(entry.depth)}\u25b8 ${headerName}`);
        }
    }
}

/**
 * Active-scope manager and message router. The single shared instance lives
 * inside this module; the public `logger` surface is a thin façade over it.
 */
class LoggerCore {
    /** Stack of currently-open scopes (innermost last). */
    readonly stack: Scope[] = [];

    /**
     * Per-depth numbered-series counter. A counter exists at depth D between
     * the moment `pushGroup(_, { total })` is called at depth D and the
     * moment the parent scope (at depth D-1) closes.
     */
    private counters = new Map<number, { index: number; total: number }>();

    private renderer: Renderer = new PlainRenderer();

    private verbosity: Verbosity = 'normal';

    setRenderer(r: Renderer): void {
        this.renderer = r;
        this.renderer.setVerbosity(this.verbosity);
    }

    setVerbosity(v: Verbosity): void {
        this.verbosity = v;
        this.renderer.setVerbosity(v);
    }

    getVerbosity(): Verbosity {
        return this.verbosity;
    }

    emit(event: LogEvent): void {
        this.renderer.handle(event);
    }

    /**
     * Drop any counters whose depth is strictly greater than `depth`. Called
     * after popping a scope so children's series state doesn't leak across
     * sibling scopes. Counters at `depth` itself survive so the popped
     * scope's siblings (at the same depth) can continue counting.
     *
     * @param depth - The popped scope's own depth.
     */
    private clearDeeperCounters(depth: number): void {
        for (const key of this.counters.keys()) {
            if (key > depth) this.counters.delete(key);
        }
    }

    /**
     * Pop the scope at the top of the stack (no-op if empty) and emit the
     * matching `*End` event. Centralizes the "drop counters at deeper depths
     * + emit end" sequence used by every close path.
     *
     * @param failed - When true, mark the closed scope as having failed.
     */
    private popScope(failed = false): void {
        if (this.stack.length === 0) return;
        const top = this.stack[this.stack.length - 1];
        this.stack.pop();
        this.clearDeeperCounters(top.depth);
        const durationMs = now() - top.start;
        if (top.kind === 'bar') {
            this.emit({
                kind: 'barEnd',
                depth: top.depth,
                name: top.name,
                durationMs,
                current: top.current,
                total: top.total,
                failed
            });
            return;
        }
        const numbering = top.kind === 'group' && top.index !== undefined && top.total !== undefined ?
            { index: top.index, total: top.total } :
            {};
        this.emit({ kind: 'scopeEnd', depth: top.depth, name: top.name, durationMs, failed, ...numbering });
    }

    /**
     * Open a named, timed group at the current depth. Pass `{ total: N }` to
     * start a fresh numbered series at this depth - this group renders as
     * `[1/N]`. Subsequent `pushGroup(name)` calls (no options) at the same
     * depth advance the active series until exhausted; once exhausted (or if
     * no series is active), they render plain.
     *
     * @param name - The group name.
     * @param options - Optional configuration.
     * @param options.total - When set, opens (or restarts) a numbered series
     * of `total` groups at the current depth. The first call renders as
     * `[1/total]`; subsequent siblings auto-advance.
     * @returns A handle for closing the group and writing nested log entries.
     */
    pushGroup(name: string, options: { total?: number } = {}): Group {
        const depth = this.stack.length;
        let numbering: { index: number; total: number } | undefined;
        if (options.total !== undefined) {
            numbering = { index: 1, total: options.total };
            this.counters.set(depth, { index: 1, total: options.total });
        } else {
            const active = this.counters.get(depth);
            if (active && active.index < active.total) {
                active.index += 1;
                numbering = { index: active.index, total: active.total };
            }
        }
        const scope: Scope = numbering ?
            { kind: 'group', name, depth, start: now(), index: numbering.index, total: numbering.total } :
            { kind: 'group', name, depth, start: now() };
        this.stack.push(scope);
        this.emit({ kind: 'scopeStart', depth, name, ...(numbering ?? {}) });
        return this.makeGroup(scope);
    }

    /**
     * Open a labelled progress bar at the current stack depth (i.e. nested
     * directly under whatever scope is currently on top of the stack). This
     * is a pure-push operation: it does not pop or auto-close anything.
     * Callers control nesting purely by the order in which they open and
     * close scopes.
     *
     * @param name - The bar's label, displayed alongside the progress indicator.
     * @param total - Total number of ticks the bar will report before completing.
     * @returns A handle for advancing and closing the bar.
     */
    pushBar(name: string, total: number): Bar {
        const scope: Scope = {
            kind: 'bar',
            name,
            depth: this.stack.length,
            start: now(),
            total: Math.max(1, total),
            current: 0
        };
        this.stack.push(scope);
        this.emit({ kind: 'barStart', depth: scope.depth, name: scope.name, total: scope.total });
        return this.makeBar(scope);
    }

    private makeBar(scope: Scope & { kind: 'bar' }): Bar {
        let closed = false;
        return {
            tick: (n = 1) => {
                if (closed) return;
                if (this.stack.indexOf(scope) === -1) {
                    // scope was popped from underneath us (e.g. a sibling
                    // bar opened inside a function call). Silently retire
                    // the handle so further ticks are no-ops.
                    closed = true;
                    return;
                }
                scope.current = Math.min(scope.total, scope.current + Math.max(0, n));
                this.emit({ kind: 'barTick', depth: scope.depth, name: scope.name, current: scope.current, total: scope.total });
            },
            end: () => {
                if (closed) return;
                closed = true;
                const idx = this.stack.indexOf(scope);
                if (idx === -1) return;
                while (this.stack.length > idx + 1) this.popScope();
                this.popScope();
            }
        };
    }

    private makeGroup(scope: Scope & { kind: 'group' }): Group {
        let closed = false;

        return {
            end: () => {
                if (closed) return;
                closed = true;
                const idx = this.stack.indexOf(scope);
                if (idx === -1) return;
                while (this.stack.length > idx + 1) this.popScope();
                this.popScope();
            }
        };
    }

    message(level: MessageKind, text: string): void {
        this.emit({ kind: 'message', depth: this.stack.length, level, text });
        if (level === 'error') this.unwindAll(true);
    }

    /**
     * Pop every open scope, emitting end-events with optional `failed` flag.
     * Called automatically on `logger.error(...)` so that aborted work renders
     * a clean trail of `(failed)` markers without callers needing try/finally.
     *
     * @param failed - When true, mark every closed scope as having failed.
     */
    unwindAll(failed = false): void {
        while (this.stack.length > 0) this.popScope(failed);
        this.counters.clear();
    }

    output(text: string): void {
        this.emit({ kind: 'output', text });
    }
}

const core = new LoggerCore();

/**
 * Public logger surface.
 *
 * Open named, timed scopes with {@link Logger.group}. Pass `{ total: N }` to
 * the first call of a numbered series; subsequent siblings auto-advance.
 * Indeterminate progress is reported with {@link Logger.bar}. Free-form
 * messages route through `info` / `warn` / `error` / `debug`, indented under
 * whatever is on top of the active-scope stack.
 *
 * Both `group` and `bar` are pure-push operations: opening a new scope simply
 * places it on top of the stack without auto-closing siblings, so call order
 * directly determines nesting. Close scopes with `handle.end()` after the
 * body. Callers that route failures through {@link Logger.error} get scope
 * cleanup for free; embedders that swallow exceptions should call
 * {@link Logger.unwindAll} from their catch to close every still-open scope.
 */
const logger = {
    /**
     * Open a named, timed scope. Returns a {@link Group} handle. Call `end()`
     * to close it. Group children indent automatically based on call depth.
     *
     * Pass `{ total: N }` to start (or restart) a numbered series at the
     * current depth: this call renders as `[1/N] name`, and the next N-1
     * `group(name)` calls at the same depth auto-render as `[2/N]`, ...,
     * `[N/N]`. Calls beyond that revert to plain rendering.
     *
     * @param name - The group name.
     * @param options - Optional configuration.
     * @param options.total - When set, opens (or restarts) a numbered series
     * of `total` groups at the current depth.
     * @returns A handle for closing the group and writing nested log entries.
     */
    group(name: string, options?: { total?: number }): Group {
        return core.pushGroup(name, options);
    },

    /**
     * Open a labelled progress bar nested directly under whatever scope is
     * currently on top of the active-scope stack. Renders as a single line
     * at child indent: `\u25b8 name [bar] %` while ticking, finalizing as
     * `\u2713 name [bar] X.XXXs` (or `\u2717 name [bar] (failed) X.XXXs`).
     *
     * Like {@link Logger.group}, this is a pure-push operation: it does not
     * close any sibling already on the stack. Close with `bar.end()`, or let
     * an enclosing group's `end()` / {@link Logger.unwindAll} pop it.
     *
     * @param name - The bar's label.
     * @param total - Expected number of ticks.
     * @returns A handle for advancing and closing the bar.
     */
    bar(name: string, total: number): Bar {
        return core.pushBar(name, total);
    },

    /**
     * Emit an info message indented under the innermost active scope.
     * @param args - Message parts (joined with a space).
     */
    info(...args: any[]): void {
        core.message('info', fmtArgs(args));
    },

    /**
     * Emit a warning indented under the innermost active scope.
     * @param args - Message parts.
     */
    warn(...args: any[]): void {
        core.message('warn', fmtArgs(args));
    },

    /**
     * Emit an error message. Always shown, regardless of verbosity. Triggers
     * an automatic unwind of all open scopes, marking each as failed.
     * @param args - Message parts.
     */
    error(...args: any[]): void {
        core.message('error', fmtArgs(args));
    },

    /**
     * Emit a debug message. Shown only at `verbose` verbosity.
     * @param args - Message parts.
     */
    debug(...args: any[]): void {
        core.message('debug', fmtArgs(args));
    },

    /**
     * Emit raw text to stdout (for piping). Always shown, regardless of
     * verbosity.
     * @param text - The text to emit.
     */
    output(text: string): void {
        core.output(text);
    },

    /**
     * Replace the active renderer. Use this from the CLI to install a
     * TTY-aware renderer; lib consumers get the default {@link PlainRenderer}.
     * @param r - The renderer to install.
     */
    setRenderer(r: Renderer): void {
        core.setRenderer(r);
    },

    /**
     * Set verbosity: `quiet` (errors and warnings), `normal` (default),
     * `verbose` (includes debug).
     * @param v - The verbosity level.
     */
    setVerbosity(v: Verbosity): void {
        core.setVerbosity(v);
    },

    /**
     * Close every open scope and bar, optionally marking them as failed.
     * Use this from an embedder's catch when an exception is being swallowed
     * (rather than rethrown into a `logger.error()` call), to prevent
     * dangling scopes from corrupting subsequent output.
     * @param failed - When true, mark every closed scope as having failed.
     */
    unwindAll(failed = false): void {
        core.unwindAll(failed);
    },

    /**
     * Get the current verbosity level.
     * @returns The active verbosity level.
     */
    getVerbosity(): Verbosity {
        return core.getVerbosity();
    }
};

/**
 * Public type alias for the logger object. Embedders can type-hint against
 * this to inject a configured logger.
 */
type Logger = typeof logger;

export {
    fmtBytes,
    fmtTime,
    indent,
    isPhaseHeader,
    logger,
    messageVisible,
    PlainRenderer,
    taskVisible,
    verbosityRank
};
export type { Bar, Group, LogEvent, Logger, MessageKind, Renderer, Verbosity };
