/**
 * Verbosity level controlling which messages reach the renderer.
 *
 * - `quiet`   - errors only.
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
 * `groupStart` / `groupEnd` carry optional `index` / `total` fields when the
 * group is part of a numbered series; renderers can use these to switch to a
 * `[N/T] name` style.
 */
type LogEvent =
    | { kind: 'groupStart'; depth: number; name: string; index?: number; total?: number }
    | { kind: 'groupEnd'; depth: number; name: string; durationMs: number; failed?: boolean; index?: number; total?: number }
    | { kind: 'stepStart'; depth: number; name: string }
    | { kind: 'stepEnd'; depth: number; name: string; durationMs: number; failed?: boolean }
    | { kind: 'barStart'; depth: number; total: number }
    | { kind: 'barTick'; depth: number; current: number; total: number }
    | { kind: 'barEnd'; depth: number; durationMs: number; failed?: boolean }
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
 * Indeterminate progress bar handle. Auto-closes when the enclosing scope
 * advances or ends.
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
 * lifecycle only - free-form messages and bars are emitted via the global
 * `logger` (they auto-indent under the innermost active scope).
 */
interface Group {
    /**
     * Open a new step within this group, automatically closing the previous
     * step (and any open bar inside it).
     * @param name - The step name.
     */
    step(name: string): void;
    /**
     * Close the group: ends any open step / bar inside it, then emits timing.
     */
    end(): void;
}

/**
 * Internal node tracked on the active-scope stack.
 *
 * `group` scopes carry optional `index` / `total` when they are part of a
 * numbered series, so the closing event can re-emit the same numbering.
 */
type Scope =
    | { kind: 'group'; name: string; depth: number; start: number; index?: number; total?: number }
    | { kind: 'step'; name: string; depth: number; start: number }
    | { kind: 'bar'; depth: number; start: number; total: number; current: number };

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
    warn: 'normal',
    info: 'normal',
    debug: 'verbose'
};

const messageVisible = (kind: MessageKind, v: Verbosity): boolean => {
    return verbosityRank[v] >= verbosityRank[messageMinVerbosity[kind]];
};

const taskVisible = (v: Verbosity): boolean => verbosityRank[v] >= verbosityRank.normal;

const fmtTime = (ms: number): string => {
    if (ms >= 60_000) return `${(ms / 60_000).toFixed(2)}m`;
    return `${(ms / 1000).toFixed(3)}s`;
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

/**
 * Default browser-safe renderer. Uses only `console.log/warn/error`. Bars are
 * not animated - only the bar's final state is printed when it closes.
 *
 * Numbered groups at depth 0 render in the legacy `[N/T] name` flush-left
 * style; nested numbered groups use indented `[N/T] name`; un-numbered groups
 * use `\u25b8 name` / `\u25c2 name` indented by depth.
 */
class PlainRenderer implements Renderer {
    private verbosity: Verbosity = 'normal';

    setVerbosity(v: Verbosity): void {
        this.verbosity = v;
    }

    handle(event: LogEvent): void {
        switch (event.kind) {
            case 'groupStart': {
                const numbered = event.index !== undefined && event.total !== undefined;
                if (numbered && event.depth === 0) {
                    if (!taskVisible(this.verbosity)) break;
                    console.log(`\n[${event.index}/${event.total}] ${event.name}`);
                } else {
                    if (!taskVisible(this.verbosity)) break;
                    if (numbered) {
                        console.log(`${this.indent(event.depth)}[${event.index}/${event.total}] ${event.name}`);
                    } else {
                        console.log(`${this.indent(event.depth)}\u25b8 ${event.name}`);
                    }
                }
                break;
            }
            case 'groupEnd': {
                const numbered = event.index !== undefined && event.total !== undefined;
                if (numbered && event.depth === 0) {
                    if (event.failed) {
                        console.log(`${this.indent(event.depth)}\u2717 ${event.name} (failed) ${fmtTime(event.durationMs)}`);
                    }
                } else {
                    if (!taskVisible(this.verbosity)) break;
                    if (event.failed) {
                        console.log(`${this.indent(event.depth)}  ${event.name} (failed) ${fmtTime(event.durationMs)}`);
                    } else {
                        console.log(`${this.indent(event.depth)}\u25c2 ${event.name} ${fmtTime(event.durationMs)}`);
                    }
                }
                break;
            }
            case 'stepStart':
                if (!taskVisible(this.verbosity)) break;
                console.log(`${this.indent(event.depth)}\u25b8 ${event.name}`);
                break;
            case 'stepEnd':
                if (!taskVisible(this.verbosity)) break;
                if (event.failed) {
                    console.log(`${this.indent(event.depth)}  ${event.name} (failed) ${fmtTime(event.durationMs)}`);
                } else {
                    console.log(`${this.indent(event.depth)}  ${event.name} ${fmtTime(event.durationMs)}`);
                }
                break;
            case 'barStart':
            case 'barTick':
                break;
            case 'barEnd':
                if (!taskVisible(this.verbosity)) break;
                console.log(`${this.indent(event.depth)}[##########] ${fmtTime(event.durationMs)}`);
                break;
            case 'message': {
                if (!messageVisible(event.level, this.verbosity)) break;
                const prefix = event.level === 'warn' ? 'warn: ' : event.level === 'error' ? 'error: ' : '';
                const fn = event.level === 'error' ? console.error : event.level === 'warn' ? console.warn : console.log;
                fn(`${this.indent(event.depth)}${prefix}${event.text}`);
                break;
            }
            case 'output':
                console.log(event.text);
                break;
        }
    }

    private indent(depth: number): string {
        return '  '.repeat(Math.max(0, depth));
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
     * Close the bar at the top of the stack (no-op if none).
     *
     * @param failed - When true, mark the closed bar as having failed.
     */
    private endTopBar(failed = false): void {
        if (this.stack.length === 0) return;
        const top = this.stack[this.stack.length - 1];
        if (top.kind !== 'bar') return;
        this.stack.pop();
        this.clearDeeperCounters(top.depth);
        this.emit({ kind: 'barEnd', depth: top.depth, durationMs: now() - top.start, failed });
    }

    /**
     * Close the step at the top of the stack (and any bar inside it).
     *
     * @param failed - When true, mark the closed step as having failed.
     */
    private endTopStep(failed = false): void {
        this.endTopBar(failed);
        if (this.stack.length === 0) return;
        const top = this.stack[this.stack.length - 1];
        if (top.kind !== 'step') return;
        this.stack.pop();
        this.clearDeeperCounters(top.depth);
        this.emit({ kind: 'stepEnd', depth: top.depth, name: top.name, durationMs: now() - top.start, failed });
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
        this.emit({ kind: 'groupStart', depth, name, ...(numbering ?? {}) });
        return this.makeGroup(scope);
    }

    /**
     * Open a free-floating bar bound to the innermost active scope.
     *
     * @param total - Total number of ticks the bar will report before completing.
     * @returns A handle for advancing and closing the bar.
     */
    pushBar(total: number): Bar {
        if (this.stack.length > 0 && this.stack[this.stack.length - 1].kind === 'bar') {
            this.endTopBar();
        }
        const scope: Scope = {
            kind: 'bar',
            depth: this.stack.length,
            start: now(),
            total: Math.max(1, total),
            current: 0
        };
        this.stack.push(scope);
        this.emit({ kind: 'barStart', depth: scope.depth, total: scope.total });
        return this.makeBar(scope);
    }

    private makeBar(scope: Scope & { kind: 'bar' }): Bar {
        let closed = false;
        return {
            tick: (n = 1) => {
                if (closed) return;
                scope.current = Math.min(scope.total, scope.current + Math.max(0, n));
                this.emit({ kind: 'barTick', depth: scope.depth, current: scope.current, total: scope.total });
            },
            end: () => {
                if (closed) return;
                closed = true;
                const idx = this.stack.indexOf(scope);
                if (idx === -1) return;
                while (this.stack.length > idx + 1) this.stack.pop();
                this.stack.pop();
                this.clearDeeperCounters(scope.depth);
                this.emit({ kind: 'barEnd', depth: scope.depth, durationMs: now() - scope.start });
            }
        };
    }

    private makeGroup(scope: Scope & { kind: 'group' }): Group {
        let closed = false;

        const stillOpen = (): boolean => {
            if (closed) return false;
            return this.stack.indexOf(scope) !== -1;
        };

        return {
            step: (name: string) => {
                if (!stillOpen()) return;
                while (this.stack.length > 0 && this.stack[this.stack.length - 1] !== scope) {
                    const top = this.stack[this.stack.length - 1];
                    if (top.kind === 'bar') this.endTopBar();
                    else if (top.kind === 'step') this.endTopStep();
                    else break;
                }
                const stepScope: Scope = {
                    kind: 'step',
                    name,
                    depth: scope.depth + 1,
                    start: now()
                };
                this.stack.push(stepScope);
                this.emit({ kind: 'stepStart', depth: stepScope.depth, name });
            },
            end: () => {
                if (closed) return;
                closed = true;
                const idx = this.stack.indexOf(scope);
                if (idx === -1) return;
                while (this.stack.length > idx + 1) {
                    const top = this.stack[this.stack.length - 1];
                    if (top.kind === 'bar') this.endTopBar();
                    else if (top.kind === 'step') this.endTopStep();
                    else this.stack.pop();
                }
                this.stack.pop();
                this.clearDeeperCounters(scope.depth);
                const numbering = scope.index !== undefined && scope.total !== undefined ?
                    { index: scope.index, total: scope.total } :
                    {};
                this.emit({ kind: 'groupEnd', depth: scope.depth, name: scope.name, durationMs: now() - scope.start, ...numbering });
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
        while (this.stack.length > 0) {
            const top = this.stack[this.stack.length - 1];
            this.stack.pop();
            this.clearDeeperCounters(top.depth);
            switch (top.kind) {
                case 'bar':
                    this.emit({ kind: 'barEnd', depth: top.depth, durationMs: now() - top.start, failed });
                    break;
                case 'step':
                    this.emit({ kind: 'stepEnd', depth: top.depth, name: top.name, durationMs: now() - top.start, failed });
                    break;
                case 'group': {
                    const numbering = top.index !== undefined && top.total !== undefined ?
                        { index: top.index, total: top.total } :
                        {};
                    this.emit({ kind: 'groupEnd', depth: top.depth, name: top.name, durationMs: now() - top.start, failed, ...numbering });
                    break;
                }
            }
        }
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
 * the innermost active scope.
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
     * Open an indeterminate progress bar bound to the innermost active scope.
     * @param total - Expected number of ticks.
     * @returns A handle for advancing and closing the bar.
     */
    bar(total: number): Bar {
        return core.pushBar(total);
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
     * Set verbosity: `quiet` (errors only), `normal` (default), `verbose`
     * (includes debug).
     * @param v - The verbosity level.
     */
    setVerbosity(v: Verbosity): void {
        core.setVerbosity(v);
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

export { logger, PlainRenderer };
export type { Bar, Group, LogEvent, Logger, Renderer, Verbosity };
