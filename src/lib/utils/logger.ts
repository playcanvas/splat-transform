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
 * `barStart` / `barTick` / `barEnd` represent a determinate progress bar.
 * The bar's `name` is repeated on every event so the renderer can keep its
 * label stable across in-place updates while tracking progress via `current`
 * and `total`.
 *
 * `output` is the pipeable channel: each event represents a single logical
 * unit of output (typically one line - or a multi-line block treated as a
 * unit) that the renderer is expected to terminate with a newline. Callers
 * should not include a trailing `\n` themselves.
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
 * Renderer interface. Receives the full stream of semantic lifecycle
 * events ({@link LogEvent}) and decides how to display them. The core
 * does not filter scope/bar events by verbosity, so renderers see a
 * faithful record of every scope open/close and bar progress update -
 * embedders consuming the event stream can rely on this for progress
 * UIs that must close themselves on completion. Visibility decisions
 * (e.g. hiding successful `scopeEnd` footers at non-`verbose`
 * verbosity) are the renderer's responsibility; {@link logger.getVerbosity}
 * is available to consult.
 *
 * `message` events for `info`, `warn` and `debug` are gated by verbosity
 * at the façade (see {@link LoggerCore.isLevelVisible}) before reaching
 * the renderer; `error` is always delivered.
 */
interface Renderer {
    /**
     * Handle a log event.
     * @param event - The event to render.
     */
    handle(event: LogEvent): void;
}

/**
 * Determinate progress bar handle. Closed explicitly via `end()`, or
 * implicitly when an enclosing {@link Group}'s `end()` (or a
 * {@link Logger.unwindAll}) pops it as part of cleanup.
 *
 * Carries a `[Symbol.dispose]` slot directly (rather than extending the
 * built-in `Disposable` lib type) so the published `.d.ts` stays free of
 * any reference to the `Disposable` interface. `Symbol.dispose` itself is
 * still a TS 5.2+ / `esnext.disposable` (or `es2024.disposable`) lib
 * symbol, so consumers compiling against these declarations need that
 * lib enabled (or `skipLibCheck: true`). Callers on TS 5.2+ / Node 20+
 * can adopt `using bar = logger.bar(...)` because `using` only requires
 * the `[Symbol.dispose]` shape structurally.
 */
interface Bar {
    /**
     * Advance the bar by `n` ticks.
     * @param n - Number of ticks to advance (default 1).
     */
    tick(n?: number): void;
    /**
     * Set the bar's absolute progress. Clamped to `[0, total]`. Suppresses
     * a `barTick` event when the value is unchanged.
     * @param current - Absolute progress value.
     */
    update(current: number): void;
    /**
     * Close the bar and emit final timing.
     */
    end(): void;
    /** Dispose hook so `using` syntax closes the bar on scope exit. */
    [Symbol.dispose](): void;
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
 *
 * Carries a `[Symbol.dispose]` slot directly (rather than extending the
 * built-in `Disposable` lib type) so the published `.d.ts` stays free of
 * any reference to the `Disposable` interface. `Symbol.dispose` itself is
 * still a TS 5.2+ / `esnext.disposable` (or `es2024.disposable`) lib
 * symbol, so consumers compiling against these declarations need that
 * lib enabled (or `skipLibCheck: true`). Callers on TS 5.2+ / Node 20+
 * can adopt `using g = logger.group(...)` because `using` only requires
 * the `[Symbol.dispose]` shape structurally.
 */
interface Group {
    /**
     * Close the group, popping anything still open above it on the stack
     * (defensively handles forgotten inner scopes) and emit the timing event.
     */
    end(): void;
    /** Dispose hook so `using` syntax closes the group on scope exit. */
    [Symbol.dispose](): void;
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
 * Default no-op renderer. Used when no other renderer is installed (e.g. in
 * library/embedded contexts where the host wants to consume `LogEvent`s
 * directly via {@link Logger.setRenderer} but hasn't done so yet). Drops
 * every event silently.
 */
class NullRenderer implements Renderer {
    handle(_event: LogEvent): void { /* no-op */ }
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

/**
 * Active-scope manager and message router. The single shared instance lives
 * inside this module; the public `logger` surface is a thin façade over it.
 */
class LoggerCore {
    /** Stack of currently-open scopes (innermost last). */
    readonly stack: Scope[] = [];

    private renderer: Renderer = new NullRenderer();

    private verbosity: Verbosity = 'normal';

    setRenderer(r: Renderer): void {
        this.renderer = r;
    }

    setVerbosity(v: Verbosity): void {
        this.verbosity = v;
    }

    getVerbosity(): Verbosity {
        return this.verbosity;
    }

    /**
     * Whether a message at `level` would be emitted at the current
     * verbosity. Primary use: the `logger` façade calls this before
     * formatting arguments so filtered `info`/`warn`/`debug` calls don't
     * allocate the joined string that {@link emit} would only throw away.
     *
     * @param level - The message level to test.
     * @returns `true` if a message at `level` would reach the renderer.
     */
    isLevelVisible(level: MessageKind): boolean {
        return verbosityRank[this.verbosity] >= verbosityRank[messageMinVerbosity[level]];
    }

    /**
     * Hand the event to the renderer. Lifecycle events (`scopeStart`,
     * `scopeEnd`, `barStart`, `barTick`, `barEnd`) and `output` are always
     * forwarded; presentation policy (e.g. hiding successful `scopeEnd`
     * footers at non-`verbose` verbosity) lives in the renderer so
     * embedders consuming the event stream see a complete, faithful
     * record of scope and bar lifecycles. `message` is assumed already
     * gated at the façade via {@link LoggerCore.isLevelVisible} (so
     * callers can skip formatting args for filtered levels); anything
     * that reaches here is passed through.
     *
     * @param event - The event to deliver.
     */
    emit(event: LogEvent): void {
        this.renderer.handle(event);
    }

    /**
     * Pop the scope at the top of the stack (no-op if empty) and emit the
     * matching `*End` event.
     *
     * @param failed - When true, mark the closed scope as having failed.
     */
    private popScope(failed = false): void {
        if (this.stack.length === 0) return;
        const top = this.stack[this.stack.length - 1];
        this.stack.pop();
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
        const numbering = top.index !== undefined && top.total !== undefined ?
            { index: top.index, total: top.total } :
            {};
        this.emit({ kind: 'scopeEnd', depth: top.depth, name: top.name, durationMs, failed, ...numbering });
    }

    /**
     * Open a named, timed group at the current depth. Pass
     * `{ index, total }` to render the group as part of a numbered series
     * (e.g. `[2/5] name`); both must be present together.
     *
     * @param name - The group name.
     * @param options - Optional configuration.
     * @param options.index - 1-based position in the numbered series.
     * @param options.total - Total length of the numbered series.
     * @returns A handle for closing the group and writing nested log entries.
     */
    pushGroup(name: string, options: { index?: number; total?: number } = {}): Group {
        const { index, total } = options;
        if ((index === undefined) !== (total === undefined)) {
            throw new Error('logger.group: { index, total } must be passed together');
        }
        const depth = this.stack.length;
        const numbering = index !== undefined && total !== undefined ? { index, total } : undefined;
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
     * A `total` of 0 is allowed (e.g. processing an empty payload); both
     * `LoggerCore` and `TextRenderer` already handle non-positive totals.
     * @returns A handle for advancing and closing the bar.
     */
    pushBar(name: string, total: number): Bar {
        const scope: Scope = {
            kind: 'bar',
            name,
            depth: this.stack.length,
            start: now(),
            total: Math.max(0, total),
            current: 0
        };
        this.stack.push(scope);
        this.emit({ kind: 'barStart', depth: scope.depth, name: scope.name, total: scope.total });
        return this.makeBar(scope);
    }

    private makeBar(scope: Scope & { kind: 'bar' }): Bar {
        let closed = false;
        // Bars are strictly LIFO from a renderer's perspective: a `TextRenderer`
        // (or any other line-based renderer) only tracks one active bar line,
        // so ticking a bar that isn't currently on top of the stack would
        // corrupt whatever inner bar is. We still update `scope.current`
        // internally so the recap line at `barEnd` is accurate, but we
        // suppress `barTick` emission unless this bar is actually on top.
        const isTopOfStack = () => this.stack[this.stack.length - 1] === scope;
        const handle: Bar = {
            tick: (n = 1) => {
                if (closed) return;
                if (this.stack.indexOf(scope) === -1) {
                    // scope was popped from underneath us (e.g. a sibling
                    // bar opened inside a function call). Silently retire
                    // the handle so further ticks are no-ops.
                    closed = true;
                    return;
                }
                const next = Math.min(scope.total, scope.current + Math.max(0, n));
                if (next === scope.current) return;
                scope.current = next;
                if (!isTopOfStack()) return;
                this.emit({ kind: 'barTick', depth: scope.depth, name: scope.name, current: scope.current, total: scope.total });
            },
            update: (current: number) => {
                if (closed) return;
                if (this.stack.indexOf(scope) === -1) {
                    closed = true;
                    return;
                }
                const next = Math.min(scope.total, Math.max(0, current));
                if (next === scope.current) return;
                scope.current = next;
                if (!isTopOfStack()) return;
                this.emit({ kind: 'barTick', depth: scope.depth, name: scope.name, current: scope.current, total: scope.total });
            },
            end: () => {
                if (closed) return;
                closed = true;
                const idx = this.stack.indexOf(scope);
                if (idx === -1) return;
                while (this.stack.length > idx + 1) this.popScope(true);
                this.popScope();
            },
            [Symbol.dispose]: () => handle.end()
        };
        return handle;
    }

    private makeGroup(scope: Scope & { kind: 'group' }): Group {
        let closed = false;
        const handle: Group = {
            end: () => {
                if (closed) return;
                closed = true;
                const idx = this.stack.indexOf(scope);
                if (idx === -1) return;
                while (this.stack.length > idx + 1) this.popScope(true);
                this.popScope();
            },
            [Symbol.dispose]: () => handle.end()
        };
        return handle;
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
    }

    output(text: string): void {
        this.emit({ kind: 'output', text });
    }
}

const core = new LoggerCore();

/**
 * Public logger surface.
 *
 * Open named, timed scopes with {@link Logger.group}. Pass `{ index, total }`
 * to render the group as part of a numbered series. Indeterminate progress is
 * reported with {@link Logger.bar}. Free-form messages route through `info` /
 * `warn` / `error` / `debug`, indented under whatever is on top of the
 * active-scope stack.
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
     * Pass `{ index, total }` to render the group as part of a numbered
     * series (e.g. `[2/5] name`). Both fields must be supplied together.
     *
     * @param name - The group name.
     * @param options - Optional configuration.
     * @param options.index - 1-based position in the numbered series.
     * @param options.total - Total length of the numbered series.
     * @returns A handle for closing the group and writing nested log entries.
     */
    group(name: string, options?: { index?: number; total?: number }): Group {
        return core.pushGroup(name, options);
    },

    /**
     * Open a labelled progress bar nested directly under whatever scope is
     * currently on top of the active-scope stack. Renders as a single line
     * at child indent.
     *
     * Like {@link Logger.group}, this is a pure-push operation: it does not
     * close any sibling already on the stack. Close with `bar.end()`, or let
     * an enclosing group's `end()` / {@link Logger.unwindAll} pop it.
     *
     * @param name - The bar's label.
     * @param total - Expected number of ticks (or absolute total when using
     * {@link Bar.update}).
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
        if (!core.isLevelVisible('info')) return;
        core.message('info', fmtArgs(args));
    },

    /**
     * Emit a warning indented under the innermost active scope.
     * @param args - Message parts.
     */
    warn(...args: any[]): void {
        if (!core.isLevelVisible('warn')) return;
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
        if (!core.isLevelVisible('debug')) return;
        core.message('debug', fmtArgs(args));
    },

    /**
     * Emit a logical unit of pipeable output (typically one line, or a
     * multi-line block treated as a single unit). The renderer terminates
     * each unit with a newline, so callers should not include a trailing
     * `\n`. Always shown, regardless of verbosity.
     * @param text - The text to emit (without a trailing newline).
     */
    output(text: string): void {
        core.output(text);
    },

    /**
     * Replace the active renderer. Embedders install their own renderer here
     * to consume `LogEvent`s; the default renderer is a no-op. Renderers
     * receive every scope/bar lifecycle event regardless of verbosity, so
     * progress UIs can rely on `scopeStart`/`scopeEnd` and `barStart`/`barEnd`
     * to manage their state.
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

export { logger, verbosityRank };
export type { Bar, Group, LogEvent, Logger, MessageKind, Renderer, Verbosity };
