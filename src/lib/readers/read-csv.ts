import { OPACITY_LINEAR_COLUMN, SH_REST_COUNTS } from '../chunk';
import { Column, DataTable } from '../data-table';
import type { ReadSource } from '../io/read';
import { Transform } from '../utils';

const numericPattern = /^[+-]?(?:(?:\d+(?:\.\d*)?)|(?:\.\d+))(?:[eE][+-]?\d+)?$/;
const nonFiniteValues = new Map<string, number>([
    ['NaN', NaN],
    ['Infinity', Infinity],
    ['+Infinity', Infinity],
    ['-Infinity', -Infinity]
]);

const csvError = (message: string): Error => new Error(`readCsv: invalid CSV: ${message}`);

const parseValue = (value: string, line: number, column: string): number => {
    const token = value.trim();
    if (token.length === 0) {
        throw csvError(`line ${line}, column '${column}': empty value`);
    }

    if (nonFiniteValues.has(token)) {
        return nonFiniteValues.get(token)!;
    }
    if (!numericPattern.test(token)) {
        throw csvError(`line ${line}, column '${column}': invalid number '${token}'`);
    }

    return Number(token);
};

const COLUMN_GROUPS: [string, string[]][] = [
    ['position', ['x', 'y', 'z']],
    ['rotation/scale/opacity', ['rot_0', 'rot_1', 'rot_2', 'rot_3', 'scale_0', 'scale_1', 'scale_2', 'opacity']],
    ['color', ['f_dc_0', 'f_dc_1', 'f_dc_2']]
];
const SH_REST_TOTALS = Object.values(SH_REST_COUNTS).filter((n) => n > 0);

// The splat pipeline keeps a standard column only when its whole group is
// present, so a partial group (e.g. a hand-deleted rot_3) would silently drop
// its siblings. Reject it up front instead.
const checkColumnGroups = (names: Set<string>) => {
    const present = new Set(names);
    if (present.has(OPACITY_LINEAR_COLUMN)) present.add('opacity');

    for (const [label, group] of COLUMN_GROUPS) {
        const missing = group.filter((name) => !present.has(name));
        if (missing.length > 0 && missing.length < group.length) {
            throw csvError(`line 1: incomplete ${label} columns, missing ${missing.join(', ')}`);
        }
    }

    let numRest = 0;
    for (const name of names) {
        const match = /^f_rest_(\d+)$/.exec(name);
        if (match) numRest = Math.max(numRest, parseInt(match[1], 10) + 1);
    }
    if (numRest === 0) return;
    if (!present.has('f_dc_0')) {
        throw csvError('line 1: f_rest_* columns require f_dc_0, f_dc_1, f_dc_2');
    }
    const missing = [];
    for (let i = 0; i < numRest; i++) {
        if (!present.has(`f_rest_${i}`)) missing.push(`f_rest_${i}`);
    }
    if (missing.length > 0) {
        throw csvError(`line 1: incomplete f_rest columns, missing ${missing.join(', ')}`);
    }
    if (!SH_REST_TOTALS.includes(numRest)) {
        throw csvError(`line 1: expected ${SH_REST_TOTALS.join(', ')} f_rest_* columns, got ${numRest}`);
    }
};

const parseHeader = (line: string): string[] => {
    const names = line.split(',').map((name) => name.trim());
    const seen = new Set<string>();
    for (let i = 0; i < names.length; i++) {
        const name = names[i];
        if (name.length === 0) {
            throw csvError(`line 1, column ${i + 1}: empty column name`);
        }
        if (seen.has(name)) {
            throw csvError(`line 1: duplicate column name '${name}'`);
        }
        seen.add(name);
    }
    checkColumnGroups(seen);
    return names;
};

const CHUNK_BYTES = 1 << 20;
const INITIAL_ROWS = 1024;
const lineBreak = /\r\n|\n|\r/;

/**
 * Reads the numeric CSV dialect produced by `writeCsv` into a data table.
 *
 * The first row contains unquoted column names and each subsequent row contains
 * one numeric value per column. Values are in PLY coordinate space (the space
 * `writeCsv` writes), so the returned table carries `Transform.PLY`. Values are
 * stored as 32-bit floats because the CSV format does not carry column type
 * information, so integer values above 2^24 and 64-bit float values lose
 * precision. `NaN` and positive or negative infinity are accepted.
 *
 * CSV has no row index to serve random row gathers, so the file is decoded
 * eagerly into resident columns (4 bytes per value). The text is parsed in a
 * single streaming pass and is never held in memory whole.
 *
 * @param source - Source containing UTF-8 encoded CSV data. The caller owns and closes it.
 * @returns A data table containing the CSV columns and rows.
 * @throws If the CSV is not valid UTF-8, has an invalid header (including a
 * partial group of standard splat columns, such as `rot_0`-`rot_2` without
 * `rot_3`), contains rows with the wrong number of values, or contains an
 * invalid numeric value.
 */
const readCsv = async (source: ReadSource): Promise<DataTable> => {
    let names: string[] | null = null;
    let data: Float32Array[] = [];
    let capacity = INITIAL_ROWS;
    let numRows = 0;
    let lineNumber = 0;
    let blankLine = 0; // first blank line seen; only valid if nothing follows it

    const parseRow = (columns: string[], line: string) => {
        const values = line.split(',');
        if (values.length !== columns.length) {
            throw csvError(`line ${lineNumber}: expected ${columns.length} values, got ${values.length}`);
        }
        if (numRows === capacity) {
            capacity *= 2;
            data = data.map((d) => {
                const grown = new Float32Array(capacity);
                grown.set(d);
                return grown;
            });
        }
        for (let column = 0; column < columns.length; column++) {
            data[column][numRows] = parseValue(values[column], lineNumber, columns[column]);
        }
        numRows++;
    };

    const processLine = (line: string) => {
        lineNumber++;
        if (line.trim() === '') {
            if (!names) {
                throw csvError('missing header');
            }
            if (!blankLine) blankLine = lineNumber;
            return;
        }
        if (blankLine) {
            throw csvError(`line ${blankLine}: empty line`);
        }
        if (!names) {
            names = parseHeader(line);
            data = names.map(() => new Float32Array(capacity));
        } else {
            parseRow(names, line);
        }
    };

    // The default decoder strips a leading BOM; `stream` keeps multi-byte
    // characters that straddle chunk boundaries intact.
    const decoder = new TextDecoder('utf-8', { fatal: true });
    const chunk = new Uint8Array(CHUNK_BYTES);
    const stream = source.read();
    try {
        let pending = '';
        while (true) {
            const n = await stream.pull(chunk);
            let text: string;
            try {
                text = pending + decoder.decode(chunk.subarray(0, n), { stream: n > 0 });
            } catch (err) {
                if (err instanceof TypeError) {
                    throw csvError('file is not valid UTF-8');
                }
                throw err;
            }
            // Hold back a trailing '\r' in case the next chunk starts with '\n'.
            const heldCR = n > 0 && text.endsWith('\r');
            const lines = (heldCR ? text.slice(0, -1) : text).split(lineBreak);
            pending = lines.pop()! + (heldCR ? '\r' : '');
            for (const line of lines) {
                processLine(line);
            }
            if (n === 0) break;
        }
        if (pending.length > 0) {
            processLine(pending);
        }
    } finally {
        stream.close?.();
    }

    if (!names) {
        throw csvError('missing header');
    }

    const columns = (names as string[]).map(
        (name, i) => new Column(name, numRows === capacity ? data[i] : data[i].slice(0, numRows))
    );
    return new DataTable(columns, Transform.PLY);
};

export { readCsv };
