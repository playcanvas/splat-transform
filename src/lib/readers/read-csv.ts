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
 * @throws If the CSV is not valid UTF-8, has an invalid header, contains rows
 * with the wrong number of values, or contains an invalid numeric value.
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
