import { Column, DataTable } from '../data-table';
import type { ReadSource } from '../io/read';

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

/**
 * Reads the numeric CSV dialect produced by {@link writeCsv} into a data table.
 *
 * The first row contains unquoted column names and each subsequent row contains
 * one numeric value per column. Values are stored as 32-bit floats because the
 * CSV format does not carry column type information. `NaN` and positive or
 * negative infinity are accepted so every value emitted by `writeCsv` can be
 * read back.
 *
 * @param source - Source containing UTF-8 encoded CSV data. The caller owns and closes it.
 * @returns A data table containing the CSV columns and rows.
 * @throws If the CSV is not valid UTF-8, has an invalid header, contains rows
 * with the wrong number of values, or contains an invalid numeric value.
 */
const readCsv = async (source: ReadSource): Promise<DataTable> => {
    const stream = source.read();
    let bytes: Uint8Array;
    try {
        bytes = await stream.readAll();
    } finally {
        stream.close?.();
    }

    let text: string;
    try {
        text = new TextDecoder('utf-8', { fatal: true }).decode(bytes);
    } catch {
        throw csvError('file is not valid UTF-8');
    }

    if (text.charCodeAt(0) === 0xfeff) {
        text = text.slice(1);
    }

    const lines = text.split(/\r\n|\n|\r/);
    while (lines.length > 0 && lines[lines.length - 1].trim() === '') {
        lines.pop();
    }
    if (lines.length === 0 || lines[0].trim() === '') {
        throw csvError('missing header');
    }

    const names = lines[0].split(',').map((name) => name.trim());
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

    const rowCount = lines.length - 1;
    const data = names.map(() => new Float32Array(rowCount));
    for (let row = 0; row < rowCount; row++) {
        const line = row + 2;
        const values = lines[row + 1].split(',');
        if (values.length !== names.length) {
            throw csvError(`line ${line}: expected ${names.length} values, got ${values.length}`);
        }
        for (let column = 0; column < names.length; column++) {
            data[column][row] = parseValue(values[column], line, names[column]);
        }
    }

    return new DataTable(names.map((name, i) => new Column(name, data[i])));
};

export { readCsv };
