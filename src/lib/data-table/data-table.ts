/**
 * Union of all typed array types supported for column data.
 */
type TypedArray = Int8Array | Uint8Array | Int16Array | Uint16Array | Int32Array | Uint32Array | Float32Array | Float64Array;

/**
 * String identifiers for typed array element types.
 */
type ColumnType = 'int8' | 'uint8' | 'int16' | 'uint16' | 'int32' | 'uint32' | 'float32' | 'float64';

/**
 * A named column of typed array data within a DataTable.
 *
 * Columns store homogeneous numeric data efficiently using JavaScript typed arrays.
 *
 * @example
 * ```ts
 * const positions = new Column('x', new Float32Array([1.0, 2.0, 3.0]));
 * console.log(positions.name);     // 'x'
 * console.log(positions.dataType); // 'float32'
 * ```
 */
class Column {
    name: string;
    data: TypedArray;

    constructor(name: string, data: TypedArray) {
        this.name = name;
        this.data = data;
    }

    get dataType(): ColumnType | null {
        switch (this.data.constructor) {
            case Int8Array: return 'int8';
            case Uint8Array: return 'uint8';
            case Int16Array: return 'int16';
            case Uint16Array: return 'uint16';
            case Int32Array: return 'int32';
            case Uint32Array: return 'uint32';
            case Float32Array: return 'float32';
            case Float64Array: return 'float64';
        }
        return null;
    }

    clone(): Column {
        return new Column(this.name, this.data.slice());
    }
}

/**
 * A row object mapping column names to numeric values.
 * @ignore
 */
type Row = {
    [colName: string]: number;
};

/**
 * A table of columnar data representing Gaussian splat properties.
 *
 * DataTable is the core data structure for splat data. Each column represents
 * a property (e.g., position, rotation, color) as a typed array, and all columns
 * must have the same number of rows.
 *
 * Standard columns include:
 * - Position: `x`, `y`, `z`
 * - Rotation: `rot_0`, `rot_1`, `rot_2`, `rot_3` (quaternion)
 * - Scale: `scale_0`, `scale_1`, `scale_2` (log scale)
 * - Color: `f_dc_0`, `f_dc_1`, `f_dc_2` (spherical harmonics DC)
 * - Opacity: `opacity` (logit)
 * - Spherical Harmonics: `f_rest_0` through `f_rest_44`
 *
 * @example
 * ```ts
 * const table = new DataTable([
 *     new Column('x', new Float32Array([0, 1, 2])),
 *     new Column('y', new Float32Array([0, 0, 0])),
 *     new Column('z', new Float32Array([0, 0, 0]))
 * ]);
 * console.log(table.numRows);    // 3
 * console.log(table.numColumns); // 3
 * ```
 */
class DataTable {
    columns: Column[];

    constructor(columns: Column[]) {
        if (columns.length === 0) {
            throw new Error('DataTable must have at least one column');
        }

        // check all columns have the same lengths
        for (let i = 1; i < columns.length; i++) {
            if (columns[i].data.length !== columns[0].data.length) {
                throw new Error(`Column '${columns[i].name}' has inconsistent number of rows: expected ${columns[0].data.length}, got ${columns[i].data.length}`);
            }
        }

        this.columns = columns;
    }

    // rows

    get numRows() {
        return this.columns[0].data.length;
    }

    getRow(index: number, row: Row = {}, columns = this.columns): Row {
        for (const column of columns) {
            row[column.name] = column.data[index];
        }
        return row;
    }

    setRow(index: number, row: Row, columns = this.columns) {
        for (const column of columns) {
            if (row.hasOwnProperty(column.name)) {
                column.data[index] = row[column.name];
            }
        }
    }

    // columns

    get numColumns() {
        return this.columns.length;
    }

    get columnNames() {
        return this.columns.map(column => column.name);
    }

    get columnData() {
        return this.columns.map(column => column.data);
    }

    get columnTypes() {
        return this.columns.map(column => column.dataType);
    }

    getColumn(index: number): Column {
        return this.columns[index];
    }

    getColumnIndex(name: string): number {
        return this.columns.findIndex(column => column.name === name);
    }

    getColumnByName(name: string): Column | null {
        return this.columns.find(column => column.name === name);
    }

    hasColumn(name: string): boolean {
        return this.columns.some(column => column.name === name);
    }

    addColumn(column: Column) {
        if (column.data.length !== this.numRows) {
            throw new Error(`Column '${column.name}' has inconsistent number of rows: expected ${this.numRows}, got ${column.data.length}`);
        }
        this.columns.push(column);
    }

    removeColumn(name: string) {
        const index = this.columns.findIndex(column => column.name === name);
        if (index === -1) {
            return false;
        }
        this.columns.splice(index, 1);
        return true;
    }

    // general

    clone(): DataTable {
        return new DataTable(this.columns.map(c => c.clone()));
    }

    // return a new table containing the rows referenced in indices
    permuteRows(indices: Uint32Array | number[]): DataTable {
        const result = new DataTable(this.columns.map((c) => {
            const constructor = c.data.constructor as new (length: number) => TypedArray;
            return new Column(c.name, new constructor(indices.length));
        }));

        for (let i = 0; i < this.numColumns; ++i) {
            const src = this.getColumn(i).data;
            const dst = result.getColumn(i).data;
            for (let j = 0; j < indices.length; j++) {
                dst[j] = src[indices[j]];
            }
        }
        return result;
    }

    /**
     * Permutes the rows of this DataTable in-place according to the given indices.
     * After calling, row `i` will contain the data that was previously at row `indices[i]`.
     *
     * This is a memory-efficient alternative to `permuteRows` that modifies the table
     * in-place rather than creating a copy.
     *
     * @param indices - Array of indices defining the permutation. Must have the same
     * length as the number of rows, and must be a valid permutation
     * (each index 0 to n-1 appears exactly once).
     */
    permuteRowsInPlace(indices: Uint32Array | number[]): void {
        const n = this.numRows;
        const numCols = this.columns.length;
        const visited = new Uint8Array(n);
        const temps = new Array(numCols);

        for (let i = 0; i < n; i++) {
            if (visited[i] || indices[i] === i) continue;

            // Save values at position i
            for (let c = 0; c < numCols; c++) {
                temps[c] = this.columns[c].data[i];
            }

            // Walk the cycle
            let j = i;
            while (true) {
                const next = indices[j];
                visited[j] = 1;

                if (next === i) {
                    // End of cycle - place saved values
                    for (let c = 0; c < numCols; c++) {
                        this.columns[c].data[j] = temps[c];
                    }
                    break;
                }
                // Move values from next to j
                for (let c = 0; c < numCols; c++) {
                    this.columns[c].data[j] = this.columns[c].data[next];
                }
                j = next;
            }
        }
    }
}

export { Column, DataTable, TypedArray, ColumnType, Row };
