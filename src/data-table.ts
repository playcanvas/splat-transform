type TypedArray = Int8Array | Uint8Array | Int16Array | Uint16Array | Int32Array | Uint32Array | Float32Array | Float64Array;

type ColumnType = 'int8' | 'uint8' | 'int16' | 'uint16' | 'int32' | 'uint32' | 'float32' | 'float64';

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

type Row = {
    [colName: string]: number;
};

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
}

export { Column, DataTable, TypedArray };
