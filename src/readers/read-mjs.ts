import { Column, DataTable } from '../data-table';

type SplatData = {
    comments: string[];
    elements: {
        name: string,
        dataTable: DataTable
    }[];
};

type Param = {
    name: string;
    value: string;
};

type Generator = {
    constructor(params: Param[]): void;
    count: number;
    columnNames: string[];
    getRow: (index: number, row: any) => void;
};

const readMjs = async (filename: string, params: Param[]): Promise<SplatData> => {
    const module = await import(filename);
    if (!module) {
        throw new Error(`Failed to load module: ${filename}`);
    }

    const generator = new module.Generator(params) as Generator;

    const columns: Column[] = generator.columnNames.map((name: string) => {
        return new Column(name, new Float32Array(generator.count));
    });

    const row: any = {};
    for (let i = 0; i < generator.count; ++i) {
        generator.getRow(i, row);
        columns.forEach((c) => {
            c.data[i] = row[c.name];
        });
    }

    return {
        comments: [],
        elements: [{
            name: 'vertex',
            dataTable: new DataTable(columns)
        }]
    };
};

export { Param, readMjs };
