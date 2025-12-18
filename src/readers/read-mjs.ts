import { Column, DataTable } from '../data-table/data-table';
import { Param } from '../types';

type Generator = {
    count: number;
    columnNames: string[];
    getRow: (index: number, row: any) => void;
};

const readMjs = async (filename: string, params: Param[]): Promise<DataTable> => {
    const module = await import(`file://${filename}`);
    if (!module) {
        throw new Error(`Failed to load module: ${filename}`);
    }

    const generator = (await module.Generator?.create(params)) as Generator;

    if (!generator) {
        throw new Error(`Failed to create Generator instance: ${filename}`);
    }

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

    return new DataTable(columns);
};

export { readMjs };
