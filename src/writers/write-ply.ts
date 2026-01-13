import { PlyData } from '../readers/read-ply';
import { type FileSystem } from '../serialize/write';

const columnTypeToPlyType = (type: string): string => {
    switch (type) {
        case 'float32': return 'float';
        case 'float64': return 'double';
        case 'int8': return 'char';
        case 'uint8': return 'uchar';
        case 'int16': return 'short';
        case 'uint16': return 'ushort';
        case 'int32': return 'int';
        case 'uint32': return 'uint';
    }
};

type WritePlyOptions = {
    filename: string;
    plyData: PlyData;
};

const writePly = async (options: WritePlyOptions, fs: FileSystem) => {
    const { filename, plyData } = options;

    const header = [
        'ply',
        'format binary_little_endian 1.0',
        plyData.comments.map(c => `comment ${c}`),
        plyData.elements.map((element) => {
            return [
                `element ${element.name} ${element.dataTable.numRows}`,
                element.dataTable.columns.map((column) => {
                    return `property ${columnTypeToPlyType(column.dataType)} ${column.name}`;
                })
            ];
        }),
        'end_header'
    ];

    // write the header
    const writer = await fs.createWriter(filename);
    await writer.write((new TextEncoder()).encode(`${header.flat(3).join('\n')}\n`));

    for (let i = 0; i < plyData.elements.length; ++i) {
        const table = plyData.elements[i].dataTable;
        const columns = table.columns;
        const buffers = columns.map(c => Buffer.from(c.data.buffer));
        const sizes = columns.map(c => c.data.BYTES_PER_ELEMENT);
        const rowSize = sizes.reduce((total, size) => total + size, 0);

        // write to file in chunks of 1024 rows
        const chunkSize = 1024;
        const numChunks = Math.ceil(table.numRows / chunkSize);
        const chunkData = Buffer.alloc(chunkSize * rowSize);

        for (let c = 0; c < numChunks; ++c) {
            const numRows = Math.min(chunkSize, table.numRows - c * chunkSize);

            let offset = 0;

            for (let r = 0; r < numRows; ++r) {
                const rowOffset = c * chunkSize + r;

                for (let p = 0; p < columns.length; ++p) {
                    const s = sizes[p];
                    buffers[p].copy(chunkData, offset, rowOffset * s, rowOffset * s + s);
                    offset += s;
                }
            }

            // write the chunk
            await writer.write(chunkData.subarray(0, offset));
        }
    }

    await writer.close();
};

export { writePly };
