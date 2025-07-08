import { Buffer } from 'node:buffer';
import { FileHandle } from 'node:fs/promises';

import { Column, DataTable } from '../data-table';

type PlyProperty = {
    name: string;               // 'x', f_dc_0', etc
    type: string;               // 'float', 'char', etc
};

type PlyElement = {
    name: string;               // 'vertex', etc
    count: number;
    properties: PlyProperty[];
};

type PlyHeader = {
    comments: string[];
    elements: PlyElement[];
};

type PlyData = {
    comments: string[];
    elements: {
        name: string,
        dataTable: DataTable
    }[];
};

const getDataType = (type: string) => {
    switch (type) {
        case 'char': return Int8Array;
        case 'uchar': return Uint8Array;
        case 'short': return Int16Array;
        case 'ushort': return Uint16Array;
        case 'int': return Int32Array;
        case 'uint': return Uint32Array;
        case 'float': return Float32Array;
        case 'double': return Float64Array;
        default: return null;
    }
};

// parse the ply header text and return an array of Element structures and a
// string containing the ply format
const parseHeader = (data: Buffer): PlyHeader => {
    // decode header and split into lines
    const strings = new TextDecoder('ascii')
    .decode(data)
    .split('\n')
    .filter(line => line);

    const elements: PlyElement[] = [];
    const comments: string[] = [];
    let element;
    for (let i = 1; i < strings.length; ++i) {
        const words = strings[i].split(' ');

        switch (words[0]) {
            case 'ply':
            case 'format':
            case 'end_header':
                // skip
                break;
            case 'comment':
                comments.push(strings[i].substring(8)); // skip 'comment '
                break;
            case 'element': {
                if (words.length !== 3) {
                    throw new Error('invalid ply header');
                }
                element = {
                    name: words[1],
                    count: parseInt(words[2], 10),
                    properties: []
                };
                elements.push(element);
                break;
            }
            case 'property': {
                if (!element || words.length !== 3 || !getDataType(words[1])) {
                    throw new Error('invalid ply header');
                }
                element.properties.push({
                    name: words[2],
                    type: words[1]
                });
                break;
            }
            default: {
                throw new Error(`unrecognized header value '${words[0]}' in ply header`);
            }
        }
    }

    return { comments, elements };
};

const cmp = (a: Uint8Array, b: Uint8Array, aOffset = 0) => {
    for (let i = 0; i < b.length; ++i) {
        if (a[aOffset + i] !== b[i]) {
            return false;
        }
    }
    return true;
};

const magicBytes = new Uint8Array([112, 108, 121, 10]);                                                 // ply\n
const endHeaderBytes = new Uint8Array([10, 101, 110, 100, 95, 104, 101, 97, 100, 101, 114, 10]);        // \nend_header\n

const readPly = async (fileHandle: FileHandle): Promise<PlyData> => {

    // we don't support ply text header larger than 128k
    const headerBuf = Buffer.alloc(128 * 1024);

    // smallest possible header size
    let headerSize = magicBytes.length + endHeaderBytes.length;

    if ((await fileHandle.read(headerBuf, 0, headerSize)).bytesRead !== headerSize) {
        throw new Error('failed to read file header');
    }

    if (!cmp(headerBuf, magicBytes)) {
        throw new Error('invalid file header');
    }

    // read the rest of the header till we find end header byte pattern
    while (true) {
        // read the next character
        if ((await fileHandle.read(headerBuf, headerSize++, 1)).bytesRead !== 1) {
            throw new Error('failed to read file header');
        }

        // check if we've reached the end of the header
        if (cmp(headerBuf, endHeaderBytes, headerSize - endHeaderBytes.length)) {
            break;
        }
    }

    // parse the header
    const header = parseHeader(headerBuf.subarray(0, headerSize));

    // create a data table for each ply element
    const elements = [];
    for (let i = 0; i < header.elements.length; ++i) {
        const element = header.elements[i];

        const columns = element.properties.map((property) => {
            return new Column(property.name, new (getDataType(property.type))(element.count));
        });

        const buffers = columns.map(column => new Uint8Array(column.data.buffer));
        const sizes = columns.map(column => column.data.BYTES_PER_ELEMENT);
        const rowSize = sizes.reduce((total, size) => total + size, 0);

        // read data in chunks of 1024 rows at a time
        const chunkSize = 1024;
        const numChunks = Math.ceil(element.count / chunkSize);
        const chunkData = Buffer.alloc(chunkSize * rowSize);

        for (let c = 0; c < numChunks; ++c) {
            const numRows = Math.min(chunkSize, element.count - c * chunkSize);

            await fileHandle.read(chunkData, 0, rowSize * numRows);

            let offset = 0;

            // read data row at a time
            for (let r = 0; r < numRows; ++r) {
                const rowOffset = c * chunkSize + r;

                // copy into column data
                for (let p = 0; p < columns.length; ++p) {
                    const s = sizes[p];
                    chunkData.copy(buffers[p], rowOffset * s, offset, offset + s);
                    offset += s;
                }
            }
        }

        elements.push({
            name: element.name,
            dataTable: new DataTable(columns)
        });
    }

    return {
        comments: header.comments,
        elements
    };
};

export { PlyData, readPly };
