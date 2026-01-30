import { isCompressedPly, decompressPly } from './decompress-ply';
import { Column, DataTable } from '../data-table/data-table';
import { ReadSource, ReadStream } from '../io/read';

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

// Returns a function that reads a value from a DataView at the given byte offset
type ValueReader = (view: DataView, offset: number) => number;

const getReader = (type: string): ValueReader => {
    switch (type) {
        case 'char':   return (v, o) => v.getInt8(o);
        case 'uchar':  return (v, o) => v.getUint8(o);
        case 'short':  return (v, o) => v.getInt16(o, true);
        case 'ushort': return (v, o) => v.getUint16(o, true);
        case 'int':    return (v, o) => v.getInt32(o, true);
        case 'uint':   return (v, o) => v.getUint32(o, true);
        case 'float':  return (v, o) => v.getFloat32(o, true);
        case 'double': return (v, o) => v.getFloat64(o, true);
        default: throw new Error(`unsupported ply type: ${type}`);
    }
};

// Check if all properties in an element are float type (enables fast path)
const isFloatElement = (element: PlyElement): boolean => {
    return element.properties.every(p => p.type === 'float');
};

// parse the ply header text and return an array of Element structures and a
// string containing the ply format
const parseHeader = (data: Uint8Array): PlyHeader => {
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

/**
 * Helper to read exactly n bytes from a stream into a buffer at an offset.
 * @param stream - The stream to read from
 * @param buffer - Target buffer
 * @param offset - Offset in buffer to write to
 * @param length - Number of bytes to read
 * @returns Number of bytes actually read
 */
const readExact = async (
    stream: ReadStream,
    buffer: Uint8Array,
    offset: number,
    length: number
): Promise<number> => {
    let totalRead = 0;
    while (totalRead < length) {
        const target = buffer.subarray(offset + totalRead, offset + length);
        const n = await stream.pull(target);
        if (n === 0) break; // EOF
        totalRead += n;
    }
    return totalRead;
};

/**
 * Reads a PLY file containing Gaussian splat data.
 *
 * Supports both standard PLY files and compressed PLY format. The PLY format is
 * the standard output from 3D Gaussian Splatting training pipelines.
 *
 * @param source - The read source providing access to the PLY file data.
 * @returns Promise resolving to a DataTable containing the splat data.
 * @ignore
 */
const readPly = async (source: ReadSource): Promise<DataTable> => {
    const stream = source.read();

    // we don't support ply text header larger than 128k
    const headerBuf = new Uint8Array(128 * 1024);

    // smallest possible header size
    let headerSize = magicBytes.length + endHeaderBytes.length;

    if (await readExact(stream, headerBuf, 0, headerSize) !== headerSize) {
        throw new Error('failed to read file header');
    }

    if (!cmp(headerBuf, magicBytes)) {
        throw new Error('invalid file header');
    }

    // read the rest of the header till we find end header byte pattern
    while (true) {
        // read the next character
        if (await readExact(stream, headerBuf, headerSize++, 1) !== 1) {
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
            return new Column(property.name, new (getDataType(property.type)!)(element.count));
        });

        const numProperties = columns.length;
        const numRows = element.count;

        // Check if all properties are float32 (enables fast path)
        if (isFloatElement(element)) {
            // Fast path: all properties are float32
            // Use Float32Array view with property-major loop order for best performance
            const rowSize = numProperties * 4; // 4 bytes per float
            const chunkSize = 1024;
            const numChunks = Math.ceil(numRows / chunkSize);
            const chunkData = new Uint8Array(chunkSize * rowSize);
            const floatData = new Float32Array(chunkData.buffer);

            // Pre-extract storage arrays for direct access
            const storage = columns.map(c => c.data as Float32Array);

            for (let c = 0; c < numChunks; ++c) {
                const chunkRows = Math.min(chunkSize, numRows - c * chunkSize);
                const baseRowIndex = c * chunkSize;

                await readExact(stream, chunkData, 0, rowSize * chunkRows);

                // Property-major loop order: better cache locality, processes batch at once
                for (let p = 0; p < numProperties; ++p) {
                    const s = storage[p];
                    for (let r = 0; r < chunkRows; ++r) {
                        s[baseRowIndex + r] = floatData[r * numProperties + p];
                    }
                }
            }
        } else {
            // General path: mixed types, use DataView with reader functions
            let byteOffset = 0;
            const columnInfo = element.properties.map((property, idx) => {
                const size = columns[idx].data.BYTES_PER_ELEMENT;
                const info = {
                    data: columns[idx].data,
                    size,
                    byteOffset,
                    reader: getReader(property.type)
                };
                byteOffset += size;
                return info;
            });

            const rowSize = byteOffset;
            const chunkSize = 1024;
            const numChunks = Math.ceil(numRows / chunkSize);
            const chunkData = new Uint8Array(chunkSize * rowSize);

            for (let c = 0; c < numChunks; ++c) {
                const chunkRows = Math.min(chunkSize, numRows - c * chunkSize);

                await readExact(stream, chunkData, 0, rowSize * chunkRows);

                // Create DataView once per chunk
                const view = new DataView(chunkData.buffer, chunkData.byteOffset, chunkData.byteLength);

                // Row-major loop with DataView readers
                for (let r = 0; r < chunkRows; ++r) {
                    const rowIndex = c * chunkSize + r;
                    const rowByteOffset = r * rowSize;

                    for (let p = 0; p < columnInfo.length; ++p) {
                        const info = columnInfo[p];
                        info.data[rowIndex] = info.reader(view, rowByteOffset + info.byteOffset);
                    }
                }
            }
        }

        elements.push({
            name: element.name,
            dataTable: new DataTable(columns)
        });
    }

    const plyData = {
        comments: header.comments,
        elements
    };

    if (isCompressedPly(plyData)) {
        return decompressPly(plyData);
    }

    const vertexElement = plyData.elements.find(e => e.name === 'vertex');
    if (!vertexElement) {
        throw new Error('PLY file does not contain vertex element');
    }

    return vertexElement.dataTable;
};

export { PlyData, readPly };
