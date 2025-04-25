import { Buffer } from 'node:buffer';
import { FileHandle } from 'node:fs/promises';

import { calcDataSize, PlyElement, PlyHeader, PlyFile, getDataType } from './ply';

// parse the ply header text and return an array of Element structures and a
// string containing the ply format
const parsePlyHeader = (data: Buffer): PlyHeader => {
    // decode header and split into lines
    const strings = new TextDecoder('ascii')
    .decode(data)
    .split('\n')
    .filter(line => line);

    const elements: PlyElement[] = [];
    let element;
    for (let i = 1; i < strings.length; ++i) {
        const words = strings[i].split(' ');

        switch (words[0]) {
            case 'ply':
            case 'format':
            case 'comment':
            case 'end_header':
                // skip
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

    return { strings, elements };
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

const readPly = async (fileHandle: FileHandle): Promise<PlyFile> => {

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
    const header = parsePlyHeader(headerBuf.subarray(0, headerSize));

    const dataSize = calcDataSize(header);

    const data = Buffer.alloc(dataSize);
    if ((await fileHandle.read(data, 0, dataSize)).bytesRead !== dataSize) {
        throw new Error('failed reading ply data');
    }

    return { header, data };
};

export { readPly };
