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
    strings: string[];
    elements: PlyElement[];
};

type PlyFile = {
    header: PlyHeader;
    data: Buffer;
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

const calcDataSize = (plyFile: PlyHeader) => {
    let result = 0;
    for (const element of plyFile.elements) {
        for (const property of element.properties) {
            result += getDataType(property.type).BYTES_PER_ELEMENT * element.count;
        }
    }
    return result;
};

const shNames = new Array(45).fill('').map((_, i) => `f_rest_${i}`);

export { PlyProperty, PlyElement, PlyHeader, PlyFile, getDataType, calcDataSize, shNames };
