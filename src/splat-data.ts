import { PlyElement, PlyFile, getDataType, shNames } from './ply';

class SplatDescriptor {
    numGaussians: number;
    numProperties: number;
    hasPosition: boolean;
    hasRotation: boolean;
    hasScale: boolean;
    hasColor: boolean;
    hasOpacity: boolean;
    numSHBands: number;
}

type Accessor = (index: number) => void;

// wraps ply file data and adds helpers accessors
class SplatData {
    hasProperty: (name: string) => boolean;

    makeAccessor: (propNames: string[], result: number[] | Float32Array) => Accessor;

    descriptor: SplatDescriptor;

    constructor(plyFile: PlyFile) {
        const float32 = new Float32Array(plyFile.data.buffer);

        let vertex: PlyElement;
        const properties: { [key: string]: { type: string, offset: number } } = {};
        let stride = 0;

        // find vertex element and populate property offsets
        for (let i = 0; i < plyFile.header.elements.length; ++i) {
            const element = plyFile.header.elements[i];
            if (element.name === 'vertex') {
                vertex = element;
            }

            for (let j = 0; j < element.properties.length; ++j) {
                const property = element.properties[j];
                if (element === vertex) {
                    properties[property.name] = {
                        type: property.type,
                        offset: stride / 4
                    };
                }
                stride += getDataType(property.type).BYTES_PER_ELEMENT;
            }
        }

        this.hasProperty = (name: string) => {
            return properties.hasOwnProperty(name);
        };

        this.makeAccessor = (propNames: string[], result: number[] | Float32Array) => {
            const offsets = propNames.map(name => properties[name].offset);
            const len = propNames.length;
            return (index: number) => {
                const base = index * stride / 4;
                for (let i = 0; i < len; ++i) {
                    result[i] = float32[base + offsets[i]];
                }
            };
        };

        this.descriptor = {
            numGaussians: vertex.count,
            numProperties: vertex.properties.length,
            hasPosition: ['x', 'y', 'z'].every(v => this.hasProperty(v)),
            hasRotation: ['rot_0', 'rot_1', 'rot_2', 'rot_3'].every(v => this.hasProperty(v)),
            hasScale: ['scale_0', 'scale_1', 'scale_2'].every(v => this.hasProperty(v)),
            hasColor: ['f_dc_0', 'f_dc_1', 'f_dc_2'].every(v => this.hasProperty(v)),
            hasOpacity: this.hasProperty('opacity'),
            numSHBands: { '9': 1, '24': 2, '-1': 3 }[shNames.findIndex(v => !this.hasProperty(v))] ?? 0
        };
    }
}

export { Accessor, SplatData };
