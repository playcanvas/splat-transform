import { Accessor, SplatData } from './splat-data';

// wrap the data of a single gaussian
class GaussianData {
    // read data from the splat into the data array
    read: (splat: SplatData, index: number) => void;

    // map of property name to value
    data: any;

    // the data buffer of values
    buffer: Float32Array;

    constructor(members: string[]) {
        const buffer = new Float32Array(members.length);

        const data: any = {};
        members.forEach((name) => {
            data[name] = 0;
        });

        const accessors = new Map<SplatData, Accessor>();

        this.read = (splat: SplatData, index: number) => {
            if (!accessors.has(splat)) {
                const accessor = splat.makeAccessor(members, buffer);
                accessors.set(splat, accessor);
            }

            accessors.get(splat)(index);

            for (let i = 0; i < members.length; ++i) {
                data[members[i]] = buffer[i];
            }
        };

        this.data = data;
        this.buffer = buffer;
    }
}

type GaussianFilter = (gaussianData: GaussianData) => boolean;
type GaussianTransform = (gaussianData: GaussianData) => void;

export { GaussianData, GaussianFilter, GaussianTransform };
