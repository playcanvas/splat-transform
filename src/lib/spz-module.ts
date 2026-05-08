import { Column, DataTable, convertToSpace } from './data-table';
import { Transform } from './utils';

type CoordinateSystem = {
    RDF: number;
};

type GaussianCloud = {
    numPoints: number;
    shDegree: number;
    antialiased: boolean;
    extensions: unknown[];
    positions: Float32Array;
    scales: Float32Array;
    rotations: Float32Array;
    alphas: Float32Array;
    colors: Float32Array;
    sh: Float32Array;
};

type PackOptions = {
    version: number;
    from: number;
    sh1Bits?: number;
    shRestBits?: number;
};

type UnpackOptions = {
    to: number;
};

type SpzModule = {
    CoordinateSystem: CoordinateSystem;
    LATEST_SPZ_HEADER_VERSION: number;
    loadSpzFromBuffer(data: Uint8Array | ArrayBuffer, options?: UnpackOptions): GaussianCloud | Promise<GaussianCloud>;
    saveSpzToBuffer(cloud: GaussianCloud, options?: PackOptions): Uint8Array;
};

type CreateSpzModule = () => Promise<SpzModule>;

const SPZ_SH_COMPONENTS = [0, 9, 24, 45, 72] as const;

let spzModulePromise: Promise<SpzModule> | null = null;

// `@adobe/spz`'s GaussianCloud uses PLY-native conventions for scales/colors/alphas:
// scales are log-space, colors are SH DC coefficients, alphas are pre-sigmoid (logit).
// The wasm packer applies (s+10)*16, c*0.15+0.5, sigmoid(a) internally; the unpacker inverts.
// So we pass these fields straight through and only reorder the quaternion (xyzw <-> wxyz).

const getCreateSpzModule = async () => {
    const { default: createModule } = await import('@adobe/spz');
    return createModule as CreateSpzModule;
};

const getSpzModule = (): Promise<SpzModule> => {
    if (!spzModulePromise) {
        spzModulePromise = getCreateSpzModule().then(createModule => createModule());
    }
    return spzModulePromise;
};

const getShColumnCount = (dataTable: DataTable) => {
    let count = 0;
    while (dataTable.hasColumn(`f_rest_${count}`)) {
        count += 1;
    }
    return count;
};

const getShDegreeFromCount = (count: number) => {
    const degree = SPZ_SH_COMPONENTS.indexOf(count as typeof SPZ_SH_COMPONENTS[number]);
    if (degree === -1) {
        throw new Error(`Unsupported SH coefficient count for SPZ: ${count}`);
    }
    return degree;
};

const gaussianCloudToDataTable = (cloud: GaussianCloud) => {
    const { numPoints, shDegree } = cloud;
    if (shDegree < 0 || shDegree >= SPZ_SH_COMPONENTS.length) {
        throw new Error(`Unsupported SH degree ${shDegree}`);
    }

    const shColumnCount = SPZ_SH_COMPONENTS[shDegree];
    const shCoefficientsPerChannel = shColumnCount / 3;

    const columns = [
        new Column('x', new Float32Array(numPoints)),
        new Column('y', new Float32Array(numPoints)),
        new Column('z', new Float32Array(numPoints)),
        new Column('scale_0', new Float32Array(numPoints)),
        new Column('scale_1', new Float32Array(numPoints)),
        new Column('scale_2', new Float32Array(numPoints)),
        new Column('f_dc_0', new Float32Array(numPoints)),
        new Column('f_dc_1', new Float32Array(numPoints)),
        new Column('f_dc_2', new Float32Array(numPoints)),
        new Column('opacity', new Float32Array(numPoints)),
        new Column('rot_0', new Float32Array(numPoints)),
        new Column('rot_1', new Float32Array(numPoints)),
        new Column('rot_2', new Float32Array(numPoints)),
        new Column('rot_3', new Float32Array(numPoints))
    ];

    for (let i = 0; i < shColumnCount; i += 1) {
        columns.push(new Column(`f_rest_${i}`, new Float32Array(numPoints)));
    }

    const x = columns[0].data as Float32Array;
    const y = columns[1].data as Float32Array;
    const z = columns[2].data as Float32Array;
    const scale0 = columns[3].data as Float32Array;
    const scale1 = columns[4].data as Float32Array;
    const scale2 = columns[5].data as Float32Array;
    const color0 = columns[6].data as Float32Array;
    const color1 = columns[7].data as Float32Array;
    const color2 = columns[8].data as Float32Array;
    const opacity = columns[9].data as Float32Array;
    const rot0 = columns[10].data as Float32Array;
    const rot1 = columns[11].data as Float32Array;
    const rot2 = columns[12].data as Float32Array;
    const rot3 = columns[13].data as Float32Array;

    for (let i = 0; i < numPoints; i += 1) {
        const i3 = i * 3;
        const i4 = i * 4;

        x[i] = cloud.positions[i3];
        y[i] = cloud.positions[i3 + 1];
        z[i] = cloud.positions[i3 + 2];

        scale0[i] = cloud.scales[i3];
        scale1[i] = cloud.scales[i3 + 1];
        scale2[i] = cloud.scales[i3 + 2];

        color0[i] = cloud.colors[i3];
        color1[i] = cloud.colors[i3 + 1];
        color2[i] = cloud.colors[i3 + 2];

        opacity[i] = cloud.alphas[i];

        rot0[i] = cloud.rotations[i4 + 3];
        rot1[i] = cloud.rotations[i4];
        rot2[i] = cloud.rotations[i4 + 1];
        rot3[i] = cloud.rotations[i4 + 2];

        for (let coeff = 0; coeff < shCoefficientsPerChannel; coeff += 1) {
            const shBase = i * shColumnCount + coeff * 3;
            (columns[14 + coeff].data as Float32Array)[i] = cloud.sh[shBase];
            (columns[14 + shCoefficientsPerChannel + coeff].data as Float32Array)[i] = cloud.sh[shBase + 1];
            (columns[14 + shCoefficientsPerChannel * 2 + coeff].data as Float32Array)[i] = cloud.sh[shBase + 2];
        }
    }

    return new DataTable(columns, Transform.PLY);
};

const dataTableToGaussianCloud = (dataTable: DataTable): GaussianCloud => {
    const plyDataTable = convertToSpace(dataTable, Transform.PLY);
    const shColumnCount = getShColumnCount(plyDataTable);
    const shDegree = getShDegreeFromCount(shColumnCount);
    const shCoefficientsPerChannel = shColumnCount / 3;
    const numPoints = plyDataTable.numRows;

    const positions = new Float32Array(numPoints * 3);
    const scales = new Float32Array(numPoints * 3);
    const rotations = new Float32Array(numPoints * 4);
    const alphas = new Float32Array(numPoints);
    const colors = new Float32Array(numPoints * 3);
    const sh = new Float32Array(numPoints * shColumnCount);

    const x = plyDataTable.getColumnByName('x').data as Float32Array;
    const y = plyDataTable.getColumnByName('y').data as Float32Array;
    const z = plyDataTable.getColumnByName('z').data as Float32Array;
    const scale0 = plyDataTable.getColumnByName('scale_0').data as Float32Array;
    const scale1 = plyDataTable.getColumnByName('scale_1').data as Float32Array;
    const scale2 = plyDataTable.getColumnByName('scale_2').data as Float32Array;
    const color0 = plyDataTable.getColumnByName('f_dc_0').data as Float32Array;
    const color1 = plyDataTable.getColumnByName('f_dc_1').data as Float32Array;
    const color2 = plyDataTable.getColumnByName('f_dc_2').data as Float32Array;
    const opacity = plyDataTable.getColumnByName('opacity').data as Float32Array;
    const rot0 = plyDataTable.getColumnByName('rot_0').data as Float32Array;
    const rot1 = plyDataTable.getColumnByName('rot_1').data as Float32Array;
    const rot2 = plyDataTable.getColumnByName('rot_2').data as Float32Array;
    const rot3 = plyDataTable.getColumnByName('rot_3').data as Float32Array;

    const shColumns = Array.from(
        { length: shColumnCount },
        (_, index) => plyDataTable.getColumnByName(`f_rest_${index}`).data as Float32Array
    );

    for (let i = 0; i < numPoints; i += 1) {
        const i3 = i * 3;
        const i4 = i * 4;

        positions[i3] = x[i];
        positions[i3 + 1] = y[i];
        positions[i3 + 2] = z[i];

        scales[i3] = scale0[i];
        scales[i3 + 1] = scale1[i];
        scales[i3 + 2] = scale2[i];

        colors[i3] = color0[i];
        colors[i3 + 1] = color1[i];
        colors[i3 + 2] = color2[i];

        alphas[i] = opacity[i];

        rotations[i4] = rot1[i];
        rotations[i4 + 1] = rot2[i];
        rotations[i4 + 2] = rot3[i];
        rotations[i4 + 3] = rot0[i];

        for (let coeff = 0; coeff < shCoefficientsPerChannel; coeff += 1) {
            const shBase = i * shColumnCount + coeff * 3;
            sh[shBase] = shColumns[coeff][i];
            sh[shBase + 1] = shColumns[shCoefficientsPerChannel + coeff][i];
            sh[shBase + 2] = shColumns[shCoefficientsPerChannel * 2 + coeff][i];
        }
    }

    return {
        numPoints,
        shDegree,
        antialiased: false,
        extensions: [],
        positions,
        scales,
        rotations,
        alphas,
        colors,
        sh
    };
};

// splat-transform treats SPZ as RDF on both sides: data is converted to PLY/RDF
// space before saving (wasm flips RDF→RUB to produce a spec-compliant on-disk file),
// and the reader requests `to: RDF` to flip back. The SPZ format itself stores no
// coordinate-system metadata (see NgspFileHeader in the spec), so the convention is
// purely by agreement; this matches the Niantic spec and lets splat-transform's
// SPZ readers/writers round-trip through PLY losslessly.
const makeSpzPackOptions = async (overrides: Partial<PackOptions> = {}): Promise<PackOptions> => {
    const spz = await getSpzModule();
    return {
        version: spz.LATEST_SPZ_HEADER_VERSION,
        from: spz.CoordinateSystem.RDF,
        sh1Bits: 5,
        shRestBits: 4,
        ...overrides
    };
};

export {
    SPZ_SH_COMPONENTS,
    dataTableToGaussianCloud,
    gaussianCloudToDataTable,
    getSpzModule,
    makeSpzPackOptions
};
