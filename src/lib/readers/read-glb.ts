import {
    POSITION_STRIDE,
    GEOMETRIC_STRIDE,
    SH_REST_COUNTS,
    colorStride,
    positionFields,
    geometricFields,
    colorFields
} from '../chunk';
import type {
    ChunkDataPool,
    ChunkLayer,
    ChunkSource,
    ChunkSourceMetadata,
    LayerLayout,
    ReadRequest,
    SHBands
} from '../chunk';
import type { ReadSource } from '../io/read';
import { Transform } from '../utils';

import { fileChunkSource, gatherRuns, readExact, sortGatherSlots } from './reader-utils';

const GLB_MAGIC = 0x46546c67;
const GLB_VERSION = 2;
const JSON_CHUNK_TYPE = 0x4e4f534a;
const BIN_CHUNK_TYPE = 0x004e4942;
const POINTS = 0;
const MAX_ACCESSOR_READ_BYTES = 8 * 1024 * 1024;
const COMPONENT_BYTE = 5120;
const COMPONENT_UNSIGNED_BYTE = 5121;
const COMPONENT_SHORT = 5122;
const COMPONENT_UNSIGNED_SHORT = 5123;
const COMPONENT_UNSIGNED_INT = 5125;
const COMPONENT_FLOAT = 5126;

type GltfAccessor = {
    bufferView?: number;
    byteOffset?: number;
    componentType: number;
    normalized?: boolean;
    count: number;
    type: string;
    sparse?: unknown;
};

type GltfBufferView = {
    buffer: number;
    byteOffset?: number;
    byteLength: number;
    byteStride?: number;
};

type GltfPrimitive = {
    attributes?: Record<string, number>;
    indices?: number;
    mode?: number;
    extensions?: Record<string, unknown>;
};

type Gltf = {
    asset?: { version?: string };
    buffers?: { byteLength: number; uri?: string }[];
    bufferViews?: GltfBufferView[];
    accessors?: GltfAccessor[];
    meshes?: { primitives?: GltfPrimitive[] }[];
    nodes?: {
        mesh?: number;
        children?: number[];
        matrix?: number[];
        translation?: number[];
        rotation?: number[];
        scale?: number[];
    }[];
    extensionsUsed?: string[];
};

type AccessorInfo = {
    name: string;
    accessor: GltfAccessor;
    components: number;
    componentBytes: number;
    elementBytes: number;
    stride: number;
    absoluteOffset: number;
};

type RowSelection = {
    count: number;
    rowAt: (slot: number) => number;
    contiguousStart?: number;
    sortedSlots?: Uint32Array;
};

type GaussianExtension = {
    kernel?: unknown;
    colorSpace?: unknown;
    projection?: unknown;
    sortingMethod?: unknown;
};

const componentsForType = (type: string): number => {
    switch (type) {
        case 'SCALAR':
            return 1;
        case 'VEC2':
            return 2;
        case 'VEC3':
            return 3;
        case 'VEC4':
            return 4;
        default:
            throw new Error(`readGlb: unsupported accessor type '${type}'`);
    }
};

const bytesForComponent = (componentType: number): number => {
    switch (componentType) {
        case COMPONENT_BYTE:
        case COMPONENT_UNSIGNED_BYTE:
            return 1;
        case COMPONENT_SHORT:
        case COMPONENT_UNSIGNED_SHORT:
            return 2;
        case COMPONENT_UNSIGNED_INT:
        case COMPONENT_FLOAT:
            return 4;
        default:
            throw new Error(`readGlb: unsupported accessor componentType ${componentType}`);
    }
};

const isIdentityNode = (node: NonNullable<Gltf['nodes']>[number]): boolean => {
    const matrix = node.matrix;
    if (matrix) {
        const identity = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1];
        if (matrix.length !== 16 || matrix.some((v, i) => v !== identity[i])) return false;
    }
    if (node.translation && (node.translation.length !== 3 || node.translation.some((v) => v !== 0))) return false;
    if (node.rotation) {
        if (
            node.rotation.length !== 4 ||
            node.rotation[0] !== 0 ||
            node.rotation[1] !== 0 ||
            node.rotation[2] !== 0 ||
            Math.abs(node.rotation[3]) !== 1
        )
            return false;
    }
    if (node.scale && (node.scale.length !== 3 || node.scale.some((v) => v !== 1))) return false;
    return true;
};

const readComponent = (view: DataView, offset: number, type: number, normalized: boolean): number => {
    switch (type) {
        case COMPONENT_BYTE: {
            const value = view.getInt8(offset);
            return normalized ? Math.max(value / 127, -1) : value;
        }
        case COMPONENT_UNSIGNED_BYTE: {
            const value = view.getUint8(offset);
            return normalized ? value / 255 : value;
        }
        case COMPONENT_SHORT: {
            const value = view.getInt16(offset, true);
            return normalized ? Math.max(value / 32767, -1) : value;
        }
        case COMPONENT_UNSIGNED_SHORT: {
            const value = view.getUint16(offset, true);
            return normalized ? value / 65535 : value;
        }
        case COMPONENT_UNSIGNED_INT: {
            const value = view.getUint32(offset, true);
            return normalized ? value / 4294967295 : value;
        }
        case COMPONENT_FLOAT:
            return view.getFloat32(offset, true);
        default:
            throw new Error(`readGlb: unsupported accessor componentType ${type}`);
    }
};

/**
 * Read a binary glTF file containing one KHR_gaussian_splatting primitive.
 *
 * The JSON header is parsed eagerly, while accessor payloads remain in the
 * underlying file and are decoded by chunk or gathered row on demand. The
 * current writer emits one baked, identity-space primitive. Indexed point
 * primitives are supported; non-identity transforms on the Gaussian node
 * hierarchy, mesh instancing, and multiple Gaussian primitives are rejected
 * rather than being silently ignored.
 *
 * @param source - Seekable (or resident fallback) GLB data source.
 * @param pool - Pool whose chunk size controls the returned source.
 * @returns A lazy source exposing canonical splat-transform layers.
 * @ignore
 */
const readGlb = async (source: ReadSource, pool: ChunkDataPool): Promise<ChunkSource> => {
    let resident: Uint8Array | null = null;
    if (!source.seekable || source.size === undefined) {
        resident = await source.read().readAll();
    }

    const size = resident?.byteLength ?? source.size!;
    const readRange = async (start: number, length: number): Promise<Uint8Array> => {
        if (
            !Number.isSafeInteger(start) ||
            !Number.isSafeInteger(length) ||
            start < 0 ||
            length < 0 ||
            start + length > size
        ) {
            throw new Error(`readGlb: byte range [${start}, ${start + length}) is outside the file`);
        }
        if (resident) return resident.subarray(start, start + length);
        const result = new Uint8Array(length);
        if ((await readExact(source.read(start, start + length), result, 0, length)) !== length) {
            throw new Error(`readGlb: short read at byte ${start}`);
        }
        return result;
    };

    if (size < 20) throw new Error('Invalid GLB: file is too short');
    const headerBytes = await readRange(0, 12);
    const header = new DataView(headerBytes.buffer, headerBytes.byteOffset, headerBytes.byteLength);
    if (header.getUint32(0, true) !== GLB_MAGIC) throw new Error('Invalid GLB: bad magic');
    if (header.getUint32(4, true) !== GLB_VERSION) throw new Error('Invalid GLB: only version 2 is supported');
    const declaredLength = header.getUint32(8, true);
    if (declaredLength !== size) {
        throw new Error(`Invalid GLB: declared length ${declaredLength} does not match file size ${size}`);
    }

    let offset = 12;
    let jsonBytes: Uint8Array | null = null;
    let binOffset = -1;
    let binLength = 0;
    let chunkIndex = 0;
    while (offset < declaredLength) {
        if (offset + 8 > declaredLength) throw new Error('Invalid GLB: truncated chunk header');
        const chunkHeaderBytes = await readRange(offset, 8);
        const chunkHeader = new DataView(
            chunkHeaderBytes.buffer,
            chunkHeaderBytes.byteOffset,
            chunkHeaderBytes.byteLength
        );
        const chunkLength = chunkHeader.getUint32(0, true);
        const chunkType = chunkHeader.getUint32(4, true);
        const dataOffset = offset + 8;
        if (chunkLength % 4 !== 0 || dataOffset + chunkLength > declaredLength) {
            throw new Error('Invalid GLB: malformed chunk length');
        }
        if (chunkIndex === 0 && chunkType !== JSON_CHUNK_TYPE) {
            throw new Error('Invalid GLB: JSON must be the first chunk');
        }
        if (chunkType === JSON_CHUNK_TYPE) {
            if (jsonBytes) throw new Error('Invalid GLB: duplicate JSON chunk');
            jsonBytes = await readRange(dataOffset, chunkLength);
        } else if (chunkType === BIN_CHUNK_TYPE) {
            if (binOffset >= 0) throw new Error('Invalid GLB: duplicate BIN chunk');
            binOffset = dataOffset;
            binLength = chunkLength;
        }
        offset = dataOffset + chunkLength;
        chunkIndex++;
    }
    if (!jsonBytes) throw new Error('Invalid GLB: missing JSON chunk');
    if (binOffset < 0) throw new Error('Invalid GLB: missing BIN chunk');

    let gltf: Gltf;
    try {
        gltf = JSON.parse(new TextDecoder().decode(jsonBytes).trimEnd()) as Gltf;
    } catch (e) {
        throw new Error(`Invalid GLB JSON: ${e instanceof Error ? e.message : String(e)}`);
    }
    if (gltf.asset?.version !== '2.0') throw new Error("Invalid GLB: glTF asset.version must be '2.0'");
    if (!gltf.buffers || gltf.buffers.length !== 1 || gltf.buffers[0].uri !== undefined) {
        throw new Error('readGlb: exactly one embedded buffer is supported');
    }
    if (gltf.buffers[0].byteLength > binLength || binLength - gltf.buffers[0].byteLength > 3) {
        throw new Error('Invalid GLB: BIN chunk length does not match buffer byteLength');
    }

    const gaussianPrimitives: GltfPrimitive[] = [];
    for (const mesh of gltf.meshes ?? []) {
        for (const primitive of mesh.primitives ?? []) {
            if (primitive.extensions?.KHR_gaussian_splatting !== undefined) gaussianPrimitives.push(primitive);
        }
    }
    if (gaussianPrimitives.length !== 1) {
        throw new Error(
            `readGlb: expected exactly one KHR_gaussian_splatting primitive, found ${gaussianPrimitives.length}`
        );
    }
    const primitive = gaussianPrimitives[0];
    if ((primitive.mode ?? POINTS) !== POINTS) throw new Error('readGlb: Gaussian primitive mode must be POINTS');
    if (!gltf.extensionsUsed?.includes('KHR_gaussian_splatting')) {
        throw new Error("readGlb: extensionsUsed must include 'KHR_gaussian_splatting'");
    }
    const extension = primitive.extensions?.KHR_gaussian_splatting;
    if (!extension || typeof extension !== 'object' || Array.isArray(extension)) {
        throw new Error('readGlb: invalid KHR_gaussian_splatting extension object');
    }
    const gaussianExtension = extension as GaussianExtension;
    if (gaussianExtension.kernel !== 'ellipse') {
        throw new Error("readGlb: only the KHR_gaussian_splatting 'ellipse' kernel is supported");
    }
    if (gaussianExtension.colorSpace !== 'srgb_rec709_display') {
        throw new Error("readGlb: only the 'srgb_rec709_display' color space is supported");
    }
    if (gaussianExtension.projection !== undefined && gaussianExtension.projection !== 'perspective') {
        throw new Error("readGlb: only the 'perspective' projection is supported");
    }
    if (gaussianExtension.sortingMethod !== undefined && gaussianExtension.sortingMethod !== 'cameraDistance') {
        throw new Error("readGlb: only the 'cameraDistance' sorting method is supported");
    }

    const gaussianMeshIndices = new Set<number>();
    for (let meshIndex = 0; meshIndex < (gltf.meshes?.length ?? 0); meshIndex++) {
        if (gltf.meshes![meshIndex].primitives?.includes(primitive)) gaussianMeshIndices.add(meshIndex);
    }
    const nodes = gltf.nodes ?? [];
    const gaussianNodes = nodes
        .map((node, index) => ({ node, index }))
        .filter(({ node }) => node.mesh !== undefined && gaussianMeshIndices.has(node.mesh));
    if (gaussianNodes.length > 1) {
        throw new Error('readGlb: instanced Gaussian meshes are not supported');
    }
    if (gaussianNodes.length === 1) {
        const parents: number[][] = Array.from({ length: nodes.length }, (): number[] => []);
        for (let parent = 0; parent < nodes.length; parent++) {
            for (const child of nodes[parent].children ?? []) {
                if (!Number.isInteger(child) || child < 0 || child >= nodes.length) {
                    throw new Error(`readGlb: node ${parent} has an invalid child index`);
                }
                parents[child].push(parent);
            }
        }
        const pending = [gaussianNodes[0].index];
        const visited = new Set<number>();
        while (pending.length > 0) {
            const nodeIndex = pending.pop()!;
            if (visited.has(nodeIndex)) throw new Error('readGlb: cycle in Gaussian node hierarchy');
            visited.add(nodeIndex);
            if (!isIdentityNode(nodes[nodeIndex])) {
                throw new Error('readGlb: non-identity Gaussian node transforms are not supported');
            }
            if (parents[nodeIndex].length > 1) {
                throw new Error('readGlb: Gaussian node has multiple parents');
            }
            pending.push(...parents[nodeIndex]);
        }
    }

    const attributes = primitive.attributes ?? {};
    const accessors = gltf.accessors ?? [];
    const bufferViews = gltf.bufferViews ?? [];
    const accessorInfoByIndex = (
        name: string,
        accessorIndex: number | undefined,
        expectedType: string
    ): AccessorInfo => {
        if (!Number.isInteger(accessorIndex) || accessorIndex < 0 || accessorIndex >= accessors.length) {
            throw new Error(`readGlb: missing or invalid '${name}' accessor`);
        }
        const accessor = accessors[accessorIndex];
        if (accessor.type !== expectedType) {
            throw new Error(`readGlb: '${name}' must use ${expectedType}, got ${accessor.type}`);
        }
        if (accessor.sparse !== undefined) throw new Error(`readGlb: sparse accessor '${name}' is not supported`);
        if (
            !Number.isInteger(accessor.bufferView) ||
            accessor.bufferView! < 0 ||
            accessor.bufferView! >= bufferViews.length
        ) {
            throw new Error(`readGlb: '${name}' has an invalid bufferView`);
        }
        if (!Number.isInteger(accessor.count) || accessor.count <= 0) {
            throw new Error(`readGlb: '${name}' has an invalid count`);
        }
        const bufferView = bufferViews[accessor.bufferView!];
        if (bufferView.buffer !== 0) throw new Error(`readGlb: '${name}' does not reference the embedded buffer`);
        const components = componentsForType(accessor.type);
        const componentBytes = bytesForComponent(accessor.componentType);
        const elementBytes = components * componentBytes;
        const stride = bufferView.byteStride ?? elementBytes;
        const viewOffset = bufferView.byteOffset ?? 0;
        const accessorOffset = accessor.byteOffset ?? 0;
        if (
            !Number.isInteger(viewOffset) ||
            !Number.isInteger(accessorOffset) ||
            !Number.isInteger(bufferView.byteLength) ||
            !Number.isInteger(stride) ||
            viewOffset < 0 ||
            accessorOffset < 0 ||
            bufferView.byteLength < 0 ||
            stride < elementBytes ||
            stride % componentBytes !== 0
        ) {
            throw new Error(`readGlb: '${name}' has an invalid byte layout`);
        }
        const endInView = accessorOffset + (accessor.count - 1) * stride + elementBytes;
        if (endInView > bufferView.byteLength || viewOffset + bufferView.byteLength > gltf.buffers![0].byteLength) {
            throw new Error(`readGlb: '${name}' accessor is outside its bufferView`);
        }
        return {
            name,
            accessor,
            components,
            componentBytes,
            elementBytes,
            stride,
            absoluteOffset: binOffset + viewOffset + accessorOffset
        };
    };
    const accessorInfo = (name: string, expectedType: string): AccessorInfo =>
        accessorInfoByIndex(name, attributes[name], expectedType);

    const position = accessorInfo('POSITION', 'VEC3');
    const rotation = accessorInfo('KHR_gaussian_splatting:ROTATION', 'VEC4');
    const scale = accessorInfo('KHR_gaussian_splatting:SCALE', 'VEC3');
    const opacity = accessorInfo('KHR_gaussian_splatting:OPACITY', 'SCALAR');
    const dc = accessorInfo('KHR_gaussian_splatting:SH_DEGREE_0_COEF_0', 'VEC3');
    const vertexCount = position.accessor.count;
    for (const info of [rotation, scale, opacity, dc]) {
        if (info.accessor.count !== vertexCount)
            throw new Error(`readGlb: '${info.name}' count does not match POSITION`);
    }

    const shInfos: AccessorInfo[] = [];
    let shBands: SHBands = 0;
    let coefficientOffset = 0;
    for (let degree = 1 as SHBands; degree <= 3; degree = (degree + 1) as SHBands) {
        const count = degree * 2 + 1;
        const names = Array.from(
            { length: count },
            (_, coefficient) => `KHR_gaussian_splatting:SH_DEGREE_${degree}_COEF_${coefficient}`
        );
        const present = names.map((name) => attributes[name] !== undefined);
        if (!present.some(Boolean)) break;
        if (!present.every(Boolean)) throw new Error(`readGlb: incomplete spherical harmonics degree ${degree}`);
        for (const name of names) {
            const info = accessorInfo(name, 'VEC3');
            if (info.accessor.count !== vertexCount)
                throw new Error(`readGlb: '${name}' count does not match POSITION`);
            shInfos.push(info);
        }
        coefficientOffset += count;
        shBands = degree;
    }
    const knownShNames = new Set(['KHR_gaussian_splatting:SH_DEGREE_0_COEF_0', ...shInfos.map((info) => info.name)]);
    for (const name of Object.keys(attributes)) {
        if (/^KHR_gaussian_splatting:SH_DEGREE_\d+_COEF_\d+$/.test(name) && !knownShNames.has(name)) {
            throw new Error(`readGlb: unsupported or non-contiguous spherical harmonics attribute '${name}'`);
        }
    }
    if (coefficientOffset !== SH_REST_COUNTS[shBands] / 3) {
        throw new Error('readGlb: invalid spherical harmonics layout');
    }

    const validTypes = (
        info: AccessorInfo,
        types: readonly number[],
        integerNormalization: 'required' | 'optional' | 'forbidden'
    ): void => {
        if (!types.includes(info.accessor.componentType)) {
            throw new Error(`readGlb: '${info.name}' has unsupported componentType ${info.accessor.componentType}`);
        }
        const integer = info.accessor.componentType !== COMPONENT_FLOAT;
        if (!integer && info.accessor.normalized === true) {
            throw new Error(`readGlb: float accessor '${info.name}' must not be normalized`);
        }
        if (integer && integerNormalization === 'required' && info.accessor.normalized !== true) {
            throw new Error(`readGlb: integer accessor '${info.name}' must be normalized`);
        }
        if (integer && integerNormalization === 'forbidden' && info.accessor.normalized === true) {
            throw new Error(`readGlb: integer accessor '${info.name}' must not be normalized`);
        }
    };
    validTypes(position, [COMPONENT_FLOAT], 'forbidden');
    validTypes(rotation, [COMPONENT_FLOAT, COMPONENT_BYTE, COMPONENT_SHORT], 'required');
    validTypes(scale, [COMPONENT_FLOAT, COMPONENT_UNSIGNED_BYTE, COMPONENT_UNSIGNED_SHORT], 'optional');
    validTypes(opacity, [COMPONENT_FLOAT, COMPONENT_UNSIGNED_BYTE, COMPONENT_UNSIGNED_SHORT], 'required');
    validTypes(dc, [COMPONENT_FLOAT], 'forbidden');
    for (const info of shInfos) validTypes(info, [COMPONENT_FLOAT], 'forbidden');

    const indexAccessor =
        primitive.indices === undefined ? null : accessorInfoByIndex('INDICES', primitive.indices, 'SCALAR');
    if (indexAccessor) {
        validTypes(
            indexAccessor,
            [COMPONENT_UNSIGNED_BYTE, COMPONENT_UNSIGNED_SHORT, COMPONENT_UNSIGNED_INT],
            'forbidden'
        );
    }
    const numGaussians = indexAccessor?.accessor.count ?? vertexCount;

    const chunkSize = pool.chunkSize;
    const availableLayers = new Set<ChunkLayer>(['position', 'geometric', 'color']);
    const layouts: Partial<Record<ChunkLayer, LayerLayout>> = {
        position: { stride: POSITION_STRIDE, fields: positionFields() },
        geometric: { stride: GEOMETRIC_STRIDE, fields: geometricFields() },
        color: { stride: colorStride(shBands), fields: colorFields(shBands) }
    };
    const meta: ChunkSourceMetadata = {
        numGaussians,
        numLods: 1,
        lodCounts: [numGaussians],
        chunkSize,
        numChunks: [Math.ceil(numGaussians / chunkSize)],
        shBands,
        model: 'default',
        extraColumns: [],
        transform: Transform.IDENTITY,
        availableLayers,
        layouts
    };

    let accessorScratch = new Uint8Array(0);
    const readAccessorRange = async (start: number, length: number): Promise<Uint8Array> => {
        if (resident) return readRange(start, length);
        if (accessorScratch.length < length) accessorScratch = new Uint8Array(length);
        const result = accessorScratch.subarray(0, length);
        if ((await readExact(source.read(start, start + length), result, 0, length)) !== length) {
            throw new Error(`readGlb: short accessor read at byte ${start}`);
        }
        return result;
    };

    const readAccessor = async (
        info: AccessorInfo,
        selection: RowSelection,
        write: (slot: number, component: number, value: number) => void
    ): Promise<void> => {
        const decodeRecord = (view: DataView, recordOffset: number, slot: number): void => {
            for (let component = 0; component < info.components; component++) {
                const value = readComponent(
                    view,
                    recordOffset + component * info.componentBytes,
                    info.accessor.componentType,
                    info.accessor.normalized === true
                );
                write(slot, component, value);
            }
        };

        if (selection.contiguousStart !== undefined) {
            const rowsPerWindow = Math.max(
                1,
                Math.floor((MAX_ACCESSOR_READ_BYTES - info.elementBytes) / info.stride) + 1
            );
            for (let firstSlot = 0; firstSlot < selection.count; firstSlot += rowsPerWindow) {
                const rows = Math.min(rowsPerWindow, selection.count - firstSlot);
                const bytes = await readAccessorRange(
                    info.absoluteOffset + (selection.contiguousStart + firstSlot) * info.stride,
                    (rows - 1) * info.stride + info.elementBytes
                );
                const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
                for (let localSlot = 0; localSlot < rows; localSlot++) {
                    decodeRecord(view, localSlot * info.stride, firstSlot + localSlot);
                }
            }
            return;
        }

        const sortedSlots = selection.sortedSlots!;
        const rowAtSorted = (i: number) => selection.rowAt(sortedSlots[i]);
        for (const run of gatherRuns(
            selection.count,
            (i) => info.absoluteOffset + rowAtSorted(i) * info.stride,
            info.stride
        )) {
            const bytes = await readAccessorRange(
                run.firstByte,
                (run.recordCount - 1) * info.stride + info.elementBytes
            );
            const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
            const first = rowAtSorted(run.j0);
            for (let i = run.j0; i < run.j1; i++) {
                decodeRecord(view, (rowAtSorted(i) - first) * info.stride, sortedSlots[i]);
            }
        }
    };

    const planSelection = (count: number, rowAt: (slot: number) => number, contiguousStart?: number): RowSelection => {
        if (count <= 0) return { count, rowAt, contiguousStart: 0 };
        if (contiguousStart !== undefined) return { count, rowAt, contiguousStart };
        const first = rowAt(0);
        let contiguous = true;
        for (let slot = 1; slot < count; slot++) {
            if (rowAt(slot) !== first + slot) {
                contiguous = false;
                break;
            }
        }
        return contiguous
            ? { count, rowAt, contiguousStart: first }
            : { count, rowAt, sortedSlots: sortGatherSlots(count, rowAt) };
    };

    const read = async (request: ReadRequest): Promise<void> => {
        if ((request.lod ?? 0) !== 0) throw new Error('readGlb: only lod 0 is supported');
        let selection: RowSelection;
        if ('indices' in request) {
            const { indices, indexOffset, count } = request;
            for (let i = 0; i < count; i++) {
                const row = indices[indexOffset + i];
                if (row >= numGaussians) throw new Error(`readGlb: row index ${row} is out of range`);
            }
            selection = planSelection(count, (slot) => indices[indexOffset + slot]);
        } else {
            const start = request.chunkIndex * chunkSize;
            const count = Math.min(chunkSize, numGaussians - start);
            if (count <= 0) throw new Error(`readGlb: chunkIndex ${request.chunkIndex} out of range`);
            selection = planSelection(count, (slot) => start + slot, start);
        }
        if (selection.count <= 0) return;
        if (!request.position && !request.geometric && !request.color) return;

        if (indexAccessor) {
            const vertexIndices = new Uint32Array(selection.count);
            await readAccessor(indexAccessor, selection, (slot, _component, value) => {
                if (!Number.isInteger(value) || value < 0 || value >= vertexCount) {
                    throw new Error(`readGlb: primitive index ${value} is out of range`);
                }
                vertexIndices[slot] = value;
            });
            selection = planSelection(selection.count, (slot) => vertexIndices[slot]);
        }

        if (request.position) {
            const output = new Float32Array(request.position.data);
            await readAccessor(position, selection, (slot, component, value) => {
                output[slot * 3 + component] = value;
            });
        }
        if (request.geometric) {
            const output = new Float32Array(request.geometric.data);
            await readAccessor(rotation, selection, (slot, component, value) => {
                output[slot * 8 + ((component + 1) % 4)] = value;
            });
            await readAccessor(scale, selection, (slot, component, value) => {
                if (!Number.isFinite(value) || value < 0) throw new Error(`readGlb: invalid scale value ${value}`);
                output[slot * 8 + 4 + component] = Math.log(value);
            });
            await readAccessor(opacity, selection, (slot, _component, value) => {
                if (!Number.isFinite(value) || value < 0 || value > 1) {
                    throw new Error(`readGlb: invalid opacity value ${value}`);
                }
                output[slot * 8 + 7] = Math.log(value / (1 - value));
            });
        }
        if (request.color) {
            const output = new Float32Array(request.color.data);
            const stride = 3 + SH_REST_COUNTS[shBands];
            await readAccessor(dc, selection, (slot, component, value) => {
                output[slot * stride + component] = value;
            });
            const coefficientsPerChannel = SH_REST_COUNTS[shBands] / 3;
            for (let coefficient = 0; coefficient < shInfos.length; coefficient++) {
                await readAccessor(shInfos[coefficient], selection, (slot, component, value) => {
                    output[slot * stride + 3 + coefficient + component * coefficientsPerChannel] = value;
                });
            }
        }
    };

    return fileChunkSource(source, meta, read);
};

export { readGlb };
