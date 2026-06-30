/**
 * ChunkLayer identifiers — the disjoint storage tiers of a gaussian source.
 *
 * - `position`        - xyz (vec3<f32>)
 * - `geometric`       - rotation quaternion + scale + opacity
 * - `color`           - DC + spherical harmonics rest coefficients
 * - `other`           - user-defined extra columns (e.g. blind data)
 */
type ChunkLayer = 'position' | 'geometric' | 'color' | 'other';

/**
 * Spherical harmonics band count.
 *
 * - 0 - DC only (3 coefficients)
 * - 1 - DC + 1st band (3 + 9 = 12 coefficients)
 * - 2 - DC + 1st + 2nd (3 + 24 = 27 coefficients)
 * - 3 - DC + 1st + 2nd + 3rd (3 + 45 = 48 coefficients)
 */
type SHBands = 0 | 1 | 2 | 3;

/** Number of `f_rest_*` SH coefficients (excluding the 3 DC coefficients) per band level. */
const SH_REST_COUNTS: Readonly<Record<SHBands, number>> = {
    0: 0,
    1: 9,
    2: 24,
    3: 45
};

/** Bytes per gaussian for the `position` layer. */
const POSITION_STRIDE = 12;

/** Bytes per gaussian for the `geometric` layer (rot4 + scale3 + opacity1 = 8 floats). */
const GEOMETRIC_STRIDE = 32;

/**
 * Bytes per gaussian for the `color` layer at the given SH band count.
 * Layout: `[dc0, dc1, dc2, sh_rest_0..N]` packed as f32.
 * @param shBands - The SH band count.
 * @returns The byte stride of the color layer.
 */
const colorStride = (shBands: SHBands): number => (3 + SH_REST_COUNTS[shBands]) * 4;

/** Default chunk size: 1M gaussians per chunk. */
const DEFAULT_CHUNK_SIZE = 1 << 20;

/**
 * Description of a single named field within a layer's per-gaussian record.
 *
 * `byteOffset` is the offset from the start of one gaussian's record;
 * `components` is the count of `type` elements that make up the field.
 */
type ChunkField = {
    readonly byteOffset: number;
    readonly components: number;
    readonly type: 'float32' | 'uint32';
};

/** Map from field name to its layout within a chunk's stride. */
type ChunkFieldMap = Readonly<Record<string, ChunkField>>;

/**
 * The byte stride and per-field map for a single layer of a source.
 *
 * Sources publish a `LayerLayout` per available layer in their metadata.
 * Callers pass a layout (along with a gaussian count) to a `ChunkDataPool`'s
 * `acquire` to receive a properly-sized `ChunkData`.
 */
type LayerLayout = {
    readonly stride: number;
    readonly fields: ChunkFieldMap;
};

const positionFields = (): ChunkFieldMap => ({
    position: { byteOffset: 0, components: 3, type: 'float32' }
});

const geometricFields = (): ChunkFieldMap => ({
    rotation: { byteOffset: 0, components: 4, type: 'float32' },
    scale: { byteOffset: 16, components: 3, type: 'float32' },
    opacity: { byteOffset: 28, components: 1, type: 'float32' }
});

const colorFields = (shBands: SHBands): ChunkFieldMap => {
    const map: Record<string, ChunkField> = {
        dc: { byteOffset: 0, components: 3, type: 'float32' }
    };
    if (shBands > 0) {
        map.shRest = {
            byteOffset: 12,
            components: SH_REST_COUNTS[shBands],
            type: 'float32'
        };
    }
    return map;
};

/** Descriptor for a single column in the `other` layer. */
type ExtraColumn = {
    readonly name: string;
    readonly type: 'float32' | 'uint32';
};

/**
 * Compute the byte stride and field map for an `other` layer given its extra columns.
 * Each column contributes 4 bytes (one f32 or u32) at its assigned offset.
 * @param extras - The extra columns, in storage order.
 * @returns The byte stride and field map for the `other` layer.
 */
const otherLayout = (
    extras: ReadonlyArray<ExtraColumn>
): { stride: number; fields: ChunkFieldMap } => {
    const fields: Record<string, ChunkField> = {};
    let offset = 0;
    for (const e of extras) {
        fields[e.name] = { byteOffset: offset, components: 1, type: e.type };
        offset += 4;
    }
    return { stride: offset, fields };
};

export {
    type ChunkLayer,
    type SHBands,
    type ChunkField,
    type ChunkFieldMap,
    type LayerLayout,
    type ExtraColumn,
    SH_REST_COUNTS,
    POSITION_STRIDE,
    GEOMETRIC_STRIDE,
    DEFAULT_CHUNK_SIZE,
    colorStride,
    positionFields,
    geometricFields,
    colorFields,
    otherLayout
};
