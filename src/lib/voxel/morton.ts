/** All 64 bits set (as unsigned 32-bit) */
const SOLID_MASK = 0xFFFFFFFF >>> 0;

/**
 * Encode block coordinates to Morton code (17 bits per axis = 51 bits total).
 * Supports up to 131,072 blocks per axis.
 *
 * @param x - Block X coordinate
 * @param y - Block Y coordinate
 * @param z - Block Z coordinate
 * @returns Morton code with interleaved bits: ...z2y2x2 z1y1x1 z0y0x0
 */
function xyzToMorton(x: number, y: number, z: number): number {
    let result = 0;
    let shift = 1;
    for (let i = 0; i < 17; i++) {
        if (x & 1) result += shift;
        if (y & 1) result += shift * 2;
        if (z & 1) result += shift * 4;
        x >>>= 1;
        y >>>= 1;
        z >>>= 1;
        shift *= 8;
    }
    return result;
}

/**
 * Decode Morton code to block coordinates.
 *
 * @param m - Morton code
 * @returns Tuple of [x, y, z] block coordinates
 */
function mortonToXYZ(m: number): [number, number, number] {
    let x = 0, y = 0, z = 0;
    let bit = 1;
    while (m > 0) {
        const triplet = m % 8;
        if (triplet & 1) x |= bit;
        if (triplet & 2) y |= bit;
        if (triplet & 4) z |= bit;
        bit <<= 1;
        m = Math.trunc(m / 8);
    }
    return [x, y, z];
}

/**
 * Count the number of set bits in a 32-bit integer.
 *
 * @param n - 32-bit integer
 * @returns Number of bits set to 1
 */
function popcount(n: number): number {
    n >>>= 0;
    n -= ((n >>> 1) & 0x55555555);
    n = (n & 0x33333333) + ((n >>> 2) & 0x33333333);
    return (((n + (n >>> 4)) & 0x0F0F0F0F) * 0x01010101) >>> 24;
}

/**
 * Check if a voxel mask represents a solid block (all 64 bits set).
 *
 * @param lo - Lower 32 bits of mask
 * @param hi - Upper 32 bits of mask
 * @returns True if all 64 voxels are solid
 */
function isSolid(lo: number, hi: number): boolean {
    return (lo >>> 0) === SOLID_MASK && (hi >>> 0) === SOLID_MASK;
}

/**
 * Check if a voxel mask represents an empty block (no bits set).
 *
 * @param lo - Lower 32 bits of mask
 * @param hi - Upper 32 bits of mask
 * @returns True if all 64 voxels are empty
 */
function isEmpty(lo: number, hi: number): boolean {
    return lo === 0 && hi === 0;
}

export {
    SOLID_MASK,
    xyzToMorton,
    mortonToXYZ,
    popcount,
    isSolid,
    isEmpty
};
