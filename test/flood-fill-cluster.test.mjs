/**
 * Tests for findClusterVoxelFlood and buildInvertedGrid.
 * Uses synthetic BlockMaskBuffer inputs to exercise the BFS connectivity
 * pipeline and catch indexing bugs.
 */

import { describe, it } from 'node:test';
import assert from 'node:assert';
import { BlockMaskBuffer } from '../src/lib/voxel/block-mask-buffer.js';
import {
    BLOCK_EMPTY,
    BLOCK_MIXED,
    BLOCK_SOLID,
    SparseVoxelGrid
} from '../src/lib/voxel/sparse-voxel-grid.js';
import {
    buildInvertedGrid,
    findClusterVoxelFlood
} from '../src/lib/voxel/filter-cluster.js';

const SOLID_LO = 0xFFFFFFFF >>> 0;
const SOLID_HI = 0xFFFFFFFF >>> 0;

// -- test helpers --

function blockIdx(bx, by, bz, nbx, nby) {
    return bx + by * nbx + bz * nbx * nby;
}

// Pick smallest grid block-dimensions that contain every coord in `coords`.
function dimsForCoords(coords) {
    let nbx = 1, nby = 1, nbz = 1;
    for (const c of coords) {
        if (c[0] + 1 > nbx) nbx = c[0] + 1;
        if (c[1] + 1 > nby) nby = c[1] + 1;
        if (c[2] + 1 > nbz) nbz = c[2] + 1;
    }
    return { nbx, nby, nbz };
}

function bufferFromSolidBlocks(coords, nbx, nby) {
    if (nbx === undefined || nby === undefined) {
        const d = dimsForCoords(coords);
        nbx = nbx ?? d.nbx;
        nby = nby ?? d.nby;
    }
    const buffer = new BlockMaskBuffer();
    for (const [bx, by, bz] of coords) {
        buffer.addBlock(blockIdx(bx, by, bz, nbx, nby), SOLID_LO, SOLID_HI);
    }
    return buffer;
}

function bufferFromMixedBlocks(entries, nbx, nby) {
    if (nbx === undefined || nby === undefined) {
        const d = dimsForCoords(entries.map(e => [e.bx, e.by, e.bz]));
        nbx = nbx ?? d.nbx;
        nby = nby ?? d.nby;
    }
    const buffer = new BlockMaskBuffer();
    for (const { bx, by, bz, lo, hi } of entries) {
        buffer.addBlock(blockIdx(bx, by, bz, nbx, nby), lo, hi);
    }
    return buffer;
}

/**
 * Block-level BFS from seed through face-adjacent ccSet members.
 * Returns array of ccSet blocks NOT reachable from the seed.
 */
function verifyCcSetConnectivity(ccSet, seedBlockIdx, nbx, nby, nbz) {
    if (!ccSet.has(seedBlockIdx)) return [...ccSet];
    const bStride = nbx * nby;
    const reachable = new Set([seedBlockIdx]);
    const queue = [seedBlockIdx];
    while (queue.length > 0) {
        const bi = queue.pop();
        const bx = bi % nbx;
        const byBz = (bi / nbx) | 0;
        const by = byBz % nby;
        const bz = (byBz / nby) | 0;
        for (const ni of [
            bx > 0 ? bi - 1 : -1,
            bx < nbx - 1 ? bi + 1 : -1,
            by > 0 ? bi - nbx : -1,
            by < nby - 1 ? bi + nbx : -1,
            bz > 0 ? bi - bStride : -1,
            bz < nbz - 1 ? bi + bStride : -1
        ]) {
            if (ni >= 0 && ccSet.has(ni) && !reachable.has(ni)) {
                reachable.add(ni);
                queue.push(ni);
            }
        }
    }
    return [...ccSet].filter(bi => !reachable.has(bi));
}

/**
 * Naive voxel-level BFS through the visited grid from the seed.
 * Returns { totalVisited, totalReachable }.
 */
function verifyVisitedVoxelConnectivity(visited, seedIx, seedIy, seedIz) {
    const { nx, ny, nz, nbx, nby, nbz } = visited;
    let totalVisited = 0;
    const visitedVoxels = new Set();
    const voxelKey = (ix, iy, iz) => ix + iy * nx + iz * nx * ny;

    const EVEN = 0x55555555 >>> 0;
    for (let w = 0; w < visited.types.length; w++) {
        const word = visited.types[w];
        if (word === 0) continue;
        let nonEmpty = ((word & EVEN) | ((word >>> 1) & EVEN)) >>> 0;
        const baseIdx = w * 16;
        while (nonEmpty) {
            const bp = 31 - Math.clz32(nonEmpty & -nonEmpty);
            const bi = baseIdx + (bp >>> 1);
            nonEmpty &= nonEmpty - 1;
            if (bi >= nbx * nby * nbz) continue;
            const bx = bi % nbx;
            const byBz = (bi / nbx) | 0;
            const by = byBz % nby;
            const bz = (byBz / nby) | 0;
            const bt = visited.getBlockType(bi);
            if (bt === BLOCK_SOLID) {
                for (let lz = 0; lz < 4; lz++)
                    for (let ly = 0; ly < 4; ly++)
                        for (let lx = 0; lx < 4; lx++) {
                            visitedVoxels.add(voxelKey((bx << 2) + lx, (by << 2) + ly, (bz << 2) + lz));
                            totalVisited++;
                        }
            } else if (bt === BLOCK_MIXED) {
                const s = visited.masks.slot(bi);
                if (visited.masks.keys[s] < 0) continue;
                const lo = visited.masks.lo[s];
                const hi = visited.masks.hi[s];
                for (let b = 0; b < 64; b++) {
                    if ((b < 32 ? (lo >>> b) & 1 : (hi >>> (b - 32)) & 1)) {
                        visitedVoxels.add(voxelKey((bx << 2) + (b & 3), (by << 2) + ((b >> 2) & 3), (bz << 2) + (b >> 4)));
                        totalVisited++;
                    }
                }
            }
        }
    }

    const reached = new Set([voxelKey(seedIx, seedIy, seedIz)]);
    const queue = [seedIx, seedIy, seedIz];
    let qi = 0;
    while (qi < queue.length) {
        const ix = queue[qi++], iy = queue[qi++], iz = queue[qi++];
        for (const [dx, dy, dz] of [[-1,0,0],[1,0,0],[0,-1,0],[0,1,0],[0,0,-1],[0,0,1]]) {
            const nx2 = ix + dx, ny2 = iy + dy, nz2 = iz + dz;
            if (nx2 < 0 || nx2 >= nx || ny2 < 0 || ny2 >= ny || nz2 < 0 || nz2 >= nz) continue;
            const k = voxelKey(nx2, ny2, nz2);
            if (visitedVoxels.has(k) && !reached.has(k)) {
                reached.add(k);
                queue.push(nx2, ny2, nz2);
            }
        }
    }
    return { totalVisited, totalReachable: reached.size };
}

/**
 * Cross-check a BlockMaskBuffer against an inverted grid.
 * Returns array of error strings (empty = OK).
 */
function verifyInvertedGrid(buffer, grid, nbx, nby, nbz) {
    const errors = [];
    const bStride = nbx * nby;
    const known = new Set();
    const NAMES = ['EMPTY', 'SOLID', 'MIXED'];
    // Decode a linear blockIdx back to (bx, by, bz) for diagnostic messages.
    const decode = (bi) => {
        const bx = bi % nbx;
        const by = ((bi / nbx) | 0) % nby;
        const bz = (bi / bStride) | 0;
        return [bx, by, bz];
    };
    for (const bi of buffer.getSolidBlocks()) {
        const [bx, by, bz] = decode(bi);
        known.add(bi);
        if (grid.getBlockType(bi) !== BLOCK_EMPTY) errors.push(`solid(${bx},${by},${bz}) expected EMPTY got ${NAMES[grid.getBlockType(bi)]}`);
    }
    const mixed = buffer.getMixedBlocks();
    for (let i = 0; i < mixed.blockIdx.length; i++) {
        const bi = mixed.blockIdx[i];
        const [bx, by, bz] = decode(bi);
        known.add(bi);
        if (grid.getBlockType(bi) !== BLOCK_MIXED) {
            errors.push(`mixed(${bx},${by},${bz}) expected MIXED got ${NAMES[grid.getBlockType(bi)]}`);
        } else {
            const s = grid.masks.slot(bi);
            const elo = (~mixed.masks[i * 2]) >>> 0, ehi = (~mixed.masks[i * 2 + 1]) >>> 0;
            if (grid.masks.lo[s] !== elo || grid.masks.hi[s] !== ehi) errors.push(`mixed(${bx},${by},${bz}) mask mismatch`);
        }
    }
    for (let bi = 0; bi < nbx * nby * nbz; bi++) {
        if (!known.has(bi) && grid.getBlockType(bi) !== BLOCK_SOLID) errors.push(`block ${bi} expected SOLID`);
    }
    return errors;
}

// ============================================================================
// buildInvertedGrid verification
// ============================================================================

describe('buildInvertedGrid', () => {
    it('should map solid buffer blocks to BLOCK_EMPTY', () => {
        const nx = 12, ny = 12, nz = 4;
        const nbx = nx >> 2, nby = ny >> 2;
        const buffer = bufferFromSolidBlocks([[0, 0, 0], [1, 0, 0], [0, 1, 0]], nbx, nby);
        const grid = buildInvertedGrid(buffer, nx, ny, nz);

        assert.strictEqual(grid.getBlockType(blockIdx(0, 0, 0, nbx, nby)), BLOCK_EMPTY);
        assert.strictEqual(grid.getBlockType(blockIdx(1, 0, 0, nbx, nby)), BLOCK_EMPTY);
        assert.strictEqual(grid.getBlockType(blockIdx(0, 1, 0, nbx, nby)), BLOCK_EMPTY);

        assert.strictEqual(grid.getBlockType(blockIdx(2, 0, 0, nbx, nby)), BLOCK_SOLID);
        assert.strictEqual(grid.getBlockType(blockIdx(0, 2, 0, nbx, nby)), BLOCK_SOLID);
    });

    it('should map mixed buffer blocks to BLOCK_MIXED with inverted masks', () => {
        const lo = 0x000000FF >>> 0;
        const hi = 0x00000000 >>> 0;
        const nx = 12, ny = 12, nz = 4;
        const nbx = nx >> 2, nby = ny >> 2;
        const buffer = bufferFromMixedBlocks([{ bx: 1, by: 1, bz: 0, lo, hi }], nbx, nby);
        const grid = buildInvertedGrid(buffer, nx, ny, nz);
        const bi = blockIdx(1, 1, 0, nbx, nby);

        assert.strictEqual(grid.getBlockType(bi), BLOCK_MIXED);
        const s = grid.masks.slot(bi);
        assert.strictEqual(grid.masks.lo[s], (~lo) >>> 0);
        assert.strictEqual(grid.masks.hi[s], (~hi) >>> 0);
    });

    it('should pass verifyInvertedGrid for a mixed buffer', () => {
        const nx = 16, ny = 12, nz = 4;
        const nbx = nx >> 2, nby = ny >> 2, nbz = nz >> 2;
        const buffer = new BlockMaskBuffer();
        buffer.addBlock(blockIdx(0, 0, 0, nbx, nby), SOLID_LO, SOLID_HI);
        buffer.addBlock(blockIdx(1, 0, 0, nbx, nby), 0x0F0F0F0F >>> 0, 0xF0F0F0F0 >>> 0);
        buffer.addBlock(blockIdx(2, 1, 0, nbx, nby), SOLID_LO, SOLID_HI);
        const grid = buildInvertedGrid(buffer, nx, ny, nz);

        const errors = verifyInvertedGrid(buffer, grid, nbx, nby, nbz);
        assert.deepStrictEqual(errors, []);
    });
});

// ============================================================================
// verifyCcSetConnectivity
// ============================================================================

describe('verifyCcSetConnectivity', () => {
    it('should return empty for a single block', () => {
        const ccSet = new Set([0]);
        assert.deepStrictEqual(verifyCcSetConnectivity(ccSet, 0, 4, 4, 4), []);
    });

    it('should return empty for a face-connected line of blocks', () => {
        const nbx = 8, nby = 4, nbz = 4;
        const set = new Set([
            blockIdx(0, 0, 0, nbx, nby),
            blockIdx(1, 0, 0, nbx, nby),
            blockIdx(2, 0, 0, nbx, nby)
        ]);
        const seed = blockIdx(0, 0, 0, nbx, nby);
        assert.deepStrictEqual(verifyCcSetConnectivity(set, seed, nbx, nby, nbz), []);
    });

    it('should detect disconnected blocks', () => {
        const nbx = 8, nby = 4, nbz = 4;
        const seedBi = blockIdx(0, 0, 0, nbx, nby);
        const disconnected = blockIdx(5, 3, 3, nbx, nby);
        const set = new Set([seedBi, blockIdx(1, 0, 0, nbx, nby), disconnected]);
        const result = verifyCcSetConnectivity(set, seedBi, nbx, nby, nbz);
        assert.strictEqual(result.length, 1);
        assert.strictEqual(result[0], disconnected);
    });
});

// ============================================================================
// findClusterVoxelFlood -- deterministic tests
// ============================================================================

describe('findClusterVoxelFlood', () => {

    it('should find a single solid block', () => {
        const nx = 4, ny = 4, nz = 4;
        const nbx = nx >> 2, nby = ny >> 2;
        const buffer = bufferFromSolidBlocks([[0, 0, 0]], nbx, nby);
        const result = findClusterVoxelFlood(buffer, nx, ny, nz, 1, 1, 1);
        assert.ok(result);
        assert.strictEqual(result.ccSet.size, 1);
        assert.ok(result.ccSet.has(0));
    });

    it('two solid cubes separated by gap -- should only reach seeded cluster', () => {
        const coordsA = [];
        const coordsB = [];
        for (let bz = 0; bz < 2; bz++) {
            for (let by = 0; by < 2; by++) {
                for (let bx = 0; bx < 2; bx++) {
                    coordsA.push([bx, by, bz]);
                    coordsB.push([bx + 4, by, bz]);
                }
            }
        }
        const nx = 24, ny = 8, nz = 8;
        const nbx = nx >> 2, nby = ny >> 2;
        const buffer = bufferFromSolidBlocks([...coordsA, ...coordsB], nbx, nby);

        const result = findClusterVoxelFlood(buffer, nx, ny, nz, 1, 1, 1);
        assert.ok(result);
        assert.strictEqual(result.ccSet.size, 8);

        for (const [bx, by, bz] of coordsA) {
            assert.ok(result.ccSet.has(blockIdx(bx, by, bz, nbx, nby)),
                `cluster A block (${bx},${by},${bz}) should be in ccSet`);
        }
        for (const [bx, by, bz] of coordsB) {
            assert.ok(!result.ccSet.has(blockIdx(bx, by, bz, nbx, nby)),
                `cluster B block (${bx},${by},${bz}) should NOT be in ccSet`);
        }

        const unreachable = verifyCcSetConnectivity(result.ccSet,
            blockIdx(0, 0, 0, nbx, nby), nbx, nby, nz >> 2);
        assert.strictEqual(unreachable.length, 0, 'ccSet should be fully connected');
    });

    it('two mixed clusters separated by gap -- should only reach seeded cluster', () => {
        const lo = 0x0000FFFF >>> 0;
        const hi = 0x0000FFFF >>> 0;
        const entries = [];
        for (let bz = 0; bz < 2; bz++) {
            for (let by = 0; by < 2; by++) {
                for (let bx = 0; bx < 2; bx++) {
                    entries.push({ bx, by, bz, lo, hi });
                    entries.push({ bx: bx + 4, by, bz, lo, hi });
                }
            }
        }
        const nx = 24, ny = 8, nz = 8;
        const nbx = nx >> 2, nby = ny >> 2;
        const buffer = bufferFromMixedBlocks(entries, nbx, nby);

        const result = findClusterVoxelFlood(buffer, nx, ny, nz, 1, 1, 1);
        assert.ok(result);

        for (let bz = 0; bz < 2; bz++) {
            for (let by = 0; by < 2; by++) {
                for (let bx = 4; bx < 6; bx++) {
                    assert.ok(!result.ccSet.has(blockIdx(bx, by, bz, nbx, nby)),
                        `cluster B block (${bx},${by},${bz}) should NOT be in ccSet`);
                }
            }
        }

        const seedBi = blockIdx(0, 0, 0, nbx, nby);
        const unreachable = verifyCcSetConnectivity(result.ccSet, seedBi, nbx, nby, nz >> 2);
        assert.strictEqual(unreachable.length, 0, 'ccSet should be fully connected');
    });

    it('L-shaped cluster -- all blocks should be in ccSet', () => {
        const coords = [];
        for (let bx = 0; bx <= 3; bx++) coords.push([bx, 0, 0]);
        for (let by = 1; by <= 3; by++) coords.push([0, by, 0]);
        const nx = 16, ny = 16, nz = 4;
        const nbx = nx >> 2, nby = ny >> 2;
        const buffer = bufferFromSolidBlocks(coords, nbx, nby);

        const result = findClusterVoxelFlood(buffer, nx, ny, nz, 1, 1, 1);
        assert.ok(result);
        assert.strictEqual(result.ccSet.size, coords.length);

        for (const [bx, by, bz] of coords) {
            assert.ok(result.ccSet.has(blockIdx(bx, by, bz, nbx, nby)),
                `block (${bx},${by},${bz}) should be in ccSet`);
        }
    });

    it('two cubes connected by a single solid bridge block', () => {
        const coordsA = [];
        const coordsB = [];
        for (let bz = 0; bz < 2; bz++) {
            for (let by = 0; by < 2; by++) {
                for (let bx = 0; bx < 2; bx++) {
                    coordsA.push([bx, by, bz]);
                    coordsB.push([bx + 3, by, bz]);
                }
            }
        }
        const bridge = [[2, 0, 0]];
        const nx = 20, ny = 8, nz = 8;
        const nbx = nx >> 2, nby = ny >> 2;
        const buffer = bufferFromSolidBlocks([...coordsA, ...bridge, ...coordsB], nbx, nby);

        const result = findClusterVoxelFlood(buffer, nx, ny, nz, 1, 1, 1);
        assert.ok(result);
        const allCoords = [...coordsA, ...bridge, ...coordsB];
        assert.strictEqual(result.ccSet.size, allCoords.length);

        for (const [bx, by, bz] of allCoords) {
            assert.ok(result.ccSet.has(blockIdx(bx, by, bz, nbx, nby)),
                `block (${bx},${by},${bz}) should be in ccSet`);
        }
    });

    it('two solid cubes connected by a mixed bridge with connected path', () => {
        const coordsA = [[0, 0, 0], [1, 0, 0]];
        const coordsB = [[3, 0, 0], [4, 0, 0]];
        const bridgeLo = 0x0000000F >>> 0;
        const bridgeHi = 0x00000000 >>> 0;

        const nx = 20, ny = 4, nz = 4;
        const nbx = nx >> 2, nby = ny >> 2;

        const buffer = new BlockMaskBuffer();
        for (const [bx, by, bz] of [...coordsA, ...coordsB]) {
            buffer.addBlock(blockIdx(bx, by, bz, nbx, nby), SOLID_LO, SOLID_HI);
        }
        buffer.addBlock(blockIdx(2, 0, 0, nbx, nby), bridgeLo, bridgeHi);

        const result = findClusterVoxelFlood(buffer, nx, ny, nz, 1, 1, 1);
        assert.ok(result);

        const allBlocks = [...coordsA, [2, 0, 0], ...coordsB];
        assert.strictEqual(result.ccSet.size, allBlocks.length);
        for (const [bx, by, bz] of allBlocks) {
            assert.ok(result.ccSet.has(blockIdx(bx, by, bz, nbx, nby)),
                `block (${bx},${by},${bz}) should be in ccSet`);
        }
    });

    it('mixed bridge with NO face connectivity should NOT connect cubes', () => {
        const interiorLo = (0x22222222 | 0x44444444) >>> 0;
        const interiorHi = (0x22222222 | 0x44444444) >>> 0;

        const nx = 20, ny = 4, nz = 4;
        const nbx = nx >> 2, nby = ny >> 2;

        const buffer = new BlockMaskBuffer();
        buffer.addBlock(blockIdx(0, 0, 0, nbx, nby), SOLID_LO, SOLID_HI);
        buffer.addBlock(blockIdx(1, 0, 0, nbx, nby), SOLID_LO, SOLID_HI);
        buffer.addBlock(blockIdx(2, 0, 0, nbx, nby), interiorLo, interiorHi);
        buffer.addBlock(blockIdx(3, 0, 0, nbx, nby), SOLID_LO, SOLID_HI);
        buffer.addBlock(blockIdx(4, 0, 0, nbx, nby), SOLID_LO, SOLID_HI);

        const result = findClusterVoxelFlood(buffer, nx, ny, nz, 1, 1, 1);
        assert.ok(result);

        assert.ok(result.ccSet.has(blockIdx(0, 0, 0, nbx, nby)));
        assert.ok(result.ccSet.has(blockIdx(1, 0, 0, nbx, nby)));
        assert.ok(!result.ccSet.has(blockIdx(3, 0, 0, nbx, nby)),
            'cluster B should not be reached through interior-only bridge');
        assert.ok(!result.ccSet.has(blockIdx(4, 0, 0, nbx, nby)),
            'cluster B should not be reached through interior-only bridge');
    });

    it('seed in unoccupied voxel should find nearest occupied', () => {
        const nx = 16, ny = 16, nz = 16;
        const nbx = nx >> 2, nby = ny >> 2;
        const buffer = bufferFromSolidBlocks([[2, 2, 2]], nbx, nby);

        const result = findClusterVoxelFlood(buffer, nx, ny, nz, 0, 0, 0);
        assert.ok(result);
        assert.strictEqual(result.ccSet.size, 1);
        assert.ok(result.ccSet.has(blockIdx(2, 2, 2, nbx, nby)));
    });
});

// ============================================================================
// Grid dimension sweep
// ============================================================================

describe('findClusterVoxelFlood grid dimension sweep', () => {
    const dimensions = [8, 12, 16, 20, 24, 28, 32, 36, 48, 52, 64, 128];

    for (const nx of dimensions) {
        for (const ny of [8, 20, 36]) {
            for (const nz of [4, 12, 28]) {
                const nbx = nx >> 2;
                const nby = ny >> 2;
                const nbz = nz >> 2;
                if (nbx < 6 || nby < 2 || nbz < 2) continue;

                it(`two clusters in ${nx}x${ny}x${nz} grid (${nbx}x${nby}x${nbz} blocks)`, () => {
                    const coordsA = [[0, 0, 0], [1, 0, 0]];
                    const coordsB = [[nbx - 2, nby - 1, nbz - 1], [nbx - 1, nby - 1, nbz - 1]];
                    const buffer = bufferFromSolidBlocks([...coordsA, ...coordsB], nbx, nby);

                    const result = findClusterVoxelFlood(buffer, nx, ny, nz, 1, 1, 1);
                    assert.ok(result, 'should find a cluster');
                    assert.strictEqual(result.ccSet.size, 2,
                        `should only have 2 blocks from cluster A, got ${result.ccSet.size}`);

                    for (const [bx, by, bz] of coordsA) {
                        assert.ok(result.ccSet.has(blockIdx(bx, by, bz, nbx, nby)),
                            `cluster A block (${bx},${by},${bz}) should be in ccSet`);
                    }
                    for (const [bx, by, bz] of coordsB) {
                        assert.ok(!result.ccSet.has(blockIdx(bx, by, bz, nbx, nby)),
                            `cluster B block (${bx},${by},${bz}) should NOT be in ccSet`);
                    }

                    const seedBi = blockIdx(0, 0, 0, nbx, nby);
                    const unreachable = verifyCcSetConnectivity(result.ccSet, seedBi, nbx, nby, nbz);
                    assert.strictEqual(unreachable.length, 0, 'ccSet should be fully connected');
                });
            }
        }
    }
});

// ============================================================================
// Randomized layouts
// ============================================================================

describe('findClusterVoxelFlood randomized layouts', () => {
    function seededRandom(seed) {
        let s = seed;
        return () => {
            s = (Math.imul(s, 1103515245) + 12345) & 0x7fffffff;
            return s / 0x7fffffff;
        };
    }

    function generateDisconnectedClusters(rng, nbx, nby, nbz, clusterSize, separation) {
        const coordsA = [];
        const coordsB = [];

        const ax = Math.floor(rng() * Math.max(1, nbx / 4));
        const ay = Math.floor(rng() * Math.max(1, nby / 4));
        const az = Math.floor(rng() * Math.max(1, nbz / 4));
        for (let i = 0; i < clusterSize; i++) {
            const base = coordsA.length > 0 ? coordsA[Math.floor(rng() * coordsA.length)] : [ax, ay, az];
            const dir = Math.floor(rng() * 6);
            const dx = dir === 0 ? 1 : dir === 1 ? -1 : 0;
            const dy = dir === 2 ? 1 : dir === 3 ? -1 : 0;
            const dz = dir === 4 ? 1 : dir === 5 ? -1 : 0;
            const nx = Math.max(0, Math.min(nbx - 1, base[0] + dx));
            const ny = Math.max(0, Math.min(nby - 1, base[1] + dy));
            const nz = Math.max(0, Math.min(nbz - 1, base[2] + dz));
            if (!coordsA.some(c => c[0] === nx && c[1] === ny && c[2] === nz)) {
                coordsA.push([nx, ny, nz]);
            }
        }

        const offset = separation + Math.ceil(nbx / 2);
        for (const [cx, cy, cz] of coordsA) {
            const bx = cx + offset;
            if (bx < nbx && cy < nby && cz < nbz) {
                coordsB.push([bx, cy, cz]);
            }
        }

        return { coordsA, coordsB };
    }

    for (let trial = 0; trial < 20; trial++) {
        const rng = seededRandom(42 + trial);
        const nbx = 8 + Math.floor(rng() * 24);
        const nby = 4 + Math.floor(rng() * 12);
        const nbz = 2 + Math.floor(rng() * 8);
        const nx = nbx * 4;
        const ny = nby * 4;
        const nz = nbz * 4;

        it(`random trial ${trial}: ${nbx}x${nby}x${nbz} blocks`, () => {
            const { coordsA, coordsB } = generateDisconnectedClusters(rng, nbx, nby, nbz, 10, 3);
            if (coordsA.length === 0 || coordsB.length === 0) return;

            const buffer = bufferFromSolidBlocks([...coordsA, ...coordsB], nbx, nby);
            const seedCoord = coordsA[0];
            const seedVx = seedCoord[0] * 4 + 1;
            const seedVy = seedCoord[1] * 4 + 1;
            const seedVz = seedCoord[2] * 4 + 1;

            const result = findClusterVoxelFlood(buffer, nx, ny, nz, seedVx, seedVy, seedVz);
            assert.ok(result, 'should find a cluster');

            for (const [bx, by, bz] of coordsB) {
                const bi = blockIdx(bx, by, bz, nbx, nby);
                assert.ok(!result.ccSet.has(bi),
                    `cluster B block (${bx},${by},${bz}) should NOT be in ccSet`);
            }

            const seedBi = blockIdx(seedCoord[0], seedCoord[1], seedCoord[2], nbx, nby);
            const unreachable = verifyCcSetConnectivity(result.ccSet, seedBi, nbx, nby, nbz);
            assert.strictEqual(unreachable.length, 0,
                `ccSet should be fully connected, but ${unreachable.length} blocks are disconnected`);
        });
    }
});

// ============================================================================
// Randomized layouts with mixed blocks
// ============================================================================

describe('findClusterVoxelFlood randomized mixed blocks', () => {
    function seededRandom(seed) {
        let s = seed;
        return () => {
            s = (Math.imul(s, 1103515245) + 12345) & 0x7fffffff;
            return s / 0x7fffffff;
        };
    }

    for (let trial = 0; trial < 10; trial++) {
        const rng = seededRandom(100 + trial);
        const nbx = 6 + Math.floor(rng() * 10);
        const nby = 4 + Math.floor(rng() * 6);
        const nbz = 2 + Math.floor(rng() * 4);
        const nx = nbx * 4;
        const ny = nby * 4;
        const nz = nbz * 4;

        it(`random mixed trial ${trial}: ${nbx}x${nby}x${nbz} blocks`, () => {
            const buffer = new BlockMaskBuffer();

            const clusterABlocks = [];
            for (let by = 0; by < 2; by++) {
                for (let bx = 0; bx < 2; bx++) {
                    const lo = (rng() > 0.5 ? SOLID_LO : (0xFFFF0000 | Math.floor(rng() * 0xFFFF)) >>> 0);
                    const hi = (rng() > 0.5 ? SOLID_HI : (0x0000FFFF | Math.floor(rng() * 0xFFFF0000)) >>> 0);
                    buffer.addBlock(blockIdx(bx, by, 0, nbx, nby), lo, hi);
                    clusterABlocks.push([bx, by, 0]);
                }
            }

            const farX = nbx - 2;
            const farY = nby - 2;
            const farZ = nbz - 1;
            if (farX > 3 && farY > 3) {
                for (let by = 0; by < 2; by++) {
                    for (let bx = 0; bx < 2; bx++) {
                        buffer.addBlock(blockIdx(farX + bx, farY + by, farZ, nbx, nby), SOLID_LO, SOLID_HI);
                    }
                }
            }

            const result = findClusterVoxelFlood(buffer, nx, ny, nz, 1, 1, 1);
            assert.ok(result, 'should find a cluster');

            if (farX > 3 && farY > 3) {
                for (let by = 0; by < 2; by++) {
                    for (let bx = 0; bx < 2; bx++) {
                        const bi = blockIdx(farX + bx, farY + by, farZ, nbx, nby);
                        assert.ok(!result.ccSet.has(bi),
                            `cluster B block (${farX + bx},${farY + by},${farZ}) should NOT be in ccSet`);
                    }
                }
            }

            const seedBi = blockIdx(0, 0, 0, nbx, nby);
            const unreachable = verifyCcSetConnectivity(result.ccSet, seedBi, nbx, nby, nbz);
            assert.strictEqual(unreachable.length, 0,
                `ccSet should be fully connected, but ${unreachable.length} blocks are disconnected`);
        });
    }
});

// ============================================================================
// Inverted grid + BFS consistency
// ============================================================================

describe('buildInvertedGrid + BFS consistency', () => {
    it('inverted grid voxels match original buffer occupancy', () => {
        const nx = 12, ny = 8, nz = 4;
        const nbx = nx >> 2, nby = ny >> 2;
        const buffer = new BlockMaskBuffer();
        buffer.addBlock(blockIdx(0, 0, 0, nbx, nby), SOLID_LO, SOLID_HI);
        buffer.addBlock(blockIdx(1, 0, 0, nbx, nby), 0xAAAAAAAA >>> 0, 0x55555555 >>> 0);
        buffer.addBlock(blockIdx(0, 1, 0, nbx, nby), 0x0F0F0F0F >>> 0, 0xF0F0F0F0 >>> 0);

        const grid = buildInvertedGrid(buffer, nx, ny, nz);
        const original = SparseVoxelGrid.fromBuffer(buffer, nx, ny, nz);

        for (let iz = 0; iz < nz; iz++) {
            for (let iy = 0; iy < ny; iy++) {
                for (let ix = 0; ix < nx; ix++) {
                    const origOccupied = original.getVoxel(ix, iy, iz);
                    const invertedBlocked = grid.getVoxel(ix, iy, iz);
                    assert.strictEqual(invertedBlocked, origOccupied ? 0 : 1,
                        `voxel (${ix},${iy},${iz}): original=${origOccupied}, inverted blocked=${invertedBlocked}`);
                }
            }
        }
    });

    it('BFS visits exactly the connected occupied voxels', () => {
        const nx = 16, ny = 4, nz = 4;
        const nbx = nx >> 2, nby = ny >> 2;
        const buffer = bufferFromSolidBlocks([[0, 0, 0], [3, 0, 0]], nbx, nby);

        const result = findClusterVoxelFlood(buffer, nx, ny, nz, 1, 1, 1);
        assert.ok(result);
        assert.strictEqual(result.ccSet.size, 1, 'should only reach 1 block');
        assert.ok(result.ccSet.has(0), 'should reach block at (0,0,0)');
    });
});

// ============================================================================
// Voxel-level connectivity verification
// ============================================================================

describe('verifyVisitedVoxelConnectivity', () => {
    it('single solid block: all 64 voxels reachable', () => {
        const nx = 4, ny = 4, nz = 4;
        const nbx = nx >> 2, nby = ny >> 2;
        const buffer = bufferFromSolidBlocks([[0, 0, 0]], nbx, nby);
        const result = findClusterVoxelFlood(buffer, nx, ny, nz, 1, 1, 1);
        assert.ok(result);
        const check = verifyVisitedVoxelConnectivity(result.visited, result.resolvedSeed.ix, result.resolvedSeed.iy, result.resolvedSeed.iz);
        assert.strictEqual(check.totalVisited, 64);
        assert.strictEqual(check.totalReachable, check.totalVisited);
    });

    it('two separated solid blocks: visited voxels are connected', () => {
        const nx = 16, ny = 4, nz = 4;
        const nbx = nx >> 2, nby = ny >> 2;
        const buffer = bufferFromSolidBlocks([[0, 0, 0], [3, 0, 0]], nbx, nby);
        const result = findClusterVoxelFlood(buffer, nx, ny, nz, 1, 1, 1);
        assert.ok(result);
        const check = verifyVisitedVoxelConnectivity(result.visited, result.resolvedSeed.ix, result.resolvedSeed.iy, result.resolvedSeed.iz);
        assert.strictEqual(check.totalVisited, 64, 'only one block should be visited');
        assert.strictEqual(check.totalReachable, check.totalVisited, 'all visited voxels should be reachable');
    });

    it('two adjacent solid blocks: all 128 voxels connected', () => {
        const nx = 8, ny = 4, nz = 4;
        const nbx = nx >> 2, nby = ny >> 2;
        const buffer = bufferFromSolidBlocks([[0, 0, 0], [1, 0, 0]], nbx, nby);
        const result = findClusterVoxelFlood(buffer, nx, ny, nz, 1, 1, 1);
        assert.ok(result);
        const check = verifyVisitedVoxelConnectivity(result.visited, result.resolvedSeed.ix, result.resolvedSeed.iy, result.resolvedSeed.iz);
        assert.strictEqual(check.totalVisited, 128);
        assert.strictEqual(check.totalReachable, check.totalVisited);
    });

    it('mixed block with disconnected internal faces: BFS visits only connected voxels', () => {
        const nx = 12, ny = 4, nz = 4;
        const nbx = nx >> 2, nby = ny >> 2;
        const buffer = new BlockMaskBuffer();
        buffer.addBlock(blockIdx(0, 0, 0, nbx, nby), SOLID_LO, SOLID_HI);
        buffer.addBlock(blockIdx(1, 0, 0, nbx, nby), 0x11111111 >>> 0, 0x11111111 >>> 0);
        buffer.addBlock(blockIdx(2, 0, 0, nbx, nby), SOLID_LO, SOLID_HI);

        const result = findClusterVoxelFlood(buffer, nx, ny, nz, 1, 1, 1);
        assert.ok(result);

        const check = verifyVisitedVoxelConnectivity(result.visited, result.resolvedSeed.ix, result.resolvedSeed.iy, result.resolvedSeed.iz);
        assert.strictEqual(check.totalReachable, check.totalVisited,
            `all visited voxels should be reachable, but ${check.totalVisited - check.totalReachable} are not`);

        assert.ok(!result.ccSet.has(blockIdx(2, 0, 0, nbx, nby)),
            'block (2,0,0) should not be reachable through lx=0-only bridge');
    });

    it('all visited voxels connected in dimension sweep', () => {
        const dims = [8, 20, 32, 52];
        for (const nx of dims) {
            for (const ny of [8, 20]) {
                const nz = 8;
                const nbx = nx >> 2, nby = ny >> 2;
                if (nbx < 4) continue;

                const buffer = bufferFromSolidBlocks([
                    [0, 0, 0], [1, 0, 0],
                    [nbx - 1, nby - 1, 0]
                ], nbx, nby);
                const result = findClusterVoxelFlood(buffer, nx, ny, nz, 1, 1, 1);
                assert.ok(result);

                const check = verifyVisitedVoxelConnectivity(result.visited,
                    result.resolvedSeed.ix, result.resolvedSeed.iy, result.resolvedSeed.iz);
                assert.strictEqual(check.totalReachable, check.totalVisited,
                    `${nx}x${ny}x${nz}: ${check.totalVisited - check.totalReachable} orphaned voxels`);
            }
        }
    });
});
