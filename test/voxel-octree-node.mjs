/**
 * Node.js test for voxel octree visualization.
 * 
 * This script generates PLY files for each octree level to visualize the voxelization:
 * - Solid blocks: Red (255, 0, 0)
 * - Mixed blocks: Shades of gray based on fill percentage
 * 
 * Run with: node test/voxel-octree-node.mjs <path-to-file> [voxel-resolution] [opacity-cutoff]
 * 
 * Supports any input format: PLY, SPLAT, KSPLAT, SOG, SPZ, LCC, etc.
 */

import { writeFile as fsWriteFile, mkdir, open, stat } from 'fs/promises';
import { resolve, basename, dirname, extname } from 'path';

// Import the compiled library
import { 
    // File reading
    readFile,
    getInputFormat,
    ReadStream,
    BufferedReadStream,
    // Voxelization pipeline
    computeGaussianExtents,
    GaussianBVH,
    GpuVoxelization,
    BlockAccumulator,
    buildSparseOctree,
    alignGridBounds,
    xyzToMorton,
    mortonToXYZ,
    // PLY output
    writePly,
    Column,
    DataTable,
    MemoryFileSystem
} from '../dist/index.mjs';

// ============================================================================
// Node.js File System (matches CLI implementation)
// ============================================================================

class NodeReadStream extends ReadStream {
    constructor(fileHandle, start, end, progress, totalSize) {
        super(end - start);
        this.fileHandle = fileHandle;
        this.position = start;
        this.end = end;
        this.closed = false;
        this.progress = progress;
        this.totalSize = totalSize;
    }

    async pull(target) {
        if (this.closed) return 0;

        const remaining = this.end - this.position;
        if (remaining <= 0) return 0;

        const bytesToRead = Math.min(target.length, remaining);
        const { bytesRead } = await this.fileHandle.read(target, 0, bytesToRead, this.position);

        this.position += bytesRead;
        this.bytesRead += bytesRead;

        if (this.progress) {
            this.progress(this.bytesRead, this.totalSize);
        }

        return bytesRead;
    }

    close() {
        this.closed = true;
    }
}

class NodeReadSource {
    constructor(fileHandle, size, progress) {
        this.fileHandle = fileHandle;
        this.size = size;
        this.seekable = true;
        this.closed = false;
        this.progress = progress;
    }

    read(start = 0, end = this.size) {
        if (this.closed) {
            throw new Error('Source has been closed');
        }

        const clampedStart = Math.max(0, Math.min(start, this.size));
        const clampedEnd = Math.max(clampedStart, Math.min(end, this.size));

        const raw = new NodeReadStream(
            this.fileHandle,
            clampedStart,
            clampedEnd,
            this.progress,
            this.size
        );
        return new BufferedReadStream(raw, 4 * 1024 * 1024);
    }

    close() {
        this.closed = true;
        this.fileHandle.close();
    }
}

class NodeReadFileSystem {
    async createSource(filename, progress) {
        const fileStats = await stat(filename);
        const fileHandle = await open(filename, 'r');

        if (progress) {
            progress(0, fileStats.size);
        }

        return new NodeReadSource(fileHandle, fileStats.size, progress);
    }
}

// Import device creation
import { WebgpuGraphicsDevice } from 'playcanvas';
import { create, globals } from 'webgpu';

// ============================================================================
// Node.js WebGPU Setup
// ============================================================================

const initializeGlobals = () => {
    Object.assign(globalThis, globals);

    globalThis.window = {
        navigator: { userAgent: 'node.js' }
    };

    globalThis.document = {
        createElement: (type) => {
            if (type === 'canvas') {
                return {
                    getContext: () => null,
                    getBoundingClientRect: () => ({
                        left: 0, top: 0, width: 300, height: 150, right: 300, bottom: 150
                    }),
                    width: 300,
                    height: 150
                };
            }
        }
    };
};

initializeGlobals();

const createDevice = async () => {
    window.navigator.gpu = create([]);
    
    const canvas = document.createElement('canvas');
    canvas.width = 1024;
    canvas.height = 512;

    const graphicsDevice = new WebgpuGraphicsDevice(canvas, {
        antialias: false,
        depth: false,
        stencil: false
    });

    await graphicsDevice.createDevice();
    return graphicsDevice;
};

// SH coefficient for color conversion
const C0 = 0.28209479177387814;

// ============================================================================
// Main
// ============================================================================

async function main() {
    const args = process.argv.slice(2);
    if (args.length === 0) {
        console.log('Usage: node test/voxel-octree-node.mjs <path-to-file> [voxel-resolution] [opacity-cutoff]');
        console.log('Example: node test/voxel-octree-node.mjs ../supersplat/dist/shoe.ply 0.05 0.5');
        console.log('');
        console.log('Supports: PLY, SPLAT, KSPLAT, SOG, SPZ, LCC formats');
        console.log('');
        console.log('  voxel-resolution: size of each voxel in world units, default 0.05');
        console.log('  opacity-cutoff: 0.0-1.0, default 0.5');
        process.exit(1);
    }

    const inputPath = resolve(args[0]);
    const voxelResolution = args[1] ? parseFloat(args[1]) : 0.05;
    const opacityCutoff = args[2] ? parseFloat(args[2]) : 0.5;
    
    console.log(`Loading: ${inputPath}`);
    console.log(`Voxel resolution: ${voxelResolution}`);
    console.log(`Opacity cutoff: ${opacityCutoff}`);

    // =========================================================================
    // Read input file (any supported format)
    // =========================================================================
    const inputFilename = basename(inputPath);
    const inputFormat = getInputFormat(inputFilename);
    console.log(`Detected format: ${inputFormat}`);

    const fileSystem = new NodeReadFileSystem();
    const dataTables = await readFile({
        filename: inputPath,
        inputFormat,
        options: {},
        params: [],
        fileSystem
    });
    const dataTable = dataTables[0];
    console.log(`Loaded ${dataTable.numRows} Gaussians`);

    // =========================================================================
    // Phase 1: Compute Gaussian extents
    // =========================================================================
    console.log('\nComputing Gaussian extents...');
    const extentsResult = computeGaussianExtents(dataTable);
    const bounds = extentsResult.sceneBounds;
    console.log(`Scene bounds: (${bounds.min.x.toFixed(2)}, ${bounds.min.y.toFixed(2)}, ${bounds.min.z.toFixed(2)}) to (${bounds.max.x.toFixed(2)}, ${bounds.max.y.toFixed(2)}, ${bounds.max.z.toFixed(2)})`);

    // =========================================================================
    // Phase 2: Build BVH
    // =========================================================================
    console.log('Building BVH...');
    const bvh = new GaussianBVH(dataTable, extentsResult.extents);
    console.log(`BVH built with ${dataTable.numRows} Gaussians`);

    // =========================================================================
    // Phase 3: GPU Voxelization
    // =========================================================================
    console.log('\nCreating WebGPU device...');
    const device = await createDevice();
    console.log(`Device type: ${device.deviceType}`);

    console.log('Creating GPU Voxelization...');
    const gpuVoxelization = new GpuVoxelization(device);
    
    console.log('Uploading all Gaussians to GPU...');
    gpuVoxelization.uploadAllGaussians(dataTable, extentsResult.extents);
    console.log(`Uploaded ${dataTable.numRows} Gaussians`);

    // Calculate grid dimensions
    const blockSize = 4 * voxelResolution;
    const numBlocksX = Math.ceil((bounds.max.x - bounds.min.x) / blockSize);
    const numBlocksY = Math.ceil((bounds.max.y - bounds.min.y) / blockSize);
    const numBlocksZ = Math.ceil((bounds.max.z - bounds.min.z) / blockSize);
    const totalBlocks = numBlocksX * numBlocksY * numBlocksZ;

    console.log(`Full grid: ${numBlocksX} x ${numBlocksY} x ${numBlocksZ} = ${totalBlocks} blocks`);

    // =========================================================================
    // Phase 4: Voxelization with BlockAccumulator
    // =========================================================================
    console.log('\nVoxelizing scene...');
    const accumulator = new BlockAccumulator();
    const batchSize = 16;
    let processedBlocks = 0;

    const startTime = performance.now();

    for (let bz = 0; bz < numBlocksZ; bz += batchSize) {
        for (let by = 0; by < numBlocksY; by += batchSize) {
            for (let bx = 0; bx < numBlocksX; bx += batchSize) {
                const currBatchX = Math.min(batchSize, numBlocksX - bx);
                const currBatchY = Math.min(batchSize, numBlocksY - by);
                const currBatchZ = Math.min(batchSize, numBlocksZ - bz);

                const blockMinX = bounds.min.x + bx * blockSize;
                const blockMinY = bounds.min.y + by * blockSize;
                const blockMinZ = bounds.min.z + bz * blockSize;
                const blockMaxX = blockMinX + currBatchX * blockSize;
                const blockMaxY = blockMinY + currBatchY * blockSize;
                const blockMaxZ = blockMinZ + currBatchZ * blockSize;

                // Query BVH for overlapping Gaussians
                const overlapping = bvh.queryOverlappingRaw(
                    blockMinX, blockMinY, blockMinZ,
                    blockMaxX, blockMaxY, blockMaxZ
                );

                processedBlocks += currBatchX * currBatchY * currBatchZ;

                if (overlapping.length === 0) {
                    continue;
                }

                // Run GPU voxelization
                const result = await gpuVoxelization.voxelizeBlocks(
                    overlapping,
                    { x: blockMinX, y: blockMinY, z: blockMinZ },
                    currBatchX, currBatchY, currBatchZ,
                    voxelResolution,
                    opacityCutoff
                );

                // Accumulate blocks with Morton codes
                for (let blockIdx = 0; blockIdx < result.blocks.length; blockIdx++) {
                    const block = result.blocks[blockIdx];
                    const maskLo = result.masks[blockIdx * 2];
                    const maskHi = result.masks[blockIdx * 2 + 1];

                    const absBlockX = bx + block.x;
                    const absBlockY = by + block.y;
                    const absBlockZ = bz + block.z;

                    const morton = xyzToMorton(absBlockX, absBlockY, absBlockZ);
                    accumulator.addBlock(morton, maskLo, maskHi);
                }

                // Progress update
                const progress = Math.floor((processedBlocks / totalBlocks) * 100);
                if (processedBlocks % 100000 < currBatchX * currBatchY * currBatchZ) {
                    process.stdout.write(`\rProgress: ${progress}% (${accumulator.count} non-empty blocks)`);
                }
            }
        }
    }

    console.log(`\rProgress: 100% (${accumulator.count} non-empty blocks)`);

    // Cleanup GPU
    gpuVoxelization.destroy();
    device.destroy();

    const voxelizeTime = performance.now() - startTime;
    console.log(`Voxelization time: ${(voxelizeTime / 1000).toFixed(1)}s`);
    console.log(`  Solid blocks: ${accumulator.solidCount}`);
    console.log(`  Mixed blocks: ${accumulator.mixedCount}`);

    // =========================================================================
    // Phase 5: Build Sparse Octree
    // =========================================================================
    console.log('\nBuilding sparse octree...');

    const gridBounds = alignGridBounds(
        bounds.min.x, bounds.min.y, bounds.min.z,
        bounds.max.x, bounds.max.y, bounds.max.z,
        voxelResolution
    );

    const octree = buildSparseOctree(
        accumulator,
        gridBounds,
        bounds,
        voxelResolution
    );

    console.log(`Octree depth: ${octree.treeDepth}`);
    console.log(`Interior nodes: ${octree.numInteriorNodes}`);
    console.log(`Mixed leaves: ${octree.numMixedLeaves}`);
    console.log(`Nodes array: ${octree.nodes.length} entries`);
    console.log(`Leaf data: ${octree.leafData.length} entries`);

    // =========================================================================
    // Phase 6: Generate Multi-Level PLY Visualization (Bottom-Up Approach)
    // =========================================================================
    console.log('\nGenerating multi-level PLY files for visualization...');

    // Determine output base path
    const inputDir = dirname(inputPath);
    const inputBase = basename(inputPath, extname(inputPath));
    const outputDir = resolve(inputDir, 'voxel-viz');
    
    // Create output directory
    await mkdir(outputDir, { recursive: true });

    // Get leaf block Morton codes from the accumulator
    const solidBlockSet = new Set(accumulator.getSolidBlocks());
    const mixedBlockMortons = accumulator.getMixedBlocks().morton;

    console.log(`Total leaf blocks: ${solidBlockSet.size + mixedBlockMortons.length} (${solidBlockSet.size} solid, ${mixedBlockMortons.length} mixed)`);

    // For each level from leaf to root, compute parent blocks
    const leafLevel = octree.treeDepth;

    for (let level = leafLevel; level >= 0; level--) {
        // Compute parent Morton codes at this level
        // levelShift = how many levels up from leaves
        const levelShift = leafLevel - level;
        const divisor = Math.pow(8, levelShift);
        
        // Map parent Morton -> { isSolid: boolean, leafMorton: number (for color variation) }
        const blocksAtLevel = new Map();

        // Process solid blocks first
        for (const morton of solidBlockSet) {
            const parentMorton = Math.floor(morton / divisor);
            if (!blocksAtLevel.has(parentMorton)) {
                blocksAtLevel.set(parentMorton, { isSolid: true, leafMorton: morton });
            }
        }

        // Process mixed blocks - any parent with a mixed child is not solid
        for (const morton of mixedBlockMortons) {
            const parentMorton = Math.floor(morton / divisor);
            const existing = blocksAtLevel.get(parentMorton);
            if (existing) {
                existing.isSolid = false;  // Has at least one mixed child
            } else {
                blocksAtLevel.set(parentMorton, { isSolid: false, leafMorton: morton });
            }
        }

        const blocks = Array.from(blocksAtLevel.entries());
        const numBlocks = blocks.length;

        if (numBlocks === 0) continue;

        // Count solid vs mixed at this level
        let solidCount = 0;
        for (const [, info] of blocks) {
            if (info.isSolid) solidCount++;
        }

        // Block size at this level (doubles each level up from leaf)
        const levelBlockSize = blockSize * Math.pow(2, levelShift);
        const splatScale = Math.log(levelBlockSize * 0.4);

        // Create arrays for Gaussian splat output
        const xArr = new Float32Array(numBlocks);
        const yArr = new Float32Array(numBlocks);
        const zArr = new Float32Array(numBlocks);
        const scale0 = new Float32Array(numBlocks);
        const scale1 = new Float32Array(numBlocks);
        const scale2 = new Float32Array(numBlocks);
        const rot0 = new Float32Array(numBlocks);
        const rot1 = new Float32Array(numBlocks);
        const rot2 = new Float32Array(numBlocks);
        const rot3 = new Float32Array(numBlocks);
        const fdc0 = new Float32Array(numBlocks);
        const fdc1 = new Float32Array(numBlocks);
        const fdc2 = new Float32Array(numBlocks);
        const opacityArr = new Float32Array(numBlocks);

        for (let i = 0; i < numBlocks; i++) {
            const [parentMorton, info] = blocks[i];
            const [bx, by, bz] = mortonToXYZ(parentMorton);

            // Block center in world coordinates
            xArr[i] = gridBounds.min.x + (bx + 0.5) * levelBlockSize;
            yArr[i] = gridBounds.min.y + (by + 0.5) * levelBlockSize;
            zArr[i] = gridBounds.min.z + (bz + 0.5) * levelBlockSize;

            // Scale (log of half block size)
            scale0[i] = splatScale;
            scale1[i] = splatScale;
            scale2[i] = splatScale;

            // Identity quaternion
            rot0[i] = 1.0;
            rot1[i] = 0.0;
            rot2[i] = 0.0;
            rot3[i] = 0.0;

            // Color based on solid vs mixed
            let r, g, b;
            if (info.isSolid) {
                // Solid blocks: red
                r = 0.9;
                g = 0.1;
                b = 0.1;
            } else {
                // Mixed blocks: shades of gray, varying by Morton code for visual distinction
                const gray = 0.3 + (info.leafMorton * 0.618033988749895 % 1.0) * 0.5;
                r = gray;
                g = gray;
                b = gray;
            }

            // Convert RGB to SH DC
            fdc0[i] = (r - 0.5) / C0;
            fdc1[i] = (g - 0.5) / C0;
            fdc2[i] = (b - 0.5) / C0;

            // High opacity (logit value)
            opacityArr[i] = 5.0;
        }

        const outputTable = new DataTable([
            new Column('x', xArr),
            new Column('y', yArr),
            new Column('z', zArr),
            new Column('scale_0', scale0),
            new Column('scale_1', scale1),
            new Column('scale_2', scale2),
            new Column('rot_0', rot0),
            new Column('rot_1', rot1),
            new Column('rot_2', rot2),
            new Column('rot_3', rot3),
            new Column('f_dc_0', fdc0),
            new Column('f_dc_1', fdc1),
            new Column('f_dc_2', fdc2),
            new Column('opacity', opacityArr)
        ]);

        const outputFs = new MemoryFileSystem();
        await writePly({
            filename: 'level.ply',
            plyData: {
                comments: [`Octree level ${level} visualization - ${solidCount} solid, ${numBlocks - solidCount} mixed`],
                elements: [{
                    name: 'vertex',
                    dataTable: outputTable
                }]
            }
        }, outputFs);

        const outputPath = resolve(outputDir, `${inputBase}_level_${level}.ply`);
        await fsWriteFile(outputPath, outputFs.results.get('level.ply'));
        console.log(`  Level ${level}: ${numBlocks.toLocaleString()} blocks (${solidCount} solid, ${numBlocks - solidCount} mixed) -> ${outputPath}`);
    }

    const totalTime = performance.now() - startTime;
    console.log(`\nTotal time: ${(totalTime / 1000).toFixed(1)}s`);
    console.log('Done!');
}

main().catch(e => {
    console.error('Error:', e);
    console.error(e.stack);
    process.exit(1);
});
