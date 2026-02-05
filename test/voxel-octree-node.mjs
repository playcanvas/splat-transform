/**
 * Node.js test for GPU voxelization with sparse octree construction.
 * 
 * This script demonstrates the full voxelization pipeline:
 * 1. Read PLY file
 * 2. Compute Gaussian extents and build BVH
 * 3. GPU voxelization in batches
 * 4. Accumulate blocks with BlockAccumulator
 * 5. Build sparse octree using buildSparseOctree
 * 
 * Run with: node test/voxel-octree-node.mjs <path-to-ply> [voxel-resolution] [opacity-cutoff]
 */

import { readFile, writeFile as fsWriteFile } from 'fs/promises';
import { resolve } from 'path';

// Import the compiled library
import { 
    readPly,
    writePly,
    computeGaussianExtents, 
    GaussianBVH, 
    GpuVoxelization,
    MemoryReadFileSystem,
    MemoryFileSystem,
    Column,
    DataTable,
    // Phase 4: Sparse octree components
    BlockAccumulator,
    buildSparseOctree,
    alignGridBounds,
    xyzToMorton,
    mortonToXYZ
} from '../dist/index.mjs';

// Import device creation
import { WebgpuGraphicsDevice } from 'playcanvas';
import { create, globals } from 'webgpu';

// Initialize globals for Node.js WebGPU
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

async function main() {
    const args = process.argv.slice(2);
    if (args.length === 0) {
        console.log('Usage: node test/voxel-octree-node.mjs <path-to-ply> [voxel-resolution] [opacity-cutoff]');
        console.log('Example: node test/voxel-octree-node.mjs ../supersplat/dist/shoe.ply 0.05 0.05');
        console.log('  voxel-resolution: size of each voxel in world units, default 0.05');
        console.log('  opacity-cutoff: 0.0-1.0, default 0.05');
        process.exit(1);
    }

    const plyPath = resolve(args[0]);
    const voxelRes = args[1] ? parseFloat(args[1]) : 0.05;
    const opacityCutoff = args[2] ? parseFloat(args[2]) : 0.05;
    
    console.log(`Loading PLY: ${plyPath}`);
    console.log(`Voxel resolution: ${voxelRes}`);
    console.log(`Opacity cutoff: ${opacityCutoff}`);

    // =========================================================================
    // Phase 1: Read PLY file
    // =========================================================================
    const fileData = await readFile(plyPath);
    console.log(`File size: ${(fileData.length / 1024 / 1024).toFixed(2)} MB`);

    console.log('Parsing PLY...');
    const memFs = new MemoryReadFileSystem();
    memFs.set('input.ply', new Uint8Array(fileData.buffer));
    const source = await memFs.createSource('input.ply');
    const dataTable = await readPly(source);
    console.log(`Loaded ${dataTable.numRows} Gaussians`);

    // =========================================================================
    // Phase 2: Compute extents and build BVH
    // =========================================================================
    console.log('Computing Gaussian extents...');
    const extentsResult = computeGaussianExtents(dataTable);
    const bounds = extentsResult.sceneBounds;
    console.log(`Scene bounds: (${bounds.min.x.toFixed(2)}, ${bounds.min.y.toFixed(2)}, ${bounds.min.z.toFixed(2)}) to (${bounds.max.x.toFixed(2)}, ${bounds.max.y.toFixed(2)}, ${bounds.max.z.toFixed(2)})`);

    console.log('Building BVH...');
    const bvh = new GaussianBVH(dataTable, extentsResult.extents);
    console.log(`BVH built with ${bvh.count} Gaussians`);

    // =========================================================================
    // Phase 3: GPU Voxelization setup
    // =========================================================================
    console.log('Creating WebGPU device...');
    const device = await createDevice();
    console.log(`Device type: ${device.deviceType}`);

    console.log('Creating GPU Voxelization...');
    const gpuVoxelization = new GpuVoxelization(device);
    
    console.log('Uploading all Gaussians to GPU...');
    gpuVoxelization.uploadAllGaussians(dataTable, extentsResult.extents);
    console.log(`Uploaded ${gpuVoxelization.numGaussians} Gaussians`);

    // Calculate grid dimensions
    const blockSize = 4 * voxelRes;  // Each block is 4x4x4 voxels
    const numBlocksX = Math.ceil((bounds.max.x - bounds.min.x) / blockSize);
    const numBlocksY = Math.ceil((bounds.max.y - bounds.min.y) / blockSize);
    const numBlocksZ = Math.ceil((bounds.max.z - bounds.min.z) / blockSize);
    const totalBlocks = numBlocksX * numBlocksY * numBlocksZ;
    console.log(`Full grid: ${numBlocksX} x ${numBlocksY} x ${numBlocksZ} = ${totalBlocks} blocks`);

    // =========================================================================
    // Phase 4: Voxelization with BlockAccumulator
    // =========================================================================
    const accumulator = new BlockAccumulator();
    const batchSize = 16;  // 16x16x16 = 4096 blocks max per batch
    
    let totalProcessedBlocks = 0;
    let totalGpuTime = 0;
    let emptyBatches = 0;
    
    console.log('\nVoxelizing entire scene...');
    const overallStart = performance.now();
    
    // Process the entire scene in batches
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
                
                if (overlapping.length === 0) {
                    // Empty batch - skip GPU work
                    emptyBatches++;
                    totalProcessedBlocks += currBatchX * currBatchY * currBatchZ;
                    continue;
                }
                
                // Run GPU voxelization
                const voxelStart = performance.now();
                const result = await gpuVoxelization.voxelizeBlocks(
                    overlapping,
                    { x: blockMinX, y: blockMinY, z: blockMinZ },
                    currBatchX, currBatchY, currBatchZ,
                    voxelRes,
                    opacityCutoff
                );
                totalGpuTime += performance.now() - voxelStart;
                
                // Accumulate blocks with Morton codes
                for (let blockIdx = 0; blockIdx < result.blocks.length; blockIdx++) {
                    const block = result.blocks[blockIdx];
                    const maskLo = result.masks[blockIdx * 2];
                    const maskHi = result.masks[blockIdx * 2 + 1];
                    
                    // Calculate absolute block coordinates
                    const absBlockX = bx + block.x;
                    const absBlockY = by + block.y;
                    const absBlockZ = bz + block.z;
                    
                    // Convert to Morton code and accumulate
                    const morton = xyzToMorton(absBlockX, absBlockY, absBlockZ);
                    accumulator.addBlock(morton, maskLo, maskHi);
                }
                
                totalProcessedBlocks += currBatchX * currBatchY * currBatchZ;
            }
        }
        
        // Progress update
        const progress = ((bz + batchSize) / numBlocksZ * 100);
        console.log(`Progress: ${Math.min(100, progress).toFixed(0)}% (${accumulator.count} non-empty blocks)`);
    }
    
    const voxelizationElapsed = performance.now() - overallStart;
    console.log(`\nVoxelization complete in ${voxelizationElapsed.toFixed(0)}ms (GPU: ${totalGpuTime.toFixed(0)}ms)`);
    console.log(`Processed ${totalProcessedBlocks} blocks (${emptyBatches} empty batches skipped)`);

    // =========================================================================
    // Build Sparse Octree
    // =========================================================================
    console.log('\nBuilding sparse octree...');
    const octreeStart = performance.now();

    // Align grid bounds to block boundaries
    const gridBounds = alignGridBounds(
        bounds.min.x, bounds.min.y, bounds.min.z,
        bounds.max.x, bounds.max.y, bounds.max.z,
        voxelRes
    );

    // Build the sparse octree
    const octree = buildSparseOctree(
        accumulator,
        gridBounds,
        bounds,  // Original Gaussian bounds
        voxelRes
    );

    const octreeElapsed = performance.now() - octreeStart;
    console.log(`Octree built in ${octreeElapsed.toFixed(0)}ms`);

    // =========================================================================
    // Print Statistics
    // =========================================================================
    console.log('\n' + '='.repeat(50));
    console.log('SPARSE OCTREE STATISTICS');
    console.log('='.repeat(50));
    
    console.log('\nGrid Configuration:');
    console.log(`  Grid bounds: (${gridBounds.min.x.toFixed(2)}, ${gridBounds.min.y.toFixed(2)}, ${gridBounds.min.z.toFixed(2)}) to (${gridBounds.max.x.toFixed(2)}, ${gridBounds.max.y.toFixed(2)}, ${gridBounds.max.z.toFixed(2)})`);
    console.log(`  Voxel resolution: ${octree.voxelResolution}`);
    console.log(`  Leaf size: ${octree.leafSize} voxels`);
    console.log(`  Block size: ${(octree.leafSize * octree.voxelResolution).toFixed(3)} world units`);
    
    console.log('\nOctree Structure:');
    console.log(`  Tree depth: ${octree.treeDepth}`);
    console.log(`  Interior nodes: ${octree.numInteriorNodes}`);
    console.log(`  Mixed leaves: ${octree.numMixedLeaves}`);
    
    console.log('\nMemory Usage:');
    const nodesBytes = octree.nodes.byteLength;
    const leafBytes = octree.leafData.byteLength;
    const totalBytes = nodesBytes + leafBytes;
    console.log(`  Nodes array: ${octree.nodes.length} entries (${(nodesBytes / 1024).toFixed(1)} KB)`);
    console.log(`  Leaf data: ${octree.leafData.length} entries (${(leafBytes / 1024).toFixed(1)} KB)`);
    console.log(`  Total octree size: ${(totalBytes / 1024).toFixed(1)} KB`);
    
    console.log('\nBlock Classification:');
    console.log(`  Total non-empty blocks: ${accumulator.count}`);
    console.log(`  Solid blocks: ${accumulator.solidCount} (${(accumulator.solidCount / accumulator.count * 100).toFixed(1)}%)`);
    console.log(`  Mixed blocks: ${accumulator.mixedCount} (${(accumulator.mixedCount / accumulator.count * 100).toFixed(1)}%)`);
    
    // Estimate voxel count
    const solidVoxels = accumulator.solidCount * 64;  // Each solid block has 64 voxels
    // For mixed blocks, we'd need to count bits - just estimate ~50% fill
    const estimatedMixedVoxels = accumulator.mixedCount * 32;
    const estimatedTotalVoxels = solidVoxels + estimatedMixedVoxels;
    console.log(`  Estimated total voxels: ~${estimatedTotalVoxels.toLocaleString()}`);
    
    console.log('\nPerformance:');
    console.log(`  Total time: ${(voxelizationElapsed + octreeElapsed).toFixed(0)}ms`);
    console.log(`    - Voxelization: ${voxelizationElapsed.toFixed(0)}ms`);
    console.log(`    - Octree construction: ${octreeElapsed.toFixed(0)}ms`);
    console.log(`  Blocks/second: ${(totalProcessedBlocks / (voxelizationElapsed / 1000)).toFixed(0)}`);

    console.log('\n' + '='.repeat(50));

    // =========================================================================
    // Multi-Level PLY Visualization Output
    // =========================================================================
    console.log('\nGenerating multi-level PLY files for visualization...');

    // Collect leaf block Morton codes, tracking solid vs mixed
    const solidBlockSet = new Set(accumulator.getSolidBlocks());
    const mixedBlockMortons = accumulator.getMixedBlocks().morton;
    const allBlockMortons = [...solidBlockSet, ...mixedBlockMortons];

    console.log(`Total leaf blocks: ${allBlockMortons.length} (${solidBlockSet.size} solid, ${mixedBlockMortons.length} mixed)`);

    // For each level from leaf to root, compute parent blocks
    const leafLevel = octree.treeDepth;
    const basePath = plyPath.replace('.ply', '');

    // SH coefficient for color conversion
    const C0 = 0.28209479177387814;

    for (let level = leafLevel; level >= 0; level--) {
        // Compute parent Morton codes at this level, tracking if all children are solid
        const levelShift = leafLevel - level;
        const divisor = Math.pow(8, levelShift);
        
        // Map parent Morton -> { isSolid: boolean, leafMorton: number (for color) }
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
                // Mixed blocks: shades of gray, varying by Morton code
                // Range from 0.3 to 0.8 for visible variation
                const gray = 0.3 + (info.leafMorton * 0.618033988749895 % 1.0) * 0.5;
                r = gray;
                g = gray;
                b = gray;
            }

            // Convert RGB to SH DC
            fdc0[i] = (r - 0.5) / C0;
            fdc1[i] = (g - 0.5) / C0;
            fdc2[i] = (b - 0.5) / C0;

            // High opacity
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

        const levelPath = `${basePath}_level${level}.ply`;
        await fsWriteFile(levelPath, outputFs.results.get('level.ply'));
        console.log(`  Level ${level}: ${numBlocks.toLocaleString()} blocks (${solidCount} solid, ${numBlocks - solidCount} mixed) -> ${levelPath}`);
    }

    console.log('\nVisualization files written!');

    // Cleanup
    gpuVoxelization.destroy();
    device.destroy();

    console.log('\nDone!');
}

main().catch(e => {
    console.error('Error:', e);
    console.error(e.stack);
    process.exit(1);
});
