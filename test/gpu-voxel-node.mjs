/**
 * Node.js test for GPU voxelization using Dawn WebGPU.
 * 
 * Run with: node test/gpu-voxel-node.mjs <path-to-ply>
 */

import { readFile } from 'fs/promises';
import { resolve, dirname } from 'path';

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
    DataTable
} from '../dist/index.mjs';

// Import device creation from CLI
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

// Morton decode function (same as shader)
const mortonToXYZ = (m) => {
    return {
        x: (m & 1) | ((m >> 2) & 2),
        y: ((m >> 1) & 1) | ((m >> 3) & 2),
        z: ((m >> 2) & 1) | ((m >> 4) & 2)
    };
};

async function main() {
    const args = process.argv.slice(2);
    if (args.length === 0) {
        console.log('Usage: node test/gpu-voxel-node.mjs <path-to-ply> [opacity-cutoff]');
        console.log('Example: node test/gpu-voxel-node.mjs ../supersplat/dist/shoe.ply 0.05');
        console.log('  opacity-cutoff: 0.0-1.0, default 0.05');
        process.exit(1);
    }

    const plyPath = resolve(args[0]);
    const opacityCutoff = args[1] ? parseFloat(args[1]) : 0.05;
    console.log(`Loading PLY: ${plyPath}`);

    // Read file
    const fileData = await readFile(plyPath);
    console.log(`File size: ${(fileData.length / 1024 / 1024).toFixed(2)} MB`);

    // Parse PLY
    console.log('Parsing PLY...');
    const memFs = new MemoryReadFileSystem();
    memFs.set('input.ply', new Uint8Array(fileData.buffer));
    const source = await memFs.createSource('input.ply');
    const dataTable = await readPly(source);
    console.log(`Loaded ${dataTable.numRows} Gaussians`);

    // Compute extents
    console.log('Computing Gaussian extents...');
    const extentsResult = computeGaussianExtents(dataTable);
    const bounds = extentsResult.sceneBounds;
    console.log(`Scene bounds: (${bounds.min.x.toFixed(2)}, ${bounds.min.y.toFixed(2)}, ${bounds.min.z.toFixed(2)}) to (${bounds.max.x.toFixed(2)}, ${bounds.max.y.toFixed(2)}, ${bounds.max.z.toFixed(2)})`);

    // Build BVH
    console.log('Building BVH...');
    const bvh = new GaussianBVH(dataTable, extentsResult.extents);
    console.log(`BVH built`);

    // Create GPU device
    console.log('Creating WebGPU device...');
    const device = await createDevice();
    console.log(`Device type: ${device.deviceType}`);

    // Create voxelization instance
    console.log('Creating GPU Voxelization...');
    const gpuVoxelization = new GpuVoxelization(device);
    
    // Upload ALL Gaussian data to GPU once (including pre-computed AABB extents)
    console.log('Uploading all Gaussians to GPU...');
    gpuVoxelization.uploadAllGaussians(dataTable, extentsResult.extents);
    console.log(`Uploaded ${gpuVoxelization.numGaussians} Gaussians, Max Indices: ${gpuVoxelization.maxIndices}, Max Blocks: ${gpuVoxelization.maxBlocks}`);

    // Voxelization parameters
    const voxelRes = 0.05;
    console.log(`Opacity cutoff: ${opacityCutoff}`);
    const blockSize = 4 * voxelRes;  // Each block is 4x4x4 voxels

    // Calculate full grid dimensions
    const numBlocksX = Math.ceil((bounds.max.x - bounds.min.x) / blockSize);
    const numBlocksY = Math.ceil((bounds.max.y - bounds.min.y) / blockSize);
    const numBlocksZ = Math.ceil((bounds.max.z - bounds.min.z) / blockSize);
    const totalBlocks = numBlocksX * numBlocksY * numBlocksZ;
    console.log(`Full grid: ${numBlocksX} x ${numBlocksY} x ${numBlocksZ} = ${totalBlocks} blocks`);

    // Batch size for processing (limited by GPU)
    const batchSize = 16;  // 16x16x16 = 4096 blocks max per batch
    
    // Collect all solid voxel centers
    const voxelCenters = [];
    let totalProcessedBlocks = 0;
    let totalGpuTime = 0;
    
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
                    // Empty batch - skip
                    totalProcessedBlocks += currBatchX * currBatchY * currBatchZ;
                    continue;
                }
                
                // Pass overlapping Gaussian indices - references pre-uploaded Gaussian data
                
                // Run GPU voxelization with proper alpha blending
                const voxelStart = performance.now();
                const result = await gpuVoxelization.voxelizeBlocks(
                    overlapping,
                    { x: blockMinX, y: blockMinY, z: blockMinZ },
                    currBatchX, currBatchY, currBatchZ,
                    voxelRes,
                    opacityCutoff
                );
                totalGpuTime += performance.now() - voxelStart;
                
                // Extract solid voxel centers
                for (let blockIdx = 0; blockIdx < result.blocks.length; blockIdx++) {
                    const block = result.blocks[blockIdx];
                    const maskLow = result.masks[blockIdx * 2];
                    const maskHigh = result.masks[blockIdx * 2 + 1];
                    
                    if (maskLow === 0 && maskHigh === 0) continue;
                    
                    // Check each voxel in the 4x4x4 block
                    for (let voxelIdx = 0; voxelIdx < 64; voxelIdx++) {
                        const isSet = voxelIdx < 32
                            ? (maskLow & (1 << voxelIdx)) !== 0
                            : (maskHigh & (1 << (voxelIdx - 32))) !== 0;
                        if (isSet) {
                            // Voxel is solid - compute world position
                            const localPos = mortonToXYZ(voxelIdx);
                            
                            const worldX = blockMinX + block.x * blockSize + (localPos.x + 0.5) * voxelRes;
                            const worldY = blockMinY + block.y * blockSize + (localPos.y + 0.5) * voxelRes;
                            const worldZ = blockMinZ + block.z * blockSize + (localPos.z + 0.5) * voxelRes;
                            
                            voxelCenters.push({ x: worldX, y: worldY, z: worldZ });
                        }
                    }
                }
                
                totalProcessedBlocks += result.masks.length;
            }
        }
        
        // Progress update
        const progress = ((bz + batchSize) / numBlocksZ * 100);
        console.log(`Progress: ${Math.min(100, progress).toFixed(0)}% (${voxelCenters.length} solid voxels found)`);
    }
    
    const overallElapsed = performance.now() - overallStart;
    console.log(`\nVoxelization complete in ${overallElapsed.toFixed(0)}ms (GPU: ${totalGpuTime.toFixed(0)}ms)`);
    console.log(`Processed ${totalProcessedBlocks} blocks`);
    console.log(`Found ${voxelCenters.length} solid voxels`);

    if (voxelCenters.length === 0) {
        console.log('No solid voxels found!');
        gpuVoxelization.destroy();
        device.destroy();
        process.exit(1);
    }

    // Create DataTable for output
    console.log('\nCreating output DataTable...');
    const numVoxels = voxelCenters.length;
    
    // Allocate arrays
    const xArr = new Float32Array(numVoxels);
    const yArr = new Float32Array(numVoxels);
    const zArr = new Float32Array(numVoxels);
    const scale0 = new Float32Array(numVoxels);
    const scale1 = new Float32Array(numVoxels);
    const scale2 = new Float32Array(numVoxels);
    const rot0 = new Float32Array(numVoxels);
    const rot1 = new Float32Array(numVoxels);
    const rot2 = new Float32Array(numVoxels);
    const rot3 = new Float32Array(numVoxels);
    const fdc0 = new Float32Array(numVoxels);
    const fdc1 = new Float32Array(numVoxels);
    const fdc2 = new Float32Array(numVoxels);
    const opacityArr = new Float32Array(numVoxels);
    
    // Voxel scale (log of half voxel size)
    const voxelScale = Math.log(voxelRes * 0.4);
    // High opacity logit for solid appearance
    const opacityLogit = 5.0;
    // SH coefficient for color conversion
    const C0 = 0.28209479177387814;
    
    // Scene extent for color normalization
    const sceneExtentX = bounds.max.x - bounds.min.x;
    const sceneExtentY = bounds.max.y - bounds.min.y;
    const sceneExtentZ = bounds.max.z - bounds.min.z;
    
    for (let i = 0; i < numVoxels; i++) {
        const v = voxelCenters[i];
        
        xArr[i] = v.x;
        yArr[i] = v.y;
        zArr[i] = v.z;
        
        scale0[i] = voxelScale;
        scale1[i] = voxelScale;
        scale2[i] = voxelScale;
        
        // Identity quaternion
        rot0[i] = 1.0;
        rot1[i] = 0.0;
        rot2[i] = 0.0;
        rot3[i] = 0.0;
        
        // Color based on position (normalized to scene bounds)
        const r = (v.x - bounds.min.x) / sceneExtentX;
        const g = (v.y - bounds.min.y) / sceneExtentY;
        const b = (v.z - bounds.min.z) / sceneExtentZ;
        
        // Convert RGB [0,1] to SH DC
        fdc0[i] = (r - 0.5) / C0;
        fdc1[i] = (g - 0.5) / C0;
        fdc2[i] = (b - 0.5) / C0;
        
        opacityArr[i] = opacityLogit;
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
    
    // Write PLY using splat-transform API
    console.log('Writing PLY...');
    const outputPath = plyPath.replace('.ply', '_voxels.ply');
    
    const outputFs = new MemoryFileSystem();
    await writePly({
        filename: 'output.ply',
        plyData: {
            comments: ['Voxelized Gaussian splat data'],
            elements: [{
                name: 'vertex',
                dataTable: outputTable
            }]
        }
    }, outputFs);
    
    // Write to disk
    const outputData = outputFs.results.get('output.ply');
    await (await import('fs/promises')).writeFile(outputPath, outputData);
    console.log(`Wrote ${numVoxels} voxels to: ${outputPath}`);

    // Cleanup
    gpuVoxelization.destroy();
    device.destroy();

    console.log('\nDone!');
}

main().catch(e => {
    console.error('Error:', e);
    process.exit(1);
});
