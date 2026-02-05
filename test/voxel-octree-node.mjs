/**
 * Node.js test for voxel output using writeVoxel.
 * 
 * This script demonstrates the simplified voxelization API:
 * 1. Read PLY file
 * 2. Call writeVoxel with the DataTable
 * 3. writeVoxel handles all voxelization internally
 * 
 * Run with: node test/voxel-octree-node.mjs <path-to-ply> [voxel-resolution] [opacity-cutoff]
 */

import { readFile, writeFile as fsWriteFile } from 'fs/promises';
import { resolve } from 'path';

// Import the compiled library
import { 
    readPly,
    writeVoxel,
    MemoryReadFileSystem,
    MemoryFileSystem
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
        console.log('Example: node test/voxel-octree-node.mjs ../supersplat/dist/shoe.ply 0.05 0.5');
        console.log('  voxel-resolution: size of each voxel in world units, default 0.05');
        console.log('  opacity-cutoff: 0.0-1.0, default 0.5');
        process.exit(1);
    }

    const plyPath = resolve(args[0]);
    const voxelResolution = args[1] ? parseFloat(args[1]) : 0.05;
    const opacityCutoff = args[2] ? parseFloat(args[2]) : 0.5;
    
    console.log(`Loading PLY: ${plyPath}`);
    console.log(`Voxel resolution: ${voxelResolution}`);
    console.log(`Opacity cutoff: ${opacityCutoff}`);

    // =========================================================================
    // Read PLY file
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
    // Write Voxel Output using writeVoxel
    // =========================================================================
    console.log('\nStarting voxelization...');
    const startTime = performance.now();

    // Track device for cleanup (caller manages device lifecycle)
    let device = null;
    const createDeviceWithTracking = async () => {
        device = await createDevice();
        return device;
    };

    const outputFs = new MemoryFileSystem();
    await writeVoxel({
        filename: 'output.voxel.json',
        dataTable,
        voxelResolution,
        opacityCutoff,
        createDevice: createDeviceWithTracking
    }, outputFs);

    // Cleanup device
    if (device) {
        device.destroy();
    }

    const elapsed = performance.now() - startTime;
    console.log(`\nTotal time: ${(elapsed / 1000).toFixed(1)}s`);

    // Write to disk
    const basePath = plyPath.replace('.ply', '');
    const jsonData = outputFs.results.get('output.voxel.json');
    const binData = outputFs.results.get('output.voxel.bin');
    
    await fsWriteFile(basePath + '.voxel.json', jsonData);
    await fsWriteFile(basePath + '.voxel.bin', binData);
    
    console.log(`\nOutput files:`);
    console.log(`  Metadata: ${basePath}.voxel.json (${(jsonData.byteLength / 1024).toFixed(1)} KB)`);
    console.log(`  Binary:   ${basePath}.voxel.bin (${(binData.byteLength / 1024).toFixed(1)} KB)`);

    // Parse and display metadata
    const metadata = JSON.parse(new TextDecoder().decode(jsonData));
    console.log('\nVoxel Octree Metadata:');
    console.log(`  Version: ${metadata.version}`);
    console.log(`  Tree depth: ${metadata.treeDepth}`);
    console.log(`  Interior nodes: ${metadata.numInteriorNodes}`);
    console.log(`  Mixed leaves: ${metadata.numMixedLeaves}`);
    console.log(`  Voxel resolution: ${metadata.voxelResolution}`);

    console.log('\nDone!');
}

main().catch(e => {
    console.error('Error:', e);
    console.error(e.stack);
    process.exit(1);
});
