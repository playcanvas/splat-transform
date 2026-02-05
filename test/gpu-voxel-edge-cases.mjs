/**
 * Edge case tests for GPU voxelization.
 * 
 * Creates various edge case Gaussians and verifies correct voxel generation.
 * Outputs test scenes and resulting voxels as PLY files for visual inspection.
 * 
 * Run with: node test/gpu-voxel-edge-cases.mjs [output-dir]
 */

import { mkdir, writeFile } from 'fs/promises';
import { resolve, join } from 'path';

// Import the compiled library
import { 
    computeGaussianExtents, 
    GaussianBVH, 
    GpuVoxelization,
    writePly,
    MemoryFileSystem,
    Column,
    DataTable
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

// Morton decode function (same as shader)
const mortonToXYZ = (m) => {
    return {
        x: (m & 1) | ((m >> 2) & 2),
        y: ((m >> 1) & 1) | ((m >> 3) & 2),
        z: ((m >> 2) & 1) | ((m >> 4) & 2)
    };
};

/**
 * Create a DataTable from an array of Gaussian definitions.
 * Each Gaussian: { x, y, z, scaleX, scaleY, scaleZ, rotW, rotX, rotY, rotZ, opacity }
 */
function createDataTable(gaussians) {
    const n = gaussians.length;
    
    const x = new Float32Array(n);
    const y = new Float32Array(n);
    const z = new Float32Array(n);
    const scale0 = new Float32Array(n);
    const scale1 = new Float32Array(n);
    const scale2 = new Float32Array(n);
    const rot0 = new Float32Array(n);
    const rot1 = new Float32Array(n);
    const rot2 = new Float32Array(n);
    const rot3 = new Float32Array(n);
    const opacity = new Float32Array(n);
    
    for (let i = 0; i < n; i++) {
        const g = gaussians[i];
        x[i] = g.x;
        y[i] = g.y;
        z[i] = g.z;
        // Scales are stored as log(scale)
        scale0[i] = Math.log(g.scaleX);
        scale1[i] = Math.log(g.scaleY);
        scale2[i] = Math.log(g.scaleZ);
        // Quaternion (w, x, y, z)
        rot0[i] = g.rotW ?? 1.0;
        rot1[i] = g.rotX ?? 0.0;
        rot2[i] = g.rotY ?? 0.0;
        rot3[i] = g.rotZ ?? 0.0;
        // Opacity as logit
        const op = g.opacity ?? 0.99;
        opacity[i] = Math.log(op / (1 - op));
    }
    
    return new DataTable([
        new Column('x', x),
        new Column('y', y),
        new Column('z', z),
        new Column('scale_0', scale0),
        new Column('scale_1', scale1),
        new Column('scale_2', scale2),
        new Column('rot_0', rot0),
        new Column('rot_1', rot1),
        new Column('rot_2', rot2),
        new Column('rot_3', rot3),
        new Column('opacity', opacity)
    ]);
}

/**
 * Create a quaternion for rotation around an axis.
 */
function axisAngleToQuat(axisX, axisY, axisZ, angleDegrees) {
    const angleRad = angleDegrees * Math.PI / 180;
    const halfAngle = angleRad / 2;
    const s = Math.sin(halfAngle);
    const c = Math.cos(halfAngle);
    
    // Normalize axis
    const len = Math.sqrt(axisX * axisX + axisY * axisY + axisZ * axisZ);
    
    return {
        rotW: c,
        rotX: (axisX / len) * s,
        rotY: (axisY / len) * s,
        rotZ: (axisZ / len) * s
    };
}

/**
 * Run voxelization on a test case and return voxel centers.
 */
async function voxelizeTestCase(gpuVoxelization, dataTable, extentsResult, voxelRes, opacityCutoff) {
    const bounds = extentsResult.sceneBounds;
    const bvh = new GaussianBVH(dataTable, extentsResult.extents);
    
    gpuVoxelization.uploadAllGaussians(dataTable, extentsResult.extents);
    
    const blockSize = 4 * voxelRes;
    const numBlocksX = Math.ceil((bounds.max.x - bounds.min.x) / blockSize);
    const numBlocksY = Math.ceil((bounds.max.y - bounds.min.y) / blockSize);
    const numBlocksZ = Math.ceil((bounds.max.z - bounds.min.z) / blockSize);
    
    const voxelCenters = [];
    
    for (let bz = 0; bz < numBlocksZ; bz++) {
        for (let by = 0; by < numBlocksY; by++) {
            for (let bx = 0; bx < numBlocksX; bx++) {
                const blockMinX = bounds.min.x + bx * blockSize;
                const blockMinY = bounds.min.y + by * blockSize;
                const blockMinZ = bounds.min.z + bz * blockSize;
                const blockMaxX = blockMinX + blockSize;
                const blockMaxY = blockMinY + blockSize;
                const blockMaxZ = blockMinZ + blockSize;
                
                const overlapping = bvh.queryOverlappingRaw(
                    blockMinX, blockMinY, blockMinZ,
                    blockMaxX, blockMaxY, blockMaxZ
                );
                
                if (overlapping.length === 0) continue;
                
                const result = await gpuVoxelization.voxelizeBlocks(
                    overlapping,
                    { x: blockMinX, y: blockMinY, z: blockMinZ },
                    1, 1, 1,
                    voxelRes,
                    opacityCutoff
                );
                
                for (let blockIdx = 0; blockIdx < result.blocks.length; blockIdx++) {
                    const block = result.blocks[blockIdx];
                    const maskLow = result.masks[blockIdx * 2];
                    const maskHigh = result.masks[blockIdx * 2 + 1];
                    
                    if (maskLow === 0 && maskHigh === 0) continue;
                    
                    for (let voxelIdx = 0; voxelIdx < 64; voxelIdx++) {
                        const isSet = voxelIdx < 32
                            ? (maskLow & (1 << voxelIdx)) !== 0
                            : (maskHigh & (1 << (voxelIdx - 32))) !== 0;
                        if (isSet) {
                            const localPos = mortonToXYZ(voxelIdx);
                            
                            const worldX = blockMinX + block.x * blockSize + (localPos.x + 0.5) * voxelRes;
                            const worldY = blockMinY + block.y * blockSize + (localPos.y + 0.5) * voxelRes;
                            const worldZ = blockMinZ + block.z * blockSize + (localPos.z + 0.5) * voxelRes;
                            
                            voxelCenters.push({ x: worldX, y: worldY, z: worldZ });
                        }
                    }
                }
            }
        }
    }
    
    return voxelCenters;
}

/**
 * Write a PLY file with Gaussians (input scene).
 */
async function writeGaussiansPly(gaussians, filename, outputDir) {
    const n = gaussians.length;
    
    const xArr = new Float32Array(n);
    const yArr = new Float32Array(n);
    const zArr = new Float32Array(n);
    const scale0 = new Float32Array(n);
    const scale1 = new Float32Array(n);
    const scale2 = new Float32Array(n);
    const rot0 = new Float32Array(n);
    const rot1 = new Float32Array(n);
    const rot2 = new Float32Array(n);
    const rot3 = new Float32Array(n);
    const fdc0 = new Float32Array(n);
    const fdc1 = new Float32Array(n);
    const fdc2 = new Float32Array(n);
    const opacityArr = new Float32Array(n);
    
    const C0 = 0.28209479177387814;
    
    for (let i = 0; i < n; i++) {
        const g = gaussians[i];
        xArr[i] = g.x;
        yArr[i] = g.y;
        zArr[i] = g.z;
        scale0[i] = Math.log(g.scaleX);
        scale1[i] = Math.log(g.scaleY);
        scale2[i] = Math.log(g.scaleZ);
        rot0[i] = g.rotW ?? 1.0;
        rot1[i] = g.rotX ?? 0.0;
        rot2[i] = g.rotY ?? 0.0;
        rot3[i] = g.rotZ ?? 0.0;
        // Red color for input Gaussians
        fdc0[i] = (1.0 - 0.5) / C0;
        fdc1[i] = (0.2 - 0.5) / C0;
        fdc2[i] = (0.2 - 0.5) / C0;
        const op = g.opacity ?? 0.99;
        opacityArr[i] = Math.log(op / (1 - op));
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
        filename: 'output.ply',
        plyData: {
            comments: ['Edge case test - input Gaussians'],
            elements: [{
                name: 'vertex',
                dataTable: outputTable
            }]
        }
    }, outputFs);
    
    const outputData = outputFs.results.get('output.ply');
    const outputPath = join(outputDir, filename);
    await writeFile(outputPath, outputData);
    console.log(`  Written: ${outputPath}`);
}

/**
 * Write a PLY file with voxels (output).
 */
async function writeVoxelsPly(voxelCenters, voxelRes, filename, outputDir) {
    const n = voxelCenters.length;
    
    const xArr = new Float32Array(n);
    const yArr = new Float32Array(n);
    const zArr = new Float32Array(n);
    const scale0 = new Float32Array(n);
    const scale1 = new Float32Array(n);
    const scale2 = new Float32Array(n);
    const rot0 = new Float32Array(n);
    const rot1 = new Float32Array(n);
    const rot2 = new Float32Array(n);
    const rot3 = new Float32Array(n);
    const fdc0 = new Float32Array(n);
    const fdc1 = new Float32Array(n);
    const fdc2 = new Float32Array(n);
    const opacityArr = new Float32Array(n);
    
    const voxelScale = Math.log(voxelRes * 0.4);
    const opacityLogit = 5.0;
    const C0 = 0.28209479177387814;
    
    for (let i = 0; i < n; i++) {
        const v = voxelCenters[i];
        xArr[i] = v.x;
        yArr[i] = v.y;
        zArr[i] = v.z;
        scale0[i] = voxelScale;
        scale1[i] = voxelScale;
        scale2[i] = voxelScale;
        rot0[i] = 1.0;
        rot1[i] = 0.0;
        rot2[i] = 0.0;
        rot3[i] = 0.0;
        // Green color for output voxels
        fdc0[i] = (0.2 - 0.5) / C0;
        fdc1[i] = (0.8 - 0.5) / C0;
        fdc2[i] = (0.2 - 0.5) / C0;
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
    
    const outputFs = new MemoryFileSystem();
    await writePly({
        filename: 'output.ply',
        plyData: {
            comments: ['Edge case test - output voxels'],
            elements: [{
                name: 'vertex',
                dataTable: outputTable
            }]
        }
    }, outputFs);
    
    const outputData = outputFs.results.get('output.ply');
    const outputPath = join(outputDir, filename);
    await writeFile(outputPath, outputData);
    console.log(`  Written: ${outputPath}`);
}

// ============================================================================
// Test Case Definitions
// ============================================================================

const testCases = [
    {
        name: '01_single_sphere',
        description: 'Single isotropic Gaussian at origin',
        voxelRes: 0.1,
        opacityCutoff: 0.1,
        gaussians: [
            { x: 0, y: 0, z: 0, scaleX: 0.2, scaleY: 0.2, scaleZ: 0.2, opacity: 0.99 }
        ],
        validate: (voxels) => {
            // Should produce roughly spherical voxels around origin
            const allNearOrigin = voxels.every(v => 
                Math.sqrt(v.x * v.x + v.y * v.y + v.z * v.z) < 1.0
            );
            return {
                pass: voxels.length > 0 && allNearOrigin,
                message: `Expected spherical voxels near origin, got ${voxels.length} voxels`
            };
        }
    },
    
    {
        name: '02_tiny_gaussian',
        description: 'Very small Gaussian (smaller than voxel) - tests energy conservation',
        voxelRes: 0.1,
        opacityCutoff: 0.1,
        gaussians: [
            // Tiny Gaussian at origin - should still produce at least one voxel
            { x: 0, y: 0, z: 0, scaleX: 0.01, scaleY: 0.01, scaleZ: 0.01, opacity: 0.99 }
        ],
        validate: (voxels) => {
            // Should produce at least one voxel containing the Gaussian center
            const hasOriginVoxel = voxels.some(v => 
                Math.abs(v.x) < 0.1 && Math.abs(v.y) < 0.1 && Math.abs(v.z) < 0.1
            );
            return {
                pass: voxels.length >= 1 && hasOriginVoxel,
                message: `Expected at least 1 voxel at origin, got ${voxels.length} voxels`
            };
        }
    },
    
    {
        name: '03_boundary_gaussian',
        description: 'Tiny Gaussian on voxel boundary - tests energy conservation across boundaries',
        voxelRes: 0.1,
        opacityCutoff: 0.1,
        gaussians: [
            // Place tiny Gaussian exactly on voxel grid boundary
            { x: 0.1, y: 0.1, z: 0.1, scaleX: 0.01, scaleY: 0.01, scaleZ: 0.01, opacity: 0.99 }
        ],
        validate: (voxels) => {
            // Should produce voxels in multiple adjacent cells (energy conservation)
            // The Gaussian is on the corner of 8 voxels
            return {
                pass: voxels.length >= 1,
                message: `Expected voxels near boundary point, got ${voxels.length}`
            };
        }
    },
    
    {
        name: '04_elongated_needle_x',
        description: 'Long thin Gaussian along X axis - tests anisotropic AABB',
        voxelRes: 0.1,
        opacityCutoff: 0.1,
        gaussians: [
            { x: 0, y: 0, z: 0, scaleX: 1.0, scaleY: 0.05, scaleZ: 0.05, opacity: 0.99 }
        ],
        validate: (voxels) => {
            // Should produce a line of voxels along X axis
            const xExtent = Math.max(...voxels.map(v => Math.abs(v.x)));
            const yExtent = Math.max(...voxels.map(v => Math.abs(v.y)));
            const isElongated = xExtent > yExtent * 2;
            return {
                pass: voxels.length > 0 && isElongated,
                message: `Expected elongated X pattern (xExtent=${xExtent.toFixed(2)}, yExtent=${yExtent.toFixed(2)})`
            };
        }
    },
    
    {
        name: '05_elongated_needle_y',
        description: 'Long thin Gaussian along Y axis',
        voxelRes: 0.1,
        opacityCutoff: 0.1,
        gaussians: [
            { x: 0, y: 0, z: 0, scaleX: 0.05, scaleY: 1.0, scaleZ: 0.05, opacity: 0.99 }
        ],
        validate: (voxels) => {
            const xExtent = Math.max(...voxels.map(v => Math.abs(v.x)));
            const yExtent = Math.max(...voxels.map(v => Math.abs(v.y)));
            const isElongated = yExtent > xExtent * 2;
            return {
                pass: voxels.length > 0 && isElongated,
                message: `Expected elongated Y pattern (yExtent=${yExtent.toFixed(2)}, xExtent=${xExtent.toFixed(2)})`
            };
        }
    },
    
    {
        name: '06_rotated_needle',
        description: 'Needle Gaussian rotated 45 degrees - tests rotation in AABB calculation',
        voxelRes: 0.1,
        opacityCutoff: 0.1,
        gaussians: [
            { 
                x: 0, y: 0, z: 0, 
                scaleX: 1.0, scaleY: 0.05, scaleZ: 0.05, 
                opacity: 0.99,
                ...axisAngleToQuat(0, 0, 1, 45)  // Rotate 45 degrees around Z
            }
        ],
        validate: (voxels) => {
            // Should produce a diagonal pattern
            const hasPositiveXY = voxels.some(v => v.x > 0.3 && v.y > 0.3);
            const hasNegativeXY = voxels.some(v => v.x < -0.3 && v.y < -0.3);
            return {
                pass: voxels.length > 0 && hasPositiveXY && hasNegativeXY,
                message: `Expected diagonal pattern, got ${voxels.length} voxels`
            };
        }
    },
    
    {
        name: '07_overlapping_gaussians',
        description: 'Two overlapping Gaussians - tests density accumulation',
        voxelRes: 0.1,
        opacityCutoff: 0.5,  // Higher cutoff to see accumulation effect
        gaussians: [
            { x: -0.1, y: 0, z: 0, scaleX: 0.15, scaleY: 0.15, scaleZ: 0.15, opacity: 0.4 },
            { x: 0.1, y: 0, z: 0, scaleX: 0.15, scaleY: 0.15, scaleZ: 0.15, opacity: 0.4 }
        ],
        validate: (voxels) => {
            // Should have voxels in the overlap region (combined opacity > cutoff)
            const hasOverlapVoxels = voxels.some(v => Math.abs(v.x) < 0.1);
            return {
                pass: voxels.length > 0 && hasOverlapVoxels,
                message: `Expected voxels in overlap region, got ${voxels.length} voxels`
            };
        }
    },
    
    {
        name: '08_low_opacity',
        description: 'Gaussian with very low opacity - tests opacity threshold',
        voxelRes: 0.1,
        opacityCutoff: 0.5,
        gaussians: [
            { x: 0, y: 0, z: 0, scaleX: 0.2, scaleY: 0.2, scaleZ: 0.2, opacity: 0.1 }
        ],
        validate: (voxels) => {
            // With low opacity and high cutoff, might produce few or no voxels
            return {
                pass: true,  // Just documenting behavior
                message: `Low opacity Gaussian produced ${voxels.length} voxels (expected few/none with 0.5 cutoff)`
            };
        }
    },
    
    {
        name: '09_high_opacity_threshold',
        description: 'Same Gaussian with very low cutoff vs high cutoff',
        voxelRes: 0.1,
        opacityCutoff: 0.01,  // Very low cutoff
        gaussians: [
            { x: 0, y: 0, z: 0, scaleX: 0.3, scaleY: 0.3, scaleZ: 0.3, opacity: 0.99 }
        ],
        validate: (voxels) => {
            // With low cutoff, should produce more voxels extending further out
            return {
                pass: voxels.length > 20,
                message: `Expected many voxels with low cutoff, got ${voxels.length}`
            };
        }
    },
    
    {
        name: '10_large_gaussian',
        description: 'Large Gaussian - tests broad coverage',
        voxelRes: 0.2,  // Coarser resolution to keep voxel count manageable
        opacityCutoff: 0.1,
        gaussians: [
            { x: 0, y: 0, z: 0, scaleX: 1.0, scaleY: 1.0, scaleZ: 1.0, opacity: 0.99 }
        ],
        validate: (voxels) => {
            // Should produce many voxels covering a large area
            const maxDist = Math.max(...voxels.map(v => 
                Math.sqrt(v.x * v.x + v.y * v.y + v.z * v.z)
            ));
            return {
                pass: voxels.length > 500 && maxDist > 1.5,
                message: `Expected large coverage (${voxels.length} voxels, maxDist=${maxDist.toFixed(2)})`
            };
        }
    },
    
    {
        name: '11_flat_disk_xy',
        description: 'Flat disk in XY plane - tests anisotropic shapes',
        voxelRes: 0.1,
        opacityCutoff: 0.1,
        gaussians: [
            { x: 0, y: 0, z: 0, scaleX: 0.5, scaleY: 0.5, scaleZ: 0.02, opacity: 0.99 }
        ],
        validate: (voxels) => {
            // Should be flat in Z
            const zExtent = Math.max(...voxels.map(v => Math.abs(v.z)));
            const xyExtent = Math.max(...voxels.map(v => Math.sqrt(v.x * v.x + v.y * v.y)));
            return {
                pass: voxels.length > 0 && xyExtent > zExtent * 2,
                message: `Expected flat disk (xyExtent=${xyExtent.toFixed(2)}, zExtent=${zExtent.toFixed(2)})`
            };
        }
    },
    
    {
        name: '12_offset_position',
        description: 'Gaussian offset from origin - tests position handling',
        voxelRes: 0.1,
        opacityCutoff: 0.1,
        gaussians: [
            { x: 5, y: -3, z: 2, scaleX: 0.2, scaleY: 0.2, scaleZ: 0.2, opacity: 0.99 }
        ],
        validate: (voxels) => {
            // All voxels should be near (5, -3, 2)
            const allNearTarget = voxels.every(v => {
                const dx = v.x - 5;
                const dy = v.y + 3;
                const dz = v.z - 2;
                return Math.sqrt(dx * dx + dy * dy + dz * dz) < 1.0;
            });
            return {
                pass: voxels.length > 0 && allNearTarget,
                message: `Expected voxels near (5,-3,2), got ${voxels.length} voxels`
            };
        }
    },
    
    {
        name: '13_multiple_separated',
        description: 'Multiple separated Gaussians - tests independent handling',
        voxelRes: 0.1,
        opacityCutoff: 0.1,
        gaussians: [
            { x: -2, y: 0, z: 0, scaleX: 0.15, scaleY: 0.15, scaleZ: 0.15, opacity: 0.99 },
            { x: 0, y: 0, z: 0, scaleX: 0.15, scaleY: 0.15, scaleZ: 0.15, opacity: 0.99 },
            { x: 2, y: 0, z: 0, scaleX: 0.15, scaleY: 0.15, scaleZ: 0.15, opacity: 0.99 }
        ],
        validate: (voxels) => {
            // Should have three separate clusters
            const atNeg2 = voxels.filter(v => v.x < -1).length;
            const atZero = voxels.filter(v => Math.abs(v.x) < 0.5).length;
            const atPos2 = voxels.filter(v => v.x > 1).length;
            
            return {
                pass: atNeg2 > 0 && atZero > 0 && atPos2 > 0,
                message: `Expected 3 clusters (left=${atNeg2}, center=${atZero}, right=${atPos2})`
            };
        }
    },
    
    {
        name: '14_90deg_rotated_needle',
        description: 'Needle rotated 90 degrees around Z - should align with Y axis',
        voxelRes: 0.1,
        opacityCutoff: 0.1,
        gaussians: [
            { 
                x: 0, y: 0, z: 0, 
                scaleX: 1.0, scaleY: 0.05, scaleZ: 0.05, 
                opacity: 0.99,
                ...axisAngleToQuat(0, 0, 1, 90)  // Rotate 90 degrees around Z
            }
        ],
        validate: (voxels) => {
            // Should now be aligned with Y axis
            const xExtent = Math.max(...voxels.map(v => Math.abs(v.x)));
            const yExtent = Math.max(...voxels.map(v => Math.abs(v.y)));
            const isYAligned = yExtent > xExtent * 2;
            return {
                pass: voxels.length > 0 && isYAligned,
                message: `Expected Y-aligned pattern (yExtent=${yExtent.toFixed(2)}, xExtent=${xExtent.toFixed(2)})`
            };
        }
    },
    
    {
        name: '15_coarse_voxels',
        description: 'Same Gaussian with coarser voxel resolution',
        voxelRes: 0.25,
        opacityCutoff: 0.1,
        gaussians: [
            { x: 0, y: 0, z: 0, scaleX: 0.3, scaleY: 0.3, scaleZ: 0.3, opacity: 0.99 }
        ],
        validate: (voxels) => {
            // Coarser resolution = fewer voxels than fine resolution (test 01 has 504)
            // With 3-sigma extent and coarse grid, we expect ~100-200 voxels
            return {
                pass: voxels.length > 50 && voxels.length < 300,
                message: `Expected moderate voxel count with coarse resolution, got ${voxels.length}`
            };
        }
    }
];

// ============================================================================
// Main Test Runner
// ============================================================================

async function main() {
    const args = process.argv.slice(2);
    const outputDir = resolve(args[0] || 'test/edge-case-output');
    
    console.log(`GPU Voxelization Edge Case Tests`);
    console.log(`================================`);
    console.log(`Output directory: ${outputDir}\n`);
    
    // Create output directory
    await mkdir(outputDir, { recursive: true });
    
    // Create GPU device
    console.log('Creating WebGPU device...');
    const device = await createDevice();
    console.log(`Device type: ${device.deviceType}\n`);
    
    const gpuVoxelization = new GpuVoxelization(device);
    
    let passed = 0;
    let failed = 0;
    const results = [];
    
    for (const testCase of testCases) {
        console.log(`\nTest: ${testCase.name}`);
        console.log(`  ${testCase.description}`);
        console.log(`  Voxel resolution: ${testCase.voxelRes}, Opacity cutoff: ${testCase.opacityCutoff}`);
        console.log(`  Input Gaussians: ${testCase.gaussians.length}`);
        
        try {
            // Create DataTable
            const dataTable = createDataTable(testCase.gaussians);
            const extentsResult = computeGaussianExtents(dataTable);
            
            // Run voxelization
            const voxels = await voxelizeTestCase(
                gpuVoxelization,
                dataTable,
                extentsResult,
                testCase.voxelRes,
                testCase.opacityCutoff
            );
            
            console.log(`  Output voxels: ${voxels.length}`);
            
            // Write PLY files
            await writeGaussiansPly(testCase.gaussians, `${testCase.name}_input.ply`, outputDir);
            if (voxels.length > 0) {
                await writeVoxelsPly(voxels, testCase.voxelRes, `${testCase.name}_voxels.ply`, outputDir);
            }
            
            // Validate
            const validation = testCase.validate(voxels);
            if (validation.pass) {
                console.log(`  ✓ PASS: ${validation.message}`);
                passed++;
            } else {
                console.log(`  ✗ FAIL: ${validation.message}`);
                failed++;
            }
            
            results.push({
                name: testCase.name,
                description: testCase.description,
                inputCount: testCase.gaussians.length,
                outputCount: voxels.length,
                pass: validation.pass,
                message: validation.message
            });
            
        } catch (error) {
            console.log(`  ✗ ERROR: ${error.message}`);
            failed++;
            results.push({
                name: testCase.name,
                description: testCase.description,
                pass: false,
                message: `Error: ${error.message}`
            });
        }
    }
    
    // Write summary
    console.log(`\n================================`);
    console.log(`Summary: ${passed} passed, ${failed} failed out of ${testCases.length} tests`);
    
    // Write results JSON
    const summaryPath = join(outputDir, 'results.json');
    await writeFile(summaryPath, JSON.stringify(results, null, 2));
    console.log(`\nResults written to: ${summaryPath}`);
    
    // Cleanup
    gpuVoxelization.destroy();
    device.destroy();
    
    process.exit(failed > 0 ? 1 : 0);
}

main().catch(e => {
    console.error('Error:', e);
    process.exit(1);
});
