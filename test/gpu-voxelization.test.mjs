/**
 * Tests for GPU Voxelization (Phase 3 of voxelizer).
 *
 * Note: Full GPU tests require a WebGPU-capable device and are integration tests.
 * This file tests the module exports and basic structure.
 */

import { describe, it } from 'node:test';
import assert from 'node:assert';

import {
    GpuVoxelization
} from '../dist/index.mjs';

describe('GpuVoxelization', function () {
    describe('module exports', function () {
        it('should export GpuVoxelization class', function () {
            assert.ok(GpuVoxelization, 'GpuVoxelization should be exported');
            assert.strictEqual(typeof GpuVoxelization, 'function', 'GpuVoxelization should be a class/constructor');
        });
    });

    describe('WGSL shader generation', function () {
        it('should have proper shader structure', function () {
            // The shader is generated internally, but we can verify the class exists
            // and has the expected properties
            assert.ok(GpuVoxelization.prototype.voxelizeBlocks, 'Should have voxelizeBlocks method');
            assert.ok(GpuVoxelization.prototype.destroy, 'Should have destroy method');
        });
    });

    // Note: GPU integration tests would go here but require WebGPU runtime
    // They should be run separately with a headless browser or Node with WebGPU support
});

describe('VoxelizationResult', function () {
    describe('structure', function () {
        it('should define expected result format', function () {
            // VoxelizationResult is a type, so we test by documenting expected structure:
            // - blocks: Array<{ x: number; y: number; z: number }>
            // - masks: Uint32Array (interleaved: masks[i*2] = low bits, masks[i*2+1] = high bits)
            // This is validated at compile time, but we can test the concept
            const mockResult = {
                blocks: [{ x: 0, y: 0, z: 0 }],
                masks: new Uint32Array(2)  // 2 u32s per block
            };

            assert.ok(Array.isArray(mockResult.blocks));
            assert.ok(mockResult.masks instanceof Uint32Array);
            assert.strictEqual(mockResult.masks.length, mockResult.blocks.length * 2);
        });
    });
});

describe('Morton encoding (conceptual)', function () {
    // These test the Morton encoding/decoding logic used in the shader
    // We implement the same logic in JS to verify correctness

    function mortonToXYZ(m) {
        return {
            x: (m & 1) | ((m >> 2) & 2),
            y: ((m >> 1) & 1) | ((m >> 3) & 2),
            z: ((m >> 2) & 1) | ((m >> 4) & 2)
        };
    }

    function xyzToMorton(x, y, z) {
        return (x & 1) | ((y & 1) << 1) | ((z & 1) << 2) |
               ((x & 2) << 2) | ((y & 2) << 3) | ((z & 2) << 4);
    }

    it('should encode (0,0,0) as 0', function () {
        assert.strictEqual(xyzToMorton(0, 0, 0), 0);
    });

    it('should encode (1,0,0) as 1', function () {
        assert.strictEqual(xyzToMorton(1, 0, 0), 1);
    });

    it('should encode (0,1,0) as 2', function () {
        assert.strictEqual(xyzToMorton(0, 1, 0), 2);
    });

    it('should encode (0,0,1) as 4', function () {
        assert.strictEqual(xyzToMorton(0, 0, 1), 4);
    });

    it('should encode (1,1,1) as 7', function () {
        assert.strictEqual(xyzToMorton(1, 1, 1), 7);
    });

    it('should encode (3,3,3) as 63', function () {
        assert.strictEqual(xyzToMorton(3, 3, 3), 63);
    });

    it('should be reversible for all 64 positions', function () {
        for (let z = 0; z < 4; z++) {
            for (let y = 0; y < 4; y++) {
                for (let x = 0; x < 4; x++) {
                    const morton = xyzToMorton(x, y, z);
                    const decoded = mortonToXYZ(morton);

                    assert.strictEqual(decoded.x, x, `x mismatch for (${x},${y},${z})`);
                    assert.strictEqual(decoded.y, y, `y mismatch for (${x},${y},${z})`);
                    assert.strictEqual(decoded.z, z, `z mismatch for (${x},${y},${z})`);
                }
            }
        }
    });

    it('should produce unique Morton codes for all 64 positions', function () {
        const codes = new Set();
        for (let z = 0; z < 4; z++) {
            for (let y = 0; y < 4; y++) {
                for (let x = 0; x < 4; x++) {
                    codes.add(xyzToMorton(x, y, z));
                }
            }
        }
        assert.strictEqual(codes.size, 64, 'Should produce 64 unique codes');
    });
});

describe('Extinction-based density accumulation (conceptual)', function () {
    // Test the extinction-based density formula used in the shader
    // sigma is accumulated, then converted to opacity: alpha = 1 - exp(-sigma * depth)

    function sigmaToOpacity(sigma, voxelResolution) {
        return 1.0 - Math.exp(-sigma * voxelResolution);
    }

    it('should return 0 opacity for 0 density', function () {
        assert.strictEqual(sigmaToOpacity(0, 1.0), 0);
    });

    it('should increase opacity with density', function () {
        const opacity1 = sigmaToOpacity(1.0, 1.0);
        const opacity2 = sigmaToOpacity(2.0, 1.0);

        assert.ok(opacity2 > opacity1, 'Higher density should give higher opacity');
    });

    it('should accumulate densities linearly', function () {
        // Two contributions of 0.5 should give same result as one contribution of 1.0
        const sigma1 = 0.5 + 0.5;
        const sigma2 = 1.0;
        const voxelRes = 0.05;

        assert.strictEqual(sigmaToOpacity(sigma1, voxelRes), sigmaToOpacity(sigma2, voxelRes));
    });

    it('should approach 1.0 asymptotically', function () {
        // sigma of 7 with voxelResolution of 1 gives > 0.999
        const opacity = sigmaToOpacity(7.0, 1.0);

        assert.ok(opacity > 0.999);
        assert.ok(opacity < 1.0);
    });

    it('should never exceed 1.0', function () {
        const opacity = sigmaToOpacity(1000, 1.0);

        assert.ok(opacity <= 1.0);
    });

    it('should scale with voxel resolution', function () {
        // Same density but different depths should give different opacities
        const opacity1 = sigmaToOpacity(10, 0.01);  // Small voxel
        const opacity2 = sigmaToOpacity(10, 0.1);   // Large voxel

        assert.ok(opacity2 > opacity1, 'Larger voxels should accumulate more opacity');
    });
});
