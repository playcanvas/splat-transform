import {
    BUFFERUSAGE_COPY_DST,
    BUFFERUSAGE_COPY_SRC,
    SHADERLANGUAGE_WGSL,
    SHADERSTAGE_COMPUTE,
    UNIFORMTYPE_FLOAT,
    UNIFORMTYPE_UINT,
    BindGroupFormat,
    BindStorageBufferFormat,
    BindUniformBufferFormat,
    Compute,
    GraphicsDevice,
    Shader,
    StorageBuffer,
    UniformBufferFormat,
    UniformFormat
} from 'playcanvas';

import { type RenderCamera, buildCameraBasis } from './camera';
import { preprocess, RECORD_STRIDE_F32 } from './preprocess';
import { rasterizeWgsl, TILE_SIZE } from './rasterize-shader';
import { DataTable } from '../data-table';

/**
 * Rasterize a Gaussian splat scene through a pinhole camera and return an
 * RGBA byte buffer in row-major order (length = width * height * 4).
 *
 * The function:
 *   1. Builds the camera basis on the CPU.
 *   2. Projects, EWA-fits, SH-evaluates, tile-bins and depth-sorts every
 *      splat on the CPU.
 *   3. Uploads the sorted per-tile records and per-tile range table to the
 *      GPU.
 *   4. Dispatches a single compute shader that runs one workgroup per tile
 *      and one thread per pixel, alpha-blending front-to-back into a
 *      packed-RGBA u32 storage buffer.
 *   5. Reads the buffer back to CPU.
 *
 * The shader stage is the only GPU work; everything else is plain TS so it
 * can be stepped through and validated against a reference.
 *
 * @param device - PlayCanvas WebGPU graphics device.
 * @param dataTable - Gaussian splat data with the standard columns.
 * @param camera - Camera parameters (position, target, up, fovY, width, height, near).
 * @param background - RGBA background composited under the running transmittance.
 * @param background.r - Red channel in [0, 1].
 * @param background.g - Green channel in [0, 1].
 * @param background.b - Blue channel in [0, 1].
 * @param background.a - Alpha channel in [0, 1].
 * @returns RGBA byte array of length width*height*4.
 */
const renderSplats = async (
    device: GraphicsDevice,
    dataTable: DataTable,
    camera: RenderCamera,
    background: { r: number; g: number; b: number; a: number }
): Promise<Uint8Array> => {
    if (!Number.isInteger(camera.width) || !Number.isInteger(camera.height) || camera.width <= 0 || camera.height <= 0) {
        throw new Error(`Invalid resolution: ${camera.width}x${camera.height}`);
    }

    const basis = buildCameraBasis(camera);

    const projected = preprocess(dataTable, basis, camera.width, camera.height, camera.near);

    const { records, tileRanges, numRecords, tilesX, tilesY } = projected;

    const width = camera.width;
    const height = camera.height;
    const numPixels = width * height;
    const numTiles = tilesX * tilesY;

    // Build resources.
    const bindGroupFormat = new BindGroupFormat(device, [
        new BindUniformBufferFormat('uniforms', SHADERSTAGE_COMPUTE),
        new BindStorageBufferFormat('splats', SHADERSTAGE_COMPUTE, true),
        new BindStorageBufferFormat('tileRanges', SHADERSTAGE_COMPUTE, true),
        new BindStorageBufferFormat('output', SHADERSTAGE_COMPUTE)
    ]);

    const shader = new Shader(device, {
        name: 'splat-rasterize',
        shaderLanguage: SHADERLANGUAGE_WGSL,
        cshader: rasterizeWgsl(),
        // @ts-ignore - computeUniformBufferFormats / computeBindGroupFormat are not in the public Shader options type yet.
        computeUniformBufferFormats: {
            uniforms: new UniformBufferFormat(device, [
                new UniformFormat('width', UNIFORMTYPE_UINT),
                new UniformFormat('height', UNIFORMTYPE_UINT),
                new UniformFormat('tilesX', UNIFORMTYPE_UINT),
                new UniformFormat('tilesY', UNIFORMTYPE_UINT),
                new UniformFormat('bgR', UNIFORMTYPE_FLOAT),
                new UniformFormat('bgG', UNIFORMTYPE_FLOAT),
                new UniformFormat('bgB', UNIFORMTYPE_FLOAT),
                new UniformFormat('bgA', UNIFORMTYPE_FLOAT)
            ])
        },
        // @ts-ignore - see above
        computeBindGroupFormat: bindGroupFormat
    });

    // Splats storage buffer must be non-zero in size even when empty.
    const splatsBytes = Math.max(RECORD_STRIDE_F32 * 4, numRecords * RECORD_STRIDE_F32 * 4);
    const splatsBuffer = new StorageBuffer(device, splatsBytes, BUFFERUSAGE_COPY_DST);
    if (numRecords > 0) {
        splatsBuffer.write(0, records, 0, numRecords * RECORD_STRIDE_F32);
    }

    const tileRangesBuffer = new StorageBuffer(device, numTiles * 2 * 4, BUFFERUSAGE_COPY_DST);
    tileRangesBuffer.write(0, tileRanges, 0, numTiles * 2);

    const outputBuffer = new StorageBuffer(device, numPixels * 4, BUFFERUSAGE_COPY_SRC);

    const compute = new Compute(device, shader, 'splat-rasterize');
    compute.setParameter('splats', splatsBuffer);
    compute.setParameter('tileRanges', tileRangesBuffer);
    compute.setParameter('output', outputBuffer);
    compute.setParameter('width', width);
    compute.setParameter('height', height);
    compute.setParameter('tilesX', tilesX);
    compute.setParameter('tilesY', tilesY);
    compute.setParameter('bgR', background.r);
    compute.setParameter('bgG', background.g);
    compute.setParameter('bgB', background.b);
    compute.setParameter('bgA', background.a);

    compute.setupDispatch(tilesX, tilesY, 1);
    device.computeDispatch([compute], 'splat-rasterize');

    const bytes = await outputBuffer.read(0, numPixels * 4, null, true) as Uint8Array;

    // Make an owned copy so the caller can keep the buffer alive past the
    // device's lifetime.
    const result = new Uint8Array(numPixels * 4);
    result.set(bytes);

    splatsBuffer.destroy();
    tileRangesBuffer.destroy();
    outputBuffer.destroy();
    shader.destroy();
    bindGroupFormat.destroy();

    return result;
};

export { renderSplats, TILE_SIZE };
