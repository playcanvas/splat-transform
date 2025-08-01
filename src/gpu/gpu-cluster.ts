import {
    BUFFERUSAGE_COPY_DST,
    BUFFERUSAGE_COPY_SRC,
    SHADERLANGUAGE_WGSL,
    SHADERSTAGE_COMPUTE,
    UNIFORMTYPE_UINT,
    BindGroupFormat,
    BindStorageBufferFormat,
    BindUniformBufferFormat,
    Compute,
    FloatPacking,
    Shader,
    StorageBuffer,
    UniformBufferFormat,
    UniformFormat
} from 'playcanvas/debug';

import { DataTable } from '../data-table.js';
import { GpuDevice } from './gpu-device.js';

const clusterWgsl = (numColumns: number, useF16: boolean) => {
    const floatType = useF16 ? 'f16' : 'f32';

    return /* wgsl */ `
${useF16 ? 'enable f16;' : ''}

struct Uniforms {
    numPoints: u32,
    numCentroids: u32,
    pointBase: u32
};

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var<storage, read> points: array<${floatType}>;
@group(0) @binding(2) var<storage, read> centroids: array<${floatType}>;
@group(0) @binding(3) var<storage, read_write> results: array<u32>;

const numColumns = ${numColumns};   // number of columns in the points and centroids tables
const chunkSize = 128u;             // must be a multiple of 64
var<workgroup> sharedChunk: array<${floatType}, numColumns * chunkSize>;

// calculate the squared distance between the point and centroid
fn calcDistanceSqr(point: array<${floatType}, numColumns>, centroid: u32) -> f32 {
    var result = 0.0;

    var ci = centroid * numColumns;

    for (var i = 0u; i < numColumns; i++) {
        let v = f32(point[i] - sharedChunk[ci+i]);
        result += v * v;
    }

    return result;
}

@compute @workgroup_size(64)
fn main(
    @builtin(local_invocation_index) local_id : u32,
    @builtin(global_invocation_id) global_id: vec3u,
    @builtin(num_workgroups) num_workgroups: vec3u
) {
    // calculate row index for this thread point
    let pointIndex = uniforms.pointBase + global_id.x + global_id.y * num_workgroups.x * 64u;

    // copy the point data from global memory
    var point: array<${floatType}, numColumns>;
    if (pointIndex < uniforms.numPoints) {
        for (var i = 0u; i < numColumns; i++) {
            point[i] = points[pointIndex * numColumns + i];
        }
    }

    var mind = 1000000.0;
    var mini = 0u;

    // work through the list of centroids in shared memory chunks
    let numChunks = u32(ceil(f32(uniforms.numCentroids) / f32(chunkSize)));
    for (var i = 0u; i < numChunks; i++) {

        // copy this thread's slice of the centroid shared chunk data
        let dstRow = local_id * (chunkSize / 64u);
        let srcRow = min(uniforms.numCentroids, i * chunkSize + local_id * chunkSize / 64u);
        let numRows = min(uniforms.numCentroids, srcRow + chunkSize / 64u) - srcRow;

        var dst = dstRow * numColumns;
        var src = srcRow * numColumns;

        for (var c = 0u; c < numRows * numColumns; c++) {
            sharedChunk[dst + c] = centroids[src + c];
        }

        // wait for all threads to finish writing their part of centroids shared memory buffer
        workgroupBarrier();

        // loop over the next chunk of centroids finding the closest
        if (pointIndex < uniforms.numPoints) {
            let thisChunkSize = min(chunkSize, uniforms.numCentroids - i * chunkSize);
            for (var c = 0u; c < thisChunkSize; c++) {
                let d = calcDistanceSqr(point, c);
                if (d < mind) {
                    mind = d;
                    mini = i * chunkSize + c;
                }
            }
        }

        // next loop will overwrite the shared memory, so wait
        workgroupBarrier();
    }

    if (pointIndex < uniforms.numPoints) {
        results[pointIndex] = mini;
    }
}
`;
};

const roundUp = (value: number, multiple: number) => {
    return Math.ceil(value / multiple) * multiple;
};

const interleaveData = (dataTable: DataTable, useF16: boolean) => {
    const { numRows, numColumns } = dataTable;
    if (useF16) {
        const result = new Uint16Array(roundUp(numColumns * numRows, 2));
        for (let c = 0; c < numColumns; ++c) {
            const column = dataTable.columns[c];
            for (let r = 0; r < numRows; ++r) {
                result[r * numColumns + c] = FloatPacking.float2Half(column.data[r]);
            }
        }
        return result;
    }

    const result = new Float32Array(numColumns * numRows);
    for (let c = 0; c < numColumns; ++c) {
        const column = dataTable.columns[c];
        for (let r = 0; r < numRows; ++r) {
            result[r * numColumns + c] = column.data[r];
        }
    }
    return result;
};

class GpuCluster {
    execute: (centroids: DataTable, labels: Uint32Array) => Promise<void>;
    destroy: () => void;

    constructor(gpuDevice: GpuDevice, points: DataTable, numCentroids: number) {
        const device = gpuDevice.app.graphicsDevice;

        // Check if device supports f16
        const useF16 = 'supportsShaderF16' in device && device.supportsShaderF16 as boolean;
        const bytesPerFloat = useF16 ? 2 : 4;

        const numPoints = points.numRows;
        const numColumns = points.numColumns;

        const bindGroupFormat = new BindGroupFormat(device, [
            new BindUniformBufferFormat('uniforms', SHADERSTAGE_COMPUTE),
            new BindStorageBufferFormat('pointsBuffer', SHADERSTAGE_COMPUTE, true),
            new BindStorageBufferFormat('centroidsBuffer', SHADERSTAGE_COMPUTE, true),
            new BindStorageBufferFormat('resultsBuffer', SHADERSTAGE_COMPUTE)
        ]);

        const shader = new Shader(device, {
            name: 'compute-cluster',
            shaderLanguage: SHADERLANGUAGE_WGSL,
            cshader: clusterWgsl(numColumns, useF16),
            // @ts-ignore
            computeUniformBufferFormats: {
                uniforms: new UniformBufferFormat(device, [
                    new UniformFormat('numPoints', UNIFORMTYPE_UINT),
                    new UniformFormat('numCentroids', UNIFORMTYPE_UINT),
                    new UniformFormat('pointBase', UNIFORMTYPE_UINT)
                ])
            },
            // @ts-ignore
            computeBindGroupFormat: bindGroupFormat
        });

        const compute = new Compute(device, shader, 'compute-cluster');

        const pointsBuffer = new StorageBuffer(
            device,
            useF16 ? roundUp(numColumns * numPoints, 2) * 2 : numColumns * numPoints * 4,
            BUFFERUSAGE_COPY_DST
        );

        const centroidsBuffer = new StorageBuffer(
            device,
            numColumns * numCentroids * bytesPerFloat,
            BUFFERUSAGE_COPY_DST
        );

        const resultsBuffer = new StorageBuffer(
            device,
            numPoints * 4,
            BUFFERUSAGE_COPY_SRC | BUFFERUSAGE_COPY_DST
        );

        // interleave the points table data and write to gpu
        const interleavedPoints = interleaveData(points, useF16);
        pointsBuffer.write(0, interleavedPoints, 0, interleavedPoints.length);

        compute.setParameter('pointsBuffer', pointsBuffer);
        compute.setParameter('centroidsBuffer', centroidsBuffer);
        compute.setParameter('resultsBuffer', resultsBuffer);

        this.execute = async (centroids: DataTable, labels: Uint32Array) => {
            // interleave centroids and write to gpu
            const interleavedCentroids = interleaveData(centroids, useF16);
            centroidsBuffer.write(0, interleavedCentroids, 0, interleavedCentroids.length);

            compute.setParameter('numPoints', points.numRows);
            compute.setParameter('numCentroids', centroids.numRows);

            // execute in batches of 1024 worksgroups
            const workgroupSize = 64;
            const workgroupsPerBatch = 1024;
            const batchSize = workgroupsPerBatch * workgroupSize;
            const numBatches = Math.ceil(numPoints / batchSize);

            for (let batch = 0; batch < numBatches; batch++) {
                const currentBatchSize = Math.min(numPoints - batch * batchSize, batchSize);
                const groups = Math.ceil(currentBatchSize / 64);

                compute.setParameter('pointBase', batch * batchSize);

                // start compute job
                compute.setupDispatch(groups);
                device.computeDispatch([compute], `cluster-dispatch-${batch}`);

                // FIXME: submit call is required, but not public API
                // @ts-ignore
                device.submit();
            }

            // read results from gpu
            await resultsBuffer.read(0, undefined, labels, true);
        };

        this.destroy = () => {
            pointsBuffer.destroy();
            centroidsBuffer.destroy();
            resultsBuffer.destroy();
            shader.destroy();
            bindGroupFormat.destroy();
        };
    }
}

export { GpuCluster };
