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
    numCentroids: u32
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
    let pointIndex = global_id.x + global_id.y * num_workgroups.x * 64u;

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

const interleaveData = (result: Uint16Array | Float32Array, dataTable: DataTable, numRows: number, rowOffset: number) => {
    const { numColumns } = dataTable;

    if (result instanceof Uint16Array) {
        // interleave shorts
        for (let c = 0; c < numColumns; ++c) {
            const column = dataTable.columns[c];
            for (let r = 0; r < numRows; ++r) {
                result[r * numColumns + c] = FloatPacking.float2Half(column.data[rowOffset + r]);
            }
        }
    } else {
        // interleave floats
        for (let c = 0; c < numColumns; ++c) {
            const column = dataTable.columns[c];
            for (let r = 0; r < numRows; ++r) {
                result[r * numColumns + c] = column.data[rowOffset + r];
            }
        }
    }
};

class GpuClustering {
    execute: (points: DataTable, centroids: DataTable, labels: Uint32Array) => Promise<void>;
    destroy: () => void;

    constructor(gpuDevice: GpuDevice, numColumns: number, numCentroids: number) {
        const device = gpuDevice.app.graphicsDevice;

        // Check if device supports f16
        const useF16 = !!('supportsShaderF16' in device && device.supportsShaderF16);

        const workgroupSize = 64;
        const workgroupsPerBatch = 1024;
        const batchSize = workgroupsPerBatch * workgroupSize;

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
                    new UniformFormat('numCentroids', UNIFORMTYPE_UINT)
                ])
            },
            // @ts-ignore
            computeBindGroupFormat: bindGroupFormat
        });

        const interleavedPoints = useF16 ? new Uint16Array(roundUp(numColumns * batchSize, 2)) : new Float32Array(numColumns * batchSize);
        const interleavedCentroids = useF16 ? new Uint16Array(roundUp(numColumns * numCentroids, 2)) : new Float32Array(numColumns * numCentroids);
        const resultsData = new Uint32Array(batchSize);

        const pointsBuffer = new StorageBuffer(
            device,
            interleavedPoints.byteLength,
            BUFFERUSAGE_COPY_DST
        );

        const centroidsBuffer = new StorageBuffer(
            device,
            interleavedCentroids.byteLength,
            BUFFERUSAGE_COPY_DST
        );

        const resultsBuffer = new StorageBuffer(
            device,
            resultsData.byteLength,
            BUFFERUSAGE_COPY_SRC | BUFFERUSAGE_COPY_DST
        );

        const compute = new Compute(device, shader, 'compute-cluster');
        compute.setParameter('pointsBuffer', pointsBuffer);
        compute.setParameter('centroidsBuffer', centroidsBuffer);
        compute.setParameter('resultsBuffer', resultsBuffer);

        this.execute = async (points: DataTable, centroids: DataTable, labels: Uint32Array) => {
            const numPoints = points.numRows;
            const numBatches = Math.ceil(numPoints / batchSize);

            // upload centroid data to gpu
            interleaveData(interleavedCentroids, centroids, numCentroids, 0);
            centroidsBuffer.write(0, interleavedCentroids, 0, interleavedCentroids.length);
            compute.setParameter('numCentroids', numCentroids);

            for (let batch = 0; batch < numBatches; batch++) {
                const currentBatchSize = Math.min(numPoints - batch * batchSize, batchSize);
                const groups = Math.ceil(currentBatchSize / 64);

                // write this batch of point data to gpu
                interleaveData(interleavedPoints, points, currentBatchSize, batch * batchSize);
                pointsBuffer.write(0, interleavedPoints, 0, useF16 ? roundUp(numColumns * currentBatchSize, 2) : numColumns * currentBatchSize);
                compute.setParameter('numPoints', currentBatchSize);

                // start compute job
                compute.setupDispatch(groups);
                device.computeDispatch([compute], `cluster-dispatch-${batch}`);

                // read results from gpu and store in labels
                await resultsBuffer.read(0, currentBatchSize * 4, resultsData, true);
                labels.set(resultsData.subarray(0, currentBatchSize), batch * batchSize);
            }
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

export { GpuClustering };
