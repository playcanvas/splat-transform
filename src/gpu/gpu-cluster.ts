import {
    BUFFERUSAGE_COPY_DST,
    BUFFERUSAGE_COPY_SRC,
    SHADERLANGUAGE_WGSL,
    SHADERSTAGE_COMPUTE,
    BindGroupFormat,
    BindStorageBufferFormat,
    Compute,
    FloatPacking,
    Shader,
    StorageBuffer
} from 'playcanvas/debug';

import { DataTable } from '../data-table.js';
import { GpuDevice } from './gpu-device.js';

const clusterWgsl = (numColumns: number, numPoints: number, numCentroids: number) => {
    return `

enable f16;

@group(0) @binding(0) var<storage, read> points: array<f16>;
@group(0) @binding(1) var<storage, read> centroids: array<f16>;
@group(0) @binding(2) var<storage, read_write> results: array<u32>;

const numColumns = ${numColumns};
const numPoints = ${numPoints};
const numCentroids = ${numCentroids};

const chunkSize = 128u; // must be a multiple of 64
var<workgroup> sharedChunk: array<f16, numColumns * chunkSize>;

// calculate the squared distance between the point and centroid
fn calcDistanceSqr(point: array<f16, numColumns>, centroid: u32) -> f32 {
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
    let pointIndex = global_id.x + global_id.y * num_workgroups.x * 64;

    // copy the point data from global memory
    var point: array<f16, numColumns>;
    if (pointIndex < numPoints) {
        for (var i = 0u; i < numColumns; i++) {
            point[i] = points[pointIndex * numColumns + i];
        }
    }

    var mind = 1000000.0;
    var mini = 0u;

    // work through the list of centroids in shared memory chunks
    let numChunks = u32(ceil(f32(numCentroids) / f32(chunkSize)));
    for (var i = 0u; i < numChunks; i++) {

        // copy this thread's slice of the centroid shared chunk data
        let dstRow = local_id * (chunkSize / 64u);
        let srcRow = min(numCentroids, i * chunkSize + local_id * chunkSize / 64u);
        let numRows = min(numCentroids, srcRow + chunkSize / 64u) - srcRow;

        var dst = dstRow * numColumns;
        var src = srcRow * numColumns;

        for (var c = 0u; c < numRows * numColumns; c++) {
            sharedChunk[dst + c] = centroids[src + c];
        }

        // wait for all threads to finish writing their part of centroids shared memory buffer
        workgroupBarrier();

        // loop over the next chunk of centroids finding the closest
        if (pointIndex < numPoints) {
            let thisChunkSize = min(chunkSize, numCentroids - i * chunkSize);
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

    if (pointIndex < numPoints) {
        results[pointIndex] = mini;
    }
}
`;
}

const roundUp = (value: number, multiple: number) => {
    return Math.ceil(value / multiple) * multiple;
};

const interleaveData = (dataTable: DataTable) => {
    const { numRows, numColumns } = dataTable;
    const result = new Uint16Array(roundUp(numColumns * numRows, 2));
    for (let c = 0; c < numColumns; ++c) {
        const column = dataTable.columns[c];
        for (let r = 0; r < numRows; ++r) {
            result[r * numColumns + c] = FloatPacking.float2Half(column.data[r]);
        }
    }
    return result;
};

class GpuCluster {
    execute: (centroids: DataTable, labels: Uint32Array) => Promise<void>;
    destroy: () => void;

    constructor(gpuDevice: GpuDevice, points: DataTable, numCentroids: number) {
        const device = gpuDevice.app.graphicsDevice;

        const bindGroupFormat = new BindGroupFormat(device, [
            new BindStorageBufferFormat('pointsBuffer', SHADERSTAGE_COMPUTE, true),
            new BindStorageBufferFormat('centroidsBuffer', SHADERSTAGE_COMPUTE, true),
            new BindStorageBufferFormat('resultsBuffer', SHADERSTAGE_COMPUTE)
        ]);

        const numPoints = points.numRows;
        const numColumns = points.numColumns;

        const shader = new Shader(device, {
            name: 'compute-cluster',
            shaderLanguage: SHADERLANGUAGE_WGSL,
            cshader: clusterWgsl(numColumns, numPoints, numCentroids),
            // @ts-ignore
            computeBindGroupFormat: bindGroupFormat
        });

        const compute = new Compute(device, shader, 'compute-cluster');

        const pointsBuffer = new StorageBuffer(
            device,
            roundUp(numColumns * numPoints, 2) * 2,
            BUFFERUSAGE_COPY_DST
        );

        const centroidsBuffer = new StorageBuffer(
            device,
            numColumns * numCentroids * 2,
            BUFFERUSAGE_COPY_DST
        );

        const resultsBuffer = new StorageBuffer(
            device,
            numPoints * 4,
            BUFFERUSAGE_COPY_SRC | BUFFERUSAGE_COPY_DST
        );

        // interleave the points table data and write to gpu
        const interleavedPoints = interleaveData(points);
        pointsBuffer.write(0, interleavedPoints, 0, interleavedPoints.length);

        compute.setParameter('columns', numColumns);
        compute.setParameter('points', numPoints);
        compute.setParameter('centroids', numCentroids);

        compute.setParameter('pointsBuffer', pointsBuffer);
        compute.setParameter('centroidsBuffer', centroidsBuffer);
        compute.setParameter('resultsBuffer', resultsBuffer);

        this.execute = async (centroids: DataTable, labels: Uint32Array) => {
            // interleave centroids and write to gpu
            const interleavedCentroids = interleaveData(centroids);
            centroidsBuffer.write(0, interleavedCentroids, 0, interleavedCentroids.length);

            // calculate the workgroup layout to minimize the number of empty workgroups
            const groups = Math.ceil(points.numRows / 64);
            const height = Math.ceil(groups / 65536);
            const width = Math.ceil(groups / height);

            // start compute job
            compute.setupDispatch(width, height);
            device.computeDispatch([compute], 'cluster-dispatch');

            // read results from gpu
            await resultsBuffer.read(0, undefined, labels, true);
        }

        this.destroy = () => {
            pointsBuffer.destroy();
            centroidsBuffer.destroy();
            resultsBuffer.destroy();
            shader.destroy();
            bindGroupFormat.destroy();
        }
    }
}

export { GpuCluster };
