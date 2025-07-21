import { Worker } from 'node:worker_threads';
import { create, globals } from 'webgpu';
import {
    BUFFERUSAGE_COPY_DST,
    BUFFERUSAGE_COPY_SRC,
    PIXELFORMAT_BGRA8,
    SHADERLANGUAGE_WGSL,
    SHADERSTAGE_COMPUTE,
    Application,
    BindGroupFormat,
    BindStorageBufferFormat,
    Compute,
    FloatPacking,
    GraphicsDevice,
    Shader,
    StorageBuffer,
    Texture,
    WebgpuGraphicsDevice
} from 'playcanvas/debug';

import { DataTable } from '../data-table.js';

import { JSDOM } from 'jsdom';

Object.assign(globalThis, globals);

const jsdomSetup = () => {
    const html = '<!DOCTYPE html><html><head></head><body></body></html>';

    const jsdom = new JSDOM(html, {
        resources: 'usable',         // Allow the engine to load assets
        runScripts: 'dangerously',   // Allow the engine to run scripts
        url: 'http://localhost:3000' // Set the URL of the document
    });

    // Copy the window and document to global scope
    // @ts-ignore
    global.window = jsdom.window;
    global.document = jsdom.window.document;

    // Copy the DOM APIs used by the engine to global scope
    global.ArrayBuffer = jsdom.window.ArrayBuffer;
    global.Audio = jsdom.window.Audio;
    global.DataView = jsdom.window.DataView;
    global.Image = jsdom.window.Image;
    global.KeyboardEvent = jsdom.window.KeyboardEvent;
    global.MouseEvent = jsdom.window.MouseEvent;
    global.XMLHttpRequest = jsdom.window.XMLHttpRequest;
};

const clusterWgsl = (numPoints: number, numCentroids: number, numColumns: number) => {
    return `
@group(0) @binding(0) var<storage, read> points: array<f32>;
@group(0) @binding(1) var<storage, read> centroids: array<f32>;
@group(0) @binding(2) var<storage, read_write> results: array<u32>;

const numPoints = ${numPoints};
const numCentroids = ${numCentroids};
const numColumns = ${numColumns};

const chunkSize = 128u;
var<workgroup> sharedChunk: array<f32, numColumns * chunkSize>;

// calculate the squared distance between the point and centroid
fn calcDistanceSqr(point: array<f32, numColumns>, centroid: u32) -> f32 {
    var result = 0.0;

    var ci = centroid * numColumns;

    for (var i = 0u; i < numColumns; i++) {
        let v = point[i] - sharedChunk[ci+i];
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
    var point: array<f32, numColumns>;
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

const interleaveData = (dataTable: DataTable) => {
    const { numRows, numColumns } = dataTable;
    const result = new Float32Array(numRows * numColumns);
    for (let c = 0; c < numColumns; ++c) {
        const column = dataTable.columns[c];
        for (let r = 0; r < numRows; ++r) {
            result[r * numColumns + c] = column.data[r]; // FloatPacking.float2Half(column.data[r]);
        }
    }
    return result;
};

class Cluster {
    device: GraphicsDevice;

    compute: (points: number, centroids: number, columns: number) => Compute;

    constructor(device: GraphicsDevice) {
        this.device = device;

        const bindGroupFormat = new BindGroupFormat(device, [
            new BindStorageBufferFormat('pointsBuffer', SHADERSTAGE_COMPUTE, true),
            new BindStorageBufferFormat('centroidsBuffer', SHADERSTAGE_COMPUTE, true),
            new BindStorageBufferFormat('resultsBuffer', SHADERSTAGE_COMPUTE)
        ]);

        let points = 0;
        let centroids = 0;
        let columns = 0;
        let shader: Shader;
        let compute: Compute;

        this.compute = (points_: number, centroids_: number, columns_: number) => {
            if (points === points_ && centroids === centroids_ && columns === columns_) {
                return compute;
            }

            points = points_;
            centroids = centroids_;
            columns = columns_;

            if (shader) {
                shader.destroy();
            }

            shader = new Shader(device, {
                name: 'compute-cluster',
                shaderLanguage: SHADERLANGUAGE_WGSL,
                cshader: clusterWgsl(points, centroids, columns),
                // @ts-ignore
                computeBindGroupFormat: bindGroupFormat
            });

            compute = new Compute(device, shader, 'compute-cluster');

            return compute;
        }
    }

    async execute(dataTable: DataTable, centroids: DataTable) {
        const { device } = this;

        const compute = this.compute(dataTable.numRows, centroids.numRows, dataTable.numColumns);

        // construct data buffers
        const pointsBuffer = new StorageBuffer(
            device,
            dataTable.numColumns * dataTable.numRows * 4,
            BUFFERUSAGE_COPY_DST
        );

        const centroidsBuffer = new StorageBuffer(
            device,
            centroids.numColumns * centroids.numRows * 4,
            BUFFERUSAGE_COPY_DST
        );

        const resultsBuffer = new StorageBuffer(
            device,
            dataTable.numRows * 4,
            BUFFERUSAGE_COPY_SRC | BUFFERUSAGE_COPY_DST
        );

        const resultsData = new Uint32Array(dataTable.numRows);

        // interleave the table data and write to gpu
        const interleavedPoints = interleaveData(dataTable);
        const interleavedCentroids = interleaveData(centroids);

        pointsBuffer.write(0, interleavedPoints, 0, interleavedPoints.length);
        centroidsBuffer.write(0, interleavedCentroids, 0, interleavedCentroids.length);

        compute.setParameter('points', dataTable.numRows);
        compute.setParameter('centroids', centroids.numRows);
        compute.setParameter('columns', dataTable.numColumns);

        compute.setParameter('pointsBuffer', pointsBuffer);
        compute.setParameter('centroidsBuffer', centroidsBuffer);
        compute.setParameter('resultsBuffer', resultsBuffer);

        // calculate the workgroup layout to try minimize the number of empty workgroups
        const groups = Math.ceil(dataTable.numRows / 64);
        const height = Math.ceil(groups / 65536);
        const width = Math.ceil(groups / height);

        // start compute job
        compute.setupDispatch(width, height);
        device.computeDispatch([compute], 'cluster-dispatch');

        // read back results
        await resultsBuffer.read(0, undefined, resultsData, true);

        const clusters: number[][] = [];
        for (let i = 0; i < centroids.numRows; ++i) {
            clusters[i] = [];
        }

        for (let i = 0; i < resultsData.length; ++i) {
            clusters[resultsData[i]].push(i);
        }

        // cleanup
        pointsBuffer.destroy();
        centroidsBuffer.destroy();
        resultsBuffer.destroy();

        return clusters;
    }
}

class GpuDevice {
    app: Application;
    backbuffer: Texture;
    cluster: Cluster;

    constructor(app: Application, backbuffer: Texture) {
        this.app = app;
        this.backbuffer = backbuffer;
        this.cluster = new Cluster(app.graphicsDevice);
    }
};

const createDevice = async () => {
    jsdomSetup();

    // @ts-ignore
    globalThis.Worker = Worker;

    // @ts-ignore
    window.navigator.gpu = create([]);

    const canvas = document.createElement('canvas');

    canvas.width = 1024;
    canvas.height = 512;

    const graphicsDevice = new WebgpuGraphicsDevice(canvas, {
        antialias: false,
        depth: true,
        stencil: false
    });

    await graphicsDevice.createDevice();

    // create the application
    const app = new Application(canvas, { graphicsDevice });

    // create external backbuffer
    const backbuffer = new Texture(graphicsDevice, {
        width: 1024,
        height: 512,
        name: 'WebgpuInternalBackbuffer',
        mipmaps: false,
        format: PIXELFORMAT_BGRA8
    });

    // @ts-ignore
    graphicsDevice.externalBackbuffer = backbuffer;

    app.start();

    return new GpuDevice(app, backbuffer);
};

export { createDevice, GpuDevice };
