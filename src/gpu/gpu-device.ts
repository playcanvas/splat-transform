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

const clusterWgsl = /* wgsl */ `

@group(0) @binding(0) var<storage, read> points: array<f32>;
@group(0) @binding(1) var<storage, read> centroids: array<f32>;
@group(0) @binding(2) var<storage, read_write> results: array<u32>;

struct DataShape {
    numRows: u32,
    numColumns: u32,
    numCentroids: u32
};

// calculate the squared distance between the point and centroid
fn calcDistanceSqr(dataShape: ptr<function, DataShape>, point: array<f32, 45>, centroid: u32) -> f32 {
    var result = 0.0;

    var ci = centroid * dataShape.numColumns;

    for (var i = 0u; i < dataShape.numColumns; i++) {
        let v = point[i] - centroids[ci+i];
        result += v * v;
    }

    return result;
}

// return the index of the nearest centroid to the point
fn findNearest(dataShape: ptr<function, DataShape>, point: array<f32, 45>) -> u32 {
    var mind = 1000000.0;
    var mini = 0u;

    for (var i = 0u; i < dataShape.numCentroids; i++) {
        let d = calcDistanceSqr(dataShape, point, i);
        if (d < mind) {
            mind = d;
            mini = i;
        }
    }

    return mini;
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3u, @builtin(num_workgroups) num_workgroups: vec3u) {
    // calculate data shape given array lengths
    var dataShape: DataShape;
    dataShape.numRows = arrayLength(&results);
    dataShape.numColumns = arrayLength(&points) / dataShape.numRows;
    dataShape.numCentroids = arrayLength(&centroids) / dataShape.numColumns;

    let index = global_id.x + global_id.y * num_workgroups.x * 64;
    if (index < dataShape.numRows) {

        // read the point data from main memory
        var point: array<f32, 45>;
        for (var i = 0u; i < dataShape.numColumns; i++) {
            point[i] = points[index * dataShape.numColumns + i];
        }

        results[index] = findNearest(&dataShape, point);
    }
}
`;

const interleaveData = (dataTable: DataTable) => {
    const result = new Float32Array(dataTable.numRows * dataTable.numColumns);
    for (let c = 0; c < dataTable.numColumns; ++c) {
        const column = dataTable.columns[c];
        for (let r = 0; r < dataTable.numRows; ++r) {
            result[r * dataTable.numColumns + c] = column.data[r];
        }
    }
    return result;
};

class Cluster {
    device: GraphicsDevice;

    compute: Compute;

    constructor(device: GraphicsDevice) {
        this.device = device;

        const bindGroupFormat = new BindGroupFormat(device, [
            new BindStorageBufferFormat('pointsBuffer', SHADERSTAGE_COMPUTE, true),
            new BindStorageBufferFormat('centroidsBuffer', SHADERSTAGE_COMPUTE, true),
            new BindStorageBufferFormat('resultsBuffer', SHADERSTAGE_COMPUTE)
        ]);

        const shader = new Shader(device, {
            name: 'compute-cluster',
            shaderLanguage: SHADERLANGUAGE_WGSL,
            cshader: clusterWgsl,
            // @ts-ignore
            computeBindGroupFormat: bindGroupFormat
        });

        this.compute = new Compute(device, shader, 'compute-cluster');
    }

    async execute(dataTable: DataTable, centroids: DataTable) {
        const { device, compute } = this;

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
