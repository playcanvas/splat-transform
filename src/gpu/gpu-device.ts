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
import { KdTree } from '../utils/kd-tree.js';

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
@group(0) @binding(2) var<storage, read> kdTree: array<u32>;
@group(0) @binding(3) var<storage, read_write> results: array<u32>;

// calculate the squared distance between the point and centroid
fn calcDistance(point: u32, centroid: u32) -> f32 {
    let numRows = arrayLength(&results);
    let numColumns = arrayLength(&points) / numRows;
    let numCentroids = arrayLength(&centroids) / numColumns;

    var result = 0.0;

    for (var i = 0u; i < numColumns; i++) {
        let p = points[point + i * numRows];
        let c = centroids[centroid + i * numCentroids];
        let v = p - c;
        result += v * v;
    }

    return result;
}

// return the index of the nearest centroid to the point
fn findNearest(point: u32) -> u32 {
    let numRows = arrayLength(&results);
    let numColumns = arrayLength(&points) / numRows;
    let numCentroids = arrayLength(&centroids) / numColumns;

    var mind = 1000000.0;
    var mini = 0u;

    for (var i = 0u; i < numCentroids; i++) {
        let d = calcDistance(point, i);
        if (d < mind) {
            mind = d;
            mini = i;
        }
    }

    return mini;
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_invocation_id: vec3u) {
    let index = global_invocation_id.x;
    if (index >= arrayLength(&results)) {
        return;
    }
    results[index] = findNearest(index);
}
`;

class Cluster {
    device: GraphicsDevice;

    compute: Compute;

    constructor(device: GraphicsDevice) {
        this.device = device;

        const bindGroupFormat = new BindGroupFormat(device, [
            new BindStorageBufferFormat('pointsBuffer', SHADERSTAGE_COMPUTE, true),
            new BindStorageBufferFormat('centroidsBuffer', SHADERSTAGE_COMPUTE, true),
            new BindStorageBufferFormat('kdTreeBuffer', SHADERSTAGE_COMPUTE, true),
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

    async execute(dataTable: DataTable, kdTree: KdTree) {
        const { device, compute } = this;

        const kdTreeData = kdTree.flatten();
        const { centroids } = kdTree;

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

        const kdTreeBuffer = new StorageBuffer(
            device,
            kdTreeData.length * 4,
            BUFFERUSAGE_COPY_DST
        );

        const resultsBuffer = new StorageBuffer(
            device,
            dataTable.numRows * 4,
            BUFFERUSAGE_COPY_SRC | BUFFERUSAGE_COPY_DST
        );

        const resultsData = new Uint32Array(dataTable.numRows);

        // write dataTable in columns to gpu
        for (let c = 0; c < dataTable.numColumns; ++c) {
            const { data } = dataTable.columns[c];
            pointsBuffer.write(c * dataTable.numRows * 4, data, 0, data.length);
        }

        // write centroids in columns to gpu
        for (let c = 0; c < centroids.numColumns; ++c) {
            const { data } = centroids.columns[c];
            centroidsBuffer.write(c * centroids.numRows * 4, data, 0, data.length);
        }

        compute.setParameter('pointsBuffer', pointsBuffer);
        compute.setParameter('centroidsBuffer', centroidsBuffer);
        compute.setParameter('kdTreeBuffer', kdTreeBuffer);
        compute.setParameter('resultsBuffer', resultsBuffer);

        // start compute job
        compute.setupDispatch(Math.ceil(dataTable.numRows / 64));
        device.computeDispatch([compute], 'cluster-dispatch');

        // read back results
        const result = await resultsBuffer.read(0, undefined, resultsData, true);

        if (result === resultsData) {
            console.log('same');
        }

        const clusters: number[][] = [];
        for (let i = 0; i < kdTree.centroids.numRows; ++i) {
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
