import { Worker } from 'node:worker_threads';

import { JSDOM } from 'jsdom';
import {
    // components
    AnimComponentSystem,
    RenderComponentSystem,
    CameraComponentSystem,
    LightComponentSystem,
    GSplatComponentSystem,
    ScriptComponentSystem,
    // handlers
    AnimClipHandler,
    AnimStateGraphHandler,
    BinaryHandler,
    ContainerHandler,
    CubemapHandler,
    GSplatHandler,
    RenderHandler,
    TextureHandler,
    // rest
    PIXELFORMAT_BGRA8,
    AppBase,
    AppOptions,
    Lightmapper,
    Texture,
    WebgpuGraphicsDevice
} from 'playcanvas/debug';
import { create, globals } from 'webgpu';


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

class Application extends AppBase {
    constructor(canvas: HTMLCanvasElement, options: any = {}) {
        super(canvas);

        const appOptions = new AppOptions();

        appOptions.graphicsDevice = options.graphicsDevice;

        appOptions.componentSystems = [
            AnimComponentSystem,
            CameraComponentSystem,
            GSplatComponentSystem,
            LightComponentSystem,
            RenderComponentSystem,
            ScriptComponentSystem
        ];

        appOptions.resourceHandlers = [
            AnimClipHandler,
            AnimStateGraphHandler,
            BinaryHandler,
            ContainerHandler,
            CubemapHandler,
            GSplatHandler,
            RenderHandler,
            TextureHandler
        ];

        appOptions.elementInput = options.elementInput;
        appOptions.keyboard = options.keyboard;
        appOptions.mouse = options.mouse;
        appOptions.touch = options.touch;
        appOptions.gamepads = options.gamepads;

        appOptions.scriptPrefix = options.scriptPrefix;
        appOptions.assetPrefix = options.assetPrefix;
        appOptions.scriptsOrder = options.scriptsOrder;

        appOptions.lightmapper = Lightmapper;

        this.init(appOptions);
    }
}

class GpuDevice {
    app: Application;
    backbuffer: Texture;

    constructor(app: Application, backbuffer: Texture) {
        this.app = app;
        this.backbuffer = backbuffer;
    }
}

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
