import { Worker } from 'node:worker_threads';

// import { JSDOM } from 'jsdom';
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
    Texture,
    WebgpuGraphicsDevice
} from 'playcanvas/debug';
import { create, globals } from 'webgpu';

const initializeGlobals = () => {
    Object.assign(globalThis, globals);

    // window stub
    (globalThis as any).window = {
        navigator: { userAgent: 'node.js' },
    };

    // document stub
    (globalThis as any).document = {
        createElement: (type: string) => {
            if (type === 'canvas') {
                return {
                    getContext: (): null => {
                        return null;
                    },
                    getBoundingClientRect: () => {
                        return {
                            left: 0,
                            top: 0,
                            width: 300,
                            height: 150,
                            right: 300,
                            bottom: 150
                        };
                    },
                    width: 300,
                    height: 150
                };
            }
        }
    };
};

initializeGlobals();

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
