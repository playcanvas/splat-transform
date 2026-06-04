import { GraphicsDevice, WebgpuGraphicsDevice } from 'playcanvas';
import { create, globals } from 'webgpu';

import { logger } from '../lib';

const initializeGlobals = () => {
    Object.assign(globalThis, globals);

    // window stub
    (globalThis as any).window = {
        navigator: { userAgent: 'node.js' }
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

// Get Dawn's actual adapter names by triggering its error message.
// This is the official documented method for enumerating adapters:
// https://github.com/dawn-gpu/node-webgpu?tab=readme-ov-file#usage
const getDawnAdapterNames = async (): Promise<string[]> => {
    try {
        const gpu = create(['adapter=__list_adapters__']);
        await gpu.requestAdapter();
    } catch (e) {
        // Parse Dawn's error message to extract adapter names
        const message = e instanceof Error ? e.message : String(e);
        const lines = message.split('\n');
        const names: string[] = [];

        for (const line of lines) {
            // Look for lines like: " * backend: 'd3d12', name: 'NVIDIA RTX A2000 8GB Laptop GPU'"
            const match = line.match(/name:\s*'([^']+)'/);
            if (match) {
                names.push(match[1]);
            }
        }

        return names;
    }

    // Unexpected: requestAdapter should have thrown with invalid adapter name
    logger.warn('Expected adapter enumeration to throw an error, but it did not.');
    return [];
};

// Cache enumerated adapters so we don't query Dawn multiple times
let cachedAdapters: Array<{ index: number; name: string }> | null = null;

const enumerateAdapters = async () => {
    if (cachedAdapters) {
        return cachedAdapters;
    }

    try {
        logger.info('Detecting GPU adapters...');

        // Get the actual adapter names directly from Dawn
        const dawnAdapterNames = await getDawnAdapterNames();

        // Cache and return the list
        cachedAdapters = dawnAdapterNames.map((name, index) => ({
            index,
            name
        }));

        return cachedAdapters;
    } catch (e) {
        logger.error('Failed to enumerate adapters. Error:', e);
        logger.error('\nThis usually means WebGPU is not available. Please ensure:');
        logger.error('  - Your GPU drivers are up to date');
        logger.error('  - Your GPU supports Vulkan, D3D12, or Metal');
        return [];
    }
};

const createDevice = async (adapterName?: string): Promise<GraphicsDevice> => {
    // Use Dawn's adapter selection if a specific adapter name is provided
    const dawnOptions = adapterName ? [`adapter=${adapterName}`] : [];

    // @ts-ignore
    window.navigator.gpu = create(dawnOptions);

    const canvas = document.createElement('canvas');

    canvas.width = 1024;
    canvas.height = 512;

    const graphicsDevice = new WebgpuGraphicsDevice(canvas, {
        antialias: false,
        depth: false,
        stencil: false
    });

    await graphicsDevice.createDevice();

    // Centralized GPU error handling. WebGPU never throws OOM/validation errors
    // into JS — they arrive asynchronously via `uncapturederror` (when no error
    // scope is active) or as device-lost, and PlayCanvas only Debug.warns then
    // tries to recreate the device, burying the cause. Left unsurfaced, an OOM
    // (e.g. a large scene on a smaller GPU) leaves buffers invalid/zeroed and
    // consumers silently produce degenerate output — which is how a streamed-SOG
    // LOD chain ended up with several identical full-resolution levels. Handling
    // it here once means every GPU consumer (decimate, filters, voxelization, …)
    // fails loudly without wrapping each call site in its own error scope.
    // @ts-ignore - wgpu is private on WebgpuGraphicsDevice but exposed in practice
    const wgpu = (graphicsDevice as any).wgpu;

    // A corrupted GPU result must never be written out, so we escalate to a hard
    // failure: re-raise on the next tick so main()'s uncaughtException handler
    // turns it into a non-zero exit. logger.error first so the precise cause is
    // recorded even if the rethrow races process teardown.
    const escalateGpuError = (summary: string) => {
        logger.error(`WebGPU ${summary} — aborting (a corrupted GPU result must not be written)`);
        const err = new Error(`WebGPU ${summary}`);
        setImmediate(() => {
            throw err;
        });
    };

    wgpu?.addEventListener?.('uncapturederror', (ev: any) => {
        const e = ev?.error;
        const kind = e?.constructor?.name === 'GPUOutOfMemoryError' ? 'out-of-memory' : 'error';
        escalateGpuError(`${kind}: ${e?.message || '(no message)'}`);
    });

    // Skip the `destroyed` reason — that fires on intentional device.destroy()
    // during normal shutdown.
    wgpu?.lost?.then((info: any) => {
        if (info?.reason === 'destroyed') return;
        escalateGpuError(`device lost: reason=${info?.reason || 'unknown'}, message=${info?.message || '(none)'}`);
    });

    return graphicsDevice;
};

export { createDevice, enumerateAdapters };
