import type { TypedArray } from '../data-table/data-table';
import { knnQueryBlock } from '../decimate/knn-core';
import { mergeGroup, createMergeScratch, splatMass } from '../decimate/moment-match';
import { KdTree, type FlatKdTree } from '../spatial/kd-tree';
import { quantize1dColumns, type QuantizedColumns } from '../spatial/quantize-1d-core';
import { WebPCodec } from '../utils/webp-codec';

/**
 * Named task handlers runnable on a worker thread or inline on the calling
 * thread. Both transports execute these exact functions, so results are
 * identical regardless of where a task runs.
 *
 * Handlers take a single structured-clone-friendly argument object (typed
 * arrays, plain objects) and return the result plus the list of buffers to
 * transfer back to the host (ignored when running inline).
 *
 * NOTE: this module is bundled into the worker, so its runtime import graph
 * must stay lean - in particular free of DataTable, whose Transform member
 * drags in the playcanvas engine.
 */

type TaskOutput<T> = { result: T, transfer: ArrayBuffer[] };

const taskHandlers = {
    quantize1d: (args: { columns: { name: string, data: TypedArray }[], k?: number, alpha?: number }):
        TaskOutput<QuantizedColumns> => {
        const result = quantize1dColumns(args.columns, args.k, args.alpha);
        return {
            result,
            transfer: [result.centroids.buffer as ArrayBuffer, ...result.labels.map(c => c.data.buffer as ArrayBuffer)]
        };
    },

    encodeWebp: async (args: { rgba: Uint8Array, width: number, height: number }): Promise<TaskOutput<Uint8Array>> => {
        // create() memoizes the wasm module per realm (each worker compiles
        // its own copy on first use)
        const codec = await WebPCodec.create();
        const webp = codec.encodeLosslessRGBA(args.rgba, args.width, args.height);
        return { result: webp, transfer: [webp.buffer as ArrayBuffer] };
    },

    // Build + flatten a KD-tree over interleaved local positions (decimation
    // GPU path: the flattened arrays upload straight into GpuKnn).
    flattenKdTree: (args: { positions: Float32Array }): TaskOutput<FlatKdTree> => {
        const n = args.positions.length / 3;
        const x = new Float32Array(n);
        const y = new Float32Array(n);
        const z = new Float32Array(n);
        for (let i = 0; i < n; i++) {
            x[i] = args.positions[i * 3];
            y[i] = args.positions[i * 3 + 1];
            z[i] = args.positions[i * 3 + 2];
        }
        const flat = new KdTree([x, y, z]).flatten();
        return {
            result: flat,
            transfer: [
                flat.nodeSplatIdx.buffer, flat.nodeX.buffer, flat.nodeY.buffer,
                flat.nodeZ.buffer, flat.nodeLeft.buffer, flat.nodeRight.buffer
            ] as ArrayBuffer[]
        };
    },

    // Decimation CPU-fallback block KNN: exact k-NN of the owned prefix
    // within the local point set, as local indices.
    knnBlock: (args: { positions: Float32Array, ownedCount: number, k: number }): TaskOutput<Uint32Array> => {
        const result = knnQueryBlock(args.positions, args.ownedCount, args.k);
        return { result, transfer: [result.buffer as ArrayBuffer] };
    },

    // Decimation merge stream: n-ary moment match of packed member-major
    // groups. Inputs are member-major (pos 3 / geo 8 / color colorDim floats
    // per member, groups back to back per `sizes`); outputs are group-major.
    // `other` columns (when present) copy from the dominant-mass member.
    mergeGroups: (args: {
        pos: Float32Array,
        geo: Float32Array,
        color: Float32Array,
        sizes: Uint32Array,
        colorDim: number,
        other?: Uint32Array,
        otherDim?: number
    }): TaskOutput<{ pos: Float32Array, geo: Float32Array, color: Float32Array, other?: Uint32Array }> => {
        const { sizes, colorDim } = args;
        const g = sizes.length;
        const otherDim = args.otherDim ?? 0;
        const view = { pos: args.pos, geo: args.geo, color: args.color, colorDim };
        const outPos = new Float32Array(g * 3);
        const outGeo = new Float32Array(g * 8);
        const outColor = new Float32Array(g * colorDim);
        const outOther = args.other && otherDim > 0 ? new Uint32Array(g * otherDim) : undefined;
        const merged = {
            pos: new Float64Array(3),
            geo: new Float64Array(8),
            color: new Float64Array(colorDim)
        };
        const scratch = createMergeScratch();
        const members: number[] = [];
        let base = 0;
        for (let gi = 0; gi < g; gi++) {
            const size = sizes[gi];
            members.length = size;
            for (let m = 0; m < size; m++) members[m] = base + m;
            mergeGroup(view, members, size, merged, scratch);
            outPos.set(merged.pos, gi * 3);
            outGeo.set(merged.geo, gi * 8);
            outColor.set(merged.color, gi * colorDim);
            if (outOther) {
                let dominant = base, best = -Infinity;
                for (let m = 0; m < size; m++) {
                    const mass = splatMass(args.geo, base + m);
                    if (mass > best) {
                        best = mass;
                        dominant = base + m;
                    }
                }
                for (let c = 0; c < otherDim; c++) {
                    outOther[gi * otherDim + c] = args.other![dominant * otherDim + c];
                }
            }
            base += size;
        }
        const transfer: ArrayBuffer[] = [outPos.buffer as ArrayBuffer, outGeo.buffer as ArrayBuffer, outColor.buffer as ArrayBuffer];
        if (outOther) transfer.push(outOther.buffer as ArrayBuffer);
        return { result: { pos: outPos, geo: outGeo, color: outColor, other: outOther }, transfer };
    }
};

type TaskName = keyof typeof taskHandlers;
type TaskArgs<T extends TaskName> = Parameters<typeof taskHandlers[T]>[0];
type TaskResult<T extends TaskName> = Awaited<ReturnType<typeof taskHandlers[T]>>['result'];

// Message protocol between host and worker. There are no per-task ids: each
// worker runs strictly one task at a time (enforced by the host's slot state
// machine), so every reply pairs with the single in-flight `run`.
type HostMessage =
    | { type: 'init', wasmUrl: string }
    | { type: 'run', task: TaskName, args: any };

type WorkerMessage =
    | { type: 'ready' }
    | { type: 'result', result: any }
    | { type: 'error', message: string, stack?: string };

export { taskHandlers, type TaskName, type TaskArgs, type TaskResult, type HostMessage, type WorkerMessage };
