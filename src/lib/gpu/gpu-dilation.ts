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
    GraphicsDevice,
    Shader,
    StorageBuffer,
    UniformBufferFormat,
    UniformFormat
} from 'playcanvas';

import {
    clearWgsl,
    compactWgsl,
    dilateXWgsl,
    dilateYZWgsl,
    extractWgsl
} from './shaders/dilation';
import type { SparseVoxelGrid } from '../voxel/sparse-voxel-grid';

/**
 * One double-buffered slot — the four `Compute` instances (X, Y, Z, clear)
 * each own a uniform buffer that mustn't be overwritten by a sibling
 * dispatch on the same submit, plus the ping-pong storage buffers. Two
 * slots let the CPU prepare chunk N+1 while the GPU is busy with chunk N.
 */
interface DilationSlot {
    bufferA: StorageBuffer;
    bufferB: StorageBuffer;
    capacity: number;
    dilateXCompute: Compute;
    dilateYCompute: Compute;
    dilateZCompute: Compute;
    clearCompute: Compute;
    extractCompute: Compute;
    compactCompute: Compute;

    typesOutBuffer: StorageBuffer;
    masksOutBuffer: StorageBuffer;
    typesOutCapacity: number;
    masksOutCapacity: number;
}

/**
 * Separable 3D dilation on the GPU using a row-aligned dense bit grid
 * (1 bit per voxel, packed into u32 words; each row of bits along X starts
 * on a word boundary so per-word access is trivial). Each pass owns its
 * own `Compute` instance because their uniform buffers must not collide
 * within a single submit.
 */
class GpuDilation {
    private device: GraphicsDevice;
    private dilateXShader: Shader;
    private dilateYZShader: Shader;
    private clearShader: Shader;
    private extractShader: Shader;
    private compactShader: Shader;
    private dilateXBindGroupFormat: BindGroupFormat;
    private dilateYZBindGroupFormat: BindGroupFormat;
    private clearBindGroupFormat: BindGroupFormat;
    private extractBindGroupFormat: BindGroupFormat;
    private compactBindGroupFormat: BindGroupFormat;

    private slots: DilationSlot[];

    // Source SparseVoxelGrid uploaded once per `gpuDilate3` call (shared
    // across all chunks and slots). `extract` shaders read these.
    private srcTypesBuffer: StorageBuffer | null = null;
    private srcKeysBuffer: StorageBuffer | null = null;
    private srcLoBuffer: StorageBuffer | null = null;
    private srcHiBuffer: StorageBuffer | null = null;
    private srcTypesCapacity = 0;
    private srcMasksCapacity = 0;
    private srcMeta = { nbx: 0, nby: 0, nbz: 0, bStride: 0, capMinusOne: 0 };

    /** Number of double-buffered dispatch slots. */
    static readonly NUM_SLOTS = 2;

    constructor(device: GraphicsDevice) {
        this.device = device;

        this.dilateXBindGroupFormat = new BindGroupFormat(device, [
            new BindUniformBufferFormat('uniforms', SHADERSTAGE_COMPUTE),
            new BindStorageBufferFormat('src', SHADERSTAGE_COMPUTE, true),
            new BindStorageBufferFormat('dst', SHADERSTAGE_COMPUTE)
        ]);

        this.dilateYZBindGroupFormat = new BindGroupFormat(device, [
            new BindUniformBufferFormat('uniforms', SHADERSTAGE_COMPUTE),
            new BindStorageBufferFormat('src', SHADERSTAGE_COMPUTE, true),
            new BindStorageBufferFormat('dst', SHADERSTAGE_COMPUTE)
        ]);

        this.clearBindGroupFormat = new BindGroupFormat(device, [
            new BindUniformBufferFormat('uniforms', SHADERSTAGE_COMPUTE),
            new BindStorageBufferFormat('clearDst', SHADERSTAGE_COMPUTE)
        ]);

        this.extractBindGroupFormat = new BindGroupFormat(device, [
            new BindUniformBufferFormat('uniforms', SHADERSTAGE_COMPUTE),
            new BindStorageBufferFormat('srcTypes', SHADERSTAGE_COMPUTE, true),
            new BindStorageBufferFormat('srcKeys', SHADERSTAGE_COMPUTE, true),
            new BindStorageBufferFormat('srcLo', SHADERSTAGE_COMPUTE, true),
            new BindStorageBufferFormat('srcHi', SHADERSTAGE_COMPUTE, true),
            new BindStorageBufferFormat('dstDense', SHADERSTAGE_COMPUTE)
        ]);

        this.compactBindGroupFormat = new BindGroupFormat(device, [
            new BindUniformBufferFormat('uniforms', SHADERSTAGE_COMPUTE),
            new BindStorageBufferFormat('dilatedDense', SHADERSTAGE_COMPUTE, true),
            new BindStorageBufferFormat('typesOut', SHADERSTAGE_COMPUTE),
            new BindStorageBufferFormat('masksOut', SHADERSTAGE_COMPUTE)
        ]);

        this.dilateXShader = new Shader(device, {
            name: 'gpu-dilation-x',
            shaderLanguage: SHADERLANGUAGE_WGSL,
            cshader: dilateXWgsl(),
            // @ts-ignore
            computeUniformBufferFormats: {
                uniforms: new UniformBufferFormat(device, [
                    new UniformFormat('numXWords', UNIFORMTYPE_UINT),
                    new UniformFormat('ny', UNIFORMTYPE_UINT),
                    new UniformFormat('nz', UNIFORMTYPE_UINT),
                    new UniformFormat('halfExtent', UNIFORMTYPE_UINT)
                ])
            },
            // @ts-ignore
            computeBindGroupFormat: this.dilateXBindGroupFormat
        });

        this.dilateYZShader = new Shader(device, {
            name: 'gpu-dilation-yz',
            shaderLanguage: SHADERLANGUAGE_WGSL,
            cshader: dilateYZWgsl(),
            // @ts-ignore
            computeUniformBufferFormats: {
                uniforms: new UniformBufferFormat(device, [
                    new UniformFormat('numXWords', UNIFORMTYPE_UINT),
                    new UniformFormat('ny', UNIFORMTYPE_UINT),
                    new UniformFormat('nz', UNIFORMTYPE_UINT),
                    new UniformFormat('halfExtent', UNIFORMTYPE_UINT),
                    new UniformFormat('stride', UNIFORMTYPE_UINT),
                    new UniformFormat('axisLen', UNIFORMTYPE_UINT)
                ])
            },
            // @ts-ignore
            computeBindGroupFormat: this.dilateYZBindGroupFormat
        });

        this.clearShader = new Shader(device, {
            name: 'gpu-dilation-clear',
            shaderLanguage: SHADERLANGUAGE_WGSL,
            cshader: clearWgsl(),
            // @ts-ignore
            computeUniformBufferFormats: {
                uniforms: new UniformBufferFormat(device, [
                    new UniformFormat('clearNumWords', UNIFORMTYPE_UINT),
                    new UniformFormat('clearRowStride', UNIFORMTYPE_UINT)
                ])
            },
            // @ts-ignore
            computeBindGroupFormat: this.clearBindGroupFormat
        });

        this.extractShader = new Shader(device, {
            name: 'gpu-dilation-extract',
            shaderLanguage: SHADERLANGUAGE_WGSL,
            cshader: extractWgsl(),
            // @ts-ignore
            computeUniformBufferFormats: {
                uniforms: new UniformBufferFormat(device, [
                    new UniformFormat('minBx', UNIFORMTYPE_UINT),  // signed reinterpret
                    new UniformFormat('minBy', UNIFORMTYPE_UINT),
                    new UniformFormat('minBz', UNIFORMTYPE_UINT),
                    new UniformFormat('outerBx', UNIFORMTYPE_UINT),
                    new UniformFormat('outerBy', UNIFORMTYPE_UINT),
                    new UniformFormat('outerBz', UNIFORMTYPE_UINT),
                    new UniformFormat('numXWords', UNIFORMTYPE_UINT),
                    new UniformFormat('srcNbx', UNIFORMTYPE_UINT),
                    new UniformFormat('srcNby', UNIFORMTYPE_UINT),
                    new UniformFormat('srcNbz', UNIFORMTYPE_UINT),
                    new UniformFormat('srcBStride', UNIFORMTYPE_UINT),
                    new UniformFormat('srcCapMinusOne', UNIFORMTYPE_UINT)
                ])
            },
            // @ts-ignore
            computeBindGroupFormat: this.extractBindGroupFormat
        });

        this.compactShader = new Shader(device, {
            name: 'gpu-dilation-compact',
            shaderLanguage: SHADERLANGUAGE_WGSL,
            cshader: compactWgsl(),
            // @ts-ignore
            computeUniformBufferFormats: {
                uniforms: new UniformBufferFormat(device, [
                    new UniformFormat('haloBx', UNIFORMTYPE_UINT),
                    new UniformFormat('haloBy', UNIFORMTYPE_UINT),
                    new UniformFormat('haloBz', UNIFORMTYPE_UINT),
                    new UniformFormat('numXWords', UNIFORMTYPE_UINT),
                    new UniformFormat('innerBx', UNIFORMTYPE_UINT),
                    new UniformFormat('innerBy', UNIFORMTYPE_UINT),
                    new UniformFormat('innerBz', UNIFORMTYPE_UINT),
                    new UniformFormat('outerBy', UNIFORMTYPE_UINT)
                ])
            },
            // @ts-ignore
            computeBindGroupFormat: this.compactBindGroupFormat
        });

        this.slots = [];
        for (let i = 0; i < GpuDilation.NUM_SLOTS; i++) {
            const initialCapacity = 1024 * 1024 * 4;
            const initialTypesOut = 64 * 1024;       // 16K blocks worth packed
            const initialMasksOut = 1024 * 1024;     // 128K blocks worth (lo, hi)
            this.slots.push({
                bufferA: new StorageBuffer(device, initialCapacity, BUFFERUSAGE_COPY_DST | BUFFERUSAGE_COPY_SRC),
                bufferB: new StorageBuffer(device, initialCapacity, BUFFERUSAGE_COPY_DST | BUFFERUSAGE_COPY_SRC),
                capacity: initialCapacity,
                dilateXCompute: new Compute(device, this.dilateXShader, `gpu-dilate-x-${i}`),
                dilateYCompute: new Compute(device, this.dilateYZShader, `gpu-dilate-y-${i}`),
                dilateZCompute: new Compute(device, this.dilateYZShader, `gpu-dilate-z-${i}`),
                clearCompute: new Compute(device, this.clearShader, `gpu-dilate-clear-${i}`),
                extractCompute: new Compute(device, this.extractShader, `gpu-dilate-extract-${i}`),
                compactCompute: new Compute(device, this.compactShader, `gpu-dilate-compact-${i}`),
                typesOutBuffer: new StorageBuffer(device, initialTypesOut, BUFFERUSAGE_COPY_DST | BUFFERUSAGE_COPY_SRC),
                masksOutBuffer: new StorageBuffer(device, initialMasksOut, BUFFERUSAGE_COPY_DST | BUFFERUSAGE_COPY_SRC),
                typesOutCapacity: initialTypesOut,
                masksOutCapacity: initialMasksOut
            });
        }
    }

    private ensureSlotBuffers(slot: DilationSlot, numWords: number): void {
        const neededBytes = numWords * 4;
        if (neededBytes <= slot.capacity) return;

        let cap = slot.capacity;
        while (cap < neededBytes) cap *= 2;

        slot.bufferA.destroy();
        slot.bufferB.destroy();
        slot.bufferA = new StorageBuffer(this.device, cap, BUFFERUSAGE_COPY_DST | BUFFERUSAGE_COPY_SRC);
        slot.bufferB = new StorageBuffer(this.device, cap, BUFFERUSAGE_COPY_DST | BUFFERUSAGE_COPY_SRC);
        slot.capacity = cap;
    }

    /**
     * Dispatch a compute clear of `dst` to zero for the first `numWords` words.
     * Uses the command encoder so it's correctly ordered with subsequent
     * dilation passes (unlike `queue.writeBuffer`, which is queued separately
     * and would race against the dispatches).
     * @param slot - Per-chunk slot whose `clearCompute` pipeline is dispatched.
     * @param dst - Destination buffer to zero.
     * @param numWords - Number of leading u32 words to clear.
     */
    private dispatchClear(slot: DilationSlot, dst: StorageBuffer, numWords: number): void {
        const totalWg = Math.ceil(numWords / 256);
        const MAX_DIM = 65535;
        const wgX = Math.min(totalWg, MAX_DIM);
        const wgY = Math.ceil(totalWg / wgX);
        const rowStride = wgX * 256;

        slot.clearCompute.setParameter('clearDst', dst);
        slot.clearCompute.setParameter('clearNumWords', numWords);
        slot.clearCompute.setParameter('clearRowStride', rowStride);
        slot.clearCompute.setupDispatch(wgX, wgY, 1);
        this.device.computeDispatch([slot.clearCompute], 'gpu-dilate-clear');
    }

    /**
     * Upload a `SparseVoxelGrid` to GPU storage buffers used by the extract
     * shader. Reuses the existing buffers if they're large enough; otherwise
     * destroys and reallocates. Designed to be called once per
     * `gpuDilate3` call (the same `src` is read across all chunks).
     * @param src - Source sparse grid to upload.
     */
    uploadSrc(src: SparseVoxelGrid): void {
        const types = src.types;
        const keys = src.masks.keys;     // Int32Array; -1 sentinel reads as 0xFFFFFFFF when interpreted as u32
        const lo = src.masks.lo;
        const hi = src.masks.hi;

        const typesBytes = types.byteLength;
        if (this.srcTypesBuffer === null || this.srcTypesCapacity < typesBytes) {
            this.srcTypesBuffer?.destroy();
            this.srcTypesBuffer = new StorageBuffer(this.device, typesBytes, BUFFERUSAGE_COPY_DST);
            this.srcTypesCapacity = typesBytes;
        }
        this.srcTypesBuffer.write(0, types, 0, types.length);

        const masksBytes = keys.byteLength;
        if (this.srcKeysBuffer === null || this.srcMasksCapacity < masksBytes) {
            this.srcKeysBuffer?.destroy();
            this.srcLoBuffer?.destroy();
            this.srcHiBuffer?.destroy();
            this.srcKeysBuffer = new StorageBuffer(this.device, masksBytes, BUFFERUSAGE_COPY_DST);
            this.srcLoBuffer = new StorageBuffer(this.device, masksBytes, BUFFERUSAGE_COPY_DST);
            this.srcHiBuffer = new StorageBuffer(this.device, masksBytes, BUFFERUSAGE_COPY_DST);
            this.srcMasksCapacity = masksBytes;
        }
        // Treat keys (Int32) as Uint32 — same byte pattern; -1 reads as 0xFFFFFFFF.
        const keysU32 = new Uint32Array(keys.buffer, keys.byteOffset, keys.length);
        this.srcKeysBuffer.write(0, keysU32, 0, keys.length);
        this.srcLoBuffer.write(0, lo, 0, lo.length);
        this.srcHiBuffer.write(0, hi, 0, hi.length);

        this.srcMeta = {
            nbx: src.nbx,
            nby: src.nby,
            nbz: src.nbz,
            bStride: src.bStride,
            capMinusOne: keys.length - 1
        };
    }

    /** Free uploaded `src` buffers. Caller can call after `gpuDilate3` finishes. */
    releaseSrc(): void {
        this.srcTypesBuffer?.destroy();
        this.srcKeysBuffer?.destroy();
        this.srcLoBuffer?.destroy();
        this.srcHiBuffer?.destroy();
        this.srcTypesBuffer = null;
        this.srcKeysBuffer = null;
        this.srcLoBuffer = null;
        this.srcHiBuffer = null;
        this.srcTypesCapacity = 0;
        this.srcMasksCapacity = 0;
    }

    private ensureSlotOutputBuffers(slot: DilationSlot, innerBlocks: number): void {
        // typesOut: 2 bits per inner block, packed into u32 words.
        const typesBytes = (((innerBlocks + 15) >>> 4) * 4);
        if (slot.typesOutCapacity < typesBytes) {
            slot.typesOutBuffer.destroy();
            slot.typesOutBuffer = new StorageBuffer(this.device, typesBytes, BUFFERUSAGE_COPY_DST | BUFFERUSAGE_COPY_SRC);
            slot.typesOutCapacity = typesBytes;
        }
        // masksOut: (lo, hi) per inner block.
        const masksBytes = innerBlocks * 8;
        if (slot.masksOutCapacity < masksBytes) {
            slot.masksOutBuffer.destroy();
            slot.masksOutBuffer = new StorageBuffer(this.device, masksBytes, BUFFERUSAGE_COPY_DST | BUFFERUSAGE_COPY_SRC);
            slot.masksOutCapacity = masksBytes;
        }
    }

    /**
     * Sparse-path submit. Reads from the previously-uploaded `src` (via
     * `uploadSrc`), runs extract → dilate → compact as GPU passes, and returns
     * Promises for the per-block `typesOut` (packed 2-bit) and `masksOut`
     * (lo/hi pairs). Caller integrates these into `dst` directly.
     * @param slotIdx - Round-robin slot index (`0..NUM_SLOTS-1`).
     * @param minBx - Outer chunk origin block X (in `src`'s block coords).
     * @param minBy - Outer chunk origin block Y.
     * @param minBz - Outer chunk origin block Z.
     * @param outerBx - Outer chunk size in blocks along X.
     * @param outerBy - Outer chunk size in blocks along Y.
     * @param outerBz - Outer chunk size in blocks along Z.
     * @param haloBx - Halo size in blocks along X (one side).
     * @param haloBy - Halo size in blocks along Y (one side).
     * @param haloBz - Halo size in blocks along Z (one side).
     * @param innerBx - Inner (output) region size in blocks along X.
     * @param innerBy - Inner region size in blocks along Y.
     * @param innerBz - Inner region size in blocks along Z.
     * @param halfExtentXZ - Dilation half-extent in voxels along X and Z.
     * @param halfExtentY - Dilation half-extent in voxels along Y.
     * @returns Promises for the inner region's packed types and `[lo, hi]` masks.
     */
    submitChunkSparse(
        slotIdx: number,
        // outer chunk in block coords (each is voxel/4)
        minBx: number, minBy: number, minBz: number,
        outerBx: number, outerBy: number, outerBz: number,
        // halo in blocks
        haloBx: number, haloBy: number, haloBz: number,
        // inner chunk in block coords
        innerBx: number, innerBy: number, innerBz: number,
        halfExtentXZ: number,
        halfExtentY: number
    ): { types: Promise<Uint32Array>, masks: Promise<Uint32Array> } {
        if (this.srcTypesBuffer === null) {
            throw new Error('GpuDilation: must call uploadSrc() before submitChunkSparse()');
        }
        const slot = this.slots[slotIdx];

        const outerNx = outerBx * 4;
        const outerNy = outerBy * 4;
        const outerNz = outerBz * 4;
        const numXWords = (outerNx + 31) >>> 5;
        const numWords = numXWords * outerNy * outerNz;
        this.ensureSlotBuffers(slot, numWords);

        const innerBlocks = innerBx * innerBy * innerBz;
        this.ensureSlotOutputBuffers(slot, innerBlocks);

        const typesOutWords = (innerBlocks + 15) >>> 4;

        // Extract: clear bufferA, dispatch extract from sparse src into bufferA.
        this.dispatchClear(slot, slot.bufferA, numWords);
        this.dispatchExtract(slot, minBx, minBy, minBz, outerBx, outerBy, outerBz, numXWords);

        // Force a queue submission boundary between the extract pass (which
        // writes bufferA via atomicOr — bound as `storage, read_write` with
        // `array<atomic<u32>>`) and the dilate-X pass (which reads bufferA
        // bound as `storage, read` with `array<u32>`). Without this, the
        // atomic writes are not reliably visible to the next pass — dilation
        // produces empty output. Other inter-pass transitions in this
        // pipeline (clear→extract, dilateX→Z→Y, Y→compact) rely on automatic
        // intra-encoder synchronization without issue. Verified raw Dawn
        // synchronizes this case correctly, so the bug is somewhere in
        // PlayCanvas's compute dispatch / bind-group path.
        (this.device as unknown as { submit: () => void }).submit();

        // X-pass: A → B
        this.dispatchX(slot, slot.bufferA, slot.bufferB, numXWords, outerNy, outerNz, halfExtentXZ);

        // Z-pass: B → A
        this.dispatchYZ(slot.dilateZCompute, slot.bufferB, slot.bufferA,
            numXWords, outerNy, outerNz, halfExtentXZ, numXWords * outerNy, outerNz);

        // Y-pass: A → B
        this.dispatchYZ(slot.dilateYCompute, slot.bufferA, slot.bufferB,
            numXWords, outerNy, outerNz, halfExtentY, numXWords, outerNy);

        // Compact: clear typesOut (atomicOr accumulates), dispatch compact.
        // masksOut is written non-atomically — no clear needed.
        this.dispatchClear(slot, slot.typesOutBuffer, typesOutWords);
        this.dispatchCompact(slot, haloBx, haloBy, haloBz, innerBx, innerBy, innerBz, numXWords, outerBy);

        const typesPromise = slot.typesOutBuffer.read(0, typesOutWords * 4, null, true)
        .then((readData: Uint8Array) => new Uint32Array(readData.buffer, readData.byteOffset, typesOutWords));
        const masksPromise = slot.masksOutBuffer.read(0, innerBlocks * 8, null, true)
        .then((readData: Uint8Array) => new Uint32Array(readData.buffer, readData.byteOffset, innerBlocks * 2));

        return { types: typesPromise, masks: masksPromise };
    }

    private dispatchExtract(
        slot: DilationSlot,
        minBx: number, minBy: number, minBz: number,
        outerBx: number, outerBy: number, outerBz: number,
        numXWords: number
    ): void {
        const c = slot.extractCompute;
        c.setParameter('srcTypes', this.srcTypesBuffer!);
        c.setParameter('srcKeys', this.srcKeysBuffer!);
        c.setParameter('srcLo', this.srcLoBuffer!);
        c.setParameter('srcHi', this.srcHiBuffer!);
        c.setParameter('dstDense', slot.bufferA);
        // Reinterpret signed minB* as u32 bits via the i32 parameter slot —
        // PlayCanvas treats UNIFORMTYPE_UINT as raw u32, and the WGSL struct
        // declares these as i32 so they read back signed.
        c.setParameter('minBx', (minBx >>> 0));
        c.setParameter('minBy', (minBy >>> 0));
        c.setParameter('minBz', (minBz >>> 0));
        c.setParameter('outerBx', outerBx);
        c.setParameter('outerBy', outerBy);
        c.setParameter('outerBz', outerBz);
        c.setParameter('numXWords', numXWords);
        c.setParameter('srcNbx', this.srcMeta.nbx);
        c.setParameter('srcNby', this.srcMeta.nby);
        c.setParameter('srcNbz', this.srcMeta.nbz);
        c.setParameter('srcBStride', this.srcMeta.bStride);
        c.setParameter('srcCapMinusOne', this.srcMeta.capMinusOne);

        const wgX = Math.ceil(outerBx / 8);
        const wgY = Math.ceil(outerBy / 4);
        const wgZ = Math.ceil(outerBz / 8);
        c.setupDispatch(wgX, wgY, wgZ);
        this.device.computeDispatch([c], c.name);
    }

    private dispatchCompact(
        slot: DilationSlot,
        haloBx: number, haloBy: number, haloBz: number,
        innerBx: number, innerBy: number, innerBz: number,
        numXWords: number,
        outerBy: number
    ): void {
        const c = slot.compactCompute;
        c.setParameter('dilatedDense', slot.bufferB);
        c.setParameter('typesOut', slot.typesOutBuffer);
        c.setParameter('masksOut', slot.masksOutBuffer);
        c.setParameter('haloBx', haloBx);
        c.setParameter('haloBy', haloBy);
        c.setParameter('haloBz', haloBz);
        c.setParameter('numXWords', numXWords);
        c.setParameter('innerBx', innerBx);
        c.setParameter('innerBy', innerBy);
        c.setParameter('innerBz', innerBz);
        c.setParameter('outerBy', outerBy);

        const wgX = Math.ceil(innerBx / 8);
        const wgY = Math.ceil(innerBy / 4);
        const wgZ = Math.ceil(innerBz / 8);
        c.setupDispatch(wgX, wgY, wgZ);
        this.device.computeDispatch([c], c.name);
    }

    private dispatchX(
        slot: DilationSlot,
        src: StorageBuffer, dst: StorageBuffer,
        numXWords: number, ny: number, nz: number,
        halfExtent: number
    ): void {
        const c = slot.dilateXCompute;
        c.setParameter('src', src);
        c.setParameter('dst', dst);
        c.setParameter('numXWords', numXWords);
        c.setParameter('ny', ny);
        c.setParameter('nz', nz);
        c.setParameter('halfExtent', halfExtent);

        const wgX = Math.ceil(numXWords / 8);
        const wgY = Math.ceil(ny / 4);
        const wgZ = Math.ceil(nz / 8);
        c.setupDispatch(wgX, wgY, wgZ);
        this.device.computeDispatch([c], c.name);
    }

    private dispatchYZ(
        compute: Compute,
        src: StorageBuffer, dst: StorageBuffer,
        numXWords: number, ny: number, nz: number,
        halfExtent: number,
        stride: number, axisLen: number
    ): void {
        compute.setParameter('src', src);
        compute.setParameter('dst', dst);
        compute.setParameter('numXWords', numXWords);
        compute.setParameter('ny', ny);
        compute.setParameter('nz', nz);
        compute.setParameter('halfExtent', halfExtent);
        compute.setParameter('stride', stride);
        compute.setParameter('axisLen', axisLen);

        const wgX = Math.ceil(numXWords / 8);
        const wgY = Math.ceil(ny / 4);
        const wgZ = Math.ceil(nz / 8);
        compute.setupDispatch(wgX, wgY, wgZ);
        this.device.computeDispatch([compute], compute.name);
    }

    destroy(): void {
        this.releaseSrc();
        for (const slot of this.slots) {
            slot.bufferA.destroy();
            slot.bufferB.destroy();
            slot.typesOutBuffer.destroy();
            slot.masksOutBuffer.destroy();
            slot.dilateXCompute.destroy();
            slot.dilateYCompute.destroy();
            slot.dilateZCompute.destroy();
            slot.clearCompute.destroy();
            slot.extractCompute.destroy();
            slot.compactCompute.destroy();
        }
        this.dilateXShader.destroy();
        this.dilateYZShader.destroy();
        this.clearShader.destroy();
        this.extractShader.destroy();
        this.compactShader.destroy();
        this.dilateXBindGroupFormat.destroy();
        this.dilateYZBindGroupFormat.destroy();
        this.clearBindGroupFormat.destroy();
        this.extractBindGroupFormat.destroy();
        this.compactBindGroupFormat.destroy();
    }
}

export { GpuDilation };
