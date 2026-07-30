import {
    SHADERLANGUAGE_WGSL,
    SHADERSTAGE_COMPUTE,
    UNIFORMTYPE_UINT,
    BindGroupFormat,
    BindStorageBufferFormat,
    BindUniformBufferFormat,
    Compute,
    GraphicsDevice,
    Shader,
    UniformBufferFormat,
    UniformFormat
} from 'playcanvas';

type Kernel = {
    compute: Compute;
    destroy: () => void;
};

/**
 * Shader + bind group format + compute boilerplate shared by compute kernels.
 * Uniforms are u32 (the struct in the WGSL must match `uniformNames` order);
 * storage bindings are `[name, readOnly]` pairs in binding order after the
 * uniform buffer.
 *
 * @param device - PlayCanvas GraphicsDevice (WebGPU).
 * @param name - Kernel name (shader + compute label).
 * @param source - WGSL source.
 * @param uniformNames - u32 uniform names, struct order.
 * @param storageBindings - Storage buffer bindings, `[name, readOnly]`.
 * @returns The compute wrapper and its destroy.
 */
const makeKernel = (
    device: GraphicsDevice,
    name: string,
    source: string,
    uniformNames: string[],
    storageBindings: [string, boolean][]
): Kernel => {
    const bindGroupFormat = new BindGroupFormat(device, [
        new BindUniformBufferFormat('uniforms', SHADERSTAGE_COMPUTE),
        ...storageBindings.map(([bname, readOnly]) => new BindStorageBufferFormat(bname, SHADERSTAGE_COMPUTE, readOnly))
    ]);

    const shader = new Shader(device, {
        name,
        shaderLanguage: SHADERLANGUAGE_WGSL,
        cshader: source,
        // @ts-ignore
        computeUniformBufferFormats: {
            uniforms: new UniformBufferFormat(device, uniformNames.map(u => new UniformFormat(u, UNIFORMTYPE_UINT)))
        },
        // @ts-ignore
        computeBindGroupFormat: bindGroupFormat
    });

    return {
        compute: new Compute(device, shader, name),
        destroy: () => {
            shader.destroy();
            bindGroupFormat.destroy();
        }
    };
};

export { makeKernel, type Kernel };
