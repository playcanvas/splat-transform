import { type ChunkSource, type ChunkSourceMetadata } from '../chunk';
import { type SogCamera, transformSogCamera } from '../sog-camera';
import { type Transform } from '../utils';

/**
 * Set (or, with `undefined`, clear) a source's camera block, lazily. Reads pass
 * through unchanged.
 *
 * `space` is the pending transform the camera's values were stored under — by
 * default the source's own, i.e. the camera is in the same raw coordinates as
 * the source's gaussians. When it differs (the source was re-bridged or baked
 * since), the camera is re-expressed so that it still means the same pose.
 *
 * @param src - The parent source.
 * @param camera - The camera block, or `undefined` to drop it.
 * @param space - The pending transform `camera` is stored under. Default: `src.meta.transform`.
 * @returns A derived source carrying the camera.
 */
const withCamera = (src: ChunkSource, camera: SogCamera | undefined, space?: Transform): ChunkSource => {
    const { transform } = src.meta;
    const cam = camera && space && !space.equals(transform) ?
        transformSogCamera(camera, transform.clone().invert().mul(space)) :
        camera;
    const meta: ChunkSourceMetadata = { ...src.meta, camera: cam };
    return {
        meta,
        read: req => src.read(req),
        close: () => src.close()
    };
};

export { withCamera };
