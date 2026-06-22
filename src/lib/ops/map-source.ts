import { type ChunkSource, type ChunkSourceMetadata } from '../source';
import { type Transform } from '../utils';

/**
 * Pre-apply a coordinate-space transform onto a {@link ChunkSource}, lazily.
 *
 * The transform is composed onto the source's pending `meta.transform` by
 * left-multiplication (`transform * existing`), matching the deferred-transform
 * semantics of `processDataTable`'s translate/rotate/scale actions
 * (`new Transform(value).mul(existing)`). No data is touched: reads pass through
 * to the parent unchanged, and consumers (writers, kernels that care about
 * coordinate space) bake `meta.transform` when they emit.
 *
 * @param src - The parent source.
 * @param transform - The transform to pre-apply.
 * @returns A derived source exposing the composed transform; reads delegate to the parent.
 */
const mapSource = (src: ChunkSource, transform: Transform): ChunkSource => {
    const composed = transform.clone().mul(src.meta.transform);
    const meta: ChunkSourceMetadata = { ...src.meta, transform: composed };
    return {
        meta,
        read: req => src.read(req),
        close: () => src.close()
    };
};

export { mapSource };
