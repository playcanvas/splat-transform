/**
 * How a scene was trained, and therefore how a renderer must evaluate it.
 *
 * - `default`     - ordinary gaussians, no special evaluation
 * - `antialiased` - trained with antialiasing (mip-splatting style screen-space filter)
 * - `2dgs`        - trained as 2D gaussian surfels (no third scale axis)
 *
 * The variants are mutually exclusive, hence one enum rather than independent
 * flags. A source that carries no tag reads as `default`.
 */
type SplatModel = 'default' | 'antialiased' | '2dgs';

/** Every model value, for validating externally-supplied strings. */
const SPLAT_MODELS: ReadonlyArray<SplatModel> = ['default', 'antialiased', '2dgs'];

/**
 * Narrow an externally-supplied string to a {@link SplatModel}. Per-format
 * readers use this on whatever their container spells the tag as.
 *
 * @param value - The candidate value.
 * @returns True if `value` is a model name.
 */
const isSplatModel = (value: unknown): value is SplatModel => {
    return SPLAT_MODELS.includes(value as SplatModel);
};

/**
 * Resolve the model of a combined scene. Mixing models can't be represented in
 * one output, so any disagreement falls back to `default` — the safe read, since
 * every variant renders acceptably (if not optimally) as ordinary gaussians.
 *
 * @param models - The models of the inputs being combined.
 * @returns The agreed model, or `default` when they disagree.
 */
const resolveSplatModel = (models: SplatModel[]): SplatModel => {
    const first = models[0] ?? 'default';
    return models.every(m => m === first) ? first : 'default';
};

export { type SplatModel, isSplatModel, resolveSplatModel };
