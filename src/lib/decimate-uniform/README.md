# decimate-uniform — the pre-3.2 decimator

The decimator that shipped up to 3.1.x, reached through `--decimate-uniform` /
`decimateSourceUniform()`. `src/lib/decimate/` holds the adaptive one
(`--decimate` / `decimateSource()`).

The names describe how each allocates removal, not a ranking. Both are
supported and neither is a fallback for the other: they win on different
content, and the choice is the user's.

- **uniform** — KL-style pairwise cost with a full-SH colour term, uniform 50%
  matching per level, so every region loses the same fraction. Lower memory,
  and measurably better at depth on scenes of uniformly-sized Gaussians:
  uniform texture, single objects, snow. See the `old` column in
  `scenes/DECIMATION-RESULTS.md` (leads at L3–L6 on `crop-snow` and `fr-snow`).
- **adaptive** — field-L2 cost with the scale-free colour term and re-costed
  selection, so removal follows local error and redundant regions collapse
  deeper than distinct ones. Large wins on mixed-scale content, skies
  especially (+9 to +11 dB on `fr-sky`), at higher memory cost.

## The contract

This directory is **bit-for-bit output-compatible with the 3.1.6 binary**. That
is its value: a decimation you can reproduce exactly against a known-good
reference, and the baseline every quality comparison in
`scenes/DECIMATION-RESULTS.md` is measured against.

Every file is a copy of its `src/lib/decimate/` counterpart at the last 3.1.x
commit. You can prove it, per file:

```bash
git diff main:src/lib/decimate/select.ts src/lib/decimate-uniform/select.ts
```

Empty output means the file is untouched. Deviations are limited to these, all
mechanical:

- **Import paths.** `../gpu/gpu-edge-cost` → `./gpu-edge-cost`,
  `../gpu/gpu-knn` → `./gpu-knn`, `./moment-match` →
  `../decimate/moment-match`.
- **`gpu-knn.ts`** consumes the current `FlatKdTree` (interleaved
  `nodePositions` / `nodeChildren`) and so drops the packing loops that built
  that same layout internally. `buildFlatKdTree` is verified structurally
  identical to the 3.1.x `KdTree.flatten()` at every size, so the uploaded
  bytes are unchanged.

## Shared dependencies

Only two, both deliberate:

- `../decimate/moment-match.ts` — has no diff against 3.1.x, and its
  `mergeGroups` worker handler is shared. Duplicating it would mean a
  duplicate worker task for no benefit.
- `../spatial/kd-tree.ts` — `KdTree`'s build and query paths are unchanged
  from 3.1.x, and it is shared with k-means.

Otherwise this directory imports nothing from `../decimate/`, so work on the
adaptive path cannot change uniform output.

## Changing things here

Changes are fine — bug fixes, performance work, new capability — but they are
output changes to a path whose selling point is reproducibility, so they need
to be deliberate rather than incidental. Before landing one:

- Re-run the whole-scene comparison if you expect output to be unchanged. That
  is what `tools/decimate-parity.mjs` is for — it chains halvings through a
  reference binary's `--decimate` and this tree's `--decimate-uniform`, compares
  byte for byte, reports PSNR for both, and exits non-zero on any mismatch:

  ```bash
  node tools/decimate-parity.mjs sky --ref splat-transform
  node tools/decimate-parity.mjs snow --ref splat-transform
  ```

  Equivalence was last verified against 3.1.6 on both study scenes, `fr-sky`
  (5.81M, 3 SH bands, multi-block) and `fr-snow` (26.1M, DC only, 13 blocks),
  six chained halvings each: every level byte-identical, PSNR matching the
  published `old` columns exactly.
- If output *should* change, re-baseline `scenes/DECIMATION-RESULTS.md` — the
  `old` column is this path, and the study's conclusions are stated relative
  to it. That document is local-only (`scenes/` is gitignored).
- Repin the digest in `test/decimate-uniform-parity.test.mjs`, which is the
  in-suite tripwire for accidental drift.

## If it is ever retired

Nothing in `src/lib/decimate/` refers to this directory, so: `rm -rf` it, drop
`decimateSourceUniform` from `src/lib/index.ts`, drop `--decimate-uniform`
from the CLI, and delete `test/decimate-uniform-parity.test.mjs` and
`tools/decimate-parity.mjs`.
