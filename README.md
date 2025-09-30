# SplatTransform - 3D Gaussian Splat Converter

[![NPM Version](https://img.shields.io/npm/v/@playcanvas/splat-transform.svg)](https://www.npmjs.com/package/@playcanvas/splat-transform)
[![NPM Downloads](https://img.shields.io/npm/dw/@playcanvas/splat-transform)](https://npmtrends.com/@playcanvas/splat-transform)
[![License](https://img.shields.io/npm/l/@playcanvas/splat-transform.svg)](https://github.com/playcanvas/splat-transform/blob/main/LICENSE)
[![Discord](https://img.shields.io/badge/Discord-5865F2?style=flat&logo=discord&logoColor=white&color=black)](https://discord.gg/RSaMRzg)
[![Reddit](https://img.shields.io/badge/Reddit-FF4500?style=flat&logo=reddit&logoColor=white&color=black)](https://www.reddit.com/r/PlayCanvas)
[![X](https://img.shields.io/badge/X-000000?style=flat&logo=x&logoColor=white&color=black)](https://x.com/intent/follow?screen_name=playcanvas)

| [User Guide](https://developer.playcanvas.com/user-manual/gaussian-splatting/editing/splat-transform/) | [Blog](https://blog.playcanvas.com/) | [Forum](https://forum.playcanvas.com/) | [Discord](https://discord.gg/RSaMRzg) |

SplatTransform is an open source CLI tool for converting and editing Gaussian splats. It can:

📥 Read PLY, Compressed PLY, SPLAT, KSPLAT, SPZ, SOG (bundled .sog or unbundled meta.json) formats  
📤 Write PLY, Compressed PLY, CSV, SOG (bundled or unbundled) and HTML viewer formats  
🔗 Merge multiple splats  
🔄 Apply transformations to input splats  
🎛️ Filter out Gaussians or spherical harmonic bands

## Installation

Install or update to the latest version:

```bash
npm install -g @playcanvas/splat-transform
```

## Usage

```bash
splat-transform [GLOBAL] input [ACTIONS]  ...  output [ACTIONS]
```

**Key points:**
- Input files become the working set; ACTIONS are applied in order
- The last file is the output; actions after it modify the final result

## Supported Formats

| Format | Input | Output | Description |
| ------ | ----- | ------ | ----------- |
| `.ply` | ✅ | ✅ | Standard PLY format |
| `.sog` | ✅ | ✅ | Bundled super-compressed format (recommended) |
| `meta.json` | ✅ | ✅ | Unbundled super-compressed format (accompanied by `.webp` textures) |
| `.compressed.ply` | ✅ | ✅ | Compressed PLY format (auto-detected and decompressed on read) |
| `.ksplat` | ✅ | ❌ | Compressed splat format (mkkellogg format) |
| `.splat` | ✅ | ❌ | Compressed splat format (antimatter15 format) |
| `.spz` | ✅ | ❌ | Compressed splat format (Niantic format) |
| `.mjs` | ✅ | ❌ | Generate a scene using an mjs script (Beta) |
| `.csv` | ❌ | ✅ | Comma-separated values spreadsheet |
| `.html` | ❌ | ✅ | Standalone HTML viewer app |
## Actions

Actions can be repeated and applied in any order:

```bash
-t, --translate        <x,y,z>             Translate splats by (x, y, z).
-r, --rotate           <x,y,z>             Rotate splats by euler angles (x, y, z), in degrees.
-s, --scale            <factor>            Uniformly scale splats by factor.
-N, --filter-nan                           Remove Gaussians with NaN or Inf values.
-V, --filter-value     <name,cmp,value>    Keep splats where <name> <cmp> <value>
                                           cmp ∈ {lt,lte,gt,gte,eq,neq}
-H, --filter-harmonics <0|1|2|3>           Remove spherical harmonic bands > n.
-B, --filter-box       <mx,my,mz,Mx,My,Mz> Remove Gaussians outside bounding box.
-S, --filter-sphere    <x,y,z,radius>      Remove Gaussians outside sphere.
-p, --params           <key=val,...>       Pass parameters to .mjs generator script.
```

## Global Options

```bash
-h, --help                                 Show this help and exit.
-v, --version                              Show version and exit.
-w, --overwrite                            Overwrite output file if it exists.
-c, --cpu                                  Use CPU for spherical harmonic compression.
-i, --iterations       <n>                 Iterations for SH compression (more = better). Default: 10.
-C, --camera-pos       <x,y,z>             HTML viewer camera position. Default: (2, 2, -2).
-T, --camera-target    <x,y,z>             HTML viewer target position. Default: (0, 0, 0).
```

## Examples

### Basic Operations

```bash
# Simple format conversion
splat-transform input.ply output.csv

# Convert from .splat format
splat-transform input.splat output.ply

# Convert from .ksplat format
splat-transform input.ksplat output.ply

# Convert to compressed PLY
splat-transform input.ply output.compressed.ply

# Uncompress a compressed PLY back to standard PLY
# (compressed .ply is detected automatically on read)
splat-transform input.compressed.ply output.ply

# Convert to SOG bundled format
splat-transform input.ply output.sog

# Convert to SOG unbundled format
splat-transform input.ply output/meta.json

# Convert from SOG (bundled) back to PLY
splat-transform scene.sog restored.ply

# Convert from SOG (unbundled folder) back to PLY
splat-transform output/meta.json restored.ply

# Convert to HTML viewer with target and camera location
splat-transform -C 0,0,0 -T 0,0,10 input.ply output.html
```

### Transformations

```bash
# Scale and translate
splat-transform bunny.ply -s 0.5 -t 0,0,10 bunny_scaled.ply

# Rotate by 90 degrees around Y axis
splat-transform input.ply -r 0,90,0 output.ply

# Chain multiple transformations
splat-transform input.ply -s 2 -t 1,0,0 -r 0,0,45 output.ply
```

### Filtering

```bash
# Remove entries containing NaN and Inf
splat-transform input.ply --filter-nan output.ply

# Filter by opacity values (keep only splats with opacity > 0.5)
splat-transform input.ply -V opacity,gt,0.5 output.ply

# Strip spherical harmonic bands higher than 2
splat-transform input.ply --filter-harmonics 2 output.ply
```

### Advanced Usage

```bash
# Combine multiple files with different transforms
splat-transform -w cloudA.ply -r 0,90,0 cloudB.ply -s 2 merged.compressed.ply

# Apply final transformations to combined result
splat-transform input1.ply input2.ply output.ply -t 0,0,10 -s 0.5
```

### Generators (Beta)

Generator scripts can be used to synthesize gaussian splat data. See [gen-grid.mjs](generators/gen-grid.mjs) for an example.

```bash
splat-transform gen-grid.mjs -p width=10,height=10,scale=10,color=0.1 scenes/grid.ply -w
```

## Getting Help

```bash
# Show version
splat-transform --version

# Show help
splat-transform --help
```
