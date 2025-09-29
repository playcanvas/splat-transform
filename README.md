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
splat-transform [GLOBAL]  <input.{ply|compressed.ply|splat|ksplat|spz|sog|meta.json|mjs}> [ACTIONS]  ...  <output.{ply|compressed.ply|sog|meta.json|csv|html}> [ACTIONS]
```

**Key points:**
- Every time an `*.ply*` appears, it becomes the current working set; the following ACTIONS are applied in the order listed
- The last file on the command line is treated as the output; anything after it is interpreted as actions that modify the final result

## Supported Formats

| Format | Input | Output | Description |
| ------ | ----- | ------ | ----------- |
| `.ply` | ✅ | ✅ | Standard PLY format |
| `.compressed.ply` | ✅ | ✅ | Compressed PLY format (auto-detected and decompressed on read) |
| `.ksplat` | ✅ | ❌ | Compressed splat format (mkkellogg format) |
| `.splat` | ✅ | ❌ | Compressed splat format (antimatter15 format) |
| `.spz` | ✅ | ❌ | Compressed splat format (Niantic format) |
| `.mjs` | ✅ | ❌ | Generate a scene using an mjs script (Beta) |
| `.sog` | ✅ | ✅ | Bundled super-compressed format (recommended) |
| `meta.json` | ✅ | ✅ | Unbundled super-compressed format (accompanied by `.webp` textures) |
| `.csv` | ❌ | ✅ | Comma-separated values spreadsheet |
| `.html` | ❌ | ✅ | Standalone HTML viewer app |
## Actions

Actions can be repeated and applied in any order:

```bash
-t, --translate  x,y,z                  Translate splats by (x, y, z)
-r, --rotate     x,y,z                  Rotate splats by Euler angles (deg)
-s, --scale      x                      Uniformly scale splats by factor x
-n, --filterNaN                         Remove any Gaussian containing NaN/Inf
-c, --filterByValue name,cmp,value      Keep splats where <name> <cmp> <value>
                                        cmp ∈ {lt,lte,gt,gte,eq,neq}
-b, --filterBands  {0|1|2|3}            Strip spherical-harmonic bands > N
-P, --params name=value[,name=value...] Pass parameters to .mjs generator script
```

## Global Options

```bash
-w, --overwrite                         Overwrite output file if it already exists
-h, --help                              Show help and exit
-v, --version                           Show version and exit
-g, --no-gpu                            Disable gpu when compressing spherical harmonics.
-i, --iterations  <number>              Specify the number of iterations when compressing spherical harmonics. More iterations generally lead to better results. Default is 10.
-p, --cameraPos                         Specify the viewer starting position. Default is 2,2,-2.
-e, --cameraTarget                      Specify the viewer starting target. Default is 0,0,0.
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
splat-transform -a 0,0,0 -e 0,0,10 input.ply output.html
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
splat-transform input.ply --filterNaN output.ply

# Filter by opacity values (keep only splats with opacity > 0.5)
splat-transform input.ply -c opacity,gt,0.5 output.ply

# Strip spherical harmonic bands higher than 2
splat-transform input.ply --filterBands 2 output.ply
```

### Advanced Usage

```bash
# Combine multiple files with different transforms
splat-transform -w cloudA.ply -r 0,90,0 cloudB.ply -s 2 merged.compressed.ply

# Apply final transformations to combined result
splat-transform input1.ply input2.ply output.ply -p 0,0,10 -e 0.5
```

### Generators (Beta)

Generator scripts can be used to synthesize gaussian splat data. See [gen-grid.mjs](generators/gen-grid.mjs) for an example.

```bash
splat-transform gen-grid.mjs -P width=10,height=10,scale=10,color=0.1 scenes/grid.ply -w
```

## Getting Help

```bash
# Show version
splat-transform --version

# Show help
splat-transform --help
```
