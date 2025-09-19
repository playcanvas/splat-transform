# SplatTransform - 3D Gaussian Splat Converter

[![NPM Version](https://img.shields.io/npm/v/@playcanvas/splat-transform.svg)](https://www.npmjs.com/package/@playcanvas/splat-transform)
[![NPM Downloads](https://img.shields.io/npm/dw/@playcanvas/splat-transform)](https://npmtrends.com/@playcanvas/splat-transform)
[![License](https://img.shields.io/npm/l/@playcanvas/splat-transform.svg)](https://github.com/playcanvas/splat-transform/blob/main/LICENSE)

| [User Guide](https://developer.playcanvas.com/user-manual/gaussian-splatting/editing/splat-transform/) | [Blog](https://blog.playcanvas.com/) | [Forum](https://forum.playcanvas.com/) | [Discord](https://discord.gg/RSaMRzg) |

SplatTransform is an open source CLI tool for converting and editing Gaussian splats. It can:

📥 Read PLY, Compressed PLY, SPLAT, KSPLAT formats  
📤 Write PLY, Compressed PLY, CSV, SOG and HTML viewer formats  
🔗 Merge multiple splats  
🔄 Apply transformations to input splats  
🎛️ Filter out Gaussians or spherical harmonic bands

## Installation

Install or update to the latest version:

```bash
npm install -g @playcanvas/splat-transform
```

> [!IMPORTANT]
> **Windows Users:** You must have the debug version of the Microsoft Visual C++ runtime installed. This is because splat-transform currently depends on [Dawn](https://dawn.googlesource.com/dawn) which is built against the debug VC runtime. We have logged an [issue](https://issues.chromium.org/issues/443906265) against the Dawn project.
>
> In the meantime, to work around this problem, you need to install the debug VC runtime as follows:
>
> 1. Install [Visual Studio 2022 Community](https://visualstudio.microsoft.com/vs/community/)
> 2. In the installer, select "Desktop development with C++"

## Usage

```bash
splat-transform [GLOBAL]  <input.{ply|compressed.ply|splat|ksplat}> [ACTIONS]  ...  <output.{ply|compressed.ply|sog|meta.json|csv|html}> [ACTIONS]
```

**Key points:**
- Every time an `*.ply*` appears, it becomes the current working set; the following ACTIONS are applied in the order listed
- The last file on the command line is treated as the output; anything after it is interpreted as actions that modify the final result

## Supported Formats

**Input:**
- `.ply` - Standard PLY format
- `.compressed.ply` - Compressed PLY format (auto-detected and decompressed on read)
- `.splat` - Binary splat format (antimatter15 format)
- `.ksplat` - Compressed binary splat format (mkkellogg format)

**Output:**
- `.ply` - Standard PLY format
- `.compressed.ply` - Compressed PLY format
- `.sog` - SOG bundled format
- `meta.json` - SOG unbundled format (JSON + WebP images)
- `.csv` - Comma-separated values
- `.html` - Standalone HTML viewer app

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

## Getting Help

```bash
# Show version
splat-transform --version

# Show help
splat-transform --help
```
