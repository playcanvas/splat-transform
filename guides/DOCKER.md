# Docker Backend Guide

This guide explains how to run `splat-transform` as a containerised backend service on a Linux host with an NVIDIA GPU.

## When you need this

`splat-transform` is a Node.js CLI and the published npm package runs anywhere Node 18+ runs — Docker is not strictly required.

A few features are **GPU-only** and will not run without WebGPU:

- `--filter-cluster` and `--filter-floaters`.
- `.voxel.json` output and `-K` / `--collision-mesh` (see the [Collision Mesh Guide](COLLISION.md)).

A few are **GPU-accelerated but optional**:

- SOG / `meta.json` / `lod-meta.json` / `.html` viewer output (see [SOG Compression Options](../README.md#sog-compression-options) in the README). The only step inside this writer that uses the GPU is k-means clustering of spherical-harmonic coefficients, so:
  - Inputs **without** SH bands (e.g. `.splat`, SH-stripped PLYs, or anything piped through `-H 0` / `--filter-harmonics 0`) write SOG fully on the CPU.
  - Inputs **with** SH bands work on the CPU too via `-g cpu`, but SH clustering is roughly 5-10x slower without a GPU.

All GPU paths go through WebGPU, which on Linux is implemented over Vulkan. That sets the host and container requirements below.

If you don't need the GPU-only features and you're happy paying the CPU cost for SH compression (or your inputs have no SH), skip to the [CPU-only variant](#cpu-only-variant).

## Host prerequisites (outside Docker)

These must be configured on the host machine — they cannot be installed from inside a container, because the container reuses the host's NVIDIA driver and Vulkan ICD.

1. **NVIDIA GPU** with a driver that exposes a Vulkan ICD (most consumer drivers do; some headless cloud drivers do not by default).
2. **Docker Engine** with the **[NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)** installed and configured. This is what makes `--gpus all` work and what mounts the host's NVIDIA Vulkan loader into the container.
3. **Vulkan loader and tools on the host** (used by the toolkit when projecting Vulkan into the container).

### AWS GPU instances

On AWS GPU instances (`g4dn`, `g5`, `g6`, etc.) running Amazon Linux 2023, the default driver is the *compute-only* Tesla driver and does not include Vulkan. To use any of the WebGPU paths you need:

- **NVIDIA GRID Driver v19.2 or newer** (this is the variant that ships the Vulkan ICD). Install it on the host following AWS's [NVIDIA driver installation instructions](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/install-nvidia-driver.html) and pick the *GRID* driver, not the *Tesla* one.
- **Vulkan packages on the host OS**:

  ```bash
  sudo dnf install vulkan vulkan-tools libXext
  ```

This step is *below* Docker — it lives on the EC2 instance itself. The container in the next section assumes both have already been done.

## Minimal Dockerfile

A minimal single-stage Dockerfile that installs the published CLI on top of an NVIDIA CUDA + Vulkan base image:

```dockerfile
# syntax=docker/dockerfile:1.4
FROM nvidia/cuda:12.8.1-devel-ubuntu22.04

# Expose compute + graphics + utility caps so the NVIDIA Container Toolkit
# mounts the host's Vulkan ICD into the container at runtime.
ENV NVIDIA_DRIVER_CAPABILITIES=compute,graphics,utility

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates curl gnupg \
        vulkan-tools \
        libgl1 libglvnd0 libglx0 libegl1 libxext6 \
    && curl -fsSL https://deb.nodesource.com/setup_22.x | bash - \
    && apt-get install -y --no-install-recommends nodejs \
    && npm install -g @playcanvas/splat-transform \
    && rm -rf /var/lib/apt/lists/*

RUN useradd --create-home --uid 2000 splat
USER splat
WORKDIR /work

ENTRYPOINT ["splat-transform"]
```

A few notes on what each part is doing:

- `nvidia/cuda:...-devel-ubuntu22.04` — gives you a CUDA-aware base image that the NVIDIA Container Toolkit recognises. The CUDA toolkit itself is not used by `splat-transform`; we just need a base where `--gpus all` projects the host driver and Vulkan loader correctly.
- `NVIDIA_DRIVER_CAPABILITIES=compute,graphics,utility` — `graphics` is the important one; without it the toolkit will not mount the Vulkan ICD.
- `vulkan-tools` — provides `vulkaninfo`, useful for debugging.
- `libgl1`, `libglvnd0`, `libglx0`, `libegl1`, `libxext6` — runtime libraries Vulkan/EGL drivers depend on.
- `npm install -g @playcanvas/splat-transform` — installs the published CLI on the `PATH`.
- The non-root `splat` user is a good default for batch workers and means files written into a host-mounted directory aren't owned by root.

## Build and run

Build the image:

```bash
docker build -t splat-transform .
```

Run a conversion, mounting the current directory as `/work`:

```bash
docker run --rm --gpus all -v "$PWD":/work splat-transform \
    input.ply output.sog
```

The image runs as a non-root user (UID 2000), so any files written into a host-mounted directory will be owned by UID 2000 rather than your host user, and the container will fail to write at all if the host directory isn't writable by UID 2000. The portable fix is to pass `--user "$(id -u):$(id -g)"`:

```bash
docker run --rm --gpus all --user "$(id -u):$(id -g)" \
    -v "$PWD":/work splat-transform input.ply output.sog
```

The remaining examples omit `--user` for brevity — add it to any invocation that mounts a host directory.

Any normal `splat-transform` invocation works — see the [README](../README.md) for the full CLI surface. For example:

```bash
# GPU SOG compression
docker run --rm --gpus all -v "$PWD":/work splat-transform \
    -i 20 input.ply output.sog

# Voxel + collision mesh
docker run --rm --gpus all -v "$PWD":/work splat-transform \
    room.ply --filter-cluster --seed-pos 0,1,0 \
    --voxel-external-fill --voxel-carve \
    -K room.voxel.json
```

## Verifying GPU access

Two quick checks confirm the GPU is visible.

List GPU adapters via the CLI:

```bash
docker run --rm --gpus all splat-transform --list-gpus
```

You should see at least one Vulkan adapter named after your GPU (e.g. `NVIDIA L4`, `Tesla T4`).

Sanity-check Vulkan directly:

```bash
docker run --rm --gpus all --entrypoint vulkaninfo splat-transform --summary
```

`vulkaninfo` should print a `GPU0` section with an NVIDIA `driverID`. If this fails, the host setup (driver / NVIDIA Container Toolkit / Vulkan ICD) is the problem, not the image.

## CPU-only variant

If you don't need the GPU-only features (`--filter-cluster`, `--filter-floaters`, `.voxel.json`, `-K`) you can skip the GPU setup entirely and use a much smaller base image. SOG / `meta.json` / `lod-meta.json` / `.html` outputs still work in this image — the SH compression step falls back to CPU:

```dockerfile
FROM node:22-slim
RUN npm install -g @playcanvas/splat-transform
WORKDIR /work
ENTRYPOINT ["splat-transform"]
```

Run it without `--gpus`:

```bash
docker run --rm -v "$PWD":/work splat-transform input.ply output.ply
```

Unlike the GPU image, `node:22-slim` runs as root by default, so without `--user` the files written into a bind-mounted host directory will be owned by root on the host. Pass `--user "$(id -u):$(id -g)"` (as described in [Build and run](#build-and-run)), or add a non-root `USER` line to the Dockerfile if this image is for shared use.

If you keep the GPU image but want to force CPU mode for a single run, pass `-g cpu`:

```bash
docker run --rm -v "$PWD":/work splat-transform -g cpu input.ply output.sog
```

Note that `-g cpu` is incompatible with the GPU-only features (`--filter-cluster`, `--filter-floaters`, `.voxel.json`, `-K`). CPU-side SH compression also runs roughly 5-10x slower than GPU.

## Troubleshooting

- **`--list-gpus` reports no adapters.** The Vulkan ICD is not reaching the container. Check that you ran with `--gpus all`, that `NVIDIA_DRIVER_CAPABILITIES` includes `graphics`, and that the NVIDIA Container Toolkit is installed and configured on the host.
- **`vulkaninfo` reports `Could not find a Vulkan ICD`.** The host NVIDIA driver does not expose Vulkan. On AWS, switch from the Tesla driver to the GRID driver (v19.2+) and reinstall, then install `vulkan vulkan-tools libXext` on the host.
- **`vulkaninfo` segfaults or fails to load `libXext.so.6`.** `libxext6` is missing from the image — make sure the `apt-get install` line in the Dockerfile includes it (and on AWS, that `libXext` is also installed on the host).
- **`docker: Error response from daemon: could not select device driver "" with capabilities: [[gpu]]`.** The NVIDIA Container Toolkit is not installed or the Docker daemon has not picked up its config. Reinstall the toolkit and restart Docker.
- **Outputs in the bind-mounted directory aren't owned by my host user.** The container's default user (UID 2000 in the GPU image, root in the slim image) doesn't match your host UID. Pass `--user "$(id -u):$(id -g)"` to `docker run` to align them, or `chown` the output directory to the container UID before mounting.
- **`Permission denied` writing to the bind-mounted directory.** The container's default user has no write access to the host directory. Use `--user "$(id -u):$(id -g)"`, or `chown` the host directory so the container user can write.
- **`splat-transform: command not found` inside the container.** Make sure `npm install -g @playcanvas/splat-transform` ran successfully during the build. The default entrypoint already invokes the binary, so `docker run <image> --help` should work without specifying `splat-transform`.
