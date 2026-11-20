# syntax=docker/dockerfile:1

FROM node:22-slim AS build

WORKDIR /src

COPY package.json package-lock.json ./
RUN npm ci

COPY . .
RUN npm run build \
    && mkdir /tmp/package \
    && npm pack --pack-destination /tmp/package

FROM node:22-slim AS node

FROM ubuntu:22.04

# The NVIDIA Container Toolkit uses these variables to expose the host's
# graphics driver and Vulkan ICD when the container is run with --gpus.
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,graphics,utility

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
        vulkan-tools \
        libgl1 \
        libglvnd0 \
        libglx0 \
        libegl1 \
        libxext6 \
        mesa-vulkan-drivers \
    && rm -rf /var/lib/apt/lists/*

COPY --from=node /usr/local/bin/ /usr/local/bin/
COPY --from=node /usr/local/lib/node_modules/ /usr/local/lib/node_modules/
COPY --from=build /tmp/package/*.tgz /tmp/splat-transform.tgz

RUN npm install --global /tmp/splat-transform.tgz \
    && rm /tmp/splat-transform.tgz \
    && useradd --create-home --uid 2000 --user-group splat \
    && mkdir /work \
    && chown splat:splat /work

USER splat
WORKDIR /work

ENTRYPOINT ["splat-transform"]
