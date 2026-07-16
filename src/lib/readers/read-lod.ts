import { containerSource, type ContainerSegment } from './container-source';
import { readSogSource } from './read-sog';
import { type ChunkDataPool, type ChunkSource } from '../chunk';
import { dirname, join, readFile, type ReadFileSystem } from '../io/read';
import { type Options } from '../types';

type LodReference = {
    file: number;
    offset: number;
    count: number;
};

type LodNode = {
    children?: LodNode[];
    lods?: Record<string, LodReference>;
};

type LodMeta = {
    version: number;
    count: number;
    counts: number[];
    lodLevels: number;
    environment?: string;
    filenames: string[];
    tree: LodNode;
};

const parseLodMeta = (text: string, filename: string): LodMeta => {
    let meta: LodMeta;
    try {
        meta = JSON.parse(text) as LodMeta;
    } catch (err) {
        const reason = err instanceof Error ? err.message : String(err);
        throw new Error(`Failed to parse lod-meta.json: ${filename}: ${reason}`);
    }

    if (meta.version !== 1) {
        throw new Error(`Unsupported lod-meta.json version: ${meta.version}`);
    }
    if (!Number.isInteger(meta.lodLevels) || meta.lodLevels <= 0) {
        throw new Error(`Invalid lod-meta.json lodLevels: ${filename}`);
    }
    if (!Array.isArray(meta.counts) || meta.counts.length !== meta.lodLevels ||
        meta.counts.some(count => !Number.isInteger(count) || count < 0)) {
        throw new Error(`Invalid lod-meta.json counts: ${filename}`);
    }
    if (!Array.isArray(meta.filenames) || meta.filenames.some(name => typeof name !== 'string')) {
        throw new Error(`Invalid lod-meta.json filenames: ${filename}`);
    }
    if (!meta.tree || typeof meta.tree !== 'object') {
        throw new Error(`Invalid lod-meta.json tree: ${filename}`);
    }

    return meta;
};

const collectFilesByLod = (meta: LodMeta, filename: string): Map<number, Map<number, number>> => {
    const result = new Map<number, Map<number, number>>();
    const traverse = (node: LodNode): void => {
        for (const [key, ref] of Object.entries(node.lods ?? {})) {
            const lod = Number(key);
            if (!Number.isInteger(lod) || lod < 0 || lod >= meta.lodLevels ||
                !ref || !Number.isInteger(ref.file) || ref.file < 0 || ref.file >= meta.filenames.length ||
                !Number.isInteger(ref.offset) || ref.offset < 0 ||
                !Number.isInteger(ref.count) || ref.count < 0) {
                throw new Error(`Invalid lod-meta.json LOD reference: ${filename}`);
            }
            let files = result.get(lod);
            if (!files) {
                files = new Map();
                result.set(lod, files);
            }
            files.set(ref.file, (files.get(ref.file) ?? 0) + ref.count);
        }
        for (const child of node.children ?? []) traverse(child);
    };
    traverse(meta.tree);

    for (let lod = 0; lod < meta.lodLevels; lod++) {
        const count = [...(result.get(lod)?.values() ?? [])].reduce((sum, value) => sum + value, 0);
        if (count !== meta.counts[lod]) {
            throw new Error(
                `lod-meta.json LOD ${lod} count mismatch: expected ${meta.counts[lod]}, found ${count}`
            );
        }
    }

    return result;
};

const resolveLodSelection = (lodSelect: number[], numLods: number): number[] => {
    if (lodSelect.length === 0) return Array.from({ length: numLods }, (_, i) => i);
    return lodSelect
    .map(lod => (lod < 0 ? numLods + lod : lod))
    .filter(lod => lod >= 0 && lod < numLods);
};

/**
 * Open a streamed SOG (`lod-meta.json`) as a lazy, structural multi-LOD source.
 * Referenced SOG units are decoded on demand and retained in the bounded
 * {@link containerSource} cache instead of materializing the complete scene.
 *
 * @param fileSystem - File system containing lod-meta.json and its SOG units.
 * @param filename - Path to lod-meta.json.
 * @param options - Read options, including optional LOD selection.
 * @param pool - Pool whose chunk size is used by the returned source.
 * @returns A lazy multi-LOD source over the selected levels.
 * @ignore
 */
const readLodSource = async (
    fileSystem: ReadFileSystem,
    filename: string,
    options: Options,
    pool: ChunkDataPool
): Promise<ChunkSource> => {
    const baseDir = dirname(filename);
    const related = (name: string) => (baseDir ? join(baseDir, name) : name);
    const meta = parseLodMeta(new TextDecoder().decode(await readFile(fileSystem, filename)), filename);
    const inputLods = resolveLodSelection(options.lodSelect ?? [], meta.lodLevels);
    if (inputLods.length === 0) {
        throw new Error(
            `No valid LODs selected for streamed SOG input file: ${filename} lods: ${JSON.stringify(options.lodSelect ?? [])}`
        );
    }

    const filesByLod = collectFilesByLod(meta, filename);
    const segmentsByLod = inputLods.map((lod): ContainerSegment[] => {
        const files = filesByLod.get(lod);
        return [...(files?.entries() ?? [])]
        .sort(([a], [b]) => a - b)
        .map(([file, count]) => {
            const path = related(meta.filenames[file]);
            return {
                count,
                decode: () => readSogSource(fileSystem, path, pool, { logging: 'silent' })
            };
        });
    });

    return containerSource(segmentsByLod, pool);
};

/**
 * Open the optional environment SOG referenced by a streamed SOG container.
 *
 * @param fileSystem - File system containing lod-meta.json and its SOG units.
 * @param filename - Path to lod-meta.json.
 * @param pool - Pool whose chunk size is used by the returned source.
 * @returns The environment source, or `null` when none is referenced.
 * @ignore
 */
const readLodEnvironmentSource = async (
    fileSystem: ReadFileSystem,
    filename: string,
    pool: ChunkDataPool
): Promise<ChunkSource | null> => {
    const baseDir = dirname(filename);
    const related = (name: string) => (baseDir ? join(baseDir, name) : name);
    const meta = parseLodMeta(new TextDecoder().decode(await readFile(fileSystem, filename)), filename);
    return meta.environment ? readSogSource(fileSystem, related(meta.environment), pool, { logging: 'silent' }) : null;
};

export { parseLodMeta, collectFilesByLod, readLodSource, readLodEnvironmentSource };
export type { LodReference, LodNode, LodMeta };
