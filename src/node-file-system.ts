import { randomBytes } from 'crypto';
import { mkdir, open, rename, FileHandle } from 'node:fs/promises';
import { basename, dirname, join } from 'node:path';

import { FileSystem, Writer } from './serialize/file-system';
import { ReadFileSystem } from './serialize/read-file-system';
import { ReadSource } from './serialize/read-source';
import { logger } from './utils/logger';

// write data to a file stream
class FileWriter implements Writer {
    write: (data: Uint8Array) => Promise<void>;
    close: () => Promise<void>;

    constructor(fileHandle: FileHandle, filename: string, tmpFilename: string) {
        this.write = async (data: Uint8Array) => {
            await fileHandle.write(data);
        };

        this.close = async () => {
            // flush to disk
            await fileHandle.sync();
            // close the file
            await fileHandle.close();
            // atomically rename to target filename
            await rename(tmpFilename, filename);
        };
    }
}

class NodeFileSystem implements FileSystem {
    async createWriter(filename: string): Promise<Writer> {
        // write to a temporary file
        const tmpFilename = `.${basename(filename)}.${process.pid}.${Date.now()}.${randomBytes(6).toString('hex')}.tmp`;
        const tmpPathname = join(dirname(filename), tmpFilename);
        const fileHandle = await open(tmpPathname, 'wx');
        return new FileWriter(fileHandle, filename, tmpPathname);
    }

    async mkdir(path: string): Promise<void> {
        await mkdir(path, { recursive: true });
    }
}

class NodeReadFileSystem implements ReadFileSystem {
    createReader(filename: string): ReadSource {
        return new ReadSource(async () => {
            const inputFile = await open(filename, 'r');
            const stats = await inputFile.stat();
            const anim = '▁▂▃▄▅▆▇█';
            const animLen = anim.length;
            let progress = 0;

            logger.info(`reading '${filename}'...`);

            return {
                source: {
                    stream: inputFile.readableWebStream() as ReadableStream<Uint8Array>,
                    totalBytes: stats.size,
                    close: () => inputFile.close()
                },
                onProgress: (readBytes: number, totalBytes?: number) => {
                    const prevProgress = progress;
                    progress = Math.max(progress, totalBytes ? animLen * (readBytes / totalBytes) : 0);

                    for (let i = Math.floor(prevProgress); i < Math.floor(progress); ++i) {
                        logger.progress(anim[i % animLen]);
                    }

                    if (progress >= animLen) {
                        logger.progress('\n');
                    }
                }
            };
        });
    }
}

export { NodeFileSystem, NodeReadFileSystem };
