import { Writer } from './serialize/writer';
import { Platform } from './serialize/platform';

import { mkdir, readFile as pathReadFile } from 'node:fs/promises';
import { randomBytes } from 'crypto';
import { open, rename } from 'node:fs/promises';
import { FileHandle } from 'node:fs/promises';
import { basename, dirname, join } from 'node:path';

// write data to a file stream
class FileWriter implements Writer {
    write: (data: Uint8Array) => void;
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

class NodePlatform implements Platform {
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

export { NodePlatform };
