import { FileSystem } from './file-system';

const writeFile = async (fs: FileSystem, filename: string, data: Uint8Array | string) => {
    const outputFile = await fs.createWriter(filename);
    outputFile.write(data instanceof Uint8Array ? data : new TextEncoder().encode(data));
    await outputFile.close();
};

export { writeFile };
