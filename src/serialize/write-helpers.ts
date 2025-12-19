import { Platform } from './platform';

const writeFile = async (platform: Platform, filename: string, data: Uint8Array | string) => {
    const outputFile = await platform.createWriter(filename);
    outputFile.write(data instanceof Uint8Array ? data : new TextEncoder().encode(data));
    await outputFile.close();
};

export { writeFile };

