import { open } from 'node:fs/promises';

import { version } from '../package.json';
import { GaussianData } from './gaussian-data';
import { readPly } from './readPly';
import { SplatData } from './splat-data';
import { writeCompressedPly } from './writeCompressedPly';

const readSplatData = async (filename: string) => {
    // open input
    console.log(`loading '${filename}'...`);
    const inputFile = await open(filename, 'r');

    // read contents
    console.log('reading contents...');
    const plyFile = await readPly(inputFile);

    // close file
    await inputFile.close();

    return new SplatData(plyFile);
};

const filter = (gaussianData: GaussianData) => {
    const { buffer } = gaussianData;
    for (let i = 0; i < buffer.length; ++i) {
        if (!isFinite(buffer[i])) {
            return false;
        }
    }
    return true;
};

const writeData = async (filename: string, splatData: SplatData) => {
    // open output
    console.log(`writing '${filename}'...`);
    const outputFile = await open(filename, 'w');
    await writeCompressedPly(outputFile, splatData, filter);
    await outputFile.close();
};

const main = async () => {
    console.log(`splat-transform v${version}`);

    if (process.argv.length < 3) {
        console.error('Usage: splat-transform <input-file> <output-file>');
        process.exit(1);
    }

    const inputFilename = process.argv[2];
    const outputFilename = process.argv[3];

    try {
        // open input
        const splatData = await readSplatData(inputFilename);

        // write
        await writeData(outputFilename, splatData);
    } catch (err) {
        // handle errors
        console.error(`error: ${err.message}`);
        process.exit(1);
    }

    console.log('done');
};

export { main };
