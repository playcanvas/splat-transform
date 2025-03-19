import { version } from '../package.json';
import { open } from 'node:fs/promises';
import { readPly } from './readPly';
import { PlyFile } from './ply';
import { Splat } from './splat';
import { writeCompressedPly } from './writeCompressedPly';

const readData = async (filename: string) => {
    // open input
    console.log(`loading '${filename}'...`);
    const inputFile = await open(filename, 'r');

    // read contents
    console.log(`reading contents...`);
    const plyFile = await readPly(inputFile);

    // close file
    await inputFile.close();

    return plyFile;
};

const processData = (plyFile: PlyFile) => {
    // check we have the necessary elements for processing
};

const writeData = async (filename: string, plyFile: PlyFile) => {
    // open output
    console.log(`writing '${filename}'...`);
    const outputFile = await open(filename, 'w');
    await writeCompressedPly(outputFile, new Splat(plyFile));
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
        const plyFile = await readData(inputFilename);

        // process
        processData(plyFile);

        // write
        await writeData(outputFilename, plyFile);
    } catch (err) {
        // handle errors
        console.error(`error: ${err.message}`);
        process.exit(1);
    }

    console.log('done');
};

export { main };
