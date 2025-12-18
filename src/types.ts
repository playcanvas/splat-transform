type Options = {
    overwrite: boolean;
    help: boolean;
    version: boolean;
    quiet: boolean;
    iterations: number;
    listGpus: boolean;
    device: number;  // -1 = auto, -2 = CPU, 0+ = GPU index

    // lcc input options
    lodSelect: number[];

    // html output options
    viewerSettingsPath: string;
    unbundled: boolean;

    // lod output options
    lodChunkCount: number;
    lodChunkExtent: number;
};

type Param = {
    name: string;
    value: string;
};

export type { Options, Param };
