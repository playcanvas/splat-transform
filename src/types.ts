type Options = {
    overwrite: boolean;
    help: boolean;
    version: boolean;
    quiet: boolean;
    cpu: boolean;
    iterations: number;

    // lcc input options
    lodSelect: number[];

    // html output options
    viewerSettingsPath: string;

    // lod output options
    lodChunkCount: number;
    lodChunkExtent: number;

};

type Param = {
    name: string;
    value: string;
};

export type { Options, Param };
