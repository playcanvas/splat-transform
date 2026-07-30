// The pre-3.2 decimator — see README.md. Its `decimateSource` is exported
// under the `Uniform` name so the two paths can coexist without renaming
// anything inside this directory.
export {
    decimateSource as decimateSourceUniform,
    type DecimateOptions as DecimateUniformOptions,
    type DecimateSpill as DecimateUniformSpill
} from './decimate-source';
