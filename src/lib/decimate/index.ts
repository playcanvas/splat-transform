// The adaptive decimator (`--decimate-adaptive`). Its `decimateSource` is
// exported under the `Adaptive` name so the two paths can coexist without
// renaming anything inside this directory; `decimateSource` is the pre-3.2
// path in ../decimate-uniform/.
export {
    decimateSource as decimateSourceAdaptive,
    type DecimateOptions as DecimateAdaptiveOptions,
    type DecimateSpill as DecimateAdaptiveSpill
} from './decimate-source';
export { mergeGroup, createMergeScratch, splatMass, makeGaussianSamples, type SplatView, type MergedOut, type MergeScratch } from './moment-match';
export { kdPartition, coherenceRuns, type BlockRange, type ResidentPositions } from './partition';
