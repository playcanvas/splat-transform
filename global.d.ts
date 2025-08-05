// these declarations are so typescript considers the
// viewer source stored in submodules/supersplat-viewer/dist as strings.
// we use rollup-plugin-string to inline these files.
declare module '*.css' {
    const content: string;
    export default content;
}

declare module '*.js' {
    const content: string;
    export default content;
}

declare module '*.html' {
    const content: string;
    export default content;
}