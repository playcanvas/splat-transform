import typescriptConfig from '@playcanvas/eslint-config/typescript';

export default [
    ...typescriptConfig,
    {
        files: ['**/*.ts'],
        rules: {
            '@typescript-eslint/ban-ts-comment': 'off',
            '@typescript-eslint/no-unused-vars': 'off'
        }
    }, {
        files: ['**/*.mjs'],
        rules: {
            'import-x/no-unresolved': 'off'
        }
    }
];
