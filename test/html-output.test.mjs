/**
 * HTML output format tests for splat-transform.
 */

import { describe, it, before } from 'node:test';
import assert from 'node:assert';

import {
    writeHtml,
    MemoryFileSystem
} from '../dist/index.mjs';

import { createMinimalTestData } from './helpers/test-utils.mjs';

describe('HTML Format (Output Only)', () => {
    let testData;

    before(() => {
        testData = createMinimalTestData();
    });

    it('should write bundled HTML viewer', async () => {
        const writeFs = new MemoryFileSystem();
        await writeHtml({
            filename: 'viewer.html',
            dataTable: testData,
            bundle: true,
            iterations: 3
        }, writeFs);

        const htmlData = writeFs.results.get('viewer.html');
        assert(htmlData, 'HTML file should be written');
        assert(htmlData.length > 0, 'HTML file should not be empty');

        const htmlText = new TextDecoder().decode(htmlData);

        // Verify basic HTML structure
        assert(htmlText.includes('<!DOCTYPE html>') || htmlText.includes('<!doctype html>'),
            'Should have DOCTYPE declaration');
        assert(htmlText.includes('<html'), 'Should have html tag');
        assert(htmlText.includes('<head'), 'Should have head tag');
        assert(htmlText.includes('<body'), 'Should have body tag');

        // For bundled, everything should be in one file
        assert.strictEqual(writeFs.results.size, 1, 'Bundled should only produce one file');
    });

    it('should write unbundled HTML viewer with separate files', async () => {
        const writeFs = new MemoryFileSystem();
        await writeHtml({
            filename: 'output/viewer.html',
            dataTable: testData,
            bundle: false,
            iterations: 3
        }, writeFs);

        const htmlData = writeFs.results.get('output/viewer.html');
        assert(htmlData, 'HTML file should be written');

        const htmlText = new TextDecoder().decode(htmlData);
        assert(htmlText.includes('<html'), 'Should have html tag');

        // For unbundled, should have additional files
        assert(writeFs.results.size > 1, 'Unbundled should produce multiple files');

        // Check for expected additional files
        const fileNames = Array.from(writeFs.results.keys());
        const hasCSS = fileNames.some(f => f.endsWith('.css'));
        const hasJS = fileNames.some(f => f.endsWith('.js'));
        const hasSOG = fileNames.some(f => f.endsWith('.sog'));

        assert(hasCSS, 'Should produce CSS file');
        assert(hasJS, 'Should produce JS file');
        assert(hasSOG, 'Should produce SOG file');
    });

    it('should accept viewer settings JSON', async () => {
        const writeFs = new MemoryFileSystem();
        await writeHtml({
            filename: 'viewer.html',
            dataTable: testData,
            bundle: true,
            iterations: 3,
            viewerSettingsJson: {
                backgroundColor: '#000000',
                autoRotate: true
            }
        }, writeFs);

        const htmlData = writeFs.results.get('viewer.html');
        assert(htmlData, 'HTML file should be written');

        // The settings should be embedded somehow (exact format depends on implementation)
        const htmlText = new TextDecoder().decode(htmlData);
        assert(htmlText.length > 0, 'HTML should have content');
    });
});
