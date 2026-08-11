#!/usr/bin/env node
/**
 * The published command's entry point, whose only job is to be parseable by
 * whatever node it lands on.
 *
 * fill.js is a modern bundle: on node 10 its very first `import {` is a syntax
 * error, on 12 and 14 it is `??`, and what the user sees is a caret under a
 * brace with no indication of what is wrong or what to do. That happens before
 * any code runs, so no check inside the bundle can ever report it. This file
 * is therefore deliberately ES5 — no arrow functions, no template literals, no
 * optional chaining, and the dynamic import hidden inside a Function body,
 * since older node cannot parse that either.
 *
 * It is copied into the tarball as launch.cjs by scripts/pack-cli.mjs.
 */
'use strict';

var MIN_MAJOR = 18;
var have = process.versions.node;
var major = parseInt(have.split('.')[0], 10);

if (!(major >= MIN_MAJOR)) {
  process.stderr.write(
    'turing-surface-cache: this is node ' + have + ', and filling the cache needs node ' +
      MIN_MAJOR + ' or newer.\n' +
      '  The solver reaches the GPU through WebGPU, and the cache keys through\n' +
      '  WebCrypto, neither of which older node has.\n' +
      '  nodejs.org has current builds; nvm, fnm and asdf install one per user\n' +
      '  without touching what the system depends on.\n',
  );
  process.exit(1);
}

var path = require('path');
var url = require('url');
var target = url.pathToFileURL(path.join(__dirname, 'fill.js')).href;

// `import(target)` as written syntax would be a parse error on the versions
// this file exists to talk to, so it is built at run time instead — by which
// point the check above has already sent them away.
new Function('specifier', 'return import(specifier);')(target).catch(function (e) {
  process.stderr.write(String((e && e.stack) || e) + '\n');
  process.exit(1);
});
