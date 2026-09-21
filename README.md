# barcode-vision

Barcode and QR code detection and decoding for Node.js.

## Installation

```bash
npm install barcode-vision
```

## Usage

```js
import { readFile } from "node:fs/promises";
import { scan } from "barcode-vision";

const image = await readFile("./image.jpg");

const results = await scan(image);

console.log(results);
```

## Result

```js
[
  {
    type: "QRCode",
    confidence: 0.94,
    decoded: true,
    value: "https://example.com"
  }
]
```

A detected code that cannot be decoded:

```js
[
  {
    type: "Barcode",
    confidence: 0.91,
    decoded: false,
    value: null
  }
]
```

If no barcode or QR code is detected:

```js
[]
```

## TypeScript

```ts
import { readFile } from "node:fs/promises";
import { scan, type ScanResult } from "barcode-vision";

const image = await readFile("./image.jpg");

const results: ScanResult[] = await scan(image);
```

## Input

`scan()` accepts a Node.js `Buffer`.

```js
const image = await readFile("./image.jpg");

const results = await scan(image);
```

## Requirements

- Node.js
- Node.js-compatible image input as `Buffer`

The model and required runtime dependencies are included with the package.

## License

MIT