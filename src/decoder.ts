import sharp, { type Sharp } from "sharp";
import { readBarcodes } from "zxing-wasm";

export type DecodeResult = {
  value: string;
  format: string;
  attempt: string;
};

type Box = {
  x1: number;
  y1: number;
  x2: number;
  y2: number;
};

async function tryZXing(
  png: Buffer
): Promise<{
  value: string;
  format: string;
} | null> {
  const results = await readBarcodes(
    new Uint8Array(png),
    {
      tryHarder: true,
      maxNumberOfSymbols: 1,
    }
  );

  const result = results.find(
    (candidate) =>
      candidate.isValid &&
      candidate.text.length > 0
  );

  if (!result) {
    return null;
  }

  return {
    value: result.text,
    format: result.format,
  };
}

export async function decodeDetection(
  image: Buffer,
  box: Box
): Promise<DecodeResult | null> {
  const metadata = await sharp(image).metadata();

  if (!metadata.width || !metadata.height) {
    throw new Error(
      "Unable to determine image dimensions."
    );
  }

  const imageWidth = metadata.width;
  const imageHeight = metadata.height;

  const boxWidth = Math.max(
    1,
    box.x2 - box.x1
  );

  const boxHeight = Math.max(
    1,
    box.y2 - box.y1
  );

  // Add some space around the detected code.
  // Decoders generally benefit from having the
  // quiet zone around a barcode/QR code.
  const padding = Math.round(
    Math.max(boxWidth, boxHeight) * 0.15
  );

  const left = Math.max(
    0,
    Math.floor(box.x1 - padding)
  );

  const top = Math.max(
    0,
    Math.floor(box.y1 - padding)
  );

  const right = Math.min(
    imageWidth,
    Math.ceil(box.x2 + padding)
  );

  const bottom = Math.min(
    imageHeight,
    Math.ceil(box.y2 + padding)
  );

  const width = Math.max(
    1,
    right - left
  );

  const height = Math.max(
    1,
    bottom - top
  );

  const crop = sharp(image)
    .extract({
      left,
      top,
      width,
      height,
    })
    .removeAlpha()
    .greyscale();

  const attempts: Array<{
    name: string;
    create: () => Sharp;
  }> = [
    {
      name: "crop",

      create: () =>
        crop.clone(),
    },

    {
      name: "crop+upscale",

      create: () =>
        crop
          .clone()
          .resize({
            width: Math.max(
              400,
              width * 2
            ),
            kernel: "lanczos3",
          }),
    },

    {
      name: "crop+upscale+quiet",

      create: () =>
        crop
          .clone()
          .resize({
            width: Math.max(
              400,
              width * 2
            ),
            kernel: "lanczos3",
          })
          .extend({
            top: 40,
            bottom: 40,
            left: 40,
            right: 40,
            background: {
              r: 255,
              g: 255,
              b: 255,
            },
          }),
    },
  ];

  for (const attempt of attempts) {
    const png = await attempt
      .create()
      .png()
      .toBuffer();

    const result = await tryZXing(png);

    if (result) {
      return {
        ...result,
        attempt: attempt.name,
      };
    }
  }

  return null;
}