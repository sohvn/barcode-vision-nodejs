import sharp from "sharp";
import * as ort from "onnxruntime-node";

const MODEL_SIZE = 320;
const PAD_VALUE = 114;

export type PreprocessResult = {
  tensor: ort.Tensor;
  originalWidth: number;
  originalHeight: number;
  scale: number;
  padX: number;
  padY: number;
};

export async function imageToTensor(
  image: Buffer
): Promise<PreprocessResult> {
  const metadata = await sharp(image).metadata();

  if (!metadata.width || !metadata.height) {
    throw new Error("Unable to determine image dimensions.");
  }

  const originalWidth = metadata.width;
  const originalHeight = metadata.height;

  const scale = Math.min(
    MODEL_SIZE / originalWidth,
    MODEL_SIZE / originalHeight
  );

  const resizedWidth = Math.round(originalWidth * scale);
  const resizedHeight = Math.round(originalHeight * scale);

  const padX = Math.floor(
    (MODEL_SIZE - resizedWidth) / 2
  );

  const padY = Math.floor(
    (MODEL_SIZE - resizedHeight) / 2
  );

  const { data } = await sharp(image)
    .resize(resizedWidth, resizedHeight, {
      fit: "fill",
    })
    .extend({
      top: padY,
      bottom: MODEL_SIZE - resizedHeight - padY,
      left: padX,
      right: MODEL_SIZE - resizedWidth - padX,
      background: {
        r: PAD_VALUE,
        g: PAD_VALUE,
        b: PAD_VALUE,
      },
    })
    .removeAlpha()
    .raw()
    .toBuffer({ resolveWithObject: true });

  const floatData = new Float32Array(
    3 * MODEL_SIZE * MODEL_SIZE
  );

  const pixelCount = MODEL_SIZE * MODEL_SIZE;

  for (let i = 0; i < pixelCount; i++) {
    const r = data[i * 3];
    const g = data[i * 3 + 1];
    const b = data[i * 3 + 2];

    floatData[i] = r / 255;
    floatData[pixelCount + i] = g / 255;
    floatData[pixelCount * 2 + i] = b / 255;
  }

  const tensor = new ort.Tensor(
    "float32",
    floatData,
    [1, 3, MODEL_SIZE, MODEL_SIZE]
  );

  return {
    tensor,
    originalWidth,
    originalHeight,
    scale,
    padX,
    padY,
  };
}