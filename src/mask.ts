import * as ort from "onnxruntime-node";

const MODEL_SIZE = 320;

const MASK_CHANNELS = 32;
const MASK_WIDTH = 80;
const MASK_HEIGHT = 80;

export type ProbabilityMask = {
  width: number;
  height: number;
  data: Float32Array;
};

export type SegmentationMask = {
  width: number;
  height: number;
  data: Uint8Array;
};

/**
 * Combines the 32 prototype masks with the
 * 32 coefficients belonging to one detection.
 *
 * The result is a continuous probability mask.
 *
 * Important:
 * We deliberately do NOT threshold here.
 *
 * Keeping the sigmoid probabilities allows us
 * to resize the mask smoothly before applying
 * the final 0.5 threshold.
 */
export function generateMask(
  maskOutput: ort.Tensor,
  coefficients: Float32Array
): ProbabilityMask {
  const prototypes =
    maskOutput.data as Float32Array;

  const data =
    new Float32Array(
      MASK_WIDTH *
        MASK_HEIGHT
    );

  for (
    let y = 0;
    y < MASK_HEIGHT;
    y++
  ) {
    for (
      let x = 0;
      x < MASK_WIDTH;
      x++
    ) {
      let value = 0;

      for (
        let channel = 0;
        channel < MASK_CHANNELS;
        channel++
      ) {
        const prototypeIndex =
          channel *
            MASK_WIDTH *
            MASK_HEIGHT +
          y * MASK_WIDTH +
          x;

        value +=
          coefficients[channel] *
          prototypes[
            prototypeIndex
          ];
      }

      /*
       * Sigmoid.
       *
       * This gives a probability-like
       * value between 0 and 1.
       */
      const probability =
        1 /
        (1 + Math.exp(-value));

      const pixelIndex =
        y * MASK_WIDTH + x;

      data[pixelIndex] =
        probability;
    }
  }

  return {
    width: MASK_WIDTH,
    height: MASK_HEIGHT,
    data,
  };
}

/**
 * Maps the continuous 80x80 probability
 * mask into original-image coordinates.
 *
 * Bilinear interpolation is used instead
 * of nearest-neighbor sampling.
 *
 * The final threshold is applied only after
 * interpolation.
 */
export function mapMaskToOriginal(
  mask: ProbabilityMask,
  detection: {
    x1: number;
    y1: number;
    x2: number;
    y2: number;
  },
  scale: number,
  padX: number,
  padY: number,
  originalWidth: number,
  originalHeight: number,
  threshold = 0.5
): SegmentationMask {
  const output =
    new Uint8Array(
      originalWidth *
        originalHeight
    );

  /*
   * One 80x80 mask pixel represents
   * 4x4 pixels in the 320x320 model input.
   */
  const modelToMask =
    MASK_WIDTH / MODEL_SIZE;

  for (
    let y = 0;
    y < originalHeight;
    y++
  ) {
    /*
     * Original image -> model coordinates.
     */
    const modelY =
      y * scale + padY;

    /*
     * Don't generate mask pixels
     * outside the detection box.
     */
    if (
      modelY < detection.y1 ||
      modelY > detection.y2
    ) {
      continue;
    }

    /*
     * Model -> mask coordinates.
     */
    const maskY =
      modelY * modelToMask;

    const y0 =
      Math.floor(maskY);

    const y1 =
      y0 + 1;

    const fy =
      maskY - y0;

    const clampedY0 =
      Math.max(
        0,
        Math.min(
          MASK_HEIGHT - 1,
          y0
        )
      );

    const clampedY1 =
      Math.max(
        0,
        Math.min(
          MASK_HEIGHT - 1,
          y1
        )
      );

    for (
      let x = 0;
      x < originalWidth;
      x++
    ) {
      /*
       * Original image -> model coordinates.
       */
      const modelX =
        x * scale + padX;

      /*
       * Restrict the mask to the
       * detected bounding box.
       */
      if (
        modelX < detection.x1 ||
        modelX > detection.x2
      ) {
        continue;
      }

      /*
       * Model -> mask coordinates.
       */
      const maskX =
        modelX * modelToMask;

      const x0 =
        Math.floor(maskX);

      const x1 =
        x0 + 1;

      const fx =
        maskX - x0;

      const clampedX0 =
        Math.max(
          0,
          Math.min(
            MASK_WIDTH - 1,
            x0
          )
        );

      const clampedX1 =
        Math.max(
          0,
          Math.min(
            MASK_WIDTH - 1,
            x1
          )
        );

      /*
       * Four neighboring mask pixels.
       */
      const topLeft =
        mask.data[
          clampedY0 *
            MASK_WIDTH +
          clampedX0
        ];

      const topRight =
        mask.data[
          clampedY0 *
            MASK_WIDTH +
          clampedX1
        ];

      const bottomLeft =
        mask.data[
          clampedY1 *
            MASK_WIDTH +
          clampedX0
        ];

      const bottomRight =
        mask.data[
          clampedY1 *
            MASK_WIDTH +
          clampedX1
        ];

      /*
       * Bilinear interpolation.
       */
      const top =
        topLeft +
        (topRight - topLeft) *
          fx;

      const bottom =
        bottomLeft +
        (bottomRight - bottomLeft) *
          fx;

      const probability =
        top +
        (bottom - top) * fy;

      const outputIndex =
        y * originalWidth + x;

      output[outputIndex] =
        probability >= threshold
          ? 255
          : 0;
    }
  }

  return {
    width: originalWidth,
    height: originalHeight,
    data: output,
  };
}