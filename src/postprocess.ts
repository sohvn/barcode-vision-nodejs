import * as ort from "onnxruntime-node";

export type Detection = {
  classId: number;
  confidence: number;
  x1: number;
  y1: number;
  x2: number;
  y2: number;
  maskCoefficients: Float32Array;
};

const NUM_CLASSES = 2;
const NUM_MASK_COEFFICIENTS = 32;
const NUM_BOX_VALUES = 4;

function calculateIoU(
  a: Detection,
  b: Detection
): number {
  const x1 = Math.max(a.x1, b.x1);
  const y1 = Math.max(a.y1, b.y1);
  const x2 = Math.min(a.x2, b.x2);
  const y2 = Math.min(a.y2, b.y2);

  const intersectionWidth = Math.max(0, x2 - x1);
  const intersectionHeight = Math.max(0, y2 - y1);

  const intersection =
    intersectionWidth * intersectionHeight;

  const areaA =
    Math.max(0, a.x2 - a.x1) *
    Math.max(0, a.y2 - a.y1);

  const areaB =
    Math.max(0, b.x2 - b.x1) *
    Math.max(0, b.y2 - b.y1);

  const union =
    areaA + areaB - intersection;

  return union > 0
    ? intersection / union
    : 0;
}

function applyNms(
  detections: Detection[],
  iouThreshold = 0.5
): Detection[] {
  const sorted = [...detections].sort(
    (a, b) => b.confidence - a.confidence
  );

  const kept: Detection[] = [];

  while (sorted.length > 0) {
    const current = sorted.shift()!;

    kept.push(current);

    for (
      let i = sorted.length - 1;
      i >= 0;
      i--
    ) {
      if (
        sorted[i].classId === current.classId &&
        calculateIoU(
          current,
          sorted[i]
        ) > iouThreshold
      ) {
        sorted.splice(i, 1);
      }
    }
  }

  return kept;
}

export function postprocess(
  output: ort.Tensor,
  confidenceThreshold = 0.25,
  iouThreshold = 0.5
): Detection[] {
  const data = output.data as Float32Array;

  const numCandidates = output.dims[2];

  const detections: Detection[] = [];

  for (
    let i = 0;
    i < numCandidates;
    i++
  ) {
    const x = data[i];

    const y =
      data[numCandidates + i];

    const width =
      data[2 * numCandidates + i];

    const height =
      data[3 * numCandidates + i];

    let bestClass = -1;
    let bestScore = 0;

    for (
      let classId = 0;
      classId < NUM_CLASSES;
      classId++
    ) {
      const score =
        data[
          (NUM_BOX_VALUES + classId) *
            numCandidates +
            i
        ];

      if (score > bestScore) {
        bestScore = score;
        bestClass = classId;
      }
    }

    if (
      bestScore <
      confidenceThreshold
    ) {
      continue;
    }

    const maskCoefficients =
      new Float32Array(
        NUM_MASK_COEFFICIENTS
      );

    for (
      let j = 0;
      j < NUM_MASK_COEFFICIENTS;
      j++
    ) {
      maskCoefficients[j] =
        data[
          (
            NUM_BOX_VALUES +
            NUM_CLASSES +
            j
          ) *
            numCandidates +
          i
        ];
    }

    detections.push({
      classId: bestClass,
      confidence: bestScore,

      x1: x - width / 2,
      y1: y - height / 2,

      x2: x + width / 2,
      y2: y + height / 2,

      maskCoefficients,
    });
  }

  return applyNms(
    detections,
    iouThreshold
  );
}

export function mapDetectionToOriginal(
  detection: Detection,
  scale: number,
  padX: number,
  padY: number,
  originalWidth: number,
  originalHeight: number
): Detection {
  const x1 =
    (detection.x1 - padX) /
    scale;

  const y1 =
    (detection.y1 - padY) /
    scale;

  const x2 =
    (detection.x2 - padX) /
    scale;

  const y2 =
    (detection.y2 - padY) /
    scale;

  return {
    ...detection,

    x1: Math.max(
      0,
      Math.min(originalWidth, x1)
    ),

    y1: Math.max(
      0,
      Math.min(originalHeight, y1)
    ),

    x2: Math.max(
      0,
      Math.min(originalWidth, x2)
    ),

    y2: Math.max(
      0,
      Math.min(originalHeight, y2)
    ),
  };
}