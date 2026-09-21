import * as ort from "onnxruntime-node";

import { loadModel } from "./model.js";
import { imageToTensor } from "./preprocess.js";
import {
  postprocess,
  mapDetectionToOriginal,
} from "./postprocess.js";
import { decodeDetection } from "./decoder.js";
import type {
  BarcodeType,
  ScanResult,
} from "./types.js";

export type { BarcodeType, ScanResult };

let modelPromise: Promise<ort.InferenceSession> | null =
  null;

async function getModel(): Promise<ort.InferenceSession> {
  if (!modelPromise) {
    modelPromise = loadModel();
  }

  return await modelPromise;
}

export async function scan(
  image: Buffer
): Promise<ScanResult[]> {
  const session = await getModel();

  const preprocessing =
    await imageToTensor(image);

  const results = await session.run({
    images: preprocessing.tensor,
  });

  const output = results.output0;

  const detections = postprocess(output);

  const scanResults: ScanResult[] = [];

  for (const detection of detections) {
    const originalDetection =
      mapDetectionToOriginal(
        detection,
        preprocessing.scale,
        preprocessing.padX,
        preprocessing.padY,
        preprocessing.originalWidth,
        preprocessing.originalHeight
      );

    const type: BarcodeType =
      originalDetection.classId === 1
        ? "QRCode"
        : "Barcode";

    const decoded = await decodeDetection(
      image,
      {
        x1: originalDetection.x1,
        y1: originalDetection.y1,
        x2: originalDetection.x2,
        y2: originalDetection.y2,
      }
    );

    scanResults.push({
      type,
      confidence: originalDetection.confidence,
      decoded: decoded !== null,
      value: decoded?.value ?? null,
    });
  }

  return scanResults;
}