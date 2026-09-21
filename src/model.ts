import * as ort from "onnxruntime-node";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

const MODEL_PATH = join(__dirname, "..", "model", "best.onnx");

export async function loadModel(): Promise<ort.InferenceSession> {
  return await ort.InferenceSession.create(MODEL_PATH);
}