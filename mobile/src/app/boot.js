/* Shared runtime boot: ORT environment + model engine (loaded once). */
import { createEngine } from "../core/pipeline.js";

export function configureOrt() {
  ort.env.wasm.wasmPaths = "/vendor/ort/";
  const isolated = typeof crossOriginIsolated !== "undefined" && crossOriginIsolated;
  const cores = navigator.hardwareConcurrency || 2;
  ort.env.wasm.numThreads = isolated ? Math.min(4, Math.max(1, cores - 1)) : 1;
  ort.env.logLevel = "error";
}

let enginePromise = null;

export function bootEngine({ detectorUrl, digitUrl }) {
  if (!enginePromise) {
    configureOrt();
    enginePromise = createEngine({ detectorUrl, digitUrl });
  }
  return enginePromise;
}

export const MODEL_URLS = {
  detectorUrl: "/models/grid_detector_yolov8n_640_int8.onnx",
  digitUrl: "/models/digit_classifier_28x28.onnx",
};
