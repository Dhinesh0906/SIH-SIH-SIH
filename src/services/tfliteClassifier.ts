// TFLite classification pipeline for PWA (39-class YOLOv8n-cls)
// Loads tflite web runtime, model, preprocesses RGBA->RGB, 224x224, 0-1 float32, runs inference.

declare global {
  interface Window {
    tf?: any;
    tflite?: any;
  }
}
import { loadTFLiteModule, BEST_MODEL_PATH } from "@/lib/tfliteLoader";

export type LabelList = string[];

export interface TFLiteClassifyResult {
  topIndex: number;
  label: string;
  confidence: number; // 0-1
  probs: Float32Array;
}

export class TFLiteClassifier {
  private model: any | null = null;
  private labels: LabelList = [];
  // Match Colab requirements: 640x640 RGB float32
  private readonly size = 640;

  async initialize(options?: { labelsUrl?: string; modelUrl?: string }): Promise<void> {
    await this.ensureTF();
    await this.ensureTFLite();

    // Load labels strictly from provided or default path
    const labelsUrl = options?.labelsUrl ?? "/species_labels.json";
    try {
      const res = await fetch(labelsUrl);
      if (res.ok) {
        const data = await res.json();
        // Accept formats: { labels: [...] } or index->name object
        if (Array.isArray(data?.labels)) {
          this.labels = data.labels as string[];
        } else if (data && typeof data === "object") {
          const keys = Object.keys(data);
          const sorted = keys.sort((a, b) => Number(a) - Number(b));
          this.labels = sorted.map((k) => String((data as any)[k]));
        }
      }
    } catch {}
    if (!this.labels.length) throw new Error("Labels not loaded. Ensure /species_labels.json exists and is accessible.");

    // Load model with absolute path
    const modelUrl = options?.modelUrl ?? BEST_MODEL_PATH;
    const absoluteModelUrl = modelUrl.startsWith('/') ? modelUrl : `/${modelUrl}`;
    try {
      console.debug('[Classifier] Loading model from:', absoluteModelUrl);
      this.model = await window.tflite!.loadTFLiteModel(absoluteModelUrl, { enableWebXnnpack: false });
      if (this.model) return;
    } catch (e: any) {
      throw new Error(`Failed to load TFLite model at ${absoluteModelUrl}: ${e?.message || e}`);
    }
  }

  async classifyFile(file: File): Promise<TFLiteClassifyResult> {
    const img = await this.fileToImage(file);
    return this.classifyImage(img);
  }

  async classifyImage(img: HTMLImageElement | HTMLVideoElement | HTMLCanvasElement): Promise<TFLiteClassifyResult> {
    if (!this.model) throw new Error("Model not initialized");
    const input = this.preprocessToTensor(img);
    const output = this.model.predict(input);

    let logits: Float32Array;
    if (output && typeof output.dataSync === "function") {
      logits = output.dataSync() as Float32Array;
    } else if (output && output.values) {
      logits = output.values as Float32Array;
    } else if (Array.isArray(output)) {
      logits = new Float32Array(output as number[]);
    } else {
      throw new Error("Unexpected TFLite output format");
    }

    const probs = softmax(logits);
    let topIndex = 0;
    let topProb = probs[0];
    for (let i = 1; i < probs.length; i++) {
      if (probs[i] > topProb) { topProb = probs[i]; topIndex = i; }
    }
    const label = this.labels[topIndex] ?? `Class ${topIndex}`;

    if (output && typeof output.dispose === "function") output.dispose();
    input.dispose();

    return { topIndex, label, confidence: topProb, probs };
  }

  // Preprocess: plain resize -> 640x640, RGBA->RGB, normalize to [0,1], shape [1,640,640,3]
  private preprocessToTensor(src: HTMLImageElement | HTMLVideoElement | HTMLCanvasElement) {
    const sz = this.size;
    const canvas = document.createElement("canvas");
    canvas.width = sz; canvas.height = sz;
    const ctx = canvas.getContext("2d", { willReadFrequently: true })!;
    // Plain resize (no crop/letterbox) to match Colab
    ctx.clearRect(0, 0, sz, sz);
    ctx.drawImage(src as any, 0, 0, sz, sz);

    const imageData = ctx.getImageData(0, 0, sz, sz);
    const rgba = imageData.data; // Uint8ClampedArray length = sz*sz*4

    // RGBA -> RGB float32 normalized
    const floatData = new Float32Array(sz * sz * 3);
    let si = 0;
    let di = 0;
    while (si < rgba.length) {
      const r = rgba[si];
      const g = rgba[si + 1];
      const b = rgba[si + 2];
      // skip alpha
      floatData[di++] = r / 255;
      floatData[di++] = g / 255;
      floatData[di++] = b / 255;
      si += 4;
    }

    // Use TFJS to create [1,640,640,3]
    const tf = window.tf!;
    const input = tf.tensor4d(floatData, [1, sz, sz, 3], "float32");
    return input;
  }

  private async ensureTF() {
    if (window.tf) return;
    await loadScript("/tflite/tf.min.js");
    if (!window.tf) throw new Error("Failed to load TensorFlow.js");
  }

  private async ensureTFLite() {
    if (window.tflite) return;
    // Prime the native TFLite C++ Web API to ensure wasm path resolution works in prod
    try {
      const nativeModule = await loadTFLiteModule();
      console.debug('[Classifier] Native TFLite module initialized');
    } catch (e) {
      console.warn('[Classifier] Native TFLite module init failed (expected if using tfjs-tflite):', e);
    }
    // Load the TFJS TFLite compatibility layer used by current classifier
    await loadScript("/tflite/tf-tflite.min.js");
    if (!window.tflite) throw new Error("Failed to load TFLite Web runtime");
    // Force absolute public path for wasm assets
    console.debug('[Classifier] Setting wasm path to /tflite/');
    window.tflite.setWasmPath?.("/tflite/");
  }

  private fileToImage(file: File): Promise<HTMLImageElement> {
    return new Promise((resolve, reject) => {
      const url = URL.createObjectURL(file);
      const img = new Image();
      img.crossOrigin = "anonymous";
      img.onload = () => { URL.revokeObjectURL(url); resolve(img); };
      img.onerror = (e) => { URL.revokeObjectURL(url); reject(e); };
      img.src = url;
    });
  }
}

function loadScript(src: string): Promise<void> {
  return new Promise((resolve, reject) => {
    const s = document.createElement("script");
    s.src = src;
    s.async = true;
    s.onload = () => resolve();
    s.onerror = () => reject(new Error(`Failed to load script ${src}`));
    document.head.appendChild(s);
  });
}

function softmax(logits: Float32Array): Float32Array {
  const arr = Array.from(logits as unknown as number[]);
  const max = Math.max(...arr);
  const exps = arr.map((v) => Math.exp(v - max));
  const sum = exps.reduce((a, b) => a + b, 0);
  return new Float32Array(exps.map((v) => v / sum));
}
