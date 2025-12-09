// TFLite Species Classification Service using tfjs-tflite scripts from /public/tflite
// Loads `fish_species_40_float32.tflite` and runs inference in the browser.

type SpeciesLabels = string[];

export interface TFLitePredictionResult {
  species: string;
  confidence: number;
}

function loadScript(src: string): Promise<void> {
  return new Promise((resolve, reject) => {
    const s = document.createElement('script');
    s.src = src;
    s.async = true;
    s.onload = () => resolve();
    s.onerror = (e) => reject(new Error(`Failed to load script: ${src}`));
    document.head.appendChild(s);
  });
}

function softmax(logits: Float32Array | number[]): Float32Array {
  const arr = Array.from(logits);
  const max = Math.max(...arr);
  const exps = arr.map(v => Math.exp(v - max));
  const sum = exps.reduce((a, b) => a + b, 0);
  return new Float32Array(exps.map(v => v / sum));
}

export class TFLiteSpeciesService {
  private model: any | null = null;
  private labels: SpeciesLabels = [];
  private inputSize = 224; // common image classifier size; adjust if model differs

  async initialize(labelsUrl?: string): Promise<void> {
    // Load TF.js and TFLite runtime from public assets
    await loadScript('/tflite/tf.min.js');
    await loadScript('/tflite/tf-tflite.min.js');

    // Load species labels if available, otherwise fallback to defaults
    try {
      if (labelsUrl) {
        const res = await fetch(labelsUrl);
        const data = await res.json();
        this.labels = data.labels as SpeciesLabels;
      } else {
        // Try default location used by existing code
        const res = await fetch('/models/species.json');
        if (res.ok) {
          const data = await res.json();
          this.labels = (data.labels as SpeciesLabels) || this.getDefaultLabels();
        } else {
          this.labels = this.getDefaultLabels();
        }
      }
    } catch {
      this.labels = this.getDefaultLabels();
    }

    // Load the TFLite model from the project root served path
    // Vite will serve files from `public/` as root. If the file is at repo root,
    // ensure it is copied to `public/` or refer via `/${filename}` when served.
    const modelUrl = '/fish_species_40_float32.tflite';

    // @ts-ignore global injected by tf-tflite.min.js
    const tflite = (window as any).tflite;
    if (!tflite || !tflite.loadTFLiteModel) {
      throw new Error('TFLite web runtime not available');
    }

    this.model = await tflite.loadTFLiteModel(modelUrl);
    if (!this.model) {
      throw new Error('Failed to load TFLite model');
    }
  }

  async classify(imageDataUrl: string): Promise<TFLitePredictionResult> {
    if (!this.model) throw new Error('Model not initialized');

    // @ts-ignore TFJS global
    const tf = (window as any).tf;
    if (!tf) throw new Error('TensorFlow.js not available');

    // Create an image element from data URL
    const img = await this.dataURLToImage(imageDataUrl);

    // Convert to tensor and preprocess to [1, H, W, 3] float32 normalized
    let input = tf.browser.fromPixels(img);
    input = tf.image.resizeBilinear(input, [this.inputSize, this.inputSize], true);
    input = input.toFloat();
    // Normalize 0-255 -> 0-1 assuming model expects normalized floats
    input = tf.div(input, tf.scalar(255));
    // Add batch dimension
    input = input.expandDims(0);

    // Run inference via tfjs-tflite model wrapper
    const output = await this.model.predict(input);

    // Output can be a tensor or typed array depending on runtime wrapper
    let logits: Float32Array | number[];
    if (output && typeof output.dataSync === 'function') {
      logits = output.dataSync() as Float32Array;
    } else if (Array.isArray(output)) {
      logits = output as number[];
    } else if (output && output.values) {
      logits = output.values as number[];
    } else {
      throw new Error('Unexpected TFLite predict output');
    }

    const probs = softmax(logits);
    // Pick top class
    let topIdx = 0;
    let topProb = probs[0];
    for (let i = 1; i < probs.length; i++) {
      if (probs[i] > topProb) { topProb = probs[i]; topIdx = i; }
    }

    const species = this.labels[topIdx] || `Class ${topIdx}`;
    const confidence = Math.round(topProb * 100);

    // Dispose tensors
    if (output && typeof output.dispose === 'function') output.dispose();
    input.dispose();

    return { species, confidence };
  }

  private dataURLToImage(dataURL: string): Promise<HTMLImageElement> {
    return new Promise((resolve, reject) => {
      const img = new Image();
      img.crossOrigin = 'anonymous';
      img.onload = () => resolve(img);
      img.onerror = (e) => reject(e);
      img.src = dataURL;
    });
  }

  private getDefaultLabels(): SpeciesLabels {
    return [
      'Salmon (Atlantic)', 'Tuna (Bluefin)', 'Bass (Largemouth)', 'Trout (Rainbow)',
      'Cod (Atlantic)', 'Snapper (Red)', 'Mackerel (Spanish)', 'Grouper (Goliath)',
      'Flounder (Summer)', 'Catfish (Channel)', 'Pike (Northern)', 'Perch (Yellow)',
      'Walleye', 'Carp (Common)', 'Shark (Blacktip)', 'Mahi Mahi', 'Barracuda (Great)',
      'Pompano (Florida)', 'Kingfish (King Mackerel)', 'Amberjack (Greater)', 'Snook (Common)',
      'Tarpon (Atlantic)', 'Redfish (Red Drum)', 'Swordfish', 'Herring (Atlantic)',
      'Anchovy (European)', 'Sardine (Pacific)', 'Tilapia (Nile)', 'Rohu', 'Catla (Catla catla)',
      'Hilsa', 'Pomfret (Silver)', 'Seer Fish (Indo-Pacific King)', 'Ribbon Fish', 'Barramundi (Asian Sea Bass)',
      'Black Drum', 'Sheepshead', 'Rockfish (Striped)', 'Bluefish', 'Tilefish (Golden)'
    ];
  }
}
