export type TFLiteModule = any;

function loadScript(src: string): Promise<void> {
  return new Promise((resolve, reject) => {
    const script = document.createElement('script');
    script.src = src;
    script.async = true;
    script.onload = () => resolve();
    script.onerror = () => reject(new Error(`Failed to load script: ${src}`));
    document.head.appendChild(script);
  });
}

export async function loadTFLiteModule(): Promise<TFLiteModule> {
  // Load the native TFLite C++ Web API JS from absolute public path
  await loadScript('/tflite/tflite_web_api_cc_simd.js');
  const factory = (window as any).tflite_web_api_ModuleFactory;
  if (typeof factory !== 'function') {
    throw new Error('tflite_web_api_ModuleFactory not available after script load');
  }
  // Initialize the module with absolute wasm locator for Vercel
  const module = await factory({
    locateFile: (path: string) => {
      const absolutePath = `/tflite/${path}`;
      console.debug('[TFLite] Locating wasm:', path, '=>', absolutePath);
      return absolutePath;
    },
    noExitRuntime: true,
  });
  return module;
}

// Ensure model path is absolute and correct
export const BEST_MODEL_PATH = '/best_float32.tflite';
