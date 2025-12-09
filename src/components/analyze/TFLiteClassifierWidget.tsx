import React, { useEffect, useRef, useState } from "react";
import { TFLiteClassifier, TFLiteClassifyResult } from "@/services/tfliteClassifier";
import { Button } from "@/components/ui/button";

export const TFLiteClassifierWidget: React.FC = () => {
  const [ready, setReady] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<TFLiteClassifyResult | null>(null);
  const [busy, setBusy] = useState(false);
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const [streaming, setStreaming] = useState(false);

  const classifierRef = useRef<TFLiteClassifier | null>(null);

  useEffect(() => {
    const init = async () => {
      try {
        const clf = new TFLiteClassifier();
        classifierRef.current = clf;
        await clf.initialize({
          labelsUrl: "/models/species.json",
          // Use default model path (/best_float32.tflite)
        });
        setReady(true);
      } catch (e: any) {
        setError(e?.message || String(e));
      }
    };
    init();
  }, []);

  const handleFile = async (file?: File) => {
    if (!file || !classifierRef.current) return;
    setBusy(true);
    setError(null);
    try {
      const out = await classifierRef.current.classifyFile(file);
      setResult(out);
    } catch (e: any) {
      setError(e?.message || String(e));
    } finally {
      setBusy(false);
    }
  };

  const startCamera = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: "environment" }, audio: false });
      if (videoRef.current) {
        videoRef.current.srcObject = stream as any;
        await videoRef.current.play();
        setStreaming(true);
      }
    } catch (e: any) {
      setError(e?.message || String(e));
    }
  };

  const stopCamera = () => {
    const v = videoRef.current;
    const stream = v?.srcObject as MediaStream | undefined;
    stream?.getTracks().forEach(t => t.stop());
    if (v) {
      v.pause();
      v.srcObject = null;
    }
    setStreaming(false);
  };

  const captureAndClassify = async () => {
    if (!videoRef.current || !classifierRef.current) return;
    setBusy(true);
    setError(null);
    try {
      const out = await classifierRef.current.classifyImage(videoRef.current);
      setResult(out);
    } catch (e: any) {
      setError(e?.message || String(e));
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="space-y-4">
      <div className="text-sm">{ready ? "Model ready" : error ? `Error: ${error}` : "Loading model..."}</div>

      <div className="flex gap-3 items-center">
        <input
          type="file"
          accept="image/*"
          onChange={(e) => handleFile(e.target.files?.[0] || undefined)}
          disabled={!ready || busy}
        />
        <Button onClick={startCamera} disabled={!ready || busy || streaming} variant="secondary">Start Camera</Button>
        <Button onClick={stopCamera} disabled={!streaming} variant="outline">Stop Camera</Button>
        <Button onClick={captureAndClassify} disabled={!streaming || busy}>Capture & Classify</Button>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-4 items-start">
        <video ref={videoRef} className="w-full rounded border border-gray-200" playsInline muted />
        {result && (
          <div className="p-4 rounded border border-gray-200 bg-white/70">
            <div className="text-lg font-semibold">{result.label}</div>
            <div className="text-sm text-gray-600">Confidence: {(result.confidence * 100).toFixed(2)}%</div>
          </div>
        )}
      </div>
    </div>
  );
};
