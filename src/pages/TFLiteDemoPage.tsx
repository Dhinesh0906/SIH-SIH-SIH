import React from "react";
import { TFLiteClassifierWidget } from "@/components/analyze/TFLiteClassifierWidget";

export default function TFLiteDemoPage() {
  return (
    <div className="min-h-[100dvh] bg-slate-50 text-slate-900 p-6">
      <div className="max-w-3xl mx-auto space-y-6">
        <h1 className="text-2xl font-bold">TFLite Species Classifier (Demo)</h1>
        <p className="text-sm text-slate-600">Upload a photo or use the camera to classify one of 39 fish species.
        This uses the TensorFlow Lite Web runtime, 224x224 RGB input, and a 39-float output.
        </p>
        <TFLiteClassifierWidget />
      </div>
    </div>
  );
}
