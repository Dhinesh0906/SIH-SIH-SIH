import { createRoot } from "react-dom/client";
// Ensure native TFLite C++ Web API runtime is available (absolute public path)
import "/tflite/tflite_web_api_cc_simd.js";
import App from "./App.tsx";
import "./index.css";
import "./i18n";

createRoot(document.getElementById("root")!).render(<App />);
