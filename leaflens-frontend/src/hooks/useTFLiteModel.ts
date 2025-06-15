// src/hooks/useTFLiteModel.ts
'use client';

import { useState, useEffect } from 'react';
import * as tf from '@tensorflow/tfjs';

export function useTFLiteModel(modelPath: string) {
  const [model, setModel] = useState<any>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    if (typeof window === 'undefined') return;

    (async () => {
      try {
        // Load tfjs first to ensure backend is ready
        await tf.ready();
        
        // Use dynamic import with explicit path
        const tflite = await import('@tensorflow/tfjs-tflite/dist/tflite_model.js');
        
        // Load model with error handling
        const m = await tflite.loadTFLiteModel(modelPath);
        
        setModel(m);
      } catch (e) {
        console.error('Failed to load TFLite model:', e);
      } finally {
        setLoading(false);
      }
    })();

    return () => {
      // Cleanup model when component unmounts
      if (model) {
        model.dispose();
      }
    };
  }, [modelPath]);


  async function predict(file: File): Promise<number[]> {
    if (!model) throw new Error('Model not loaded');

    // 1. File → HTMLImageElement
    const imgEl = await new Promise<HTMLImageElement>((resolve, reject) => {
      const reader = new FileReader();
      reader.onload = () => {
        const img = new Image();
        img.src = reader.result as string;
        img.onload = () => resolve(img);
        img.onerror = reject;
      };
      reader.onerror = reject;
      reader.readAsDataURL(file);
    });

    // 2. Preprocess
    const input = tf.browser
      .fromPixels(imgEl)
      .resizeBilinear([256, 256])
      .expandDims(0)
      .toFloat()
      .div(255);

    // 3. Inference
    const rawOutput = (model.predict as any)(input);
    const outputTensor = rawOutput as tf.Tensor;
    const scores = Array.from((outputTensor.dataSync() as Float32Array));

    // Cleanup
    tf.dispose([input, outputTensor]);
    return scores;
  }

  return { predict, loading };
}
