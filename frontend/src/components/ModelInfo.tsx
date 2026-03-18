"use client";

import React, { useEffect, useState } from "react";
import { motion } from "framer-motion";

interface ModelData {
  model_name: string;
  task: string;
  classes: string[];
  num_classes: number;
  input_shape: number[];
  architecture: { base: string; custom_head: string[] };
  math_formulation: Record<string, string>;
  hyperparameters: Record<string, number | number[]>;
  training_strategy: Record<string, string | string[]>;
}

export default function ModelInfo() {
  const [data, setData] = useState<ModelData | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetch("http://localhost:8000/model-info")
      .then((r) => r.json())
      .then((d) => { setData(d); setLoading(false); })
      .catch(() => setLoading(false));
  }, []);

  const math = data?.math_formulation ?? {};
  const hp = data?.hyperparameters ?? {};
  const strategy = data?.training_strategy ?? {};

  return (
    <div className="w-full max-w-5xl mx-auto space-y-10 py-4">
      <div className="text-center">
        <h2 className="text-3xl font-black tracking-tighter mb-2">
          Architecture & <span className="text-[#00ff99]">Math Model</span>
        </h2>
        <p className="text-slate-500 text-sm uppercase tracking-widest">Mathematical Formulation</p>
      </div>

      {loading && <div className="text-center text-slate-400 animate-pulse py-16">Loading model info...</div>}

      {data && (
        <>
          {/* Architecture Pipeline */}
          <div className="glass-card p-6">
            <h3 className="text-sm font-black uppercase tracking-widest text-slate-400 mb-6">Network Architecture</h3>
            <div className="flex flex-col gap-2">
              {/* Base Model */}
              <div className="flex items-center gap-4 p-4 rounded-xl bg-[#00e5ff]/5 border border-[#00e5ff]/20">
                <div className="w-3 h-3 rounded-full bg-[#00e5ff] shrink-0" />
                <div>
                  <span className="font-bold text-[#00e5ff]">MobileNetV2</span>
                  <span className="text-slate-500 text-sm ml-2">— Pre-trained ImageNet backbone · 7×7×1280 output</span>
                </div>
              </div>

              {data.architecture.custom_head.map((layer, i) => {
                const isReg = layer.includes("L2") || layer.includes("Dropout");
                const isOut = layer.includes("Softmax");
                return (
                  <motion.div
                    key={i}
                    initial={{ opacity: 0, x: -20 }}
                    animate={{ opacity: 1, x: 0 }}
                    transition={{ delay: i * 0.07 }}
                    className={`flex items-center gap-4 p-3 rounded-xl border ${
                      isOut ? "border-[#00ff99]/30 bg-[#00ff99]/5"
                      : isReg ? "border-orange-500/20 bg-orange-500/5"
                      : "border-white/5 bg-white/3"
                    }`}
                  >
                    <div className={`w-2 h-2 rounded-full shrink-0 ${isOut ? "bg-[#00ff99]" : isReg ? "bg-orange-400" : "bg-slate-500"}`} />
                    <span className={`font-mono text-sm ${isOut ? "text-[#00ff99] font-bold" : isReg ? "text-orange-300" : "text-slate-300"}`}>
                      {layer}
                    </span>
                  </motion.div>
                );
              })}
            </div>
            <p className="text-slate-600 text-xs mt-4">Orange = regularisation layers · Green = output</p>
          </div>

          {/* Math Formulations */}
          <div className="glass-card p-6">
            <h3 className="text-sm font-black uppercase tracking-widest text-slate-400 mb-6">Mathematical Formulation</h3>
            <div className="space-y-4">
              {Object.entries(math).map(([key, val]) => (
                <div key={key} className="p-4 rounded-xl bg-white/3 border border-white/5">
                  <span className="text-xs font-black uppercase tracking-wider text-[#00ff99] block mb-2">
                    {key.replace(/_/g, " ")}
                  </span>
                  <code className="text-slate-200 text-sm font-mono break-words">{val as string}</code>
                </div>
              ))}
            </div>
          </div>

          {/* Hyperparameters Grid */}
          <div className="glass-card p-6">
            <h3 className="text-sm font-black uppercase tracking-widest text-slate-400 mb-4">Hyperparameters</h3>
            <div className="grid grid-cols-2 md:grid-cols-3 gap-3">
              {Object.entries(hp).map(([key, val]) => (
                <div key={key} className="p-3 bg-white/3 rounded-lg border border-white/5">
                  <div className="text-[10px] uppercase tracking-widest text-slate-500 mb-1">{key.replace(/_/g, " ")}</div>
                  <div className="font-bold text-[#00e5ff] font-mono text-sm">{JSON.stringify(val)}</div>
                </div>
              ))}
            </div>
          </div>

          {/* Training Strategy */}
          <div className="glass-card p-6">
            <h3 className="text-sm font-black uppercase tracking-widest text-slate-400 mb-4">Training Strategy</h3>
            <div className="space-y-3">
              {Object.entries(strategy).map(([phase, desc]) => (
                <div key={phase} className="flex gap-4 items-start">
                  <span className="text-[#00ff99] font-black uppercase text-xs tracking-widest min-w-[80px] mt-1">{phase}</span>
                  <span className="text-slate-400 text-sm">{Array.isArray(desc) ? desc.join(", ") : desc as string}</span>
                </div>
              ))}
            </div>
          </div>
        </>
      )}
    </div>
  );
}
