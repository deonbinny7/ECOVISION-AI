"use client";

import React, { useEffect, useState } from "react";
import { motion } from "framer-motion";
import { CheckCircle2 } from "lucide-react";

interface PipelineAnimationProps {
    onComplete: () => void;
}

const STAGES = [
    { id: 1, name: "Image Input",      desc: "Receiving RAW image data" },
    { id: 2, name: "Preprocess",       desc: "Resize 224×224, Normalise ÷255" },
    { id: 3, name: "MobileNetV2",      desc: "Depthwise separable CNN feature extraction" },
    { id: 4, name: "GlobalAvgPool",    desc: "Spatial feature aggregation 7×7→1280" },
    { id: 5, name: "BatchNorm",        desc: "Stabilise activations" },
    { id: 6, name: "Dense + L2 Reg",   desc: "ReLU(256) · L2(λ=0.001) · Dropout(0.4)" },
    { id: 7, name: "Dense + Dropout",  desc: "ReLU(128) · L2(λ=0.001) · Dropout(0.3)" },
    { id: 8, name: "Softmax Output",   desc: "σ(z)ᵢ = eᶻⁱ / Σ eᶻʲ → 6 classes" }
];

export default function PipelineAnimation({ onComplete }: PipelineAnimationProps) {
    const [activeStage, setActiveStage] = useState(0);

    useEffect(() => {
        if (activeStage < STAGES.length) {
            const timer = setTimeout(() => setActiveStage((p) => p + 1), 750);
            return () => clearTimeout(timer);
        } else {
            const timer = setTimeout(onComplete, 400);
            return () => clearTimeout(timer);
        }
    }, [activeStage, onComplete]);

    return (
        <div className="w-full flex flex-col items-center p-4">
            <h4 className="text-base font-bold mb-8 text-slate-300 uppercase tracking-widest text-xs">CNN Forward Pass</h4>
            <div className="w-full grid grid-cols-4 md:grid-cols-8 gap-4 relative">
                {STAGES.map((stage, index) => {
                    const isActive = activeStage === index;
                    const isPast = activeStage > index;
                    const isRegLayer = stage.name.includes("L2") || stage.name.includes("Dropout");

                    return (
                        <div key={stage.id} className="flex flex-col items-center text-center">
                            <div className="relative flex-shrink-0 z-10 mb-3 bg-slate-900 rounded-full">
                                <motion.div
                                    animate={{
                                        scale: isActive ? 1.2 : 1,
                                        backgroundColor: isPast ? (isRegLayer ? "#f97316" : "#10b981") : isActive ? "#06b6d4" : "#1e293b",
                                        boxShadow: isActive ? "0 0 20px rgba(6,182,212,0.7)" : isPast && isRegLayer ? "0 0 12px rgba(249,115,22,0.4)" : isPast ? "0 0 12px rgba(16,185,129,0.4)" : "none"
                                    }}
                                    className={`w-12 h-12 rounded-full border-2 flex items-center justify-center ${
                                        isPast && isRegLayer ? 'border-orange-400' : isPast ? 'border-green-400' : isActive ? 'border-cyan-400' : 'border-slate-600'
                                    }`}
                                >
                                    {isPast && <CheckCircle2 className="w-5 h-5 text-slate-900" />}
                                    {isActive && <div className="w-3 h-3 rounded-full bg-slate-900 animate-pulse" />}
                                </motion.div>
                            </div>
                            <motion.div
                                initial={{ opacity: 0.3 }}
                                animate={{ opacity: isActive || isPast ? 1 : 0.3 }}
                                className="flex flex-col"
                            >
                                <span className={`font-bold text-[10px] mb-1 leading-tight ${
                                    isActive ? 'text-cyan-400' : isPast && isRegLayer ? 'text-orange-300' : isPast ? 'text-slate-200' : 'text-slate-600'
                                }`}>{stage.name}</span>
                                <span className="text-[9px] text-slate-500 leading-tight px-1">{stage.desc}</span>
                            </motion.div>
                        </div>
                    );
                })}
            </div>
            <p className="text-slate-600 text-[10px] mt-6 uppercase tracking-widest">Orange = Regularisation · Green = Completed</p>
        </div>
    );
}
