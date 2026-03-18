"use client";

import React, { useState } from "react";
import { Upload, ShieldCheck, RefreshCw, Leaf, BarChart3 } from "lucide-react";
import PipelineAnimation from "./PipelineAnimation";
import { motion, AnimatePresence } from "framer-motion";

const CLASSES = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash'];

const CLASS_COLORS: Record<string, string> = {
  cardboard: '#b45309',
  glass:     '#0891b2',
  metal:     '#6366f1',
  paper:     '#059669',
  plastic:   '#d97706',
  trash:     '#dc2626',
};

interface PredictionResult {
  class: string;
  confidence: number;
  all_probabilities: Record<string, number>;
  disposal_tip: string;
  co2_impact: string;
}

export default function Dashboard() {
    const [preview, setPreview] = useState<string | null>(null);
    const [isAnalyzing, setIsAnalyzing] = useState(false);
    const [prediction, setPrediction] = useState<PredictionResult | null>(null);
    const [gradcam, setGradcam] = useState<string | null>(null);
    const [showResult, setShowResult] = useState(false);
    const [showHeatmap, setShowHeatmap] = useState(false);
    const startAnalysis = async (selectedFile: File) => {
        setIsAnalyzing(true);
        setPrediction(null);
        setGradcam(null);
        setShowResult(false);
        setShowHeatmap(false);

        const formData = new FormData();
        formData.append("file", selectedFile);

        const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

        try {
            // Fire both requests in parallel
            const [predRes, explainRes] = await Promise.all([
                fetch(`${API_URL}/predict`, { method: "POST", body: formData }),
                fetch(`${API_URL}/explain`, { method: "POST", body: (() => { const fd = new FormData(); fd.append("file", selectedFile); return fd; })() }),
            ]);

            if (predRes.ok) setPrediction(await predRes.json());
            if (explainRes.ok) {
                const exData = await explainRes.json();
                if (exData.gradcam_image) setGradcam(exData.gradcam_image);
            }
        } catch (e) {
            console.error("Failed prediction", e);
            setPrediction({ class: "error", confidence: 0, all_probabilities: {}, disposal_tip: "", co2_impact: "" });
        }
    };

    const onDrop = (e: React.DragEvent) => {
        e.preventDefault();
        const f = e.dataTransfer.files[0];
        if (f && f.type.startsWith("image/")) { setPreview(URL.createObjectURL(f)); startAnalysis(f); }
    };

    const onFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
        const f = e.target.files?.[0];
        if (f) { setPreview(URL.createObjectURL(f)); startAnalysis(f); }
    };

    const reset = () => {
        setPreview(null); setIsAnalyzing(false);
        setPrediction(null); setGradcam(null); setShowResult(false); setShowHeatmap(false);
    };

    const probs = prediction?.all_probabilities ?? {};
    const sortedClasses = CLASSES.slice().sort((a, b) => (probs[b] ?? 0) - (probs[a] ?? 0));

    return (
        <div className="w-full max-w-5xl mx-auto flex flex-col items-center">
            <AnimatePresence mode="wait">
                {/* ── UPLOAD STATE ── */}
                {!isAnalyzing && !showResult && (
                    <motion.div
                        key="upload"
                        initial={{ opacity: 0, y: 30 }}
                        animate={{ opacity: 1, y: 0 }}
                        exit={{ opacity: 0, y: -20 }}
                        className="w-full flex flex-col items-center"
                    >
                        <div className="text-center mb-10">
                            <h2 className="text-5xl md:text-8xl font-black tracking-tighter leading-[0.88] mb-6">
                                WASTE DETECTION <br /><span className="text-[#00ff99]">MADE EASY.</span>
                            </h2>
                            <p className="text-slate-500 text-base max-w-lg mx-auto font-medium uppercase tracking-[0.08em]">
                                MobileNetV2 CNN · Grad-CAM Explainability<br />6-Class Garbage Classification
                            </p>
                        </div>

                        <div
                           onDragOver={(e) => e.preventDefault()}
                           onDrop={onDrop}
                           className="glass-card glow-border w-full aspect-video md:aspect-[21/9] flex flex-col items-center justify-center p-4 cursor-pointer group transition-all"
                        >
                            <input type="file" id="file-upload" className="hidden" accept="image/*" onChange={onFileChange} />
                            <label htmlFor="file-upload" className="w-full h-full flex flex-col items-center justify-center cursor-pointer">
                                <div className="flex flex-col items-center text-center">
                                    <div className="w-20 h-20 rounded-full bg-white/5 mb-6 flex items-center justify-center border border-white/10 group-hover:scale-110 group-hover:bg-[#00ff99]/10 group-hover:border-[#00ff99]/30 transition-all duration-500">
                                        <Upload className="w-8 h-8 text-[#00ff99]" />
                                    </div>
                                    <h3 className="text-2xl font-bold tracking-tight mb-2">Drag &amp; drop your waste image</h3>
                                    <p className="text-[#00ff99] text-xs font-black uppercase tracking-widest">or browse local drive &rarr;</p>
                                    <p className="text-slate-600 text-xs mt-3">cardboard · glass · metal · paper · plastic · trash</p>
                                </div>
                            </label>
                        </div>
                    </motion.div>
                )}

                {/* ── ANALYZING STATE ── */}
                {isAnalyzing && !showResult && (
                    <motion.div
                        key="analyzing"
                        initial={{ opacity: 0 }}
                        animate={{ opacity: 1 }}
                        exit={{ opacity: 0 }}
                        className="w-full flex flex-col items-center py-16"
                    >
                        <div className="mb-10 text-center">
                            <h3 className="text-xs font-black uppercase tracking-[0.3em] text-[#00ff99] mb-3">Neural Processing In Progress</h3>
                            <div className="text-3xl font-bold tracking-tighter">Running CNN inference pipeline...</div>
                        </div>
                        <div className="w-full glass-card p-10 flex items-center justify-center overflow-hidden">
                            <PipelineAnimation onComplete={() => setShowResult(true)} />
                        </div>
                    </motion.div>
                )}

                {/* ── RESULT STATE ── */}
                {showResult && prediction && (
                    <motion.div
                        key="result"
                        initial={{ opacity: 0, scale: 0.96 }}
                        animate={{ opacity: 1, scale: 1 }}
                        className="w-full flex flex-col gap-8"
                    >
                        <div className="flex flex-col md:flex-row gap-8 items-stretch">
                            {/* Image + Grad-CAM Toggle */}
                            <div className="flex-1 flex flex-col gap-3">
                                <div className="glass-card overflow-hidden h-[320px] relative group">
                                    {/* eslint-disable-next-line @next/next/no-img-element */}
                                    <img
                                        src={showHeatmap && gradcam ? gradcam : preview!}
                                        alt="Input"
                                        className="w-full h-full object-cover transition-all duration-700"
                                    />
                                    <div className="absolute inset-0 bg-gradient-to-t from-[#05080a] via-transparent to-transparent pointer-events-none" />
                                    <div className="absolute bottom-4 left-4">
                                        <span className="text-xs font-black uppercase tracking-widest text-[#00ff99] bg-[#00ff99]/10 px-3 py-1 rounded border border-[#00ff99]/20">
                                            {showHeatmap ? "Grad-CAM Heatmap" : "Source Input"}
                                        </span>
                                    </div>
                                </div>
                                {gradcam && (
                                    <button
                                        onClick={() => setShowHeatmap((v) => !v)}
                                        className={`w-full py-3 rounded-xl text-xs font-black uppercase tracking-widest border transition-all ${
                                            showHeatmap
                                                ? "bg-orange-500/20 border-orange-500/40 text-orange-300"
                                                : "bg-white/5 border-white/10 text-slate-400 hover:border-[#00ff99]/30"
                                        }`}
                                    >
                                        {showHeatmap ? "🔥 Viewing Grad-CAM — Click to restore" : "🔥 View Grad-CAM Heatmap (Explainability)"}
                                    </button>
                                )}
                            </div>

                            {/* Classification Result */}
                            <div className="flex-1 flex flex-col justify-between py-2">
                                <div>
                                    <h4 className="text-xs font-black uppercase tracking-[0.3em] text-slate-500 mb-1">Classification Result</h4>
                                    <h1 className="text-6xl md:text-8xl font-black uppercase tracking-tighter leading-none mb-3 break-words"
                                        style={{ color: CLASS_COLORS[prediction.class] ?? '#00ff99' }}>
                                        {prediction.class}
                                    </h1>
                                    <div className="flex items-center gap-4 mb-6">
                                        <div className="h-px flex-1 bg-white/10" />
                                        <span className="text-sm font-bold tracking-widest text-[#00e5ff]">
                                            CONFIDENCE: {(prediction.confidence * 100).toFixed(1)}%
                                        </span>
                                    </div>

                                    {/* Probability Bar Chart */}
                                    <div className="space-y-2 mb-6">
                                        <p className="text-xs font-black uppercase tracking-wider text-slate-500 mb-3 flex items-center gap-2">
                                            <BarChart3 className="w-3 h-3" /> All Class Probabilities
                                        </p>
                                        {sortedClasses.map((cls) => {
                                            const p = (probs[cls] ?? 0) * 100;
                                            return (
                                                <div key={cls} className="flex items-center gap-3">
                                                    <span className="text-xs font-bold w-20 uppercase text-slate-400">{cls}</span>
                                                    <div className="flex-1 h-2 rounded-full bg-white/5 overflow-hidden">
                                                        <motion.div
                                                            initial={{ width: 0 }}
                                                            animate={{ width: `${p}%` }}
                                                            transition={{ duration: 0.8, delay: 0.1 }}
                                                            className="h-full rounded-full"
                                                            style={{ background: CLASS_COLORS[cls] ?? '#00ff99' }}
                                                        />
                                                    </div>
                                                    <span className="text-xs font-bold text-slate-500 w-12 text-right">{p.toFixed(1)}%</span>
                                                </div>
                                            );
                                        })}
                                    </div>

                                    {/* Disposal Tip */}
                                    <div className="flex items-start gap-4 p-4 rounded-2xl bg-white/5 border border-white/5 mb-3">
                                        <div className="w-9 h-9 rounded-lg bg-[#00ff99]/10 flex items-center justify-center shrink-0">
                                            <ShieldCheck className="w-4 h-4 text-[#00ff99]" />
                                        </div>
                                        <div>
                                            <p className="text-sm font-bold text-slate-200 mb-1">Disposal Guidance</p>
                                            <p className="text-sm text-slate-500 font-medium">{prediction.disposal_tip}</p>
                                        </div>
                                    </div>

                                    {/* CO2 Impact */}
                                    <div className="flex items-start gap-4 p-4 rounded-2xl bg-white/5 border border-white/5">
                                        <div className="w-9 h-9 rounded-lg bg-green-500/10 flex items-center justify-center shrink-0">
                                            <Leaf className="w-4 h-4 text-green-400" />
                                        </div>
                                        <div>
                                            <p className="text-sm font-bold text-slate-200 mb-1">CO₂ Impact</p>
                                            <p className="text-sm text-slate-500 font-medium">{prediction.co2_impact}</p>
                                        </div>
                                    </div>
                                </div>

                                <button
                                    onClick={reset}
                                    className="mt-8 flex items-center justify-center gap-3 w-full py-4 rounded-2xl bg-[#00ff99] text-black font-black uppercase tracking-widest text-sm hover:bg-[#00e5ff] transition-all transform hover:scale-[1.02] shadow-[0_0_30px_rgba(0,255,153,0.2)]"
                                >
                                    <RefreshCw className="w-5 h-5" /> New Analysis
                                </button>
                            </div>
                        </div>
                    </motion.div>
                )}
            </AnimatePresence>
        </div>
    );
}
