"use client";

import React, { useEffect, useState } from "react";
import { motion } from "framer-motion";

interface EvalData {
  overall: {
    accuracy: number;
    weighted_precision: number;
    weighted_recall: number;
    weighted_f1: number;
    num_test_samples: number;
  };
  confusion_matrix: number[][];
  class_labels: string[];
  per_class_metrics: Record<string, { precision: number; recall: number; f1_score: number; support: number }>;
}

export default function EvalPanel() {
  const [data, setData] = useState<EvalData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';
    fetch(`${API_URL}/evaluate`)
      .then(res => res.json())
      .then((d) => {
        if (d.error) setError(d.error + (d.hint ? `\n${d.hint}` : ""));
        else setData(d);
        setLoading(false);
      })
      .catch(() => { setError("Backend unreachable. Start the server first."); setLoading(false); });
  }, []);

  const maxVal = data ? Math.max(...data.confusion_matrix.flat()) : 1;

  return (
    <div className="w-full max-w-5xl mx-auto space-y-10 py-4">
      <div className="text-center">
        <h2 className="text-3xl font-black tracking-tighter mb-2">
          Model <span className="text-[#00ff99]">Evaluation</span>
        </h2>
        <p className="text-slate-500 text-sm uppercase tracking-widest">Performance Analysis</p>
      </div>

      {loading && (
        <div className="text-center text-slate-400 animate-pulse py-16">Loading evaluation results...</div>
      )}

      {error && (
        <div className="glass-card p-8 border border-red-500/30 text-red-400">
          <pre className="text-sm whitespace-pre-wrap">{error}</pre>
        </div>
      )}

      {data && (
        <>
          {/* Overall Metrics */}
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            {[
              { label: "Accuracy", value: (data.overall.accuracy * 100).toFixed(1) + "%" },
              { label: "Precision", value: (data.overall.weighted_precision * 100).toFixed(1) + "%" },
              { label: "Recall", value: (data.overall.weighted_recall * 100).toFixed(1) + "%" },
              { label: "F1-Score", value: (data.overall.weighted_f1 * 100).toFixed(1) + "%" },
            ].map((m) => (
              <motion.div
                key={m.label}
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                className="glass-card p-6 flex flex-col items-center"
              >
                <span className="text-3xl font-black text-[#00ff99]">{m.value}</span>
                <span className="text-xs font-bold uppercase tracking-widest text-slate-500 mt-1">{m.label}</span>
              </motion.div>
            ))}
          </div>

          {/* Confusion Matrix */}
          <div className="glass-card p-6">
            <h3 className="text-sm font-black uppercase tracking-widest text-slate-400 mb-4">Confusion Matrix</h3>
            <div className="overflow-x-auto">
              <table className="w-full text-xs border-collapse">
                <thead>
                  <tr>
                    <th className="p-2 text-slate-500 text-left">Actual ↓ / Pred →</th>
                    {data.class_labels.map((l) => (
                      <th key={l} className="p-2 text-[#00ff99] uppercase tracking-wider font-bold">{l.slice(0, 4)}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {data.confusion_matrix.map((row, i) => (
                    <tr key={i}>
                      <td className="p-2 text-[#00ff99] font-bold uppercase">{data.class_labels[i]}</td>
                      {row.map((val, j) => {
                        const intensity = val / maxVal;
                        const isDiag = i === j;
                        return (
                          <td key={j} className="p-2 text-center font-bold rounded"
                            style={{
                              background: isDiag
                                ? `rgba(0,255,153,${0.1 + intensity * 0.7})`
                                : `rgba(239,68,68,${intensity * 0.5})`,
                              color: intensity > 0.3 ? "#fff" : "#64748b"
                            }}>
                            {val}
                          </td>
                        );
                      })}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            <p className="text-slate-600 text-xs mt-3">Green diagonal = correct predictions. Red off-diagonal = misclassifications.</p>
          </div>

          {/* Per-Class Metrics Table */}
          <div className="glass-card p-6">
            <h3 className="text-sm font-black uppercase tracking-widest text-slate-400 mb-4">Per-Class Metrics</h3>
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-white/10">
                  <th className="text-left py-2 text-slate-500 font-bold uppercase text-xs tracking-wider">Class</th>
                  <th className="text-center py-2 text-slate-500 font-bold uppercase text-xs tracking-wider">Precision</th>
                  <th className="text-center py-2 text-slate-500 font-bold uppercase text-xs tracking-wider">Recall</th>
                  <th className="text-center py-2 text-slate-500 font-bold uppercase text-xs tracking-wider">F1-Score</th>
                  <th className="text-center py-2 text-slate-500 font-bold uppercase text-xs tracking-wider">Support</th>
                </tr>
              </thead>
              <tbody>
                {data.class_labels.map((cls) => {
                  const m = data.per_class_metrics[cls];
                  return (
                    <tr key={cls} className="border-b border-white/5 hover:bg-white/5">
                      <td className="py-3 font-bold text-[#00ff99] uppercase">{cls}</td>
                      <td className="py-3 text-center text-slate-300">{(m.precision * 100).toFixed(1)}%</td>
                      <td className="py-3 text-center text-slate-300">{(m.recall * 100).toFixed(1)}%</td>
                      <td className="py-3 text-center font-bold text-[#00e5ff]">{(m.f1_score * 100).toFixed(1)}%</td>
                      <td className="py-3 text-center text-slate-500">{m.support}</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>

          <p className="text-center text-slate-600 text-xs">
            Test samples: {data.overall.num_test_samples} · Metrics computed via sklearn classification_report
          </p>
        </>
      )}
    </div>
  );
}
