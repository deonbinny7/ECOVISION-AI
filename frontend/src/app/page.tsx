"use client";

import React, { useState } from "react";
import Dashboard from "@/components/Dashboard";
import EvalPanel from "@/components/EvalPanel";
import ModelInfo from "@/components/ModelInfo";

type Tab = "classify" | "evaluate" | "model";

export default function Home() {
  const [activeTab, setActiveTab] = useState<Tab>("classify");

  const tabs: { id: Tab; label: string }[] = [
    { id: "classify",  label: "Classify"   },
    { id: "evaluate",  label: "Evaluate"   },
    { id: "model",     label: "Model Info" },
  ];

  return (
    <main className="min-h-screen bg-[#05080a] text-slate-100 flex flex-col">


      {/* ── MAIN HEADER ── */}
      <header className="w-full py-6 px-10 flex justify-between items-center z-50">
        <div className="flex items-center gap-4 group cursor-pointer">
          <div className="relative">
            <div className="w-11 h-11 bg-gradient-to-br from-[#00ff99] to-[#00e5ff] rounded-2xl flex items-center justify-center shadow-[0_0_20px_rgba(0,255,153,0.3)] group-hover:shadow-[0_0_30px_rgba(0,255,153,0.5)] transition-all duration-500 transform group-hover:rotate-6">
              <svg viewBox="0 0 24 24" className="w-6 h-6 text-black" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
                <path d="M21 16V8a2 2 0 0 0-1-1.73l-7-4a2 2 0 0 0-2 0l-7 4A2 2 0 0 0 3 8v8a2 2 0 0 0 1 1.73l7 4a2 2 0 0 0 2 0l7-4A2 2 0 0 0 21 16z" />
                <polyline points="7.5 4.21 12 6.81 16.5 4.21" />
                <polyline points="7.5 19.79 7.5 14.6 3 12" />
                <polyline points="21 12 16.5 14.6 16.5 19.79" />
                <polyline points="3.27 6.96 12 12.01 20.73 6.96" />
                <line x1="12" y1="22.08" x2="12" y2="12" />
              </svg>
            </div>
            <div className="absolute inset-0 rounded-2xl bg-[#00ff99]/20 animate-ping pointer-events-none" />
          </div>
          <div className="flex flex-col">
            <span className="text-xl font-black tracking-tighter leading-none group-hover:text-[#00ff99] transition-colors">
              ECOVISION <span className="text-[#00ff99] group-hover:text-white">AI</span>
            </span>
            <span className="text-[9px] font-bold tracking-[0.4em] text-slate-600 uppercase mt-0.5">Intelligent Waste Classification</span>
          </div>
        </div>

        {/* Tab Navigation */}
        <nav className="flex gap-1 bg-white/5 rounded-xl p-1 border border-white/10">
          {tabs.map((t) => (
            <button
              key={t.id}
              onClick={() => setActiveTab(t.id)}
              className={`px-4 py-2 rounded-lg text-xs font-black uppercase tracking-widest transition-all duration-300 ${
                activeTab === t.id
                  ? "bg-[#00ff99] text-black shadow-[0_0_15px_rgba(0,255,153,0.3)]"
                  : "text-slate-500 hover:text-slate-300"
              }`}
            >
              {t.label}
            </button>
          ))}
        </nav>
      </header>

      {/* ── PAGE CONTENT ── */}
      <div className="flex-1 flex flex-col items-center px-6 pb-10 pt-4">
        {activeTab === "classify" && <Dashboard />}
        {activeTab === "evaluate" && <EvalPanel />}
        {activeTab === "model"    && <ModelInfo />}
      </div>

      {/* ── FOOTER ── */}
      <footer className="py-6 text-center border-t border-white/5">
        <p className="text-slate-700 text-[10px] font-bold uppercase tracking-[0.25em]">
          &copy; 2026 EcoVision AI
        </p>
      </footer>
    </main>
  );
}
