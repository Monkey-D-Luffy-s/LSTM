/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import { useState, useEffect, ChangeEvent } from 'react';
import { 
  TrendingUp, 
  TrendingDown, 
  Activity, 
  Settings, 
  Terminal, 
  FileJson, 
  AlertCircle,
  ArrowRight,
  BrainCircuit,
  Database
} from 'lucide-react';
import { motion, AnimatePresence } from 'motion/react';
import { 
  LineChart, 
  Line, 
  XAxis, 
  YAxis, 
  CartesianGrid, 
  Tooltip, 
  ResponsiveContainer,
  AreaChart,
  Area
} from 'recharts';
import { clsx, type ClassValue } from 'clsx';
import { twMerge } from 'tailwind-merge';

function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

// Mock data generator for the demo chart
const generateMockData = () => {
  const data = [];
  let price = 50000;
  for (let i = 0; i < 50; i++) {
    const change = (Math.random() - 0.48) * 1000; // Slight upward bias
    price += change;
    data.push({
      time: i,
      price: Math.round(price),
      volume: Math.round(Math.random() * 5000 + 2000),
      rsi: Math.round(Math.random() * 40 + 30),
    });
  }
  return data;
};

export default function App() {
  const [isTraining, setIsTraining] = useState(false);
  const [isPredicting, setIsPredicting] = useState(false);
  const [prediction, setPrediction] = useState<{ direction: 'UP' | 'DOWN', confidence: number } | null>(null);
  const [logs, setLogs] = useState<string[]>([]);
  const [chartData, setChartData] = useState(generateMockData());

  const addLog = (msg: string) => {
    setLogs(prev => [`[${new Date().toLocaleTimeString()}] ${msg}`, ...prev.slice(0, 10)]);
  };

  useEffect(() => {
    // Initial boot logs
    const bootLogs = [
      "Initializing AETHER-LSTM Engine...",
      "Connecting to AIS-GLOBAL-RUN1 cluster...",
      "Verifying TensorFlow.js bindings...",
      "Kernel: Python 3.10 / C++ Core v4.2",
      "SYSTEM_READY: Awaiting input stream..."
    ];
    
    bootLogs.forEach((log, i) => {
      setTimeout(() => {
        setLogs(prev => [`[${new Date().toLocaleTimeString()}] ${log}`, ...prev.slice(0, 15)]);
      }, i * 400);
    });

    const interval = setInterval(() => {
      setChartData(prev => {
        const last = prev[prev.length - 1];
        const nextPrice = last.price + (Math.random() - 0.48) * 500;
        return [...prev.slice(1), {
          time: last.time + 1,
          price: Math.round(nextPrice),
          volume: Math.round(Math.random() * 5000 + 2000),
          rsi: Math.round(Math.random() * 40 + 30),
        }];
      });
    }, 5000);
    return () => clearInterval(interval);
  }, []);

  const handleTrain = async (e: ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    setIsTraining(true);
    addLog(`Starting training with ${file.name}...`);

    const formData = new FormData();
    formData.append('file', file);

    try {
      const res = await fetch('/api/train', {
        method: 'POST',
        body: formData,
      });
      const data = await res.json();
      if (res.ok) {
        addLog('Training successful. Model saved.');
      } else {
        addLog(`Training failed: ${data.error}`);
        if (data.logs) {
          data.logs.split('\n').filter((l: string) => l.trim()).forEach((line: string) => addLog(`SYSTEM: ${line}`));
        }
      }
    } catch (err) {
      addLog(`Error: ${err}`);
    } finally {
      setIsTraining(false);
    }
  };

  const handlePredict = async (e: ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    setIsPredicting(true);
    setPrediction(null);
    addLog(`Running inference for ${file.name}...`);

    const formData = new FormData();
    formData.append('file', file);

    try {
      const res = await fetch('/api/predict', {
        method: 'POST',
        body: formData,
      });
      const data = await res.json();
      if (res.ok) {
        setPrediction({ direction: data.direction, confidence: data.confidence });
        addLog(`Inference complete: Next candle is ${data.direction}.`);
      } else {
        addLog(`Prediction failed: ${data.error}`);
        if (data.logs) {
          data.logs.split('\n').filter((l: string) => l.trim()).forEach((line: string) => addLog(`SYSTEM: ${line}`));
        }
      }
    } catch (err) {
      addLog(`Error: ${err}`);
    } finally {
      setIsPredicting(false);
    }
  };

  const handleReset = async () => {
    if (!confirm('Are you sure you want to wipe the trained model and start from scratch?')) return;
    
    addLog('Requesting system reset...');
    try {
      const res = await fetch('/api/reset', { method: 'POST' });
      if (res.ok) {
        setPrediction(null);
        addLog('System reset successful. Model wiped.');
      } else {
        addLog('Reset failed.');
      }
    } catch (err) {
      addLog(`Reset error: ${err}`);
    }
  };

  return (
    <div className="min-h-screen bg-bg-dark text-[#e0e0e0] font-sans flex flex-col overflow-y-auto overflow-x-hidden">
      {/* Top Navigation / Header */}
      <header className="h-16 border-b border-border-muted bg-header-bg flex items-center justify-between px-8 shrink-0 sticky top-0 z-50 backdrop-blur-md">
        <div className="flex items-center gap-4">
          <div className="w-3 h-3 bg-accent-cyan rounded-full shadow-[0_0_8px_#06b6d4]"></div>
          <h1 className="text-lg font-bold tracking-widest text-accent-cyan flex items-center gap-2">
            AETHER-LSTM <span className="text-[10px] font-normal text-slate-500 tracking-normal px-2 py-0.5 border border-slate-500/20 rounded">v4.2.0-STABLE</span>
          </h1>
        </div>
        <div className="flex items-center gap-4 text-sm font-medium text-gray-400">
          <button 
            onClick={handleReset}
            className="px-3 py-1 border border-red-500/30 text-red-400/80 hover:bg-red-500 hover:text-white rounded text-[10px] uppercase tracking-widest transition-all"
          >
            Reset Model
          </button>
          <div className="h-4 w-px bg-white/10" />
          <div className="flex items-center gap-2">
            <span className={cn("text-xs", isTraining ? "text-accent-cyan animate-pulse" : "text-slate-600")}>
              ● {isTraining ? "TRAINING_IN_PROGRESS" : "SYSTEM_IDLE"}
            </span>
          </div>
          <div className="flex items-center gap-2"><span className="text-accent-emerald">●</span> Scale: 5m Candle</div>
          <div className="flex items-center gap-2 text-xs md:flex hidden"><span className="text-slate-600">●</span> Cluster: AIS-RUN1</div>
        </div>
      </header>

      <main className="flex-1 flex flex-col lg:flex-row p-6 gap-6 pb-24">
        {/* Left Panel: Training & Model Config */}
        <section className="w-full lg:w-72 flex flex-col gap-6 shrink-0 h-fit">
          <div className="bg-panel-bg border border-border-subtle rounded-xl p-5 shadow-2xl flex flex-col min-h-[400px]">
            <h2 className="text-[10px] uppercase text-slate-500 tracking-widest mb-4 border-b border-border-subtle pb-2 font-bold">Training Suite</h2>
            <div className="space-y-6 flex-1">
              <div className="group relative">
                <label className="block text-[9px] uppercase font-bold text-slate-500 mb-2 tracking-widest">Dataset Upload (50k+)</label>
                <div className={cn(
                  "border-2 border-dashed rounded-xl p-4 transition-all flex flex-col items-center justify-center text-center cursor-pointer",
                  isTraining ? "border-accent-cyan bg-accent-cyan/5" : "border-slate-800 hover:border-slate-600"
                )}>
                  <Database className={cn("w-6 h-6 mb-2", isTraining ? "text-accent-cyan animate-pulse" : "text-slate-600")} />
                  <span className="text-[10px] text-slate-400 font-mono">
                    {isTraining ? 'Training Model...' : 'DROP .CSV TO TRAIN'}
                  </span>
                  <input 
                    type="file" 
                    onChange={handleTrain} 
                    accept=".csv"
                    className="absolute inset-0 opacity-0 cursor-pointer"
                    disabled={isTraining}
                  />
                </div>
              </div>

              <div className="group relative">
                <label className="block text-[9px] uppercase font-bold text-slate-500 mb-2 tracking-widest">Inference Upload (50-100)</label>
                <div className={cn(
                  "border-2 border-dashed rounded-xl p-4 transition-all flex flex-col items-center justify-center text-center cursor-pointer",
                  isPredicting ? "border-accent-emerald bg-accent-emerald/5" : "border-slate-800 hover:border-slate-600"
                )}>
                  <BrainCircuit className={cn("w-6 h-6 mb-2", isPredicting ? "text-accent-emerald animate-pulse" : "text-slate-600")} />
                  <span className="text-[10px] text-slate-400 font-mono">
                    {isPredicting ? 'Inferring...' : 'DROP .CSV TO PREDICT'}
                  </span>
                  <input 
                    type="file" 
                    onChange={handlePredict} 
                    accept=".csv"
                    className="absolute inset-0 opacity-0 cursor-pointer"
                    disabled={isPredicting || isTraining}
                  />
                </div>
              </div>
            </div>
            
            <div className="mt-auto pt-4">
              <div className="flex justify-between text-[10px] mb-2">
                <span className="text-slate-500 uppercase tracking-widest">Model Readiness</span>
                <span className="text-accent-cyan font-mono">STABLE</span>
              </div>
              <div className="w-full h-1 bg-slate-800 rounded-full overflow-hidden">
                <motion.div 
                  initial={{ width: 0 }}
                  animate={{ width: "100%" }}
                  className="h-full bg-accent-cyan shadow-[0_0_12px_rgba(6,182,212,0.8)]"
                ></motion.div>
              </div>
            </div>
          </div>

          <div className="bg-panel-bg border border-border-subtle rounded-xl p-5 shadow-2xl">
            <h2 className="text-[10px] uppercase text-slate-500 tracking-widest mb-4 border-b border-border-subtle pb-2 font-bold">Input Features</h2>
            <ul className="space-y-4 font-mono text-xs">
              <li className="flex justify-between items-center">
                <span className="text-slate-400">OHLCV</span>
                <span className="text-accent-cyan">[5 CHS]</span>
              </li>
              <li className="flex justify-between items-center text-accent-emerald">
                <span>RSI (14)</span>
                <span className="w-2 h-2 rounded-full bg-accent-emerald shadow-[0_0_5px_#10b981]"></span>
              </li>
              <li className="flex justify-between items-center text-accent-emerald">
                <span>MACD</span>
                <span className="w-2 h-2 rounded-full bg-accent-emerald shadow-[0_0_5px_#10b981]"></span>
              </li>
              <li className="flex justify-between items-center text-accent-emerald">
                <span>EMA (200)</span>
                <span className="w-2 h-2 rounded-full bg-accent-emerald shadow-[0_0_5px_#10b981]"></span>
              </li>
              <li className="flex justify-between items-center text-accent-emerald">
                <span>VWAP</span>
                <span className="w-2 h-2 rounded-full bg-accent-emerald shadow-[0_0_5px_#10b981]"></span>
              </li>
              <li className="flex justify-between items-center text-accent-cyan">
                <span>VOLATILITY</span>
                <span className="w-2 h-2 rounded-full bg-accent-cyan shadow-[0_0_5px_#06b6d4]"></span>
              </li>
            </ul>
          </div>
        </section>

        {/* Center Panel: Live Prediction Result */}
        <section className="flex-1 flex flex-col gap-6 min-h-[500px] h-fit">
          <div className="flex-1 bg-gradient-to-br from-[#0d0f14] to-[#08090c] border border-border-muted rounded-2xl relative flex flex-col items-center justify-center shadow-[inset_0_0_100px_rgba(0,0,0,0.8)] overflow-hidden group min-h-[400px]">
            {/* Atmospheric Grid Background */}
            <div className="absolute inset-0 opacity-[0.03] pointer-events-none" style={{ backgroundImage: 'radial-gradient(#fff 1px, transparent 0)', backgroundSize: '40px 40px' }}></div>
            
            <AnimatePresence mode="wait">
              {!prediction && !isPredicting ? (
                <motion.div 
                  key="idle"
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  exit={{ opacity: 0 }}
                  className="z-10 text-center px-12"
                >
                  <p className="text-slate-500 text-xs tracking-[0.4em] uppercase mb-6 font-bold">Neural Engine Offline</p>
                  <div className="text-sm text-slate-400 font-mono max-w-sm">
                    Upload a history file (50k samples) to train, then provide recent data (50+ samples) for real-time market foresight.
                  </div>
                </motion.div>
              ) : isPredicting ? (
                <motion.div 
                  key="scanning"
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  exit={{ opacity: 0 }}
                  className="z-10 text-center"
                >
                  <div className="relative w-24 h-24 mx-auto mb-6">
                    <div className="absolute inset-0 border-4 border-accent-cyan/20 rounded-full"></div>
                    <div className="absolute inset-0 border-4 border-t-accent-cyan rounded-full animate-spin"></div>
                    <Activity className="absolute inset-0 m-auto w-8 h-8 text-accent-cyan animate-pulse" />
                  </div>
                  <p className="text-accent-cyan text-xs tracking-[0.4em] uppercase animate-pulse">Running In-Memory Inference</p>
                </motion.div>
              ) : (
                <motion.div 
                  key="result"
                  initial={{ opacity: 0, scale: 0.9 }}
                  animate={{ opacity: 1, scale: 1 }}
                  className="z-10 text-center"
                >
                  <p className="text-slate-500 text-[10px] tracking-[0.4em] uppercase mb-4 font-bold">Future Horizon Scanned</p>
                  <h3 className={cn(
                    "text-[80px] md:text-[100px] font-black tracking-tighter leading-none drop-shadow-[0_0_30px_rgba(0,0,0,0.5)]",
                    prediction?.direction === 'UP' ? "text-accent-emerald text-shadow-[0_0_30px_rgba(16,185,129,0.3)]" : "text-red-500 text-shadow-[0_0_30px_rgba(239,68,68,0.3)]"
                  )}>
                    {prediction?.direction === 'UP' ? 'BULLISH' : 'BEARISH'}
                  </h3>
                  <div className="flex items-center justify-center gap-8 mt-12 bg-black/20 p-6 rounded-3xl border border-white/5 backdrop-blur-sm">
                    <div className="text-left">
                      <p className="text-[9px] text-slate-500 uppercase font-black tracking-[0.2em] mb-1">Confidence</p>
                      <p className="text-5xl font-mono text-white leading-none">{prediction?.confidence}<span className="text-xl text-slate-500">%</span></p>
                    </div>
                    <div className="w-px h-12 bg-slate-800"></div>
                    <div className="text-left">
                      <p className="text-[9px] text-slate-500 uppercase font-black tracking-[0.2em] mb-1">Signal Bias</p>
                      <div className={cn(
                        "text-5xl font-mono leading-none flex items-center gap-2",
                        prediction?.direction === 'UP' ? "text-accent-emerald" : "text-red-500"
                      )}>
                        {prediction?.direction === 'UP' ? <TrendingUp /> : <TrendingDown />}
                      </div>
                    </div>
                  </div>
                </motion.div>
              )}
            </AnimatePresence>

            {/* Decorative Overlay Rings */}
            <div className="absolute w-[800px] h-[800px] border border-white/5 rounded-full -bottom-1/2 pointer-events-none"></div>
          </div>
        </section>

        {/* Right Panel: Market Context */}
        <section className="w-full lg:w-64 flex flex-col gap-6 shrink-0 h-fit">
          {/* Console Output View */}
          <div className="h-64 bg-black/40 border border-border-subtle rounded-xl font-mono p-4 text-[11px] text-slate-400 shrink-0 group hover:border-white/10 transition-colors">
            <div className="flex justify-between items-center mb-2 border-b border-border-subtle pb-1">
              <span className="text-[9px] uppercase tracking-widest font-bold">System Console Logs</span>
              <span className="text-accent-emerald">● ACTIVE</span>
            </div>
            <div className="space-y-1.5 overflow-y-auto h-[calc(100%-25px)] pr-2 scrollbar-hide">
              {logs.map((log, i) => (
                <p key={i} className={cn(
                  "opacity-80 transition-opacity hover:opacity-100",
                  log.includes('Error') || log.includes('failed') ? "text-red-400" : log.includes('successful') || log.includes('complete') ? "text-accent-emerald" : ""
                )}>{log}</p>
              ))}
              {logs.length === 0 && <p className="text-slate-600">Waiting for system commands...</p>}
            </div>
          </div>

          <div className="bg-panel-bg border border-border-subtle rounded-xl p-5 shadow-2xl min-h-[300px] flex flex-col">
            <h2 className="text-[10px] uppercase text-slate-500 tracking-widest mb-4 border-b border-border-subtle pb-2 font-bold">Market Dynamics</h2>
            <div className="space-y-4 overflow-y-auto pr-2 scrollbar-hide">
              {chartData.slice(-6).reverse().map((d, i) => (
                <div key={i} className={cn(
                  "border-l-2 pl-3 py-1 bg-white/[0.02] rounded-r",
                  i === 0 ? "border-accent-cyan bg-accent-cyan/5" : "border-slate-800"
                )}>
                  <p className="text-[9px] text-slate-500 tracking-tighter uppercase font-mono">Candle Index {d.time}</p>
                  <p className="text-sm font-mono text-slate-200">${d.price.toLocaleString()}</p>
                </div>
              ))}
            </div>
          </div>

          <div className={cn(
            "border rounded-xl p-5 text-center transition-all duration-500",
            prediction?.direction === 'UP' ? "bg-accent-emerald/10 border-accent-emerald/20" : prediction?.direction === 'DOWN' ? "bg-red-500/10 border-red-500/20" : "bg-slate-800/10 border-slate-800/20"
          )}>
            <p className="text-[9px] uppercase tracking-[0.2em] text-slate-500 font-black mb-1">Execution Signal</p>
            <p className={cn(
              "text-2xl font-black transition-all",
              prediction?.direction === 'UP' ? "text-accent-emerald" : prediction?.direction === 'DOWN' ? "text-red-500" : "text-white"
            )}>
              {prediction ? (prediction.direction === 'UP' ? 'LONG' : 'SHORT') : 'NEUTRAL'}
            </p>
            <div className="w-full h-px bg-white/5 my-3" />
            <p className="text-[9px] text-slate-500 font-mono uppercase tracking-tight">Status: Model Verified</p>
          </div>
        </section>
      </main>

      {/* Bottom Status Bar */}
      <footer className="h-8 border-t border-border-subtle bg-bg-dark flex items-center justify-between px-8 text-[9px] text-slate-600 uppercase tracking-widest shrink-0 font-mono">
        <div className="flex gap-4">
          <span>Runtime: Node.js 22.x</span>
          <span>Engine: TensorFlow.js / C++ Core</span>
        </div>
        <div className="flex gap-6">
          <span className="flex items-center gap-1.5"><div className="w-1.5 h-1.5 bg-accent-emerald rounded-full"></div> GPU-ACCEL: AUTO</span>
          <span className="text-accent-cyan/60">Cluster: AIS-GLOBAL-RUN1</span>
        </div>
      </footer>
    </div>
  );
}
