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

// Mock data generator removed

export default function App() {
  const [isTraining, setIsTraining] = useState(false);
  const [isPredicting, setIsPredicting] = useState(false);
  const [prediction, setPrediction] = useState<{ direction: 'UP' | 'DOWN', confidence: number } | null>(null);
  const [logs, setLogs] = useState<string[]>([]);

  const addLog = (msg: string) => {
    setLogs(prev => [`[${new Date().toLocaleTimeString()}] ${msg}`, ...prev.slice(0, 10)]);
  };

  useEffect(() => {
    // Initial boot log
    addLog("Initializing AETHER-LSTM Engine UI...");
    addLog("Connecting to AIS-GLOBAL-RUN1 cluster...");
    
    // Connect to Server-Sent Events for live logs
    const eventSource = new EventSource('/api/logs');
    eventSource.onmessage = (event) => {
      addLog(event.data);
    };

    eventSource.onerror = () => {
      // Ignore errors, SSE will auto-reconnect
    };

    return () => {
      eventSource.close();
    };
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
      
      const contentType = res.headers.get('content-type');
      if (contentType && contentType.includes('application/json')) {
        const data = await res.json();
        if (res.ok) {
          addLog('Training successful. Model saved.');
        } else {
          const detailStr = data.details ? (typeof data.details === 'string' ? data.details : JSON.stringify(data.details)) : '';
          addLog(`Training failed: ${data.error || 'Unknown error'}`);
          if (detailStr) addLog(`DETAIL: ${detailStr.substring(0, 100)}...`);
          if (data.logs) {
            data.logs.split('\n').filter((l: string) => l.trim()).forEach((line: string) => addLog(`SYSTEM: ${line}`));
          }
        }
      } else {
        const text = await res.text();
        console.error('Non-JSON response:', text);
        if (text.includes('Please wait while your application starts')) {
          addLog('Status: Server overloaded (Memory limit hit during TensorFlow training).');
          addLog('RECOMMENDED: Use a smaller CSV file or reduce complexity.');
        } else if (res.status === 504 || res.status === 503) {
          addLog(`Error: Connection timed out or server overloaded (Status ${res.status}).`);
        } else {
          addLog(`Error: Server returned unexpected content (Status ${res.status}).`);
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
      
      const contentType = res.headers.get('content-type');
      if (contentType && contentType.includes('application/json')) {
        const data = await res.json();
        if (res.ok) {
          setPrediction({ direction: data.direction, confidence: data.confidence });
          addLog(`Inference complete: Next candle is ${data.direction}.`);
        } else {
          const detailStr = data.details ? (typeof data.details === 'string' ? data.details : JSON.stringify(data.details)) : '';
          addLog(`Prediction failed: ${data.error || 'Unknown error'}`);
          if (detailStr) addLog(`DETAIL: ${detailStr.substring(0, 100)}...`);
          if (data.logs) {
            data.logs.split('\n').filter((l: string) => l.trim()).forEach((line: string) => addLog(`SYSTEM: ${line}`));
          }
        }
      } else {
        const text = await res.text();
        console.error('Non-JSON response:', text);
        if (text.includes('Please wait while your application starts')) {
          addLog('Status: Server overloaded. Inference aborted.');
        } else {
          addLog(`Error: Server returned non-JSON response (Status ${res.status}).`);
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


        </section>

        {/* Center Panel: System Console Logs */}
        <section className="flex-1 flex flex-col gap-6 min-h-[500px] h-fit">
          <div className="flex-1 bg-[#080a0f] border border-border-subtle rounded-2xl p-6 shadow-2xl flex flex-col relative overflow-hidden group">
            <div className="absolute inset-0 opacity-[0.02] pointer-events-none" style={{ backgroundImage: 'radial-gradient(#fff 1px, transparent 0)', backgroundSize: '40px 40px' }}></div>
            
            <div className="flex justify-between items-center mb-6 border-b border-border-subtle pb-3 z-10">
              <div className="flex items-center gap-3">
                <Terminal className="text-slate-500 w-5 h-5" />
                <span className="text-xs uppercase tracking-[0.2em] font-bold text-slate-300">System Console Stream</span>
              </div>
              <span className="text-accent-emerald text-[10px] flex items-center gap-2 font-mono">
                <div className="w-2 h-2 rounded-full bg-accent-emerald shadow-[0_0_8px_#10b981] animate-pulse" /> 
                LIVE TELEMETRY
              </span>
            </div>
            
            <div className="flex-1 space-y-2 overflow-y-auto pr-4 scrollbar-hide z-10 font-mono text-xs">
              {logs.map((log, i) => (
                <p key={i} className={cn(
                  "opacity-80 transition-all hover:opacity-100 hover:translate-x-1",
                  log.includes('Error') || log.includes('failed') || log.includes('CRITICAL') ? "text-red-400 bg-red-500/5 py-1 px-2 -ml-2 rounded border-l-2 border-red-500" : 
                  log.includes('successful') || log.includes('complete') ? "text-accent-emerald" : 
                  log.includes('SYSTEM:') ? "text-accent-cyan" : "text-slate-400"
                )}>{log}</p>
              ))}
              {logs.length === 0 && <p className="text-slate-600 italic">Waiting for AETHER-LSTM backend connections...</p>}
            </div>
          </div>
        </section>

        {/* Right Panel: Market Context & Prediction */}
        <section className="w-full lg:w-64 flex flex-col gap-6 shrink-0 h-fit">
          {/* Prediction View */}
          <div className="h-64 bg-gradient-to-br from-[#0d0f14] to-[#08090c] border border-border-muted rounded-xl relative flex flex-col items-center justify-center shadow-2xl overflow-hidden group">
            <div className="absolute inset-0 opacity-[0.03] pointer-events-none" style={{ backgroundImage: 'radial-gradient(#fff 1px, transparent 0)', backgroundSize: '20px 20px' }}></div>
            
            <AnimatePresence mode="wait">
              {!prediction && !isPredicting ? (
                <motion.div 
                  key="idle"
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  exit={{ opacity: 0 }}
                  className="z-10 text-center px-4"
                >
                  <p className="text-slate-500 text-[10px] tracking-[0.2em] uppercase mb-3 font-bold">Model Idle</p>
                  <div className="text-[10px] text-slate-400 font-mono leading-relaxed">
                    Awaiting CSV data payload to execute neural inference engine.
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
                  <div className="relative w-16 h-16 mx-auto mb-4">
                    <div className="absolute inset-0 border-2 border-accent-cyan/20 rounded-full"></div>
                    <div className="absolute inset-0 border-2 border-t-accent-cyan rounded-full animate-spin"></div>
                    <Activity className="absolute inset-0 m-auto w-6 h-6 text-accent-cyan animate-pulse" />
                  </div>
                  <p className="text-accent-cyan text-[10px] tracking-[0.2em] uppercase animate-pulse">Inferring...</p>
                </motion.div>
              ) : (
                <motion.div 
                  key="result"
                  initial={{ opacity: 0, scale: 0.9 }}
                  animate={{ opacity: 1, scale: 1 }}
                  className="z-10 text-center w-full px-5"
                >
                  <p className="text-slate-500 text-[9px] tracking-[0.2em] uppercase mb-2 font-bold">Prediction</p>
                  <h3 className={cn(
                    "text-4xl font-black tracking-tighter leading-none drop-shadow-[0_0_20px_rgba(0,0,0,0.8)] mb-4",
                    prediction?.direction === 'UP' ? "text-accent-emerald" : "text-red-500"
                  )}>
                    {prediction?.direction === 'UP' ? 'BULL' : 'BEAR'}
                  </h3>
                  
                  <div className="flex items-center justify-between bg-black/40 p-3 rounded-lg border border-white/5 backdrop-blur-sm w-full">
                    <div className="text-left">
                      <p className="text-[8px] text-slate-500 uppercase font-black tracking-[0.1em] mb-1">Conf.</p>
                      <p className="text-xl font-mono text-white leading-none">{prediction?.confidence}<span className="text-[10px] text-slate-500">%</span></p>
                    </div>
                    <div className={cn(
                      "text-2xl",
                      prediction?.direction === 'UP' ? "text-accent-emerald" : "text-red-500"
                    )}>
                      {prediction?.direction === 'UP' ? <TrendingUp size={24} /> : <TrendingDown size={24} />}
                    </div>
                  </div>
                </motion.div>
              )}
            </AnimatePresence>
            <div className="absolute w-32 h-32 border border-white/5 rounded-full -bottom-10 pointer-events-none"></div>
          </div>



          <div className={cn(
            "border rounded-xl p-5 text-center transition-all duration-500 shadow-2xl",
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
            <p className="text-[9px] text-slate-500 font-mono uppercase tracking-tight">Status: {prediction ? 'Verified' : 'Standby'}</p>
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
