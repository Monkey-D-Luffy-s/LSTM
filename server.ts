import express from "express";
import { createServer as createViteServer } from "vite";
import path from "path";
import multer from "multer";
import { spawn } from "child_process";
import fs from "fs";
import cors from "cors";

async function startServer() {
  const app = express();
  const PORT = 3000;

  app.use(cors());
  app.use(express.json());

  // SSE endpoint for live logs
  let logClients: any[] = [];
  
  app.get("/api/logs", (req, res) => {
    res.setHeader("Content-Type", "text/event-stream");
    res.setHeader("Cache-Control", "no-cache");
    res.setHeader("Connection", "keep-alive");
    res.flushHeaders();
    
    logClients.push(res);
    
    // Send current status immediately
    if (isPythonReady) {
      res.write(`data: SYSTEM: Python dependencies verified successfully. System Ready.\n\n`);
    } else {
      res.write(`data: SYSTEM: Initializing backend python engine, please wait...\n\n`);
    }
    
    req.on("close", () => {
      logClients = logClients.filter(client => client !== res);
    });
  });

  const broadcastLog = (msg: string) => {
    logClients.forEach(client => client.write(`data: ${msg}\n\n`));
  };

  // Setup Multer for disk storage
  const uploadsDir = path.join(process.cwd(), "uploads");
  if (!fs.existsSync(uploadsDir)) {
    fs.mkdirSync(uploadsDir, { recursive: true });
  }

  const storage = multer.diskStorage({
    destination: (req, file, cb) => {
      cb(null, uploadsDir);
    },
    filename: (req, file, cb) => {
      cb(null, `${Date.now()}-${file.originalname}`);
    },
  });
  const upload = multer({ storage });

  // Python command detection and dependency installation
  let pythonCmd = "python3";
  let pipCmd = "pip3";
  let isPythonReady = false;

  // Check which python command is available and install requirements in background
  const initializePython = async () => {
    try {
      // Prioritize 'python' on Windows, then 'python3'
      const isWindows = process.platform === "win32";
      const primaryCmd = isWindows ? "python" : "python3";
      
      await new Promise((resolve, reject) => {
        const check = spawn(primaryCmd, ["--version"]);
        check.on("error", reject);
        check.on("close", (code) => {
          if (code === 0) resolve(code);
          else reject(new Error(`Command ${primaryCmd} exited with code ${code}`));
        });
      });
      pythonCmd = primaryCmd;
      pipCmd = isWindows ? "pip" : "pip3";
    } catch (e) {
      // Fallback
      const isWindows = process.platform === "win32";
      pythonCmd = isWindows ? "python3" : "python";
      pipCmd = isWindows ? "pip3" : "pip";
    }

    console.log(`Using Python command: ${pythonCmd}`);
    
    // First check if dependencies are already installed
    console.log("Checking if Python dependencies are already installed...");
    const check = spawn(pythonCmd, ["-c", "import pandas; import tensorflow; print('READY')"]);
    
    check.on("close", (code) => {
      if (code === 0) {
        console.log("Python dependencies verified successfully.");
        broadcastLog("SYSTEM: Python dependencies verified successfully. System Ready.");
        isPythonReady = true;
      } else {
        console.log("Dependencies not found. Running background pip install...");
        broadcastLog("SYSTEM: Dependencies not found. Running background pip install...");
        const install = spawn(pythonCmd, ["-m", "pip", "install", "--no-cache-dir", "-r", "requirements.txt"]);
        
        install.stdout.on("data", (data) => {
          console.log(`[PIP]: ${data.toString().trim()}`);
        });
        install.stderr.on("data", (data) => {
          console.error(`[PIP ERROR]: ${data.toString().trim()}`);
        });
        
        install.on("close", (installCode) => {
          if (installCode === 0) {
            console.log("PIP installation completed successfully.");
            isPythonReady = true;
          } else {
            console.error(`PIP installation failed with code ${installCode}.`);
            // Fallback check
            const fallbackCheck = spawn(pythonCmd, ["-c", "import pandas; import tensorflow; print('READY')"]);
            fallbackCheck.on("close", (innerCode) => {
              if (innerCode === 0) {
                console.log("Dependencies confirmed via fallback check.");
                isPythonReady = true;
              }
            });
          }
        });
      }
    });
  };

  // Start initialization
  initializePython();
  
  let isTrainingLock = false;

  // API: Train
  app.post("/api/train", upload.single("file"), (req: any, res: any) => {
    if (!isPythonReady) {
      return res.status(503).json({ 
        error: "Python engine is still warming up. Installing dependencies...", 
        logs: "The system is currently installing Pandas and TensorFlow. This takes 1-2 minutes on first boot. Please wait." 
      });
    }
    if (isTrainingLock) {
      return res.status(429).json({
        error: "Training is already in progress.",
        logs: "Please wait for the current training job to finish."
      });
    }
    if (!req.file) return res.status(400).json({ error: "No file uploaded" });

    isTrainingLock = true;
    broadcastLog("SYSTEM: Starting training process...");
    const filePath = req.file.path;
    let pythonProcess = spawn(pythonCmd, ["predictor.py", "train", filePath]);

    let logs = "";
    pythonProcess.on("error", (err: any) => {
      if (err.code === 'ENOENT') {
         console.error(`Python command "${pythonCmd}" not found.`);
         const errMsg = `[CRITICAL ERROR] "${pythonCmd}" was not found in your system path. Please install Python 3.10+.`;
         logs += errMsg;
         broadcastLog(errMsg);
      }
    });

    pythonProcess.stdout.on("data", (data) => {
      const msg = data.toString();
      logs += msg;
      console.log(`Python: ${msg}`);
      msg.split('\n').filter((l: string) => l.trim()).forEach((l: string) => broadcastLog(`[TRAIN] ${l.trim()}`));
    });

    pythonProcess.stderr.on("data", (data) => {
      const errStr = data.toString();
      logs += `[ERROR] ${errStr}`;
      console.error(`Python Error: ${errStr}`);
      errStr.split('\n').filter((l: string) => l.trim()).forEach((l: string) => broadcastLog(`[TRAIN ERROR] ${l.trim()}`));
    });

    pythonProcess.on("close", (code, signal) => {
      isTrainingLock = false;
      // Clean up upload
      if (fs.existsSync(filePath)) fs.unlinkSync(filePath);
      
      if (code === 0) {
        broadcastLog("SYSTEM: Training complete. Model saved.");
        res.json({ message: "Training complete", logs });
      } else {
        const errorMsg = signal ? `Training aborted by signal: ${signal} (Possible Out of Memory)` : `Training failed with code ${code}`;
        console.error(errorMsg);
        broadcastLog(`SYSTEM: ${errorMsg}`);
        res.status(500).json({ error: errorMsg, logs });
      }
    });
  });

  // API: Reset
  app.post("/api/reset", (req, res) => {
    try {
      const files = ["model.keras", "scaler.gz"];
      files.forEach(f => {
        if (fs.existsSync(f)) fs.unlinkSync(f);
      });
      res.json({ message: "System reset successful" });
    } catch (err) {
      res.status(500).json({ error: "Reset failed" });
    }
  });

  // API: Predict
  app.post("/api/predict", upload.single("file"), (req: any, res: any) => {
    if (!isPythonReady) {
      return res.status(503).json({ 
        error: "Python engine is still warming up. Installing dependencies...", 
        logs: "The system is currently installing Pandas and TensorFlow. This takes 1-2 minutes on first boot. Please wait." 
      });
    }
    if (!req.file) return res.status(400).json({ error: "No file uploaded" });

    const filePath = req.file.path;
    const pythonProcess = spawn(pythonCmd, ["predictor.py", "predict", filePath]);

    let output = "";
    pythonProcess.on("error", (err: any) => {
      if (err.code === 'ENOENT') {
        output += `[CRITICAL ERROR] "${pythonCmd}" was not found in your system path.`;
      }
    });

    pythonProcess.stdout.on("data", (data) => {
      output += data.toString();
    });

    pythonProcess.stderr.on("data", (data) => {
      output += `[ERROR] ${data.toString()}`;
    });

    pythonProcess.on("close", (code, signal) => {
      if (fs.existsSync(filePath)) fs.unlinkSync(filePath);

      if (code === 0) {
        const directionMatch = output.match(/RESULT_DIRECTION:(UP|DOWN)/);
        const confidenceMatch = output.match(/RESULT_CONFIDENCE:([\d.]+)/);

        if (directionMatch && confidenceMatch) {
          res.json({
            direction: directionMatch[1],
            confidence: parseFloat(confidenceMatch[1]),
            logs: output
          });
        } else {
          console.error("Prediction parse error. Output was:", output);
          res.status(500).json({ error: "Could not parse prediction result", logs: output });
        }
      } else {
        const errorMsg = signal ? `Prediction aborted by signal: ${signal}` : `Prediction failed with code ${code}`;
        console.error(errorMsg);
        res.status(500).json({ error: errorMsg, logs: output });
      }
    });
  });

  // API 404 Handler - Ensure API routes always return JSON
  app.use("/api/*", (req, res, next) => {
    res.status(404).json({ error: `API route not found: ${req.method} ${req.originalUrl}` });
  });

  // Global Error Handler for API
  app.use((err: any, req: any, res: any, next: any) => {
    console.error('SERVER ERROR:', err);
    if (req.path && req.path.startsWith('/api/')) {
      return res.status(err.status || 500).json({
        error: err.message || 'Internal Server Error',
        details: typeof err === 'object' ? err.toString() : err
      });
    }
    // Fallback for HTML
    if (!res.headersSent) {
      res.status(500).send(`<h1>Server Error</h1><p>${err.message || 'Unknown error'}</p>`);
    }
  });

  // Vite middleware for development
  if (process.env.NODE_ENV !== "production") {
    const vite = await createViteServer({
      server: { middlewareMode: true },
      appType: "spa",
    });
    app.use(vite.middlewares);
  } else {
    const distPath = path.join(process.cwd(), "dist");
    app.use(express.static(distPath));
    app.get("*", (req, res) => {
      res.sendFile(path.join(distPath, "index.html"));
    });
  }

  app.listen(PORT, "0.0.0.0", () => {
    console.log(`Server running on http://localhost:${PORT}`);
  });
}

startServer();
