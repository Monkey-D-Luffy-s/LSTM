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

  // Python command setup
  const pythonCmd = "python3";

  // API: Train
  app.post("/api/train", upload.single("file"), (req: any, res: any) => {
    if (!req.file) return res.status(400).json({ error: "No file uploaded" });

    const filePath = req.file.path;
    let pythonProcess = spawn(pythonCmd, ["predictor.py", "train", filePath]);

    let logs = "";
    pythonProcess.on("error", (err: any) => {
      if (err.code === 'ENOENT') {
         console.error(`Python command "${pythonCmd}" not found.`);
         logs += `[CRITICAL ERROR] "${pythonCmd}" was not found in your system path. Please install Python 3.10+.`;
      }
    });

    pythonProcess.stdout.on("data", (data) => {
      logs += data.toString();
      console.log(`Python: ${data}`);
    });

    pythonProcess.stderr.on("data", (data) => {
      const errStr = data.toString();
      logs += `[ERROR] ${errStr}`;
      console.error(`Python Error: ${errStr}`);
    });

    pythonProcess.on("close", (code) => {
      // Clean up upload
      if (fs.existsSync(filePath)) fs.unlinkSync(filePath);
      
      if (code === 0) {
        res.json({ message: "Training complete", logs });
      } else {
        res.status(500).json({ error: "Training failed", logs });
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

    pythonProcess.on("close", (code) => {
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
          res.status(500).json({ error: "Could not parse prediction result", logs: output });
        }
      } else {
        res.status(500).json({ error: "Prediction failed", logs: output });
      }
    });
  });

  // API 404 Handler - Ensure API routes always return JSON
  app.use("/api/*", (req, res) => {
    res.status(404).json({ error: `API route not found: ${req.method} ${req.originalUrl}` });
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
