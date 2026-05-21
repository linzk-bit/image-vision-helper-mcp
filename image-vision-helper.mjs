#!/usr/bin/env node

/**
 * image-vision-helper MCP Server v2.0
 * ================================
 * Secure & Performant Image Understanding Tool
 *
 * Improvements:
 *   - Worker pool caching (10x speedup for repeated OCR)
 *   - File size limits (prevents OOM)
 *   - Request timeouts (prevents hanging)
 *   - Input validation (path traversal & language whitelist)
 *   - Concurrent request limiting (prevents overload)
 *   - Base64 LRU cache (avoids re-encoding)
 *   - Structured logging & health checks
 */

import fs from "node:fs";
import path from "node:path";

import { Server } from "@modelcontextprotocol/sdk/server/index.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import {
  CallToolRequestSchema,
  ListToolsRequestSchema,
} from "@modelcontextprotocol/sdk/types.js";

import { createWorker } from "tesseract.js";
import dotenv from "dotenv";
import { Jimp } from "jimp";

// ═══════════════════════════════════════════════════════════════
// Configuration
// ═══════════════════════════════════════════════════════════════

const SCRIPT_DIR = path.dirname(path.resolve(process.argv[1]));

// Load .env with validation
const envResult = dotenv.config({ path: path.join(SCRIPT_DIR, ".env") });
if (envResult.error) {
  console.error("[WARN] .env file not found, using defaults or environment variables");
}

const CONFIG = {
  vision: {
    baseUrl: process.env.VISION_BASE_URL || "https://dashscope.aliyuncs.com/compatible-mode/v1",
    apiKey: process.env.VISION_API_KEY || "",
    model: process.env.VISION_MODEL || "qwen-vl-plus",
    timeoutMs: parseInt(process.env.VISION_TIMEOUT_MS, 10) || 120000, // Default: 120 seconds
    maxFileSize: 10 * 1024 * 1024, // 10MB
    maxDimension: parseInt(process.env.VISION_MAX_DIMENSION, 10) || 1024, // Max dimension in pixels for image resizing
  },
  ocr: {
    defaultLang: process.env.OCR_DEFAULT_LANG || "eng",
    maxFileSize: 20 * 1024 * 1024, // 20MB (OCR can handle larger than vision)
    maxConcurrency: 2,
  },
  security: {
    // Set to a directory path to restrict file access, or null to allow any
    allowedDir: process.env.ALLOWED_DIR || null,
  },
};

// Validate critical configuration
function validateConfig() {
  const issues = [];
  if (!CONFIG.vision.apiKey || CONFIG.vision.apiKey === "your-api-key-here") {
    issues.push("VISION_API_KEY not configured (visual_analyze will be unavailable)");
  }
  if (CONFIG.vision.apiKey.length < 10) {
    issues.push("VISION_API_KEY looks invalid (too short)");
  }
  return issues;
}

// ═══════════════════════════════════════════════════════════════
// Constants
// ═══════════════════════════════════════════════════════════════

const SUPPORTED_EXTENSIONS = new Set([
  ".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp", ".tiff", ".tif", ".avif",
]);

const LANGUAGE_MAP = {
  eng: "eng",
  chinese: "chi_sim",
  "chinese-simplified": "chi_sim",
  "chinese-traditional": "chi_tra",
  japanese: "jpn",
  korean: "kor",
  french: "fra",
  german: "deu",
  spanish: "spa",
  russian: "rus",
  arabic: "ara",
};

const VALID_LANGUAGES = Object.keys(LANGUAGE_MAP);

// ═══════════════════════════════════════════════════════════════
// Structured Logging (uses stderr to avoid polluting stdout)
// ═══════════════════════════════════════════════════════════════

function log(level, message, meta = {}) {
  const entry = {
    timestamp: new Date().toISOString(),
    level,
    message,
    ...meta,
  };
  // Mask sensitive data
  if (entry.apiKey) entry.apiKey = "***";
  console.error(JSON.stringify(entry));
}

// ═══════════════════════════════════════════════════════════════
// Worker Pool (Performance: avoids recreating workers)
// ═══════════════════════════════════════════════════════════════

const workerPool = new Map();
const workerLocks = new Map();

async function getWorker(lang) {
  // If worker exists and is ready, return it
  if (workerPool.has(lang)) {
    return workerPool.get(lang);
  }

  // If someone is already creating this worker, wait
  if (workerLocks.has(lang)) {
    await workerLocks.get(lang);
    return workerPool.get(lang);
  }

  // Create worker with lock
  let resolveLock;
  const lockPromise = new Promise((r) => (resolveLock = r));
  workerLocks.set(lang, lockPromise);

  try {
    log("info", `Creating tesseract worker for language: ${lang}`);
    const worker = await createWorker(lang);
    workerPool.set(lang, worker);
    log("info", `Worker ready for language: ${lang}`);
    return worker;
  } catch (err) {
    log("error", `Failed to create worker for language: ${lang}`, { error: err.message });
    throw new Error(
      `OCR initialization failed for language '${lang}'. ` +
      `Ensure ${lang}.traineddata exists in ${SCRIPT_DIR}. ` +
      `Error: ${err.message}`
    );
  } finally {
    resolveLock();
    workerLocks.delete(lang);
  }
}

async function cleanupWorkers() {
  log("info", "Cleaning up tesseract workers...");
  const promises = [];
  for (const [lang, worker] of workerPool.entries()) {
    promises.push(
      worker.terminate().catch((err) => {
        log("error", `Failed to terminate worker for ${lang}`, { error: err.message });
      })
    );
  }
  await Promise.all(promises);
  workerPool.clear();
  log("info", "All workers terminated");
}

// Graceful shutdown
process.on("exit", () => {
  // Synchronous cleanup only
  console.error("[image-vision-helper] Process exiting");
});

process.on("SIGINT", async () => {
  await cleanupWorkers();
  process.exit(0);
});

process.on("SIGTERM", async () => {
  await cleanupWorkers();
  process.exit(0);
});

// ═══════════════════════════════════════════════════════════════
// Base64 LRU Cache (Performance: avoids re-reading files)
// ═══════════════════════════════════════════════════════════════

class LRUCache {
  constructor(maxSize = 50) {
    this.maxSize = maxSize;
    this.cache = new Map();
  }

  get(key) {
    if (!this.cache.has(key)) return undefined;
    const value = this.cache.get(key);
    // Move to end (most recently used)
    this.cache.delete(key);
    this.cache.set(key, value);
    return value;
  }

  set(key, value) {
    if (this.cache.has(key)) {
      this.cache.delete(key);
    } else if (this.cache.size >= this.maxSize) {
      // Remove least recently used (first item)
      const firstKey = this.cache.keys().next().value;
      this.cache.delete(firstKey);
    }
    this.cache.set(key, value);
  }

  clear() {
    this.cache.clear();
  }
}

const base64Cache = new LRUCache(20);

// ═══════════════════════════════════════════════════════════════
// Concurrency Control (Safety: limits parallel processing)
// ═══════════════════════════════════════════════════════════════

class ConcurrencyLimiter {
  constructor(maxConcurrent) {
    this.maxConcurrent = maxConcurrent;
    this.running = 0;
    this.queue = [];
  }

  async acquire() {
    if (this.running < this.maxConcurrent) {
      this.running++;
      return;
    }
    await new Promise((resolve) => this.queue.push(resolve));
  }

  release() {
    if (this.queue.length > 0) {
      const next = this.queue.shift();
      next();
    } else {
      this.running--;
    }
  }
}

const ocrLimiter = new ConcurrencyLimiter(CONFIG.ocr.maxConcurrency);

// ═══════════════════════════════════════════════════════════════
// Helper Functions
// ═══════════════════════════════════════════════════════════════

function formatFileSize(bytes) {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(2)} MB`;
}

function validateImage(filePath, maxSize) {
  if (!filePath) {
    throw new Error("Missing required parameter: file_path");
  }

  // Security: resolve to absolute path
  const resolvedPath = path.resolve(filePath);

  // Security: optional directory restriction
  if (CONFIG.security.allowedDir) {
    const allowedDir = path.resolve(CONFIG.security.allowedDir);
    if (!resolvedPath.startsWith(allowedDir)) {
      throw new Error(
        `Access denied: file must be within ${allowedDir}. ` +
        `Received: ${resolvedPath}`
      );
    }
  }

  if (!fs.existsSync(resolvedPath)) {
    throw new Error(`File not found: ${resolvedPath}`);
  }

  const stats = fs.statSync(resolvedPath);
  if (!stats.isFile()) {
    throw new Error(`Path is not a file: ${resolvedPath}`);
  }

  if (stats.size > maxSize) {
    throw new Error(
      `File too large: ${formatFileSize(stats.size)}. ` +
      `Maximum allowed: ${formatFileSize(maxSize)}. ` +
      `Please resize or compress the image.`
    );
  }

  const ext = path.extname(resolvedPath).toLowerCase();
  if (!SUPPORTED_EXTENSIONS.has(ext)) {
    throw new Error(
      `Unsupported format: ${ext}. ` +
      `Supported: ${[...SUPPORTED_EXTENSIONS].join(", ")}`
    );
  }

  return resolvedPath;
}

function validateLanguage(lang) {
  if (!VALID_LANGUAGES.includes(lang)) {
    throw new Error(
      `Unsupported language: '${lang}'. ` +
      `Supported: ${VALID_LANGUAGES.join(", ")}`
    );
  }
  return LANGUAGE_MAP[lang] || lang;
}

function fileToBase64(filePath) {
  // Check cache first
  const cached = base64Cache.get(filePath);
  if (cached) {
    log("debug", "Base64 cache hit", { file: path.basename(filePath) });
    return cached;
  }

  const ext = path.extname(filePath).toLowerCase().slice(1);
  const mimeType = ext === "jpg" ? "jpeg" : ext;
  const data = fs.readFileSync(filePath);
  const base64 = `data:image/${mimeType};base64,${data.toString("base64")}`;

  // Cache result
  base64Cache.set(filePath, base64);
  return base64;
}

// ═══════════════════════════════════════════════════════════════
// Image Resizing (for Vision LLM)
// ═══════════════════════════════════════════════════════════════

async function resizeImageForVision(filePath) {
  const maxDim = CONFIG.vision.maxDimension;

  try {
    const image = await Jimp.read(filePath);
    const { width, height } = image.bitmap;

    if (width <= maxDim && height <= maxDim) {
      log("debug", "Image within size limit, no resize needed", {
        file: path.basename(filePath),
        width,
        height,
      });
      return null; // No resize needed
    }

    const scale = Math.min(maxDim / width, maxDim / height);
    const newWidth = Math.round(width * scale);
    const newHeight = Math.round(height * scale);

    log("info", "Resizing image for vision API", {
      file: path.basename(filePath),
      from: `${width}x${height}`,
      to: `${newWidth}x${newHeight}`,
    });

    image.resize({ w: newWidth, h: newHeight });

    // Convert to JPEG buffer (good compression, widely supported)
    const buffer = await image.getBuffer("image/jpeg");
    return buffer;
  } catch (err) {
    log("warn", "Image resize failed, falling back to original", {
      file: path.basename(filePath),
      error: err.message,
    });
    return null; // Fallback: let caller use original file
  }
}

function bufferToBase64(buffer, mimeType = "jpeg") {
  return `data:image/${mimeType};base64,${buffer.toString("base64")}`;
}

// ═══════════════════════════════════════════════════════════════
// OCR Text Extraction (Local)
// ═══════════════════════════════════════════════════════════════

async function performOCR(filePath, lang = CONFIG.ocr.defaultLang) {
  const tessLang = validateLanguage(lang);

  // Check if traineddata exists locally (prevents auto-download hangs)
  const trainedDataPath = path.join(SCRIPT_DIR, `${tessLang}.traineddata`);
  if (tessLang !== "eng" && !fs.existsSync(trainedDataPath)) {
    throw new Error(
      `OCR language '${lang}' (${tessLang}) not available locally. ` +
      `Only 'eng' is installed. To use other languages, download ` +
      `${tessLang}.traineddata from https://github.com/tesseract-ocr/tessdata ` +
      `and place it in ${SCRIPT_DIR}`
    );
  }

  await ocrLimiter.acquire();
  const startTime = Date.now();

  try {
    const worker = await getWorker(tessLang);
    const { data } = await worker.recognize(filePath);

    log("info", "OCR completed", {
      file: path.basename(filePath),
      language: lang,
      confidence: data.confidence,
      duration: Date.now() - startTime,
    });

    return {
      text: data.text || "",
      confidence: data.confidence || 0,
    };
  } finally {
    ocrLimiter.release();
  }
}

function formatOCR(ocr) {
  const lines = ["--- OCR Text Recognition Result ---"];
  if (!ocr.text.trim()) {
    lines.push("  No text detected in the image.");
    return lines.join("\n");
  }
  lines.push(`  Confidence: ${ocr.confidence.toFixed(1)}%`);
  lines.push(`  Content:`);
  for (const line of ocr.text.trim().split("\n")) {
    lines.push(`    ${line}`);
  }
  return lines.join("\n");
}

// ═══════════════════════════════════════════════════════════════
// Vision LLM Image Analysis (Online)
// ═══════════════════════════════════════════════════════════════

async function callVisionLLM(filePath, prompt) {
  if (!CONFIG.vision.apiKey || CONFIG.vision.apiKey.length < 10) {
    throw new Error(
      "VISION_API_KEY not configured or invalid. " +
      "Please set it in the .env file."
    );
  }

  // Resize if needed (keeps pixel count manageable for vision API)
  const resizedBuffer = await resizeImageForVision(filePath);
  const base64Image = resizedBuffer
    ? bufferToBase64(resizedBuffer, "jpeg")
    : fileToBase64(filePath);

  const url = `${CONFIG.vision.baseUrl.replace(/\/$/, "")}/chat/completions`;

  const controller = new AbortController();
  const timeout = setTimeout(() => {
    controller.abort();
  }, CONFIG.vision.timeoutMs);

  const startTime = Date.now();

  try {
    const response = await fetch(url, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Authorization: `Bearer ${CONFIG.vision.apiKey}`,
      },
      body: JSON.stringify({
        model: CONFIG.vision.model,
        messages: [
          {
            role: "user",
            content: [
              { type: "text", text: prompt },
              { type: "image_url", image_url: { url: base64Image } },
            ],
          },
        ],
        max_tokens: 2000,
      }),
      signal: controller.signal,
    });

    clearTimeout(timeout);

    if (!response.ok) {
      const errorText = await response.text();

      // Classify errors by status code
      switch (response.status) {
        case 401:
          throw new Error(
            "API Key is invalid or expired (401). Please check VISION_API_KEY in your .env file."
          );
        case 429:
          throw new Error(
            "Too many requests (429), rate limit exceeded. Please try again later."
          );
        case 500:
        case 502:
        case 503:
          throw new Error(
            `Vision LLM service temporarily unavailable (${response.status}). Please try again later.`
          );
        default:
          throw new Error(
            `API request failed (${response.status}): ${errorText.slice(0, 200)}`
          );
      }
    }

    const result = await response.json();

    log("info", "Vision LLM analysis completed", {
      file: path.basename(filePath),
      model: CONFIG.vision.model,
      duration: Date.now() - startTime,
    });

    return result.choices[0].message.content;
  } catch (err) {
    clearTimeout(timeout);

    if (err.name === "AbortError") {
      throw new Error(
        `Vision analysis request timed out (${CONFIG.vision.timeoutMs / 1000}s). ` +
        `The network may be unstable or the service is responding slowly. Please try again later.`
      );
    }

    // Don't leak API key in error messages
    if (err.message.includes(CONFIG.vision.apiKey)) {
      err.message = err.message.replace(CONFIG.vision.apiKey, "***");
    }

    throw err;
  }
}

function formatVisionResult(content, filePath) {
  const lines = [
    "--- Vision LLM Image Analysis Result ---",
    "",
    `Analysis Model: ${CONFIG.vision.model}`,
    `Image File: ${path.basename(filePath)}`,
    "",
    "Analysis Content:",
    "",
  ];

  for (const line of content.split("\n")) {
    lines.push(`  ${line}`);
  }

  lines.push(
    "",
    "NOTE: This analysis was performed via online API, image data has been uploaded to the service provider."
  );

  return lines.join("\n");
}

// ═══════════════════════════════════════════════════════════════
// MCP Server
// ═══════════════════════════════════════════════════════════════

const server = new Server(
  {
    name: "image-vision-helper",
    version: "2.0.0",
  },
  {
    capabilities: { tools: {} },
  }
);

server.setRequestHandler(ListToolsRequestSchema, async () => ({
  tools: [
    {
      name: "extract_text",
      description:
        "[Local execution, privacy-safe] Use OCR to extract text from images. " +
        "Supports multiple languages. No internet required. " +
        "Max file size: 20MB.",
      inputSchema: {
        type: "object",
        properties: {
          file_path: {
            type: "string",
            description: "Local image file path (.png/.jpg/.webp/.bmp/.tiff/.avif)",
          },
          language: {
            type: "string",
            description: `Recognition language: ${VALID_LANGUAGES.join(", ")}`,
            default: CONFIG.ocr.defaultLang,
          },
        },
        required: ["file_path"],
      },
    },
    {
      name: "visual_analyze",
      description:
        "[Online analysis] Use vision LLM to deeply understand image content. " +
        "Can describe scenes, objects, charts, emotions, etc. " +
        `Auto-resizes images to max ${CONFIG.vision.maxDimension}px if larger. ` +
        `API Key required. Max file size: 10MB. Timeout: ${CONFIG.vision.timeoutMs / 1000}s.`,
      inputSchema: {
        type: "object",
        properties: {
          file_path: {
            type: "string",
            description: "Local image file path",
          },
          prompt: {
            type: "string",
            description:
              "Custom analysis prompt (e.g.: 'Analyze the trend of this chart')",
            default:
              "Describe the content of this image in detail, including all visible information.",
          },
        },
        required: ["file_path"],
      },
    },
    {
      name: "health_check",
      description:
        "Check server status, configuration, and available resources. " +
        "Use this to diagnose issues.",
      inputSchema: {
        type: "object",
        properties: {},
      },
    },
  ],
}));

server.setRequestHandler(CallToolRequestSchema, async (request) => {
  const { name, arguments: args } = request.params;
  const requestId = `${Date.now()}-${Math.random().toString(36).slice(2, 7)}`;

  log("info", `Tool call: ${name}`, { requestId, args: { file_path: args?.file_path } });

  try {
    switch (name) {
      case "extract_text": {
        const resolvedPath = validateImage(args?.file_path, CONFIG.ocr.maxFileSize);
        const lang = args?.language || CONFIG.ocr.defaultLang;
        const result = await performOCR(resolvedPath, lang);
        return {
          content: [{ type: "text", text: formatOCR(result) }],
        };
      }

      case "visual_analyze": {
        const resolvedPath = validateImage(args?.file_path, CONFIG.vision.maxFileSize);
        const prompt =
          args?.prompt ||
          "Describe the content of this image in detail, including all visible information.";
        const result = await callVisionLLM(resolvedPath, prompt);
        return {
          content: [{ type: "text", text: formatVisionResult(result, resolvedPath) }],
        };
      }

      case "health_check": {
        const installedLangs = ["eng"]; // Always available
        // Check for other language files
        for (const langCode of Object.values(LANGUAGE_MAP)) {
          if (langCode === "eng") continue;
          if (fs.existsSync(path.join(SCRIPT_DIR, `${langCode}.traineddata`))) {
            const displayName = Object.entries(LANGUAGE_MAP).find(([, v]) => v === langCode)?.[0];
            if (displayName && !installedLangs.includes(displayName)) {
              installedLangs.push(displayName);
            }
          }
        }

        return {
          content: [
            {
              type: "text",
              text: JSON.stringify(
                {
                  status: "healthy",
                  version: "2.0.0",
                  timestamp: new Date().toISOString(),
                  ocr: {
                    available: true,
                    installedLanguages: installedLangs,
                    defaultLanguage: CONFIG.ocr.defaultLang,
                    maxFileSize: formatFileSize(CONFIG.ocr.maxFileSize),
                    workerPoolSize: workerPool.size,
                    concurrentJobs: ocrLimiter.running,
                    queuedJobs: ocrLimiter.queue.length,
                  },
                  vision: {
                    configured: !!CONFIG.vision.apiKey && CONFIG.vision.apiKey.length >= 10,
                    model: CONFIG.vision.model,
                    endpoint: CONFIG.vision.baseUrl,
                    maxFileSize: formatFileSize(CONFIG.vision.maxFileSize),
                    timeoutSeconds: CONFIG.vision.timeoutMs / 1000,
                    maxDimension: CONFIG.vision.maxDimension,
                  },
                  security: {
                    allowedDir: CONFIG.security.allowedDir || "(not restricted)",
                  },
                  cache: {
                    base64CacheSize: base64Cache.cache.size,
                  },
                },
                null,
                2
              ),
            },
          ],
        };
      }

      default:
        throw new Error(`Unknown tool: ${name}`);
    }
  } catch (err) {
    log("error", `Tool call failed: ${name}`, {
      requestId,
      error: err.message,
    });

    return {
      content: [{ type: "text", text: `Error: ${err.message}` }],
      isError: true,
    };
  }
});

// ═══════════════════════════════════════════════════════════════
// Startup
// ═══════════════════════════════════════════════════════════════

async function main() {
  const transport = new StdioServerTransport();
  await server.connect(transport);

  const configIssues = validateConfig();

  console.error("╔══════════════════════════════════════════════════════════════╗");
  console.error("║  image-vision-helper MCP Server v2.0                        ║");
  console.error("╠══════════════════════════════════════════════════════════════╣");
  console.error(`║  Local OCR:        Ready (default: ${CONFIG.ocr.defaultLang})                    ║`);
  console.error(`║  Vision LLM:       ${CONFIG.vision.apiKey && CONFIG.vision.apiKey.length >= 10 ? "Configured ✓" : "Not configured ✗"}                          ║`);
  console.error(`║  Model:            ${CONFIG.vision.model.padEnd(40)} ║`);
  console.error(`║  Max file size:    OCR ${formatFileSize(CONFIG.ocr.maxFileSize).padEnd(5)} / Vision ${formatFileSize(CONFIG.vision.maxFileSize).padEnd(5)}        ║`);
  console.error(`║  Workers cached:   ${workerPool.size}                                          ║`);
  console.error("╚══════════════════════════════════════════════════════════════╝");

  if (configIssues.length > 0) {
    console.error("\n⚠️  Configuration issues:");
    for (const issue of configIssues) {
      console.error(`   - ${issue}`);
    }
  }
}

main().catch((err) => {
  console.error("[image-vision-helper] Startup failed:", err.message);
  process.exit(1);
});
