# image-vision-helper MCP Server

An MCP (Model Context Protocol) server for image understanding, supporting **local OCR** and **online vision LLM analysis**.

## Features

| Tool | Description | Network | Privacy |
|------|-------------|---------|---------|
| `extract_text` | Local OCR text extraction with 11 language support | Offline | 100% local |
| `visual_analyze` | Vision LLM deep analysis - describe scenes, charts, objects, etc. | Online | Image uploaded to provider |

## Quick Start

### 1. Install dependencies

```bash
npm install
```

### 2. Configure Vision LLM

Copy `.env.example` to `.env` and edit:

```bash
cp .env.example .env
```

```env
# Alibaba Cloud Qwen example (recommended)
VISION_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
VISION_API_KEY=sk-your-api-key
VISION_MODEL=qwen-vl-plus

# OpenAI example
# VISION_BASE_URL=https://api.openai.com/v1
# VISION_API_KEY=sk-xxx
# VISION_MODEL=gpt-4o
```

### 3. Start the server

```bash
npm start
```

## Usage with MCP Client (e.g. Claude Desktop)

Edit your MCP client config (e.g. `claude_desktop_config.json`):

```json
{
  "mcpServers": {
    "image-vision-helper": {
      "command": "node",
      "args": ["/path/to/image-vision-helper.mjs"]
    }
  }
}
```

Restart your MCP client to use the tools.

## Tools

### extract_text

Extract text from images using local OCR (Tesseract.js). No internet required.

**Parameters:**
- `file_path` (required): Path to image file (.png/.jpg/.webp/.bmp/.tiff/.avif)
- `language` (optional): Recognition language. Default: `eng`. Supported: eng, chinese, chinese-simplified, chinese-traditional, japanese, korean, french, german, spanish, russian, arabic

**Privacy:** 100% local. Images never leave your machine.

### visual_analyze

Analyze image content using a vision-capable LLM via OpenAI-compatible API.

**Parameters:**
- `file_path` (required): Path to image file
- `prompt` (optional): Custom analysis prompt. Default: "Describe the content of this image in detail, including all visible information."

**Privacy:** Image is encoded as base64 and sent to the configured LLM provider.

### health_check

Check server status, configuration, and available resources.

## Supported Vision LLMs

- Alibaba Cloud Qwen: `qwen-vl-plus`, `qwen-vl-max`, `qwen3.5-plus`
- OpenAI: `gpt-4o`, `gpt-4o-mini`
- Any OpenAI-compatible vision model

## Files

| File | Description |
|------|-------------|
| `image-vision-helper.mjs` | MCP server main file - local OCR + online vision LLM |
| `.env` | Configuration file (API Key, etc.) |
| `.env.example` | Configuration template |
| `package.json` | Dependencies |
| `eng.traineddata` | Tesseract OCR language data for English |

## Privacy Notes

- **extract_text**: Fully local, images never leave your computer
- **visual_analyze**: Image is base64-encoded and sent to the LLM provider. For sensitive images, use `extract_text` only.

## License

MIT
