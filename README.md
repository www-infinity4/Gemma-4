# Gemma 4 Chat

A simple web chat interface for Google's **Gemma 4** model, powered by the Google Gemini API.

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Set your API key

The app reads your Google AI API key from the `API` environment variable.  
**No manual key entry in the UI** — just set the variable once and the app uses it automatically.

**Local development:**

```bash
cp .env.example .env
# Edit .env and replace the placeholder with your real Google AI API key
```

**GitHub Actions / deployment:**  
Add a repository secret named **`API`** with your Google AI API key.  
The app will pick it up automatically via `os.environ.get("API")`.

### 3. Run

```bash
python app.py
```

Then open [http://localhost:5000](http://localhost:5000) in your browser.

## Model

Uses `gemma-4-31b-it` via the Google Gemini API.

## Features

- Multi-turn conversation with full history context
- Clean dark-mode chat UI
- Shift+Enter for newlines, Enter to send
- Clear conversation button
