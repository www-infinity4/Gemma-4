"""
Gemma 4 Chat App
-----------------
Reads the Google AI API key from the 'API' environment variable (or .env file).
No manual key entry required — set the GitHub secret named 'API' and it just works.
"""

import os
from flask import Flask, request, jsonify, render_template
from google import genai
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)

# Read API key from environment — set the GitHub secret named 'API'
_API_KEY = os.environ.get("API")

# Default Gemma 4 model
GEMMA_MODEL = "gemma-4-31b-it"


def _get_client() -> genai.Client:
    """Return a configured Gemini API client using the API secret."""
    if not _API_KEY:
        raise RuntimeError(
            "No API key found. Set the 'API' environment variable (or GitHub secret)."
        )
    return genai.Client(api_key=_API_KEY)


@app.route("/")
def index():
    return render_template("index.html", model=GEMMA_MODEL)


@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json(force=True)
    history = data.get("history", [])   # list of {role, text} dicts
    user_message = data.get("message", "").strip()

    if not user_message:
        return jsonify({"error": "Empty message"}), 400

    try:
        client = _get_client()

        # Build contents list from conversation history + new user turn
        contents = []
        for turn in history:
            role = turn.get("role", "user")
            text = turn.get("text", "")
            contents.append({"role": role, "parts": [{"text": text}]})

        contents.append({"role": "user", "parts": [{"text": user_message}]})

        response = client.models.generate_content(
            model=GEMMA_MODEL,
            contents=contents,
        )
        reply = response.text
    except RuntimeError:
        return jsonify({"error": "No API key configured. Set the 'API' environment variable."}), 500
    except Exception:  # noqa: BLE001
        return jsonify({"error": "Failed to generate a response from the Gemma API. Please try again."}), 500

    return jsonify({"reply": reply})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
