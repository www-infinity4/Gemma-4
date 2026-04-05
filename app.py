"""
JAXX — AI That Goes Fast
-------------------------
Reads the Google AI API key from the 'API' environment variable (or .env file).
No manual key entry required — set the GitHub secret named 'API' and it just works.
"""

import os
import base64
import binascii
from flask import Flask, request, jsonify, render_template, redirect
from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)

# Read API key from environment — set the GitHub secret named 'API'
_API_KEY = os.environ.get("API")

# Underlying model
GEMMA_MODEL = os.environ.get("GEMMA_MODEL", "gemma-2.0-flash-001")

# Maximum characters to include from a non-image file attachment
_MAX_FILE_CHARS = 8_000

# JAXX persona injected as system instruction on every request
JAXX_SYSTEM_PROMPT = (
    "You are JAXX — a blazing-fast, brilliantly knowledgeable AI built around a rich internal knowledge base. "
    "You love speed, go-karts, and racing, but your real power is that you know more than any ordinary assistant. "

    # Knowledge domains
    "Your built-in knowledge base covers: "
    "Science & Technology (physics, chemistry, biology, astronomy, computing, AI, robotics, engineering, medicine, quantum mechanics, climate science); "
    "Mathematics (algebra, calculus, statistics, number theory, geometry, cryptography); "
    "History & Culture (world history, wars, civilizations, art movements, philosophy, religion, mythology); "
    "Current Events & News (politics, economics, business, global affairs — up to your knowledge cutoff); "
    "Sports & Games (F1, NASCAR, go-karts, all major sports, esports, chess, speedrunning world records); "
    "Entertainment (movies, TV, music, anime, books, comics, gaming — classics to modern); "
    "Food & Travel (world cuisines, restaurants, destinations, cultures, languages); "
    "Health & Wellness (nutrition, fitness, mental health, medical conditions — always recommend professionals for diagnoses); "
    "Law & Finance (legal concepts, investing, crypto, taxes — always recommend professionals for formal advice); "
    "Programming & Dev (Python, JS, TypeScript, Rust, Go, SQL, system design, algorithms, open-source, APIs); "
    "Racing & Motorsport (F1 stats, go-kart tracks worldwide, lap records, car specs, tire strategy, pit stops). "

    # Behavior
    "Speak with personality: high-energy, witty, direct, occasionally drop a racing metaphor when it fits naturally. "
    "You are NOT Gemma — you are JAXX. "
    "When a question is simple, answer it fast and sharp. "
    "When a question is complex or multi-part, slow down (metaphorically), think it through carefully, check your internal knowledge from multiple angles, and deliver a thorough answer. "
    "For very complex or multi-step questions, treat them like a full race — run multiple mental laps, cross-check facts, handle edge cases, and flag uncertainty clearly. "
    "Never refuse a question; always take a fast lap at the answer even when uncertain, flagging it as your best take. "
    "If you are not sure about something, say so clearly — JAXX is honest about the limits of the knowledge base. "
    "Be helpful, smart, and entertaining. Always bring energy to every response."
)


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


@app.route("/tricorder")
def tricorder():
    return render_template("tricorder.html", model=GEMMA_MODEL)


@app.route("/tricorder/ask", methods=["POST"])
def tricorder_ask():
    """JAXX answers a question scoped to a specific Tricorder coin type."""
    data = request.get_json(force=True)
    coin_name = data.get("coin", "").strip()
    question = data.get("question", "").strip()

    if not question:
        return jsonify({"error": "Empty question"}), 400

    coin_context = f"The user is asking about the '{coin_name}' coin from the Tricorder Coin System. " if coin_name else ""
    prompt = (
        f"{coin_context}"
        f"Answer this question concisely and with JAXX energy: {question}"
    )

    try:
        client = _get_client()
        response = client.models.generate_content(
            model=GEMMA_MODEL,
            contents=[{"role": "user", "parts": [{"text": prompt}]}],
            config=types.GenerateContentConfig(
                system_instruction=JAXX_SYSTEM_PROMPT,
            ),
        )
        reply = response.text
    except RuntimeError:
        return jsonify({"error": "No API key configured. Set the 'API' environment variable."}), 500
    except Exception:  # noqa: BLE001
        return jsonify({"error": "JAXX blew a tire — failed to get a response. Please try again."}), 500

    return jsonify({"reply": reply})


@app.route("/chat", methods=["GET", "POST"])
def chat():
    # Redirect accidental GET requests (e.g. direct browser navigation to /chat)
    if request.method == "GET":
        return redirect("/")

    data = request.get_json(force=True)
    history = data.get("history", [])        # list of {role, text} dicts
    user_message = data.get("message", "").strip()
    attachments = data.get("files", [])      # list of {name, mime_type, data (base64)}

    if not user_message and not attachments:
        return jsonify({"error": "Empty message"}), 400

    try:
        client = _get_client()

        # Build contents list from conversation history
        contents = []
        for turn in history:
            role = turn.get("role", "user")
            text = turn.get("text", "")
            contents.append({"role": role, "parts": [{"text": text}]})

        # Build user parts: images first, then text (optimal for multimodal models)
        user_parts = []
        for f in attachments:
            mime_type = f.get("mime_type", "application/octet-stream")
            file_data = f.get("data", "")
            file_name = f.get("name", "file")

            if mime_type.startswith("image/"):
                user_parts.append({
                    "inline_data": {
                        "mime_type": mime_type,
                        "data": file_data,
                    }
                })
            else:
                try:
                    decoded = base64.b64decode(file_data).decode("utf-8", errors="replace")
                    # Truncate very large files to avoid token limits
                    if len(decoded) > _MAX_FILE_CHARS:
                        decoded = decoded[:_MAX_FILE_CHARS] + "\n… [truncated]"
                    user_parts.append({
                        "text": f"[Attached file: {file_name}]\n```\n{decoded}\n```"
                    })
                except (binascii.Error, ValueError, UnicodeDecodeError):
                    user_parts.append({
                        "text": f"[Attached binary file: {file_name} — unable to read as text]"
                    })

        if user_message:
            user_parts.append({"text": user_message})

        if user_parts:
            contents.append({"role": "user", "parts": user_parts})

        response = client.models.generate_content(
            model=GEMMA_MODEL,
            contents=contents,
            config=types.GenerateContentConfig(
                system_instruction=JAXX_SYSTEM_PROMPT,
            ),
        )
        reply = response.text or "(No response generated)"
    except RuntimeError:
        return jsonify({"error": "No API key configured. Set the 'API' environment variable."}), 500
    except Exception:  # noqa: BLE001
        return jsonify({"error": "JAXX blew a tire — failed to get a response. Please try again."}), 500

    return jsonify({"reply": reply})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
