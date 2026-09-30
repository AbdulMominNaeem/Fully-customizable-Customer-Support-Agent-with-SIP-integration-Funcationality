"""
One-time setup: records the call greeting in Ayesha's own voice.

The agent plays this file the instant a call connects, so the caller hears
a voice right away instead of silence while Gemini Live is connecting.

Run it once:  python make_greeting.py
Run it again after changing AGENT_NAME, COMPANY_NAME, GEMINI_VOICE
or GREETING_TEXT in .env.local.
"""

import os
import wave
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from google.genai import types


load_dotenv(".env.local")


AGENT_NAME = os.getenv("AGENT_NAME", "Ayesha")
COMPANY_NAME = os.getenv("COMPANY_NAME", "our company")
GEMINI_VOICE = os.getenv("GEMINI_VOICE", "Aoede")
TTS_MODEL = os.getenv("GREETING_TTS_MODEL", "gemini-2.5-flash-preview-tts")

# Must match GREETING_FILE in agent.py
GREETING_FILE = Path(__file__).parent / "assets" / "greeting.wav"

GREETING_TEXT = os.getenv(
    "GREETING_TEXT",
    f"Assalam-o-Alaikum, main {AGENT_NAME} baat kar rahi hoon {COMPANY_NAME} se, "
    "main aap ki kya madad kar sakti hoon?",
)


def main():

    client = genai.Client()

    response = client.models.generate_content(
        model=TTS_MODEL,
        contents=(
            "Say this in natural Pakistani Urdu, warmly and at a normal pace, "
            "like a friendly call center agent answering the phone:\n"
            f"{GREETING_TEXT}"
        ),
        config=types.GenerateContentConfig(
            response_modalities=["AUDIO"],
            automatic_function_calling=types.AutomaticFunctionCallingConfig(
                disable=True,
            ),
            speech_config=types.SpeechConfig(
                voice_config=types.VoiceConfig(
                    prebuilt_voice_config=types.PrebuiltVoiceConfig(
                        voice_name=GEMINI_VOICE,
                    )
                )
            ),
        ),
    )

    # Gemini TTS returns raw 16-bit mono PCM at 24 kHz
    pcm = response.candidates[0].content.parts[0].inline_data.data

    GREETING_FILE.parent.mkdir(exist_ok=True)

    with wave.open(str(GREETING_FILE), "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(24000)
        f.writeframes(pcm)

    print(f"Saved {GREETING_FILE} ({len(pcm) / 48000:.1f} seconds)")
    print("Greeting:", GREETING_TEXT)


if __name__ == "__main__":
    main()
