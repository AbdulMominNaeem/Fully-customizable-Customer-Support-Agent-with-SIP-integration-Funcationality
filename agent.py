import asyncio
import os
import time
from pathlib import Path

from dotenv import load_dotenv
from google.genai import types as genai_types

from livekit import agents, rtc
from livekit.agents import (
    Agent,
    AgentServer,
    AgentSession,
    AudioConfig,
    BackgroundAudioPlayer,
    BuiltinAudioClip,
)

from livekit.plugins import (
    google,
    deepgram,
    cartesia,
)

from telemetry import setup_langfuse
from tools.call_tools import (
    book_appointment,
    log_customer_issue,
    transfer_to_human,
    end_call,
)


load_dotenv(".env.local")


# Must match AGENT_DISPATCH_NAME in setup_sip.py
AGENT_DISPATCH_NAME = "car-support"

COMPANY_NAME = os.getenv("COMPANY_NAME", "our company")
AGENT_NAME = os.getenv("AGENT_NAME", "Ayesha")

# "realtime": Gemini Live speech-to-speech, understands and speaks Urdu, English and Chinese
# "pipeline": Deepgram + Gemini + Cartesia, English only
VOICE_MODE = os.getenv("VOICE_MODE", "realtime")
GEMINI_VOICE = os.getenv("GEMINI_VOICE", "Aoede")
STT_LANGUAGE = os.getenv("STT_LANGUAGE", "en")

# Realtime latency tuning.
# gemini-3.1-flash-live-preview starts replying in ~0.7s versus ~1.3-2.5s
# for gemini-2.5-flash-native-audio-preview-12-2025 (measured with this prompt).
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash-native-audio-preview-12-202")
# Thinking budget 0 = answer straight away; raise it if answers get sloppy.
# Gemini 2.5 only; Gemini 3 models use a minimal thinking level instead.
GEMINI_THINKING_BUDGET = int(os.getenv("GEMINI_THINKING_BUDGET", "0"))
# How long the caller must be silent before Ayesha replies. Lower is faster,
# too low and she may cut in when the caller pauses mid-sentence.
TURN_SILENCE_MS = int(os.getenv("TURN_SILENCE_MS", "500"))

KNOWLEDGE_FILE = Path(__file__).parent / "knowledge" / "company_info.md"

# Pre-recorded greeting, played the instant the call connects so the caller
# never hears silence while Gemini Live is connecting.
# Create it with: python make_greeting.py
GREETING_FILE = Path(__file__).parent / "assets" / "greeting.wav"
GREETING_TEXT = os.getenv(
    "GREETING_TEXT",
    f"Assalam-o-Alaikum, main {AGENT_NAME} baat kar rahi hoon {COMPANY_NAME} se, "
    "main aap ki kya madad kar sakti hoon?",
)

# Soft keyboard typing while the caller waits for a reply, so a short pause
# sounds like someone working instead of a dead line. "off" to disable.
WAITING_SOUND = os.getenv("WAITING_SOUND", "on") == "on"

# Seconds of silence from both sides before asking "are you still there?",
# and again before hanging up, so an abandoned line doesn't keep billing
SILENCE_TIMEOUT = float(os.getenv("SILENCE_TIMEOUT", "20"))


def load_knowledge() -> str:
    if not KNOWLEDGE_FILE.exists():
        return "No company information has been provided yet."

    return KNOWLEDGE_FILE.read_text(encoding="utf-8")


class CarSupportAgent(Agent):

    def __init__(self, caller_number: str | None, greeted: bool = False):

        caller_info = (
            f"The caller's phone number is {caller_number}. "
            "Use it for bookings and tickets unless they give a different number."
            if caller_number
            else "The caller's phone number is unknown. Ask for it when needed."
        )

        if greeted:
            caller_info += (
                f'\n\nThe caller has already heard your greeting: "{GREETING_TEXT}" '
                "Don't greet or introduce yourself again. "
                "Wait for the caller and answer what they say."
            )

        super().__init__(

            instructions=f"""
You are {AGENT_NAME}, the phone support agent for {COMPANY_NAME},
a car company in Pakistan. You are answering a live phone call.

IDENTITY

When introducing yourself, just say "main {AGENT_NAME} baat kar rahi hoon {COMPANY_NAME} se"
(in English: "this is {AGENT_NAME} from {COMPANY_NAME}").
Never call yourself an agent or an assistant of the company.
You are an AI assistant. Never claim to be human.

PHONE VOICE STYLE

This is a phone call, so everything you say is spoken aloud.

Keep answers short: one to three sentences.
Never use markdown, lists, symbols, or emojis.
Say prices in lakh and crore, the way Pakistanis say them,
not as "PKR" with digits.
Always use the exact price from the company information;
never round it or change the number when converting it to words.
Read phone numbers digit by digit.
Ask only one question at a time.
If you didn't hear or understand something, politely ask the caller to repeat.

Keep the call focused on what the caller needs. Don't add long explanations,
small talk, or repeat information they already have.
Once their request is handled, ask if there is anything else.
If there isn't, say a short goodbye and end the call.

Be warm, calm, and polite. Callers may be frustrated about
a problem with their car, so be patient and reassuring.
It's natural to use greetings like "Assalam-o-Alaikum" and "JazakAllah".

LANGUAGE

You speak three languages: Urdu, English, and Chinese (Mandarin).
Urdu is your default language. Start every call in Urdu and keep speaking
Urdu unless the caller talks to you in English or Chinese.
Once they do, reply in the language the caller is speaking.
Always match the language of the caller's latest message: if they say
something in English, answer in English, even if earlier turns were in Urdu.
If they speak Urdu, reply in natural, simple, everyday Pakistani Urdu,
the way a polite call center agent in Pakistan talks.
Common English words like car, service, booking, model, and engine
are fine to use in Urdu sentences.
If they speak English, reply in English.
If they speak Chinese, reply in natural, polite Mandarin Chinese
(Putonghua), the way a helpful call center agent in China talks.
Say car model names as they are normally said, and say prices in
Chinese number words, for example "四百五十万卢比" for 45 lakh rupees.
If they mix Urdu and English, you can mix too.
If the caller switches language, switch with them immediately.
If you are not sure which language the caller is speaking, use Urdu.
Never speak Hindi; use Urdu words, for example "shukriya" not "dhanyavaad".
Never speak Cantonese or any language other than Urdu, English, and Mandarin.

WHAT YOU CAN HELP WITH

Answer any question about {COMPANY_NAME}: car models, variants,
prices, features, colors, availability, booking and delivery,
financing and installments, dealerships and branches,
timings, service and maintenance, spare parts, warranty,
and general car problems.

For general car questions such as warning lights, strange noises,
battery, AC, or tyre issues, give simple, safe first advice.
If the problem sounds serious or unsafe, like brake failure,
smoke, overheating, or a fuel smell, tell the caller to stop driving
and offer to book a service visit or connect them to a person.

SOURCE OF TRUTH

The COMPANY INFORMATION section below is your only source for
company-specific facts: prices, models, stock, branches,
timings, policies, and phone numbers.

Never invent or guess company-specific facts.
If the answer isn't in the company information, say you don't have
that detail, and offer to log a callback request so the team can
get back to them, or to transfer them to a person.

TOOLS

book_appointment: when the caller wants a test drive, a service visit,
an inspection, or a showroom visit. Collect their name,
the car model, and their preferred date and time first.
Confirm the details back to them before booking.

log_customer_issue: when the caller has a complaint, a problem to report,
or wants someone to call them back. Collect their name and a clear
description first. After logging, read out the ticket number.

transfer_to_human: when the caller asks for a person, is very upset,
or needs something you can't handle. Tell them you're connecting them
before you call the tool.

end_call: when the conversation is finished and the caller has said
goodbye. Say a short goodbye first, then call the tool.

CALLER

{caller_info}

COMPANY INFORMATION

{load_knowledge()}
""",

            tools=[
                book_appointment,
                log_customer_issue,
                transfer_to_human,
                end_call,
            ],
        )


server = AgentServer()


@server.rtc_session(agent_name=AGENT_DISPATCH_NAME)
async def phone_agent(
    ctx: agents.JobContext,
):

    caller_number = None

    # In console mode there's no real caller to wait for
    if not ctx.is_fake_job():

        # The SIP caller joins the room as a participant.
        # Its attributes include the caller's number (sip.phoneNumber).
        participant = await ctx.wait_for_participant()

        if participant.kind == rtc.ParticipantKind.PARTICIPANT_KIND_SIP:
            caller_number = participant.attributes.get("sip.phoneNumber")

    print("================================")
    print("INCOMING CALL")
    print("Room:", ctx.room.name)
    print("Caller:", caller_number)
    print("================================")

    # One Langfuse session per call, filterable by caller and voice mode
    trace_provider = setup_langfuse(
        metadata={
            "langfuse.session.id": ctx.room.name,
            "langfuse.user.id": caller_number or "unknown",
            "langfuse.trace.tags": ["phone-call", VOICE_MODE],
            "langfuse.trace.metadata.company": COMPANY_NAME,
            "langfuse.trace.metadata.agent_name": AGENT_NAME,
        }
    )

    if trace_provider is not None:

        async def flush_traces():
            trace_provider.force_flush()

        ctx.add_shutdown_callback(flush_traces)

    if VOICE_MODE == "realtime":

        # Gemini Live hears and speaks directly, in Urdu, English or Chinese
        session = AgentSession(
            llm=google.realtime.RealtimeModel(
                model=GEMINI_MODEL,
                voice=GEMINI_VOICE,
                # Keep "thinking" before each reply to a minimum
                thinking_config=genai_types.ThinkingConfig(thinking_level="minimal")
                if "gemini-3" in GEMINI_MODEL
                else genai_types.ThinkingConfig(thinking_budget=GEMINI_THINKING_BUDGET),
                # Reply sooner once the caller stops talking
                realtime_input_config=genai_types.RealtimeInputConfig(
                    automatic_activity_detection=genai_types.AutomaticActivityDetection(
                        end_of_speech_sensitivity=genai_types.EndSensitivity.END_SENSITIVITY_HIGH,
                        silence_duration_ms=TURN_SILENCE_MS,
                    ),
                ),
            ),
            user_away_timeout=SILENCE_TIMEOUT,
        )

    else:

        session = AgentSession(

            stt=deepgram.STT(
                model="nova-3",
                language=STT_LANGUAGE,
            ),

            llm=google.LLM(
                model="gemini-3.1-flash-lite",
            ),

            tts=cartesia.TTS(
                model="sonic-3",
            ),

            user_away_timeout=SILENCE_TIMEOUT,
        )

    silence_tasks: set[asyncio.Task] = set()

    async def check_if_caller_left():

        try:
            handle = session.generate_reply(
                instructions="The caller has gone quiet. Briefly ask if they are "
                "still there, in the language of the conversation.",
            )
            await handle.wait_for_playout()
            await asyncio.sleep(SILENCE_TIMEOUT)

            # Still silent after the prompt: the caller has probably left
            if session.user_state == "away":
                print("Caller silent, ending call")

                handle = session.generate_reply(
                    instructions="The caller isn't responding. Say a short, polite "
                    "goodbye in the language of the conversation.",
                )
                await handle.wait_for_playout()
                await ctx.delete_room()

        except Exception as e:
            # The call may have already ended
            print("Silence check stopped:", e)

    # Background audio (greeting and waiting sound) doesn't play in console mode
    use_background_audio = not ctx.is_fake_job()
    play_greeting = use_background_audio and GREETING_FILE.exists()

    background = BackgroundAudioPlayer(
        # LiveKit plays this during tool calls, e.g. while saving a booking
        thinking_sound=AudioConfig(BuiltinAudioClip.KEYBOARD_TYPING, volume=0.4)
        if WAITING_SOUND
        else None,
    )
    waiting_handle = None

    def start_waiting_sound():
        nonlocal waiting_handle
        if WAITING_SOUND and use_background_audio and (
            waiting_handle is None or waiting_handle.done()
        ):
            waiting_handle = background.play(
                AudioConfig(BuiltinAudioClip.KEYBOARD_TYPING, volume=0.4),
            )

    def stop_waiting_sound():
        if waiting_handle is not None:
            waiting_handle.stop()

    @session.on("user_state_changed")
    def on_user_state_changed(ev):
        if ev.new_state == "away" and not silence_tasks:
            task = asyncio.create_task(check_if_caller_left())
            silence_tasks.add(task)
            task.add_done_callback(silence_tasks.discard)

        # Caller finished talking and the reply isn't playing yet
        if (
            ev.old_state == "speaking"
            and ev.new_state == "listening"
            and session.agent_state != "speaking"
        ):
            start_waiting_sound()
        elif ev.new_state == "speaking":
            stop_waiting_sound()

    # When the caller last stopped talking, to log how long each reply took
    caller_stopped_at = None

    @session.on("agent_state_changed")
    def on_agent_state_changed(ev):
        nonlocal caller_stopped_at
        if ev.new_state == "speaking":
            stop_waiting_sound()

            if caller_stopped_at is not None:
                print(f"Reply delay: {time.time() - caller_stopped_at:.2f}s")
                caller_stopped_at = None

    @session.on("user_state_changed")
    def track_caller_stopped(ev):
        nonlocal caller_stopped_at
        if ev.old_state == "speaking" and ev.new_state == "listening":
            caller_stopped_at = time.time()
        elif ev.new_state == "speaking":
            caller_stopped_at = None

    if use_background_audio:
        await background.start(room=ctx.room, agent_session=session)
        # Stop the audio mixer cleanly when the call ends
        ctx.add_shutdown_callback(background.aclose)

    if play_greeting:
        # Caller hears the greeting immediately; Gemini connects meanwhile
        background.play(str(GREETING_FILE))

    await session.start(
        room=ctx.room,
        agent=CarSupportAgent(caller_number, greeted=play_greeting),
    )

    if play_greeting:
        return

    # No recording (or console mode): let Gemini speak the greeting
    await session.generate_reply(
        instructions=f"""
Greet the caller in Urdu with exactly this sentence and nothing more:
"Assalam-o-Alaikum, main {AGENT_NAME} baat kar rahi hoon {COMPANY_NAME} se, main aap ki kya madad kar sakti hoon?"

Keep talking in Urdu after this. Only switch to English or Chinese (Mandarin)
if the caller speaks to you in English or Chinese.
"""
    )


if __name__ == "__main__":
    agents.cli.run_app(server)
