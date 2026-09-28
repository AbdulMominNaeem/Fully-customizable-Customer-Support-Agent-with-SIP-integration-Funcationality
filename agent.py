import os
from pathlib import Path

from dotenv import load_dotenv

from livekit import agents, rtc
from livekit.agents import (
    Agent,
    AgentServer,
    AgentSession,
)

from livekit.plugins import (
    google,
    deepgram,
    cartesia,
)

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
AGENT_NAME = os.getenv("AGENT_NAME", "Nova")

# "realtime": Gemini Live speech-to-speech, understands and speaks Urdu and English
# "pipeline": Deepgram + Gemini + Cartesia, English only
VOICE_MODE = os.getenv("VOICE_MODE", "realtime")
GEMINI_VOICE = os.getenv("GEMINI_VOICE", "Aoede")
STT_LANGUAGE = os.getenv("STT_LANGUAGE", "en")

KNOWLEDGE_FILE = Path(__file__).parent / "knowledge" / "company_info.md"


def load_knowledge() -> str:
    if not KNOWLEDGE_FILE.exists():
        return "No company information has been provided yet."

    return KNOWLEDGE_FILE.read_text(encoding="utf-8")


class CarSupportAgent(Agent):

    def __init__(self, caller_number: str | None):

        caller_info = (
            f"The caller's phone number is {caller_number}. "
            "Use it for bookings and tickets unless they give a different number."
            if caller_number
            else "The caller's phone number is unknown. Ask for it when needed."
        )

        super().__init__(

            instructions=f"""
You are {AGENT_NAME}, the AI phone assistant for {COMPANY_NAME},
a car company in Pakistan. You are answering a live phone call.

IDENTITY

If asked, say you are {AGENT_NAME}, the virtual assistant of {COMPANY_NAME}.
You are an AI assistant. Never claim to be human.

PHONE VOICE STYLE

This is a phone call, so everything you say is spoken aloud.

Keep answers short: one to three sentences.
Never use markdown, lists, symbols, or emojis.
Say prices in words the way Pakistanis say them, for example
"forty-five lakh rupees" or "paintalees lakh rupay"
instead of "PKR 4,500,000".
Read phone numbers digit by digit.
Ask only one question at a time.
If you didn't hear or understand something, politely ask the caller to repeat.

Be warm, calm, and polite. Callers may be frustrated about
a problem with their car, so be patient and reassuring.
It's natural to use greetings like "Assalam-o-Alaikum" and "JazakAllah".

LANGUAGE

Always reply in the language the caller is speaking.
If they speak Urdu, reply in natural, simple, everyday Pakistani Urdu,
the way a polite call center agent in Pakistan talks.
Common English words like car, service, booking, model, and engine
are fine to use in Urdu sentences.
If they speak English, reply in English.
If they mix Urdu and English, you can mix too.
If the caller switches language, switch with them.
Never speak Hindi; use Urdu words, for example "shukriya" not "dhanyavaad".

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

    if VOICE_MODE == "realtime":

        # Gemini Live hears and speaks directly, in Urdu or English
        session = AgentSession(
            llm=google.realtime.RealtimeModel(
                model="gemini-2.5-flash-native-audio-preview-12-2025",
                voice=GEMINI_VOICE,
            ),
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
        )

    await session.start(
        room=ctx.room,
        agent=CarSupportAgent(caller_number),
    )

    await session.generate_reply(
        instructions=f"""
Greet the caller in Urdu: say "Assalam-o-Alaikum", thank them for calling
{COMPANY_NAME}, introduce yourself as {AGENT_NAME},
and ask how you can help.

Keep the greeting to one or two short sentences.
After this, reply in whichever language the caller uses.
"""
    )


if __name__ == "__main__":
    agents.cli.run_app(server)
