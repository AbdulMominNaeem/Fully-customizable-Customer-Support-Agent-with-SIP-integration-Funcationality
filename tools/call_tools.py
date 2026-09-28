import asyncio
import json
import os
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from livekit import rtc
from livekit.agents import RunContext, function_tool, get_job_context

from database import (
    appointments_collection,
    tickets_collection,
)


DATA_DIR = Path(__file__).parent.parent / "data"


def _save_record(collection, filename: str, record: dict) -> None:
    if collection is not None:
        collection.insert_one(record)
        return

    # No MongoDB configured: append to a local JSON Lines file
    DATA_DIR.mkdir(exist_ok=True)

    with open(DATA_DIR / filename, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, default=str) + "\n")


def _caller_number() -> str | None:
    room = get_job_context().room

    for participant in room.remote_participants.values():
        if participant.kind == rtc.ParticipantKind.PARTICIPANT_KIND_SIP:
            return participant.attributes.get("sip.phoneNumber")

    return None


def _sip_participant_identity() -> str | None:
    room = get_job_context().room

    for participant in room.remote_participants.values():
        if participant.kind == rtc.ParticipantKind.PARTICIPANT_KIND_SIP:
            return participant.identity

    return None


@function_tool()
async def book_appointment(
    context: RunContext,
    appointment_type: Literal["test_drive", "service", "inspection", "showroom_visit"],
    customer_name: str,
    car_model: str,
    preferred_date: str,
    preferred_time: str,
    phone_number: str = "",
    notes: str = "",
) -> str:
    """
    Book a test drive, service visit, inspection, or showroom visit for the caller.

    Only call this after collecting the caller's name, the car model,
    and their preferred date and time, and after confirming them with the caller.

    Args:
        appointment_type: The kind of appointment.
        customer_name: The caller's full name.
        car_model: The car model the appointment is for.
        preferred_date: The preferred date, as the caller said it (e.g. "Monday 5 October").
        preferred_time: The preferred time, as the caller said it (e.g. "11 am").
        phone_number: Contact number, only if the caller gave a different one.
        notes: Any extra details, like the car's problem or registration number.
    """

    appointment = {
        "type": appointment_type,
        "customer_name": customer_name,
        "phone_number": phone_number or _caller_number(),
        "car_model": car_model,
        "preferred_date": preferred_date,
        "preferred_time": preferred_time,
        "notes": notes,
        "status": "pending_confirmation",
        "room": get_job_context().room.name,
        "created_at": datetime.now(timezone.utc),
    }

    print("NEW APPOINTMENT:", appointment)

    try:
        await asyncio.to_thread(
            _save_record,
            appointments_collection,
            "appointments.jsonl",
            appointment,
        )

    except Exception as e:
        print("Failed to save appointment:", e)

        return json.dumps({
            "success": False,
            "error": "Unable to save the booking right now.",
        })

    return json.dumps({
        "success": True,
        "message": "Appointment request saved. The team will call to confirm the exact slot.",
    })


@function_tool()
async def log_customer_issue(
    context: RunContext,
    issue_type: Literal["complaint", "car_problem", "callback_request", "other"],
    customer_name: str,
    description: str,
    car_model: str = "",
    registration_number: str = "",
    phone_number: str = "",
    urgent: bool = False,
) -> str:
    """
    Log a complaint, a car problem, or a callback request, and return a ticket number.

    Use this when the caller reports a problem, makes a complaint,
    asks something you can't answer, or wants someone to call them back.

    Args:
        issue_type: The kind of issue.
        customer_name: The caller's full name.
        description: A clear summary of the issue or request in the caller's words.
        car_model: The caller's car model, if relevant.
        registration_number: The car's registration number, if given.
        phone_number: Contact number, only if the caller gave a different one.
        urgent: True if the issue is a safety risk or the caller says it's urgent.
    """

    ticket_number = str(random.randint(100000, 999999))

    ticket = {
        "ticket_number": ticket_number,
        "type": issue_type,
        "customer_name": customer_name,
        "phone_number": phone_number or _caller_number(),
        "description": description,
        "car_model": car_model,
        "registration_number": registration_number,
        "urgent": urgent,
        "status": "open",
        "room": get_job_context().room.name,
        "created_at": datetime.now(timezone.utc),
    }

    print("NEW TICKET:", ticket)

    try:
        await asyncio.to_thread(
            _save_record,
            tickets_collection,
            "tickets.jsonl",
            ticket,
        )

    except Exception as e:
        print("Failed to save ticket:", e)

        return json.dumps({
            "success": False,
            "error": "Unable to log the issue right now.",
        })

    return json.dumps({
        "success": True,
        "ticket_number": ticket_number,
    })


@function_tool()
async def transfer_to_human(
    context: RunContext,
) -> str:
    """
    Transfer the call to a human agent.

    Use this when the caller asks to speak to a person, is very upset,
    or needs help you can't provide. Tell the caller you're connecting
    them before calling this.
    """

    transfer_number = os.getenv("HUMAN_TRANSFER_NUMBER")
    participant_identity = _sip_participant_identity()

    if not transfer_number or not participant_identity:
        return json.dumps({
            "success": False,
            "error": "Transfer isn't available right now. Offer to log a callback request instead.",
        })

    # Let the "connecting you" message finish before transferring
    await context.wait_for_playout()

    try:
        await get_job_context().transfer_sip_participant(
            participant_identity,
            transfer_number,
            play_dialtone=True,
        )

    except Exception as e:
        print("Transfer failed:", e)

        return json.dumps({
            "success": False,
            "error": "The transfer failed. Offer to log a callback request instead.",
        })

    return json.dumps({
        "success": True,
    })


@function_tool()
async def end_call(
    context: RunContext,
) -> str:
    """
    Hang up the call.

    Use this only after the caller has said goodbye or the conversation
    is clearly finished. Say a short goodbye before calling this.
    """

    # Let the goodbye finish before hanging up
    await context.wait_for_playout()

    await get_job_context().delete_room()

    return json.dumps({
        "success": True,
    })
