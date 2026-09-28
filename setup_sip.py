"""
One-time setup: connects your SIP phone number to this agent on LiveKit.

It creates:
  1. An inbound SIP trunk that accepts calls to SIP_PHONE_NUMBER
  2. A dispatch rule that puts each call in its own room with the agent

Run it once:  python setup_sip.py
Run it again after changing SIP_* values in .env.local to update the trunk.
"""

import asyncio
import os

from dotenv import load_dotenv
from livekit import api


load_dotenv(".env.local")


# Must match AGENT_DISPATCH_NAME in agent.py
AGENT_DISPATCH_NAME = "car-support"

TRUNK_NAME = "car-support-inbound"
DISPATCH_RULE_NAME = "car-support-dispatch"


def _env_list(name: str) -> list[str]:
    value = os.getenv(name, "")
    return [item.strip() for item in value.split(",") if item.strip()]


async def main():

    phone_number = os.getenv("SIP_PHONE_NUMBER")

    if not phone_number:
        raise ValueError(
            "SIP_PHONE_NUMBER is missing from .env.local (e.g. +924212345678)"
        )

    lkapi = api.LiveKitAPI()

    try:

        # ==========================================
        # INBOUND TRUNK
        # ==========================================

        trunk_info = api.SIPInboundTrunkInfo(
            name=TRUNK_NAME,
            numbers=[phone_number],
            # Only accept calls from your provider's IPs, if set
            allowed_addresses=_env_list("SIP_ALLOWED_ADDRESSES"),
            # Digest auth, if your provider uses a username/password
            auth_username=os.getenv("SIP_AUTH_USERNAME", ""),
            auth_password=os.getenv("SIP_AUTH_PASSWORD", ""),
            # Telephony noise cancellation
            krisp_enabled=True,
        )

        all_trunks = await lkapi.sip.list_inbound_trunk(
            api.ListSIPInboundTrunkRequest()
        )

        existing = [t for t in all_trunks.items if t.name == TRUNK_NAME]

        if existing:
            # Re-running after changing .env.local updates the same trunk,
            # e.g. a new phone number or credentials
            trunk = await lkapi.sip.update_inbound_trunk(
                existing[0].sip_trunk_id,
                trunk_info,
            )
            print("Updated inbound trunk:", trunk.sip_trunk_id, list(trunk.numbers))

        else:
            trunk = await lkapi.sip.create_inbound_trunk(
                api.CreateSIPInboundTrunkRequest(trunk=trunk_info)
            )
            print("Created inbound trunk:", trunk.sip_trunk_id, list(trunk.numbers))

        # ==========================================
        # DISPATCH RULE
        # ==========================================

        existing_rules = await lkapi.sip.list_dispatch_rule(
            api.ListSIPDispatchRuleRequest(trunk_ids=[trunk.sip_trunk_id])
        )

        if existing_rules.items:
            print(
                "Dispatch rule already exists:",
                existing_rules.items[0].sip_dispatch_rule_id,
            )

        else:
            rule = await lkapi.sip.create_dispatch_rule(
                api.CreateSIPDispatchRuleRequest(
                    dispatch_rule=api.SIPDispatchRuleInfo(
                        name=DISPATCH_RULE_NAME,
                        trunk_ids=[trunk.sip_trunk_id],
                        # Each caller gets their own room, e.g. call-+923001234567_abc
                        rule=api.SIPDispatchRule(
                            dispatch_rule_individual=api.SIPDispatchRuleIndividual(
                                room_prefix="call-",
                            )
                        ),
                        room_config=api.RoomConfiguration(
                            agents=[
                                api.RoomAgentDispatch(
                                    agent_name=AGENT_DISPATCH_NAME,
                                )
                            ]
                        ),
                    )
                )
            )
            print("Created dispatch rule:", rule.sip_dispatch_rule_id)

    finally:
        await lkapi.aclose()

    print()
    print("Done. Now point your SIP provider's trunk at your LiveKit SIP URI")
    print("(LiveKit Cloud dashboard > Settings > SIP URI).")


if __name__ == "__main__":
    asyncio.run(main())
