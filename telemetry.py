"""
Langfuse tracing for the phone agent.

LiveKit Agents emits OpenTelemetry spans for every call: each turn, the
Gemini Live generation (model, tokens, latency), tool calls and so on.
This sends those spans to Langfuse, one trace session per call.

Needs LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY and LANGFUSE_BASE_URL
in .env.local. Without them, tracing is simply skipped.
"""

import os

from langfuse import Langfuse
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.util.types import AttributeValue

from livekit.agents.telemetry import set_tracer_provider


def setup_langfuse(
    metadata: dict[str, AttributeValue] | None = None,
) -> TracerProvider | None:

    public_key = os.getenv("LANGFUSE_PUBLIC_KEY")
    secret_key = os.getenv("LANGFUSE_SECRET_KEY")
    base_url = os.getenv("LANGFUSE_BASE_URL") or os.getenv("LANGFUSE_HOST")

    if not (public_key and secret_key and base_url):
        print("Langfuse not configured, tracing disabled")
        return None

    environment = os.getenv("LANGFUSE_TRACING_ENVIRONMENT", "development")

    # Must be set before AgentSession.start(). LiveKit Cloud's own
    # observability adds its exporter to this same provider, so both work.
    trace_provider = TracerProvider()
    set_tracer_provider(
        trace_provider,
        metadata={"langfuse.environment": environment, **(metadata or {})},
    )

    Langfuse(
        public_key=public_key,
        secret_key=secret_key,
        base_url=base_url,
        tracer_provider=trace_provider,
        environment=environment,
        # LiveKit spans aren't from a known LLM library, so export all of them
        should_export_span=lambda span: True,
    )

    return trace_provider
