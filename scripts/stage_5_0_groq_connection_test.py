
import os
from pathlib import Path

from dotenv import load_dotenv
from groq import Groq

PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")

api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    raise SystemExit(
        "ERROR: GROQ_API_KEY missing from the project-root .env file."
    )

try:
    client = Groq(api_key=api_key)

    # Discover the models returned by Groq for this API key.
    model_response = client.models.list()
    available = sorted(
        model.id
        for model in model_response.data
        if getattr(model, "active", True)
    )

    if not available:
        raise RuntimeError("Groq returned no active models.")

    print("Available model IDs:")
    for model_id in available:
        print(f"  {model_id}")

    # Prefer documented general-purpose chat models.
    preferred = [
        os.getenv("GROQ_MODEL", "").strip(),
        "openai/gpt-oss-20b",
        "openai/gpt-oss-120b",
        "qwen/qwen3.6-27b",
        "qwen/qwen3.8-27b",
    ]

    model_id = next(
        (candidate for candidate in preferred
         if candidate and candidate in available),
        None,
    )

    if model_id is None:
        raise RuntimeError(
            "No preferred chat model was found in the model list. "
            "Set GROQ_MODEL in .env to a suitable chat model ID printed above."
        )

    print(f"\nTesting chat completion with: {model_id}")

    response = client.chat.completions.create(
        model=model_id,
        messages=[
            {
                "role": "system",
                "content": (
                    "You assist with conveyor-belt crack analysis. "
                    "Never invent measurements or claim to inspect data "
                    "that was not provided."
                ),
            },
            {
                "role": "user",
                "content": "Reply with exactly: Groq connection successful.",
            },
        ],
        temperature=0,
        max_completion_tokens=60,
    )

    answer = response.choices[0].message.content
    print("\nGROQ API TEST PASSED")
    print(f"Model: {model_id}")
    print(f"Response: {answer}")

except Exception as exc:
    print("\nGROQ API TEST FAILED")
    print(f"Error type: {type(exc).__name__}")
    print(str(exc))
    raise SystemExit(1)
