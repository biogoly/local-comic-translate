"""Parsing and validation helpers for local OpenAI-compatible models."""

from __future__ import annotations

import json
from typing import Any, Callable, Iterable


class TranslationResponseError(ValueError):
    """Raised when a model response cannot be used without losing block alignment."""


def chat_completions_url(base_url: str) -> str:
    """Return a chat-completions URL from either a base URL or full endpoint."""
    normalized = base_url.strip().rstrip("/")
    if normalized.endswith("/chat/completions"):
        return normalized
    return f"{normalized}/chat/completions"


def build_chat_payload(
    *,
    model: str,
    system_prompt: str,
    user_prompt: str,
    image_data_url: str | None,
    temperature: float,
    top_p: float,
    top_k: int,
    max_tokens: int,
) -> dict[str, Any]:
    """Build the conservative OpenAI-compatible request used by local servers."""
    user_content: list[dict[str, Any]] = []
    if image_data_url:
        # Image-first works with current local VLM templates and remains valid
        # OpenAI-compatible multimodal content.
        user_content.append({
            "type": "image_url",
            "image_url": {"url": image_data_url},
        })
    user_content.append({"type": "text", "text": user_prompt})

    return {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
        "temperature": temperature,
        "top_p": top_p,
        "top_k": top_k,
        "max_tokens": max_tokens,
        "response_format": {"type": "json_object"},
    }


def request_valid_translation(
    perform_request: Callable[[str], str],
    base_prompt: str,
    expected_keys: Iterable[str],
    max_attempts: int = 2,
    required_content_keys: Iterable[str] | None = None,
) -> dict[str, str]:
    """Request a complete translation object, retrying malformed model output."""
    ordered_keys = tuple(expected_keys)
    last_error: TranslationResponseError | None = None
    for attempt in range(max(1, max_attempts)):
        prompt = base_prompt
        if attempt and last_error is not None:
            prompt += (
                "\n\nYour previous response was invalid ("
                f"{last_error}). Try again. Output only the complete JSON object with exactly "
                "the requested keys and string values."
            )

        content = perform_request(prompt)
        try:
            return parse_translation_object(
                content,
                ordered_keys,
                required_content_keys=required_content_keys,
            )
        except TranslationResponseError as exc:
            last_error = exc

    raise TranslationResponseError(
        f"invalid translation response after {max(1, max_attempts)} attempts: {last_error}"
    )


def extract_message_content(response_data: Any) -> str:
    """Extract text from an OpenAI-compatible chat completion response."""
    try:
        content = response_data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise TranslationResponseError(
            "Response did not contain choices[0].message.content."
        ) from exc

    if isinstance(content, str):
        return content

    # A few compatible servers return an array of content parts instead of a
    # string. Preserve all text parts, but reject non-text-only responses.
    if isinstance(content, list):
        text_parts = [
            part.get("text", "")
            for part in content
            if isinstance(part, dict) and isinstance(part.get("text"), str)
        ]
        if text_parts:
            return "".join(text_parts)

    raise TranslationResponseError("The completion content was not text.")


def parse_translation_object(
    content: str,
    expected_keys: Iterable[str],
    required_content_keys: Iterable[str] | None = None,
) -> dict[str, str]:
    """Find and validate a complete block-to-translation JSON object.

    Local models occasionally surround JSON with Markdown fences or reasoning
    text. We tolerate that wrapper, but require the object itself to contain
    exactly the expected block IDs and string values. This prevents a malformed
    response from shifting translations into the wrong speech bubbles.
    """
    if not isinstance(content, str) or not content.strip():
        raise TranslationResponseError("The model returned an empty response.")

    ordered_keys = tuple(expected_keys)
    expected = set(ordered_keys)
    required_content = set(required_content_keys or ())
    decoder = json.JSONDecoder()
    validation_errors: list[str] = []
    found_object = False

    for index, character in enumerate(content):
        if character != "{":
            continue
        try:
            candidate, _ = decoder.raw_decode(content[index:])
        except json.JSONDecodeError:
            continue
        if not isinstance(candidate, dict):
            continue

        found_object = True
        actual = set(candidate.keys())
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        non_string = sorted(
            key for key, value in candidate.items()
            if key in expected and not isinstance(value, str)
        )
        empty = sorted(
            key for key, value in candidate.items()
            if key in required_content
            and isinstance(value, str)
            and not any(character.isalnum() for character in value)
        )

        if not missing and not extra and not non_string and not empty:
            return {key: candidate[key] for key in ordered_keys}

        problems = []
        if missing:
            problems.append(f"missing keys: {', '.join(missing)}")
        if extra:
            problems.append(f"unexpected keys: {', '.join(extra)}")
        if non_string:
            problems.append(f"non-string values: {', '.join(non_string)}")
        if empty:
            problems.append(f"values without translated text: {', '.join(empty)}")
        validation_errors.append("; ".join(problems))

    if validation_errors:
        raise TranslationResponseError(validation_errors[0])
    if found_object:
        raise TranslationResponseError("No usable translation object was found.")
    raise TranslationResponseError("The model did not return a JSON object.")
