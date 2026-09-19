"""Direct providers and stable settings keys; no account or hosted-service routing."""

PROVIDERS = {
    "OpenAI": ("Open AI GPT", ("gpt-4.1-mini", "gpt-4.1")),
    "Google Gemini": ("Google Gemini", ("gemini-3.1-flash-lite", "gemini-2.5-pro")),
    "Anthropic Claude": ("Anthropic Claude", ("claude-sonnet-4-6", "claude-haiku-4-5-20251001")),
    "Deepseek": ("Deepseek", ("deepseek-flash", "deepseek-v4-pro")),
}


def provider_for_translator(name: str) -> str | None:
    if name in PROVIDERS:
        return name
    for prefix, provider in (("GPT", "OpenAI"), ("Gemini", "Google Gemini"),
                             ("Claude", "Anthropic Claude")):
        if name.startswith(prefix):
            return provider
    return None


def credential_service(name: str) -> str:
    if name == "Microsoft OCR":
        return "Microsoft Azure"
    if name == "Google Cloud Vision":
        return "Google Cloud"
    provider = provider_for_translator(name)
    return PROVIDERS[provider][0] if provider else name
