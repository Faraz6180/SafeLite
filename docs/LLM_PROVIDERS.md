# LLM providers for SafeLite

## Overview

SafeLite now supports multiple LLM-backed providers through a single planner interface.
The planner chooses a provider from configuration and can fall back to the offline mock provider when no API key is available.

## Free provider options

### Groq

1. Create a free account at https://console.groq.com
2. Create an API key from the Keys page.
3. Set the environment variable `GROQ_API_KEY` to that key.
4. Set `LLM_PROVIDER=groq`.

### OpenRouter

1. Create a free account at https://openrouter.ai
2. Create an API key from the Keys page.
3. Set `OPENROUTER_API_KEY`.
4. Set `LLM_PROVIDER=openrouter`.

### Hugging Face

1. Create a free account at https://huggingface.co
2. Create a read access token from Settings > Access Tokens.
3. Set `HF_API_KEY`.
4. Set `LLM_PROVIDER=huggingface`.

## Configuration

Copy [.env.example](../.env.example) to `.env` and adjust the values.

Example:

```env
LLM_PROVIDER=mock
GROQ_API_KEY=
OPENROUTER_API_KEY=
HF_API_KEY=
MODEL_NAME=llama-3.1-8b-instant
TEMPERATURE=0.1
MAX_TOKENS=512
TIMEOUT=30
```

## Switching providers

Change the `LLM_PROVIDER` value in `.env` to one of:

- `mock`
- `groq`
- `openrouter`
- `huggingface`

The planner will instantiate the matching provider automatically.

## Testing without internet

Leave `LLM_PROVIDER=mock` or omit all API keys. The planner will use the offline mock provider and never crash from missing keys.
