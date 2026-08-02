# Groq Setup Guide

## 1. Paste your API key

Open [.env](../.env) and replace the empty value for `GROQ_API_KEY` with your real Groq key.

```env
GROQ_API_KEY=your-real-key-here
LLM_PROVIDER=groq
MODEL_NAME=llama-3.1-8b-instant
TEMPERATURE=0.1
MAX_TOKENS=512
TIMEOUT=30
```

## 2. Install dependencies

```bash
pip install -r requirements.txt
```

## 3. Run the health check

```bash
python scripts/check_groq.py
```

Expected output:

```text
✓ Connected
```

## 4. Run the live demo

```bash
python scripts/demo_live.py
```

You will be prompted to enter a natural-language instruction. For example:

```text
Pick up the red cube and place it into the blue basket.
```

## Common errors

- Missing `GROQ_API_KEY`: the planner will fall back to the mock provider.
- Invalid API key: the health check will report a detailed provider error.
- Network issues: retry later and confirm your connection.
- Unsupported model name: switch to a public Groq model such as `llama-3.1-8b-instant`.

## Troubleshooting

- Verify the `.env` file exists and is in the project root.
- Confirm the `LLM_PROVIDER` value is `groq`.
- Re-run the health check after updating `.env`.
