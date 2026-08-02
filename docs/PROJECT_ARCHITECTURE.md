# Project Architecture

SafeLite is a research-oriented robotics planning stack organized around a modular pipeline:

1. Simulator
2. Planner
3. Safety Guard
4. Execution Engine
5. Self-Correction

The project now includes a provider-based LLM planning layer that allows the planner to switch between mock, Groq, OpenRouter, and Hugging Face implementations without changing the core planner agent.

## Planner

The Planner module converts natural-language instructions into structured action plans.
It now uses the provider abstraction in the `llm` package so that the planner agent depends on configuration rather than a single concrete backend.

## Safety

The Safety module validates action plans before execution and ensures they satisfy hard constraints.

## Executor

The Executor module carries out validated plans step by step and records execution outcomes.

## Simulator

The Simulator module provides a controlled environment model for experimentation and analysis of system behavior.

## Self-Correction

The Self-Correction module analyzes failed executions and proposes recovery actions or safe termination.

## LLM provider architecture

The LLM package contains the following modules:

- `llm/base.py`: abstract provider interface
- `llm/config.py`: environment-based configuration loading
- `llm/factory.py`: provider selection by configuration
- `llm/mock_provider.py`: offline deterministic provider
- `llm/groq_provider.py`: Groq chat completions provider
- `llm/openrouter_provider.py`: OpenRouter provider
- `llm/huggingface_provider.py`: Hugging Face provider
- `llm/prompt_builder.py`: prompt construction helpers
- `llm/response_parser.py`: JSON-to-ActionPlan parsing

## Provider selection

The planner uses `load_config()` to read `.env` settings and `create_provider()` to instantiate the selected provider. The selection logic is centralized in the factory so the planner remains stable while providers evolve.

## Configuration

The project expects the following environment variables:

- `LLM_PROVIDER`
- `GROQ_API_KEY`
- `OPENROUTER_API_KEY`
- `HF_API_KEY`
- `MODEL_NAME`
- `TEMPERATURE`
- `MAX_TOKENS`
- `TIMEOUT`

A template is provided in `.env.example`.

## Free usage

The stack can be used completely free with the mock provider or with free-tier access to Groq, OpenRouter, or Hugging Face.
