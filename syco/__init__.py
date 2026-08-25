"""Open-ended verbalized assumptions over persona-conditioned dilemmas.

The probe is from Chen et al., *Verbalizing LLMs' assumptions to explain and
control sycophancy* (vendored under `verbalizedassumptions/`). The model layer
is adapted from myazan/LLM-Self-Concept.

    syco.data            personas, dilemmas, prior answers
    syco.prompts         the open-ended probe, in two history modes
    syco.grid            the design and how a run subsets it
    syco.model_registry  aliases, backend routing, quantization
    syco.models          chat adapters: hf, llamacpp, openai, anthropic, mock
    syco.store           append-only JSONL results and resume
    syco.parse           raw completion -> mental models + reply
"""
__all__ = ["data", "prompts", "grid", "model_registry", "models", "store", "parse"]
