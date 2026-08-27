"""Open-ended verbalized assumptions over persona-conditioned dilemmas.

The probe is from Cheng et al., *Verbalizing LLMs' assumptions to explain and
control sycophancy* (vendored under `verbalizedassumptions/`). The model layer
is adapted from myazan/LLM-Self-Concept.

    syco.data            personas, dilemmas, prior answers
    syco.prompts         the open-ended probe, in two history modes
    syco.grid            the design and how a run subsets it
    syco.model_registry  aliases, backend routing, quantization
    syco.models          chat adapters: hf, llamacpp, openai, anthropic, mock
    syco.store           append-only JSONL results and resume
    syco.parse           raw completion -> mental models + reply
    syco.tables          reading a parsed table; shared analysis conventions
    syco.topics          n-grams and BERTopic over the assumption text
    syco.sycophancy      forced-choice sycophancy, joined to assumptions
    syco.text_analysis   text features for personas and model responses
    syco.experiments     validated experiment profiles
    syco.manifest        immutable run identity and provenance
    syco.orchestrate     multi-model scheduling and pipeline operations
"""
__all__ = [
    "data",
    "experiments",
    "grid",
    "manifest",
    "model_registry",
    "models",
    "orchestrate",
    "parse",
    "prompts",
    "store",
    "sycophancy",
    "tables",
    "text_analysis",
    "topics",
]
