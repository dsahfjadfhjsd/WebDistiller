"""
WebDistiller Memory Module

Implements the hierarchical memory architecture from the paper:
- Factual Memory (M^F): Evidence-grounded candidates with provenance
- Procedural Memory (M^P): Decision traces and reasoning trajectory
- Experiential Memory (M^E): Meta-level heuristics for action biasing

Also provides:
- Intent-Guided Memory Folding operator (Φ)
- Structured Extraction Intent synthesis
"""

from .memory_manager import (
    MemoryManager,
    Evidence,
    FactualCandidate,
    FactualEntry,
    ProceduralNode,
    ExperientialHeuristic,
    norm_entity,
    norm_attr,
    norm_value,
    validate_evidence,
)

from .memory_folding import (
    fold_memory,
    generate_factual_memory,
    generate_procedural_memory,
    generate_experiential_memory,
    generate_strategy_reflection,
)

from .extraction_intent import (
    ExtractionIntent,
    DistilledObservation,
    synthesize_extraction_intent,
    synthesize_intent_with_llm,
)

__all__ = [
    'MemoryManager',

    'Evidence',
    'FactualCandidate',
    'FactualEntry',

    'ProceduralNode',

    'ExperientialHeuristic',

    'norm_entity',
    'norm_attr',
    'norm_value',
    'validate_evidence',

    'fold_memory',
    'generate_factual_memory',
    'generate_procedural_memory',
    'generate_experiential_memory',
    'generate_strategy_reflection',

    'ExtractionIntent',
    'DistilledObservation',
    'synthesize_extraction_intent',
    'synthesize_intent_with_llm',
]
