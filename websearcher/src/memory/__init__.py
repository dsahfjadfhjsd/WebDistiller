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
    # Structured data types (Paper Section 3.4)
    Evidence,
    FactualCandidate,
    FactualEntry,
    ProceduralNode,
    ExperientialHeuristic,
    # Normalization functions
    norm_entity,
    norm_attr,
    norm_value,
    validate_evidence,
)

from .memory_folding import (
    # Main folding operator
    fold_memory,
    # Memory generation functions (paper-aligned names)
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
    # Core manager
    'MemoryManager',

    # Structured data types (M^F)
    'Evidence',
    'FactualCandidate',
    'FactualEntry',

    # Structured data types (M^P)
    'ProceduralNode',

    # Structured data types (M^E)
    'ExperientialHeuristic',

    # Normalization functions
    'norm_entity',
    'norm_attr',
    'norm_value',
    'validate_evidence',

    # Memory folding
    'fold_memory',
    'generate_factual_memory',
    'generate_procedural_memory',
    'generate_experiential_memory',
    'generate_strategy_reflection',

    # Extraction intent
    'ExtractionIntent',
    'DistilledObservation',
    'synthesize_extraction_intent',
    'synthesize_intent_with_llm',
]
