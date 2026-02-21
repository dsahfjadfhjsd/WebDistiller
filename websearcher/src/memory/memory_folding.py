"""
WebDistiller Memory Folding

Implements Intent-Guided Memory Folding (Paper Section 3.4):
- Factual Memory (M^F): Evidence-grounded candidates with provenance
- Procedural Memory (M^P): Decision traces and reasoning trajectory
- Experiential Memory (M^E): Meta-level heuristics for action biasing

The folding operator Φ projects linear interaction history into structured
memory subspaces with explicit integration rules.
"""

import asyncio
import json
import re
from typing import Dict, List, Tuple, Any, Optional
from openai import AsyncOpenAI

# Compiled regex patterns for performance
_JSON_PATTERN = re.compile(r'```json\s*(.*?)\s*```', re.DOTALL)
_FALLBACK_JSON_PATTERN = re.compile(r'\{[\s\S]*\}')


def extract_json_from_response(response: str) -> str:
    """Extract JSON content from LLM response."""
    if not response:
        return "{}"

    # Try markdown code block first
    match = _JSON_PATTERN.search(response)
    if match:
        return match.group(1).strip()

    # Fallback to raw JSON detection
    match = _FALLBACK_JSON_PATTERN.search(response)
    if match:
        return match.group(0).strip()

    return response.strip()


# =============================================================================
# Prompt Templates for Memory Generation (Paper-Aligned Terminology)
# =============================================================================

def get_factual_memory_instruction(
    question: str,
    prev_reasoning: str,
    available_tools: str = "",
    extraction_intents: str = ""
) -> str:
    """Generate prompt for Factual Memory (M^F) extraction.

    Corresponds to paper's factual memory: evidence-grounded candidates with provenance.
    """
    intent_section = ""
    if extraction_intents:
        intent_section = f"""
## Accumulated Extraction Intents
{extraction_intents}

Use these intents to prioritize which facts to extract and preserve.
"""
    return f"""You are a memory compression assistant. Extract factual evidence from the reasoning process.

## Question
{question}
{intent_section}
## Reasoning History
{prev_reasoning}

## Task
Create a structured factual memory capturing:
1. Key facts discovered (entities, attributes, values)
2. Evidence sources (URLs, domains) for each fact
3. Confidence indicators based on source agreement

## Output Format (JSON only, no other text)
```json
{{
  "task_description": "What the agent has been investigating",
  "facts_discovered": [
    {{
      "entity": "Entity name",
      "attribute": "Attribute being measured",
      "value": "Discovered value",
      "sources": ["URL or domain that provided this"],
      "confidence": "high/medium/low based on source agreement"
    }}
  ],
  "key_evidence": ["Direct quotes or snippets supporting the facts"],
  "current_progress": "What's been verified and what remains uncertain"
}}
```

Output ONLY valid JSON."""


def get_procedural_memory_instruction(
    question: str,
    prev_reasoning: str,
    available_tools: str = "",
    extraction_intents: str = ""
) -> str:
    """Generate prompt for Procedural Memory (M^P) extraction.

    Corresponds to paper's procedural memory: decision traces and reasoning trajectory.
    """
    intent_section = ""
    if extraction_intents:
        intent_section = f"""
## Accumulated Extraction Intents
{extraction_intents}

Use these intents to understand the decision trajectory and subgoal progression.
"""
    return f"""You are a procedural memory manager. Create a snapshot of the agent's decision trajectory.

## Question
{question}
{intent_section}
## Reasoning History
{prev_reasoning}

## Task
Extract the decision trace:
1. What subgoals have been pursued
2. What actions were taken and their outcomes
3. What the current reasoning state is
4. What concrete steps should come NEXT

## Output Format (JSON only, no other text)
```json
{{
  "current_subgoal": "Current specific goal being pursued",
  "decision_trace": [
    {{"subgoal": "Goal", "action": "Action taken", "outcome": "Result"}}
  ],
  "information_gaps": ["What facts/data are still needed"],
  "current_challenges": "Main obstacles or uncertainties",
  "next_actions": [
    {{"type": "tool_call", "description": "Specific action to take"}}
  ]
}}
```

Output ONLY valid JSON."""


def get_experiential_memory_instruction(
    question: str,
    prev_reasoning: str,
    tool_call_history: str,
    available_tools: str = "",
    extraction_intents: str = ""
) -> str:
    """Generate prompt for Experiential Memory (M^E) extraction.

    Corresponds to paper's experiential memory: meta-level heuristics that bias
    actions but never adjudicate factual truth.
    """
    intent_section = ""
    if extraction_intents:
        intent_section = f"""
## Accumulated Extraction Intents
{extraction_intents}

Use these intents to assess which tools and parameters were effective for each intent type.
"""
    return f"""You are an experiential memory recorder. Analyze tool usage patterns and derive heuristics.

## Question
{question}
{intent_section}
## Tool Call History
{tool_call_history}

## Task
Extract experiential heuristics:
1. Which tools worked well for which intent types
2. What parameter combinations were effective
3. Which domains/sources were reliable vs unreliable
4. Rules for future action selection (NOT for fact verification)

## Output Format (JSON only, no other text)
```json
{{
  "tool_effectiveness": [
    {{
      "tool_name": "tool_name",
      "intent_family": "search/extraction/computation/verification",
      "success_rate": 0.85,
      "effective_parameters": ["param patterns that worked"],
      "experience": "What worked and what didn't"
    }}
  ],
  "source_reliability": [
    {{"domain": "domain.com", "reliability": "high/medium/low", "notes": "why"}}
  ],
  "derived_heuristics": [
    "Actionable rule for future action selection (NOT fact verification)"
  ]
}}
```

Output ONLY valid JSON."""


# =============================================================================
# LLM Response Generation
# =============================================================================

async def generate_response(
    client: AsyncOpenAI,
    prompt: str,
    model_name: str,
    temperature: float = 0.3,
    max_tokens: int = 2048,
    max_retries: int = 2
) -> str:
    """Generate LLM response with retry logic."""
    for attempt in range(max_retries + 1):
        try:
            response = await client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens
            )
            message = response.choices[0].message
            content = message.content or ""

            # Handle DeepSeek-R1 format: reasoning_content + content
            reasoning_content = None

            # Try model_extra first (pydantic v2)
            if hasattr(message, 'model_extra') and message.model_extra:
                reasoning_content = message.model_extra.get('reasoning_content')

            # Fallback: direct attribute
            if not reasoning_content:
                reasoning_content = getattr(message, 'reasoning_content', None)

            if reasoning_content:
                if content:
                    return f"{reasoning_content}\n\n{content}"
                else:
                    return reasoning_content

            return content
        except Exception as e:
            if attempt == max_retries:
                print(f"Error generating memory response after {max_retries + 1} attempts: {e}")
                return ""
            await asyncio.sleep(0.5)
    return ""


# =============================================================================
# Memory Generation Functions (Paper-Aligned)
# =============================================================================

async def generate_factual_memory(
    client: AsyncOpenAI,
    model_name: str,
    question: str,
    prev_reasoning: str,
    available_tools: List[Dict] = None,
    temperature: float = 0.3,
    max_tokens: int = 2048,
    extraction_intents: str = ""
) -> str:
    """Generate Factual Memory (M^F) from reasoning history.

    Extracts evidence-grounded candidates with provenance.
    """
    tools_str = json.dumps(available_tools or [], ensure_ascii=False)
    instruction = get_factual_memory_instruction(
        question, prev_reasoning, tools_str, extraction_intents
    )

    response = await generate_response(
        client=client,
        prompt=instruction,
        model_name=model_name,
        temperature=temperature,
        max_tokens=max_tokens
    )

    return extract_json_from_response(response)


async def generate_procedural_memory(
    client: AsyncOpenAI,
    model_name: str,
    question: str,
    prev_reasoning: str,
    available_tools: List[Dict] = None,
    temperature: float = 0.3,
    max_tokens: int = 2048,
    extraction_intents: str = ""
) -> str:
    """Generate Procedural Memory (M^P) from reasoning history.

    Extracts decision traces and reasoning trajectory.
    """
    tools_str = json.dumps(available_tools or [], ensure_ascii=False)
    instruction = get_procedural_memory_instruction(
        question, prev_reasoning, tools_str, extraction_intents
    )

    response = await generate_response(
        client=client,
        prompt=instruction,
        model_name=model_name,
        temperature=temperature,
        max_tokens=max_tokens
    )

    return extract_json_from_response(response)


async def generate_experiential_memory(
    client: AsyncOpenAI,
    model_name: str,
    question: str,
    prev_reasoning: str,
    tool_call_history: List[Dict],
    available_tools: List[Dict] = None,
    temperature: float = 0.3,
    max_tokens: int = 2048,
    extraction_intents: str = ""
) -> str:
    """Generate Experiential Memory (M^E) from tool usage history.

    Extracts meta-level heuristics for action biasing.
    """
    tools_str = json.dumps(available_tools or [], ensure_ascii=False)
    history_str = json.dumps(tool_call_history, ensure_ascii=False)
    instruction = get_experiential_memory_instruction(
        question, prev_reasoning, history_str, tools_str, extraction_intents
    )

    response = await generate_response(
        client=client,
        prompt=instruction,
        model_name=model_name,
        temperature=temperature,
        max_tokens=max_tokens
    )

    return extract_json_from_response(response)


async def generate_strategy_reflection(
    client: AsyncOpenAI,
    model_name: str,
    question: str,
    factual_memory: str,
    procedural_memory: str,
    experiential_memory: str,
    temperature: float = 0.3,
    max_tokens: int = 512
) -> str:
    """Generate strategy reflection bridging the three memory layers.

    This provides a coherent summary connecting factual evidence,
    procedural state, and experiential heuristics.
    """
    prompt = f"""Based on the memory summaries below, provide a brief strategy reflection.

Question: {question}

Factual Memory (M^F): {factual_memory}
Procedural Memory (M^P): {procedural_memory}
Experiential Memory (M^E): {experiential_memory}

Reflect on:
1. What approach has been working?
2. What information is still missing to answer the question?
3. What should be tried next?

Keep response concise (2-3 sentences)."""

    response = await generate_response(
        client=client,
        prompt=prompt,
        model_name=model_name,
        temperature=temperature,
        max_tokens=max_tokens
    )

    return response.strip()


# =============================================================================
# Main Folding Operator (Paper Algorithm 2)
# =============================================================================

async def fold_memory(
    client: AsyncOpenAI,
    model_name: str,
    question: str,
    current_output: str,
    tool_call_history: List[Dict],
    available_tools: List[Dict] = None,
    temperature: float = 0.3,
    max_tokens: int = 2048,
    enable_reflection: bool = True,
    enable_factual: bool = True,
    enable_procedural: bool = True,
    enable_experiential: bool = True,
    extraction_intents: Optional[List[Dict]] = None
) -> Tuple[str, str, str, Optional[str]]:
    """Intent-Guided Memory Folding Operator Φ.

    Projects linear interaction history into structured memory subspaces:
    - M^F (Factual): Evidence-grounded candidates with provenance
    - M^P (Procedural): Decision traces
    - M^E (Experiential): Meta-level heuristics

    Implements Φ(H, M | i_{1:t}) — folding conditioned on accumulated intents.

    Args:
        client: AsyncOpenAI client
        model_name: Model to use for generation
        question: Original user question
        current_output: Current reasoning output to fold
        tool_call_history: History of tool calls
        available_tools: Available tool definitions
        temperature: Generation temperature
        max_tokens: Max tokens per generation
        enable_reflection: Whether to generate strategy reflection
        enable_factual: Whether to generate factual memory
        enable_procedural: Whether to generate procedural memory
        enable_experiential: Whether to generate experiential memory
        extraction_intents: List of accumulated extraction intent dicts

    Returns:
        Tuple of (factual_memory, procedural_memory, experiential_memory, strategy_reflection)
    """
    print("Executing Intent-Guided Memory Folding (Φ)...")

    # Truncate reasoning if too long
    prev_reasoning = current_output[:30000] if len(current_output) > 30000 else current_output

    # Format accumulated intents for prompt conditioning
    intents_str = ""
    if extraction_intents:
        intent_parts = []
        for i, intent in enumerate(extraction_intents[-10:], 1):  # Keep last 10
            intent_parts.append(
                f"{i}. Target: {intent.get('target', 'N/A')} | "
                f"Family: {intent.get('intent_family', 'general')} | "
                f"Subgoal: {intent.get('subgoal', 'N/A')}"
            )
        intents_str = "\n".join(intent_parts)

    # Helper for disabled layers
    async def _empty_result():
        return ""

    # Generate three memory layers in parallel
    factual_task = (
        generate_factual_memory(
            client, model_name, question, prev_reasoning,
            available_tools, temperature, max_tokens, intents_str
        ) if enable_factual else _empty_result()
    )
    procedural_task = (
        generate_procedural_memory(
            client, model_name, question, prev_reasoning,
            available_tools, temperature, max_tokens, intents_str
        ) if enable_procedural else _empty_result()
    )
    experiential_task = (
        generate_experiential_memory(
            client, model_name, question, prev_reasoning,
            tool_call_history, available_tools, temperature, max_tokens, intents_str
        ) if enable_experiential else _empty_result()
    )

    factual_memory, procedural_memory, experiential_memory = await asyncio.gather(
        factual_task, procedural_task, experiential_task
    )

    # Generate strategy reflection to bridge layers
    strategy_reflection = None
    if enable_reflection and (factual_memory or procedural_memory or experiential_memory):
        strategy_reflection = await generate_strategy_reflection(
            client=client,
            model_name=model_name,
            question=question,
            factual_memory=factual_memory,
            procedural_memory=procedural_memory,
            experiential_memory=experiential_memory,
            temperature=temperature,
            max_tokens=512
        )

    print("Memory folding completed (M^F, M^P, M^E)")
    return factual_memory, procedural_memory, experiential_memory, strategy_reflection
