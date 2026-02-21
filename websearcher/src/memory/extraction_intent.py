"""
WebDistiller Extraction Intent

Implements the structured Extraction Intent (Paper Section 3.3):
    i_t = ⟨target, constraints, output⟩

The extraction intent guides Cognitive Distillation by specifying:
- target: What information to extract
- constraints: What to ignore or filter out
- output: Expected output format/structure
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
import json
import re


@dataclass
class ExtractionIntent:
    """Structured Extraction Intent for Cognitive Distillation.

    Corresponds to paper's i_t = ⟨target, constraints, output⟩

    Attributes:
        target: What specific information to extract (e.g., "GDP growth rate")
        constraints: What to ignore (e.g., "advertisements, navigation elements")
        output_format: Expected output structure (e.g., "numerical value with unit")
        intent_family: Category of intent (search/extraction/computation/verification)
        subgoal: The current subgoal this intent serves
    """
    target: str
    constraints: str
    output_format: str
    intent_family: str = "general"
    subgoal: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "target": self.target,
            "constraints": self.constraints,
            "output_format": self.output_format,
            "intent_family": self.intent_family,
            "subgoal": self.subgoal
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ExtractionIntent':
        return cls(
            target=data.get("target", ""),
            constraints=data.get("constraints", ""),
            output_format=data.get("output_format", ""),
            intent_family=data.get("intent_family", "general"),
            subgoal=data.get("subgoal", "")
        )

    def to_prompt_string(self) -> str:
        """Convert intent to a string suitable for prompts."""
        parts = [f"Target: {self.target}"]
        if self.constraints:
            parts.append(f"Ignore: {self.constraints}")
        if self.output_format:
            parts.append(f"Format: {self.output_format}")
        return "\n".join(parts)

    def to_distillation_prompt(self) -> str:
        """Generate a distillation conditioning prompt for Cognitive Distillation.

        This produces the intent-conditioned instructions that guide the Manager
        model (G_φ) to distill raw observations into structured output per the
        extraction intent i_t = ⟨target, constraints, output⟩.
        """
        prompt = (
            f"## Extraction Intent\n"
            f"Target information: {self.target}\n"
            f"Filter out: {self.constraints}\n"
            f"Expected output format: {self.output_format}\n"
        )
        if self.subgoal:
            prompt += f"Current subgoal: {self.subgoal}\n"
        prompt += (
            f"\n## Distillation Instructions\n"
            f"1. Extract ONLY information matching the target above\n"
            f"2. Discard content matching the constraints (noise)\n"
            f"3. Structure output according to the expected format\n"
            f"4. Preserve exact values, dates, names, and numerical data\n"
            f"5. Include source attribution (URL/domain) for each extracted fact\n"
            f"6. If a factual candidate is found, output it as:\n"
            f"   CANDIDATE: entity={{...}} attr={{...}} value={{...}} source={{...}}\n"
        )
        return prompt

    @classmethod
    def from_natural_language(cls, intent_text: str, subgoal: str = "") -> 'ExtractionIntent':
        """Parse a natural language intent into structured form."""
        # Default values
        target = intent_text
        constraints = "advertisements, navigation elements, boilerplate"
        output_format = "relevant facts and evidence"

        # Try to extract structured components
        target_match = re.search(r'(?:looking for|find|extract|get)\s+(.+?)(?:\.|,|$)', intent_text, re.I)
        if target_match:
            target = target_match.group(1).strip()

        # Infer intent family
        intent_lower = intent_text.lower()
        if any(kw in intent_lower for kw in ['search', 'find', 'look for', 'query']):
            intent_family = 'search'
        elif any(kw in intent_lower for kw in ['read', 'extract', 'get', 'fetch']):
            intent_family = 'extraction'
        elif any(kw in intent_lower for kw in ['calculate', 'compute', 'solve']):
            intent_family = 'computation'
        elif any(kw in intent_lower for kw in ['verify', 'check', 'confirm']):
            intent_family = 'verification'
        else:
            intent_family = 'general'

        return cls(
            target=target,
            constraints=constraints,
            output_format=output_format,
            intent_family=intent_family,
            subgoal=subgoal
        )


@dataclass
class DistilledObservation:
    """Result of Cognitive Distillation applied to raw observation.

    Corresponds to paper's o_t = {snippets_t, cands_t}
    """
    snippets: List[Dict[str, str]] = field(default_factory=list)  # (url, domain, snippet, ts)
    candidates: List[Dict[str, Any]] = field(default_factory=list)  # (entity, attr, value, evidence)
    compression_ratio: float = 1.0
    original_tokens: int = 0
    distilled_tokens: int = 0

    def add_snippet(self, url: str, domain: str, snippet: str, timestamp: str = "") -> None:
        """Add an evidence snippet."""
        self.snippets.append({
            "url": url,
            "domain": domain,
            "snippet": snippet,
            "ts": timestamp
        })

    def add_candidate(
        self,
        entity: str,
        attr: str,
        value: str,
        url: str,
        domain: str,
        snippet: str
    ) -> None:
        """Add a factual candidate with evidence."""
        self.candidates.append({
            "entity": entity,
            "attr": attr,
            "value": value,
            "evidence": {
                "url": url,
                "domain": domain,
                "snippet": snippet
            }
        })

    def to_dict(self) -> Dict[str, Any]:
        return {
            "snippets": self.snippets,
            "candidates": self.candidates,
            "compression_ratio": self.compression_ratio,
            "original_tokens": self.original_tokens,
            "distilled_tokens": self.distilled_tokens
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DistilledObservation':
        obs = cls(
            compression_ratio=data.get("compression_ratio", 1.0),
            original_tokens=data.get("original_tokens", 0),
            distilled_tokens=data.get("distilled_tokens", 0)
        )
        obs.snippets = data.get("snippets", [])
        obs.candidates = data.get("candidates", [])
        return obs

    def to_context_string(self) -> str:
        """Convert to string for inclusion in context."""
        parts = []

        if self.candidates:
            parts.append("Extracted Candidates:")
            for cand in self.candidates:
                parts.append(f"  - {cand['entity']}.{cand['attr']} = {cand['value']}")
                if cand.get('evidence'):
                    parts.append(f"    Source: {cand['evidence'].get('domain', 'unknown')}")

        if self.snippets:
            parts.append("\nEvidence Snippets:")
            for snip in self.snippets[:5]:  # Limit to 5 snippets
                parts.append(f"  [{snip.get('domain', 'unknown')}] {snip.get('snippet', '')[:200]}")

        if self.compression_ratio < 1.0:
            parts.append(f"\n[Compressed {self.compression_ratio:.1%} of original content]")

        return "\n".join(parts)


def synthesize_extraction_intent(
    query: str,
    subgoal: str,
    action_type: str,
    action_args: Dict[str, Any]
) -> ExtractionIntent:
    """Synthesize a structured extraction intent from action context.

    This implements the Manager's intent synthesis function:
    i_t = G_φ^Intent(q, τ_t, a_t)

    Args:
        query: Original user query
        subgoal: Current subgoal (τ_t)
        action_type: Type of action being taken
        action_args: Arguments of the action

    Returns:
        Structured ExtractionIntent
    """
    # Determine intent family from action type
    if action_type in ['web_search']:
        intent_family = 'search'
        target = action_args.get('query', subgoal)
        constraints = "advertisements, sponsored content, unrelated results"
        output_format = "search results with titles, URLs, and relevant snippets"

    elif action_type in ['click_link', 'browse_pages']:
        intent_family = 'extraction'
        goal = action_args.get('goal', action_args.get('query', subgoal))
        target = goal if goal else f"information relevant to: {subgoal}"
        constraints = "navigation menus, headers, footers, advertisements, cookie notices"
        output_format = "key facts, numbers, dates, names with source attribution"

    elif action_type in ['execute_python_code', 'calculator', 'solve_math']:
        intent_family = 'computation'
        target = "computation result"
        constraints = "intermediate steps, debug output"
        output_format = "final computed value with units if applicable"

    elif action_type in ['download_file', 'process_file']:
        intent_family = 'extraction'
        target = f"content from file relevant to: {subgoal}"
        constraints = "formatting artifacts, page numbers, headers/footers"
        output_format = "extracted text, tables, or data relevant to query"

    else:
        intent_family = 'general'
        target = subgoal
        constraints = "irrelevant content"
        output_format = "relevant information"

    return ExtractionIntent(
        target=target,
        constraints=constraints,
        output_format=output_format,
        intent_family=intent_family,
        subgoal=subgoal
    )


# Prompt template for LLM-based intent synthesis
INTENT_SYNTHESIS_PROMPT = """Based on the reasoning context, synthesize a structured extraction intent.

## Query
{query}

## Current Subgoal
{subgoal}

## Action Being Taken
Type: {action_type}
Arguments: {action_args}

## Recent Reasoning Context
{reasoning_context}

## Task
Create a structured extraction intent specifying:
1. TARGET: What specific information to extract
2. CONSTRAINTS: What to ignore or filter out
3. OUTPUT: Expected format of extracted information

## Output Format (JSON only)
```json
{{
  "target": "specific information to extract",
  "constraints": "what to ignore",
  "output_format": "expected output structure",
  "intent_family": "search|extraction|computation|verification|general"
}}
```

Output ONLY valid JSON."""


async def synthesize_intent_with_llm(
    client,
    model_name: str,
    query: str,
    subgoal: str,
    action_type: str,
    action_args: Dict[str, Any],
    reasoning_context: str = ""
) -> ExtractionIntent:
    """Synthesize extraction intent using Manager LLM (G_φ^Intent).

    Implements i_t ~ G_φ^Intent(· | q, τ_t, a_t) with reasoning context.
    Falls back to rule-based synthesis if LLM fails.
    """
    try:
        prompt = INTENT_SYNTHESIS_PROMPT.format(
            query=query,
            subgoal=subgoal,
            action_type=action_type,
            action_args=json.dumps(action_args, ensure_ascii=False),
            reasoning_context=reasoning_context[-2000:] if reasoning_context else "N/A"
        )

        response = await client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=300
        )

        content = response.choices[0].message.content or ""

        # Extract JSON
        json_match = re.search(r'```json\s*(.*?)\s*```', content, re.DOTALL)
        if json_match:
            data = json.loads(json_match.group(1))
        else:
            data = json.loads(content)

        return ExtractionIntent(
            target=data.get("target", subgoal),
            constraints=data.get("constraints", ""),
            output_format=data.get("output_format", ""),
            intent_family=data.get("intent_family", "general"),
            subgoal=subgoal
        )

    except Exception as e:
        # Fallback to rule-based synthesis
        return synthesize_extraction_intent(query, subgoal, action_type, action_args)
