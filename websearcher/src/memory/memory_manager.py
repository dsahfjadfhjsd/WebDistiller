"""
WebDistiller Memory Manager

Implements the hierarchical memory architecture described in the paper:
- Factual Memory (M^F): Evidence-grounded candidates with provenance
- Procedural Memory (M^P): Decision traces and reasoning trajectory
- Experiential Memory (M^E): Meta-level heuristics for action biasing

This module provides the MemoryManager class that maintains structured memory
state and supports Intent-Guided Memory Folding operations.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple, Set
import re
from datetime import datetime


# =============================================================================
# Data Structures for Structured Memory (Paper Section 3.4)
# =============================================================================

@dataclass
class Evidence:
    """Evidence record with provenance information.

    Corresponds to: evidence = (url, domain, snippet, ts)
    """
    ev_id: str
    url: str
    domain: str
    snippet: str
    timestamp: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ev_id": self.ev_id,
            "url": self.url,
            "domain": self.domain,
            "snippet": self.snippet,
            "timestamp": self.timestamp
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Evidence':
        return cls(
            ev_id=data.get("ev_id", ""),
            url=data.get("url", ""),
            domain=data.get("domain", ""),
            snippet=data.get("snippet", ""),
            timestamp=data.get("timestamp", "")
        )


@dataclass
class FactualCandidate:
    """A candidate value with support count and evidence IDs.

    Corresponds to: (value, support, ev_ids, first_ts, last_ts)
    """
    value: str
    support: int  # Number of unique domains supporting this value
    evidence_ids: List[str]
    first_ts: str
    last_ts: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "value": self.value,
            "support": self.support,
            "evidence_ids": self.evidence_ids,
            "first_ts": self.first_ts,
            "last_ts": self.last_ts
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'FactualCandidate':
        return cls(
            value=data.get("value", ""),
            support=data.get("support", 0),
            evidence_ids=data.get("evidence_ids", []),
            first_ts=data.get("first_ts", ""),
            last_ts=data.get("last_ts", "")
        )


@dataclass
class FactualEntry:
    """Factual Memory entry for a (entity_id, attr) key.

    Corresponds to paper's M^F structure:
    - values: bounded candidate list [(value, support, ev_ids, first_ts, last_ts)]_≤k
    - evidence: ev_id → (url, domain, snippet, ts)
    """
    entity_id: str
    attr: str
    candidates: List[FactualCandidate] = field(default_factory=list)
    evidence: Dict[str, Evidence] = field(default_factory=dict)
    max_candidates: int = 2  # k=2 as per paper

    def get_current_value(self) -> Optional[str]:
        """Returns top-1 value as current value."""
        if self.candidates:
            return self.candidates[0].value
        return None

    def add_candidate(self, value: str, evidence: Evidence) -> None:
        """Add or update a candidate with new evidence.

        Implements the rule-based conflict resolution:
        1. De-duplicate evidence by (domain, snippet)
        2. Update timestamps
        3. Rank by support↓, last_ts↓, lexical↑
        4. Keep top-k
        """
        # Store evidence
        self.evidence[evidence.ev_id] = evidence

        # Find existing candidate with same value
        existing = None
        for cand in self.candidates:
            if cand.value == value:
                existing = cand
                break

        if existing:
            # Update existing candidate
            if evidence.ev_id not in existing.evidence_ids:
                existing.evidence_ids.append(evidence.ev_id)
            existing.last_ts = evidence.timestamp
            # Recalculate support (unique domains)
            domains = set()
            for ev_id in existing.evidence_ids:
                if ev_id in self.evidence:
                    domains.add(self.evidence[ev_id].domain)
            existing.support = len(domains)
        else:
            # Create new candidate
            new_cand = FactualCandidate(
                value=value,
                support=1,
                evidence_ids=[evidence.ev_id],
                first_ts=evidence.timestamp,
                last_ts=evidence.timestamp
            )
            self.candidates.append(new_cand)

        # Sort and keep top-k
        self._rank_and_prune()

    def _rank_and_prune(self) -> None:
        """Rank candidates by support↓, last_ts↓, lexical↑ and keep top-k."""
        self.candidates.sort(
            key=lambda c: (-c.support, c.last_ts or "", c.value),
            reverse=False
        )
        # Re-sort with proper ordering
        self.candidates.sort(
            key=lambda c: (-c.support, -(hash(c.last_ts) if c.last_ts else 0), c.value)
        )
        self.candidates = self.candidates[:self.max_candidates]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "entity_id": self.entity_id,
            "attr": self.attr,
            "candidates": [c.to_dict() for c in self.candidates],
            "evidence": {k: v.to_dict() for k, v in self.evidence.items()},
            "max_candidates": self.max_candidates
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'FactualEntry':
        entry = cls(
            entity_id=data.get("entity_id", ""),
            attr=data.get("attr", ""),
            max_candidates=data.get("max_candidates", 2)
        )
        entry.candidates = [
            FactualCandidate.from_dict(c)
            for c in data.get("candidates", [])
        ]
        entry.evidence = {
            k: Evidence.from_dict(v)
            for k, v in data.get("evidence", {}).items()
        }
        return entry


@dataclass
class ProceduralNode:
    """Procedural Memory node for decision trace.

    Corresponds to paper's M^P structure:
    node = (tau, intent_family, action_type, outcome, produced_fact_keys, ts)
    """
    node_id: str
    subgoal: str  # tau
    intent_family: str
    action_type: str
    outcome: str
    produced_fact_keys: List[str]  # Links to M^F keys
    timestamp: str
    is_failure: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "node_id": self.node_id,
            "subgoal": self.subgoal,
            "intent_family": self.intent_family,
            "action_type": self.action_type,
            "outcome": self.outcome,
            "produced_fact_keys": self.produced_fact_keys,
            "timestamp": self.timestamp,
            "is_failure": self.is_failure
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ProceduralNode':
        return cls(
            node_id=data.get("node_id", ""),
            subgoal=data.get("subgoal", ""),
            intent_family=data.get("intent_family", ""),
            action_type=data.get("action_type", ""),
            outcome=data.get("outcome", ""),
            produced_fact_keys=data.get("produced_fact_keys", []),
            timestamp=data.get("timestamp", ""),
            is_failure=data.get("is_failure", False)
        )


@dataclass
class ExperientialHeuristic:
    """Experiential Memory heuristic entry.

    Corresponds to paper's M^E: biases actions but never adjudicates factual truth.
    """
    intent_family: str
    preferred_tools: List[str] = field(default_factory=list)
    blacklist_domains: List[str] = field(default_factory=list)
    effective_patterns: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "intent_family": self.intent_family,
            "preferred_tools": self.preferred_tools,
            "blacklist_domains": self.blacklist_domains,
            "effective_patterns": self.effective_patterns
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ExperientialHeuristic':
        return cls(
            intent_family=data.get("intent_family", ""),
            preferred_tools=data.get("preferred_tools", []),
            blacklist_domains=data.get("blacklist_domains", []),
            effective_patterns=data.get("effective_patterns", [])
        )


# =============================================================================
# Normalization Functions (Paper Section 3.4)
# =============================================================================

def norm_entity(entity: str) -> str:
    """NormE: lowercase, trim, punctuation removal."""
    if not entity:
        return ""
    result = entity.lower().strip()
    result = re.sub(r'[^\w\s]', '', result)
    result = re.sub(r'\s+', ' ', result)
    return result.strip()


def norm_attr(attr: str) -> str:
    """NormA: lowercase + whitespace normalization."""
    if not attr:
        return ""
    result = attr.lower().strip()
    result = re.sub(r'\s+', ' ', result)
    return result


def norm_value(value: str) -> str:
    """NormV: dates → YYYY-MM-DD, numeric separators removed."""
    if not value:
        return ""

    result = value.strip()

    # Try to parse as date
    date_patterns = [
        (r'(\d{1,2})[/-](\d{1,2})[/-](\d{4})', lambda m: f"{m.group(3)}-{m.group(1).zfill(2)}-{m.group(2).zfill(2)}"),
        (r'(\d{4})[/-](\d{1,2})[/-](\d{1,2})', lambda m: f"{m.group(1)}-{m.group(2).zfill(2)}-{m.group(3).zfill(2)}"),
    ]

    for pattern, formatter in date_patterns:
        match = re.match(pattern, result)
        if match:
            try:
                return formatter(match)
            except:
                pass

    # Remove numeric separators for numbers
    if re.match(r'^[\d,.\s]+$', result):
        result = re.sub(r'[,\s]', '', result)

    return result


def validate_evidence(snippet: str, value: str) -> bool:
    """Check if snippet contains value (or regex-match)."""
    if not snippet or not value:
        return False
    return value.lower() in snippet.lower() or bool(re.search(re.escape(value), snippet, re.IGNORECASE))


# =============================================================================
# Memory Manager Class
# =============================================================================

class MemoryManager:
    """
    Hierarchical Memory Manager implementing Intent-Guided Memory Folding.

    Memory Structure (Paper Section 3.4):
    - M^F (Factual Memory): Evidence-grounded candidates with provenance
    - M^P (Procedural Memory): Decision traces
    - M^E (Experiential Memory): Meta-level heuristics

    Also maintains legacy text-based memories for backward compatibility.
    """

    __slots__ = (
        # Structured memory (paper-aligned)
        'factual_memory', 'procedural_memory', 'experiential_memory',
        # Legacy text-based memory
        'factual_memories', 'procedural_memories', 'experiential_memories',
        # State tracking
        'fold_count', 'max_folds', 'interactions', '_tool_call_cache',
        # Counters
        '_evidence_counter', '_node_counter'
    )

    def __init__(self, max_folds: int = 3):
        # Structured memory (paper-aligned naming)
        self.factual_memory: Dict[Tuple[str, str], FactualEntry] = {}  # M^F
        self.procedural_memory: List[ProceduralNode] = []  # M^P
        self.experiential_memory: Dict[str, ExperientialHeuristic] = {}  # M^E

        # Legacy text-based memory (for backward compatibility)
        self.factual_memories: List[str] = []  # Previously episode_memories
        self.procedural_memories: Optional[str] = None  # Previously working_memory
        self.experiential_memories: Optional[str] = None  # Previously tool_memory

        # State tracking
        self.fold_count: int = 0
        self.max_folds: int = max_folds
        self.interactions: List[Dict[str, Any]] = []
        self._tool_call_cache: Optional[List[Dict]] = None

        # Counters for ID generation
        self._evidence_counter: int = 0
        self._node_counter: int = 0

    # =========================================================================
    # Structured Memory Operations (Paper Section 3.4)
    # =========================================================================

    def add_fact(
        self,
        entity: str,
        attr: str,
        value: str,
        url: str,
        domain: str,
        snippet: str
    ) -> Optional[str]:
        """Add a factual candidate with evidence to M^F.

        Returns the fact key if added, None if validation fails.
        """
        # Normalize
        entity_id = norm_entity(entity)
        attr_norm = norm_attr(attr)
        value_norm = norm_value(value)

        if not entity_id or not attr_norm or not value_norm:
            return None

        # Validate evidence
        is_valid = validate_evidence(snippet, value)

        # Create evidence
        self._evidence_counter += 1
        ev_id = f"ev_{self._evidence_counter}"
        timestamp = datetime.now().isoformat()

        evidence = Evidence(
            ev_id=ev_id,
            url=url,
            domain=domain,
            snippet=snippet if is_valid else f"[tentative] {snippet}",
            timestamp=timestamp
        )

        # Get or create entry
        key = (entity_id, attr_norm)
        if key not in self.factual_memory:
            self.factual_memory[key] = FactualEntry(
                entity_id=entity_id,
                attr=attr_norm
            )

        # Add candidate
        self.factual_memory[key].add_candidate(value_norm, evidence)

        return f"{entity_id}:{attr_norm}"

    def add_procedural_node(
        self,
        subgoal: str,
        intent_family: str,
        action_type: str,
        outcome: str,
        produced_fact_keys: List[str] = None,
        is_failure: bool = False,
        max_nodes: int = 50
    ) -> str:
        """Add a procedural node to M^P.

        Only records decision-relevant nodes (those that produced facts or are failures).
        """
        self._node_counter += 1
        node_id = f"node_{self._node_counter}"
        timestamp = datetime.now().isoformat()

        node = ProceduralNode(
            node_id=node_id,
            subgoal=subgoal,
            intent_family=intent_family,
            action_type=action_type,
            outcome=outcome,
            produced_fact_keys=produced_fact_keys or [],
            timestamp=timestamp,
            is_failure=is_failure
        )

        self.procedural_memory.append(node)

        # Keep bounded size
        if len(self.procedural_memory) > max_nodes:
            # Keep most recent nodes and most recent failure per intent_family
            failures_by_family: Dict[str, ProceduralNode] = {}
            for n in self.procedural_memory:
                if n.is_failure:
                    failures_by_family[n.intent_family] = n

            recent = self.procedural_memory[-(max_nodes - len(failures_by_family)):]
            failure_nodes = list(failures_by_family.values())

            # Merge, avoiding duplicates
            seen_ids = set()
            merged = []
            for n in failure_nodes + recent:
                if n.node_id not in seen_ids:
                    merged.append(n)
                    seen_ids.add(n.node_id)

            self.procedural_memory = merged

        return node_id

    def update_heuristic(
        self,
        intent_family: str,
        preferred_tools: List[str] = None,
        blacklist_domains: List[str] = None,
        effective_patterns: List[str] = None
    ) -> None:
        """Update experiential heuristics for an intent family."""
        if intent_family not in self.experiential_memory:
            self.experiential_memory[intent_family] = ExperientialHeuristic(
                intent_family=intent_family
            )

        heuristic = self.experiential_memory[intent_family]

        if preferred_tools:
            for tool in preferred_tools:
                if tool not in heuristic.preferred_tools:
                    heuristic.preferred_tools.append(tool)

        if blacklist_domains:
            for domain in blacklist_domains:
                if domain not in heuristic.blacklist_domains:
                    heuristic.blacklist_domains.append(domain)

        if effective_patterns:
            for pattern in effective_patterns:
                if pattern not in heuristic.effective_patterns:
                    heuristic.effective_patterns.append(pattern)

    # =========================================================================
    # Query Interface (Paper Section 3.4)
    # =========================================================================

    def query_facts(self, subgoal: str, m: int = 10) -> List[FactualEntry]:
        """QueryFacts(τ, M^F, m): Return top-m facts by token overlap with subgoal."""
        if not self.factual_memory:
            return []

        subgoal_tokens = set(subgoal.lower().split())

        scored = []
        for key, entry in self.factual_memory.items():
            # Calculate token overlap
            entry_str = f"{entry.entity_id} {entry.attr} {entry.get_current_value() or ''}"
            entry_tokens = set(entry_str.lower().split())
            overlap = len(subgoal_tokens & entry_tokens)
            scored.append((overlap, entry))

        # Sort by overlap descending
        scored.sort(key=lambda x: -x[0])

        return [entry for _, entry in scored[:m]]

    def query_trace(self, subgoal: str, r: int = 6) -> List[ProceduralNode]:
        """QueryTrace(τ, M^P, r): Return recent r nodes matching intent_family."""
        if not self.procedural_memory:
            return []

        # Infer intent family from subgoal (simple heuristic)
        intent_family = self._infer_intent_family(subgoal)

        # Filter by intent family
        matching = [n for n in self.procedural_memory if n.intent_family == intent_family]

        # Get most recent r
        recent = matching[-r:] if len(matching) > r else matching

        # Also include most recent failure if any
        failures = [n for n in self.procedural_memory if n.is_failure and n.intent_family == intent_family]
        if failures and failures[-1] not in recent:
            recent = [failures[-1]] + recent

        return recent

    def query_heuristics(self, intent_family: str) -> Optional[ExperientialHeuristic]:
        """QueryHeuristics(intent_family, M^E): Return heuristics for action biasing."""
        return self.experiential_memory.get(intent_family)

    def build_context_pack(
        self,
        subgoal: str,
        m: int = 10,
        r: int = 6,
        tail: Optional[List[Dict]] = None,
        h: int = 4
    ) -> str:
        """Construct Bounded Context Pack per paper Eq.4.

        C_t = QueryFacts(τ_t, M^F, m) ⊕ QueryTrace(τ_t, M^P, r) ⊕ Tail(H, h)

        Args:
            subgoal: Current subgoal τ_t for relevance ranking
            m: Number of top facts to retrieve (default 10)
            r: Number of recent trace nodes to retrieve (default 6)
            tail: Recent interaction messages (raw tail from conversation)
            h: Number of tail entries to keep (default 4)

        Returns:
            Formatted context pack string
        """
        parts = []

        # QueryFacts(τ_t, M^F, m)
        facts = self.query_facts(subgoal, m=m)
        if facts:
            parts.append("== Factual Memory (M^F) — QueryFacts ==")
            for entry in facts:
                current = entry.get_current_value()
                if current:
                    support = entry.candidates[0].support if entry.candidates else 0
                    parts.append(f"  {entry.entity_id}.{entry.attr} = {current} (support={support})")
            parts.append("")

        # Also include text-based factual memory if available
        if self.factual_memories:
            parts.append("== Factual Memory (M^F) — Summary ==")
            parts.append(self.factual_memories[-1])
            parts.append("")

        # QueryTrace(τ_t, M^P, r)
        trace = self.query_trace(subgoal, r=r)
        if trace:
            parts.append("== Procedural Trace (M^P) — QueryTrace ==")
            for node in trace:
                status = "[FAIL]" if node.is_failure else "[OK]"
                parts.append(f"  {status} {node.subgoal} -> {node.action_type}: {node.outcome[:100]}")
            parts.append("")

        # Text-based procedural memory
        if self.procedural_memories:
            parts.append("== Procedural Memory (M^P) — Summary ==")
            parts.append(self.procedural_memories)
            parts.append("")

        # Experiential heuristics
        intent_family = self._infer_intent_family(subgoal)
        heuristic = self.query_heuristics(intent_family)
        if heuristic:
            parts.append("== Experiential Memory (M^E) ==")
            if heuristic.preferred_tools:
                parts.append(f"  Preferred tools: {', '.join(heuristic.preferred_tools)}")
            if heuristic.derived_heuristics:
                for h_text in heuristic.derived_heuristics[:3]:
                    parts.append(f"  Heuristic: {h_text}")
            parts.append("")
        elif self.experiential_memories:
            parts.append("== Experiential Memory (M^E) ==")
            parts.append(self.experiential_memories)
            parts.append("")

        # Tail(H, h) — recent interaction tail
        if tail:
            recent_tail = tail[-h:] if len(tail) > h else tail
            parts.append(f"== Recent Context (last {len(recent_tail)} interactions) ==")
            for entry in recent_tail:
                role = entry.get("role", "unknown")
                content = entry.get("content", "")
                if len(content) > 300:
                    content = content[:300] + "..."
                parts.append(f"  [{role}] {content}")
            parts.append("")

        if not parts:
            return ""

        return "\n".join(parts)

    def _infer_intent_family(self, subgoal: str) -> str:
        """Infer intent family from subgoal text."""
        subgoal_lower = subgoal.lower()

        if any(kw in subgoal_lower for kw in ['search', 'find', 'look for', 'query']):
            return 'search'
        elif any(kw in subgoal_lower for kw in ['read', 'extract', 'get', 'fetch']):
            return 'extraction'
        elif any(kw in subgoal_lower for kw in ['calculate', 'compute', 'solve']):
            return 'computation'
        elif any(kw in subgoal_lower for kw in ['verify', 'check', 'confirm']):
            return 'verification'
        else:
            return 'general'

    # =========================================================================
    # Core Interface
    # =========================================================================

    def can_fold(self) -> bool:
        return self.fold_count < self.max_folds

    def add_interaction(self, interaction: Dict[str, Any]):
        self.interactions.append(interaction)
        if interaction.get("type") == "tool_call":
            self._tool_call_cache = None

    def add_fold(
        self,
        factual_memory: str,
        procedural_memory: str,
        experiential_memory: str
    ):
        """Add a memory fold (text-based interface).

        Args:
            factual_memory: Factual memory text (M^F)
            procedural_memory: Procedural memory text (M^P)
            experiential_memory: Experiential memory text (M^E)
        """
        self.factual_memories.append(factual_memory)
        self.procedural_memories = procedural_memory
        self.experiential_memories = experiential_memory
        self.fold_count += 1

        self.interactions.append({
            "type": "memory_folding",
            "fold_number": self.fold_count,
            "factual_memory": factual_memory,
            "procedural_memory": procedural_memory,
            "experiential_memory": experiential_memory,
        })

    def get_memory_summary(self) -> str:
        """Get formatted memory summary for system prompt using paper-aligned terminology."""
        if not self.factual_memories and not self.factual_memory:
            return ""

        parts = ["Memory of previous reasoning:\n"]

        # Structured factual memory (M^F)
        if self.factual_memory:
            parts.append("== Factual Memory (M^F) ==")
            for key, entry in list(self.factual_memory.items())[:10]:
                current = entry.get_current_value()
                if current:
                    support = entry.candidates[0].support if entry.candidates else 0
                    parts.append(f"  {entry.entity_id}.{entry.attr} = {current} (support={support})")
            parts.append("")

        # Text-based factual memory (most recent fold)
        if self.factual_memories:
            parts.append("== Factual Memory (M^F) — Text ==")
            parts.append(self.factual_memories[-1])
            parts.append("")

        # Procedural memory (M^P)
        if self.procedural_memories:
            parts.append("== Procedural Memory (M^P) ==")
            parts.append(self.procedural_memories)
            parts.append("")

        # Structured procedural trace
        if self.procedural_memory:
            parts.append("== Procedural Trace (M^P) ==")
            for node in self.procedural_memory[-6:]:
                status = "[FAIL]" if node.is_failure else "[OK]"
                parts.append(f"  {status} {node.subgoal} -> {node.action_type}: {node.outcome[:80]}")
            parts.append("")

        # Experiential memory (M^E)
        if self.experiential_memories:
            parts.append("== Experiential Memory (M^E) ==")
            parts.append(self.experiential_memories)
            parts.append("")

        return "\n".join(parts)

    def get_tool_call_history(self) -> List[Dict]:
        if self._tool_call_cache is not None:
            return self._tool_call_cache

        tool_calls = [
            {
                "tool_name": interaction.get("tool_name"),
                "arguments": interaction.get("arguments"),
                "result": interaction.get("result"),
                "iteration": interaction.get("iteration")
            }
            for interaction in self.interactions
            if interaction.get("type") == "tool_call"
        ]

        self._tool_call_cache = tool_calls
        return tool_calls

    def get_stats(self) -> Dict[str, Any]:
        return {
            "fold_count": self.fold_count,
            "max_folds": self.max_folds,
            "can_fold": self.can_fold(),
            "factual_memory_entries": len(self.factual_memory),
            "procedural_memory_nodes": len(self.procedural_memory),
            "experiential_memory_families": len(self.experiential_memory),
            "episode_memory_count": len(self.factual_memories),
            "has_working_memory": self.procedural_memories is not None,
            "has_tool_memory": self.experiential_memories is not None,
            "interaction_count": len(self.interactions),
            "tool_call_count": len(self.get_tool_call_history())
        }

    def reset(self):
        # Structured memory
        self.factual_memory = {}
        self.procedural_memory = []
        self.experiential_memory = {}

        # Legacy memory
        self.factual_memories = []
        self.procedural_memories = None
        self.experiential_memories = None

        # State
        self.fold_count = 0
        self.interactions = []
        self._tool_call_cache = None
        self._evidence_counter = 0
        self._node_counter = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            # Structured memory
            "factual_memory": {
                f"{k[0]}:{k[1]}": v.to_dict()
                for k, v in self.factual_memory.items()
            },
            "procedural_memory": [n.to_dict() for n in self.procedural_memory],
            "experiential_memory": {
                k: v.to_dict() for k, v in self.experiential_memory.items()
            },
            # Legacy memory
            "factual_memories": self.factual_memories,
            "procedural_memories": self.procedural_memories,
            "experiential_memories": self.experiential_memories,
            # State
            "fold_count": self.fold_count,
            "max_folds": self.max_folds,
            "interactions": self.interactions,
            "_evidence_counter": self._evidence_counter,
            "_node_counter": self._node_counter
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MemoryManager':
        manager = cls(max_folds=data.get("max_folds", 3))

        # Structured memory
        for key_str, entry_data in data.get("factual_memory", {}).items():
            entry = FactualEntry.from_dict(entry_data)
            key = (entry.entity_id, entry.attr)
            manager.factual_memory[key] = entry

        manager.procedural_memory = [
            ProceduralNode.from_dict(n)
            for n in data.get("procedural_memory", [])
        ]

        manager.experiential_memory = {
            k: ExperientialHeuristic.from_dict(v)
            for k, v in data.get("experiential_memory", {}).items()
        }

        # Legacy memory
        manager.factual_memories = data.get("factual_memories", [])
        manager.procedural_memories = data.get("procedural_memories")
        manager.experiential_memories = data.get("experiential_memories")

        # State
        manager.fold_count = data.get("fold_count", 0)
        manager.interactions = data.get("interactions", [])
        manager._evidence_counter = data.get("_evidence_counter", 0)
        manager._node_counter = data.get("_node_counter", 0)

        return manager
