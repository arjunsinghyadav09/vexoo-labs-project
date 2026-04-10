"""
Bonus: Reasoning-Aware Plug-and-Play Adapter
=============================================
A modular routing layer that detects the nature of an input query and
activates the appropriate reasoning module.

Architecture
------------

                     ┌──────────────────────────────────┐
  Raw Query ─────►   │       QueryClassifier            │
                     │  (rule-based + score-weighted)    │
                     └──────────────┬───────────────────┘
                                    │ query_type
                        ┌───────────▼────────────┐
                        │   ReasoningRouter       │
                        └───┬──────┬──────┬──────┘
                            │      │      │
                   ┌────────▼─┐ ┌──▼───┐ ┌▼────────────┐
                   │  Math    │ │ Legal│ │  General     │
                   │  Module  │ │Module│ │  Module      │
                   └────────┬─┘ └──┬───┘ └┬────────────┘
                            │      │      │
                     ┌──────▼──────▼──────▼──────┐
                     │        Response Builder     │
                     └────────────────────────────┘

Each module is a plug-in (implements ReasoningModule protocol).
New modules can be added without touching the router — pure Open/Closed.

Author  : Candidate
Project : Vexoo Labs AI Engineer Assignment
"""

from __future__ import annotations

import re
import json
import math
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

log = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
)


# ---------------------------------------------------------------------------
# Protocol / Interface
# ---------------------------------------------------------------------------

class ReasoningModule(ABC):
    """
    Base class for all reasoning modules.
    Every module must implement `handle(query)` and expose a `name`.
    """
    name: str = "base"

    @abstractmethod
    def handle(self, query: str, context: Optional[Dict] = None) -> "ReasoningResult":
        ...

    def can_handle(self, query_type: str) -> bool:
        """Override to declare which query types this module handles."""
        return False


@dataclass
class ReasoningResult:
    """Standardised output from any reasoning module."""
    module_used  : str
    query_type   : str
    answer       : str
    confidence   : float           # 0.0 – 1.0
    reasoning    : List[str]       # step-by-step trace
    metadata     : Dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Query Classifier
# ---------------------------------------------------------------------------

CLASSIFIER_RULES: Dict[str, List[str]] = {
    "math": [
        r"\b(calculate|compute|solve|equation|how many|how much|total|sum|"
        r"product|divide|multiply|percent|ratio|rate|average|mean|median|"
        r"profit|loss|discount|tax|area|perimeter|volume|speed|distance)\b",
        r"\d+\s*[\+\-\*\/\^]\s*\d+",     # arithmetic expression
        r"\b\d+\s*(apples?|students?|items?|dollars?|\$|km|m|kg)\b",
    ],
    "legal": [
        r"\b(law|legal|court|judgment|liability|contract|clause|statute|"
        r"regulation|compliance|rights?|obligation|breach|damages?|"
        r"intellectual property|patent|copyright|trademark|GDPR|HIPAA|"
        r"lawsuit|settlement|defendant|plaintiff|appeal|jurisdiction)\b",
    ],
    "code": [
        r"\b(code|function|algorithm|debug|error|bug|implement|python|"
        r"javascript|sql|api|class|variable|loop|recursion|data structure|"
        r"time complexity|space complexity|runtime)\b",
        r"```",   # code block
        r"def |class |import |#include",
    ],
    "science": [
        r"\b(biology|chemistry|physics|quantum|molecule|atom|cell|DNA|"
        r"evolution|gravity|relativity|thermodynamics|photosynthesis|"
        r"ecosystem|neural|neuron|experiment|hypothesis|theory)\b",
    ],
    "general": [],   # fallback
}


@dataclass
class ClassificationResult:
    query_type : str
    scores     : Dict[str, float]
    flags      : Dict[str, bool]


class QueryClassifier:
    """
    Multi-signal query classifier.

    Signal pipeline:
      1. Keyword / regex pattern matching → weighted hit count
      2. Question-word heuristics (who/what/when → general; how much → math)
      3. Structural signals (numbers, operators, code blocks)

    Returns the type with the highest composite score.
    """

    # Weight for each signal type
    PATTERN_WEIGHT   = 1.0
    STRUCTURE_WEIGHT = 1.5   # structural signals are stronger evidence

    def classify(self, query: str) -> ClassificationResult:
        lower = query.lower()
        scores: Dict[str, float] = {k: 0.0 for k in CLASSIFIER_RULES}

        # ── Signal 1: regex pattern hits ──────────────────────────────
        for qtype, patterns in CLASSIFIER_RULES.items():
            for pat in patterns:
                hits = len(re.findall(pat, lower, re.IGNORECASE))
                scores[qtype] += hits * self.PATTERN_WEIGHT

        # ── Signal 2: structural heuristics ───────────────────────────
        has_numbers    = bool(re.search(r"\b\d+\b", query))
        has_operator   = bool(re.search(r"[\+\-\*\/\=\^]", query))
        has_code_fence = "```" in query or "def " in query

        if has_numbers and has_operator:
            scores["math"]  += 2 * self.STRUCTURE_WEIGHT
        if has_code_fence:
            scores["code"]  += 2 * self.STRUCTURE_WEIGHT

        # ── Signal 3: question-word override ──────────────────────────
        q_words = re.findall(r"^(who|what|when|where|why|how)\b", lower)
        if q_words:
            qw = q_words[0]
            if qw in ("how many", "how much") or "how" in lower and has_numbers:
                scores["math"] += 1.0
            elif qw in ("who", "where", "when"):
                scores["general"] += 0.5

        # Determine winner (exclude 'general' as active choice)
        active = {k: v for k, v in scores.items() if k != "general"}
        winner = max(active, key=active.get) if any(v > 0 for v in active.values()) else "general"

        flags = {
            "has_numbers"  : has_numbers,
            "has_operator" : has_operator,
            "has_code"     : has_code_fence,
        }
        log.debug("Classifier | type=%s | scores=%s", winner, scores)
        return ClassificationResult(query_type=winner, scores=scores, flags=flags)


# ---------------------------------------------------------------------------
# Concrete Reasoning Modules
# ---------------------------------------------------------------------------

class MathReasoningModule(ReasoningModule):
    """
    Handles arithmetic and word-problem reasoning.

    Strategy:
      1. Extract numbers and operators from query
      2. Attempt direct eval of simple expressions
      3. For word problems: identify operation type → template solution
      4. Return step-by-step chain-of-thought
    """
    name = "math"

    def can_handle(self, query_type: str) -> bool:
        return query_type == "math"

    def handle(self, query: str, context: Optional[Dict] = None) -> ReasoningResult:
        steps  = []
        answer = ""

        # Try direct expression evaluation (safe subset)
        expr_match = re.search(r"([\d\s\+\-\*\/\(\)\.]+)", query)
        if expr_match:
            raw = expr_match.group(1).strip()
            # Allow only digits and safe operators
            safe_expr = re.sub(r"[^\d\s\+\-\*\/\(\)\.]", "", raw).strip()
            if safe_expr and any(c in safe_expr for c in "+-*/"):
                try:
                    result = eval(safe_expr, {"__builtins__": {}})  # noqa: S307
                    steps.append(f"Identified arithmetic expression: {safe_expr}")
                    steps.append(f"Computed: {safe_expr} = {result}")
                    answer = str(round(result, 4))
                    return ReasoningResult(
                        module_used = self.name,
                        query_type  = "math",
                        answer      = f"The answer is {answer}",
                        confidence  = 0.92,
                        reasoning   = steps,
                        metadata    = {"expression": safe_expr, "result": result},
                    )
                except Exception:
                    pass

        # Word problem heuristic
        numbers = re.findall(r"\d+(?:\.\d+)?", query)
        steps.append(f"Extracted numbers: {numbers}")

        op_hints = {
            "total|sum|combined|together|add"  : "addition",
            "left|remain|differ|subtract|less" : "subtraction",
            "each|per|every|times|product"     : "multiplication",
            "split|share|divide|group|per"     : "division",
        }
        detected_op = "unknown"
        for pattern, op in op_hints.items():
            if re.search(pattern, query.lower()):
                detected_op = op
                break

        steps.append(f"Detected operation type: {detected_op}")

        if len(numbers) >= 2 and detected_op != "unknown":
            a, b = float(numbers[0]), float(numbers[1])
            ops_map = {
                "addition"      : (a + b, f"{a} + {b}"),
                "subtraction"   : (a - b, f"{a} - {b}"),
                "multiplication": (a * b, f"{a} × {b}"),
                "division"      : (a / b if b != 0 else None, f"{a} ÷ {b}"),
            }
            result, expr = ops_map.get(detected_op, (None, ""))
            if result is not None:
                steps.append(f"Apply {detected_op}: {expr} = {result}")
                answer = str(round(result, 4))
            else:
                answer = "Cannot divide by zero."
                steps.append("Error: division by zero detected")
        else:
            answer = "Unable to extract a definitive numeric answer. Please reformulate."
            steps.append("Insufficient numeric data for automatic computation.")

        return ReasoningResult(
            module_used = self.name,
            query_type  = "math",
            answer      = f"Based on analysis: {answer}",
            confidence  = 0.70,
            reasoning   = steps,
        )


class LegalReasoningModule(ReasoningModule):
    """
    Handles legal query routing.

    Strategy:
      1. Identify legal domain (contract, IP, compliance, etc.)
      2. Flag jurisdiction cues
      3. Route to structured retrieval (simulated here)
      4. Append disclaimer
    """
    name = "legal"

    LEGAL_DOMAINS = {
        "contract"    : r"\b(contract|agreement|clause|breach|obligation|party)\b",
        "intellectual": r"\b(patent|copyright|trademark|ip|intellectual property)\b",
        "compliance"  : r"\b(GDPR|HIPAA|CCPA|regulation|compliance|data protection)\b",
        "litigation"  : r"\b(lawsuit|court|judge|plaintiff|defendant|settlement|damages)\b",
        "employment"  : r"\b(employee|employer|termination|discrimination|severance|HR)\b",
    }

    def can_handle(self, query_type: str) -> bool:
        return query_type == "legal"

    def handle(self, query: str, context: Optional[Dict] = None) -> ReasoningResult:
        steps = []

        # Detect legal sub-domain
        domain_scores: Dict[str, int] = {}
        for dom, pat in self.LEGAL_DOMAINS.items():
            hits = len(re.findall(pat, query, re.IGNORECASE))
            domain_scores[dom] = hits
        top_domain = max(domain_scores, key=domain_scores.get)
        steps.append(f"Identified legal domain: {top_domain}")

        # Detect jurisdiction
        jurisdictions = ["US", "EU", "UK", "India", "California", "federal"]
        found_j = [j for j in jurisdictions if j.lower() in query.lower()]
        if found_j:
            steps.append(f"Jurisdiction signals detected: {found_j}")
        else:
            steps.append("No explicit jurisdiction detected — assuming general legal principles")

        steps.append("Routing to structured legal knowledge retrieval (simulated)")
        steps.append("Applying citation-aware response template")

        answer = (
            f"This appears to be a {top_domain} law question"
            + (f" under {', '.join(found_j)} jurisdiction" if found_j else "")
            + ". For precise legal advice, consult a qualified attorney. "
            "In general terms: [structured retrieval result would appear here]."
        )

        return ReasoningResult(
            module_used = self.name,
            query_type  = "legal",
            answer      = answer,
            confidence  = 0.60,
            reasoning   = steps,
            metadata    = {"domain": top_domain, "jurisdiction": found_j},
        )


class CodeReasoningModule(ReasoningModule):
    """
    Handles programming / algorithmic queries.

    Strategy:
      1. Detect language (Python, JS, SQL…)
      2. Identify task type (debug, implement, explain)
      3. Route to code-specific reasoning path
    """
    name = "code"

    LANGUAGES = {
        "python"    : r"\b(python|def |import |\.py|pandas|numpy|sklearn)\b",
        "javascript": r"\b(javascript|js|node|npm|async|await|const|let)\b",
        "sql"       : r"\b(sql|select|from|where|join|insert|update|delete)\b",
        "java"      : r"\b(java|class |public |private |static |void )\b",
        "c/c++"     : r"\b(c\+\+|#include|malloc|pointer|struct)\b",
    }
    TASK_TYPES = {
        "debug"     : r"\b(bug|error|fix|debug|not working|failing|exception)\b",
        "implement" : r"\b(write|implement|create|build|code|make)\b",
        "explain"   : r"\b(explain|what is|how does|understand|mean)\b",
        "optimise"  : r"\b(optim|faster|efficient|performance|complexity)\b",
    }

    def can_handle(self, query_type: str) -> bool:
        return query_type == "code"

    def handle(self, query: str, context: Optional[Dict] = None) -> ReasoningResult:
        steps = []

        # Detect language
        detected_lang = "general"
        for lang, pat in self.LANGUAGES.items():
            if re.search(pat, query, re.IGNORECASE):
                detected_lang = lang
                break
        steps.append(f"Detected language: {detected_lang}")

        # Detect task
        detected_task = "general"
        for task, pat in self.TASK_TYPES.items():
            if re.search(pat, query, re.IGNORECASE):
                detected_task = task
                break
        steps.append(f"Detected task type: {detected_task}")
        steps.append("Activating code reasoning path with syntax-aware response")

        answer = (
            f"This is a {detected_lang} {detected_task} question. "
            "[Code reasoning module would generate a structured, syntax-highlighted solution here, "
            "with complexity analysis and best-practice notes.]"
        )

        return ReasoningResult(
            module_used = self.name,
            query_type  = "code",
            answer      = answer,
            confidence  = 0.80,
            reasoning   = steps,
            metadata    = {"language": detected_lang, "task": detected_task},
        )


class ScienceReasoningModule(ReasoningModule):
    name = "science"

    def can_handle(self, query_type: str) -> bool:
        return query_type == "science"

    def handle(self, query: str, context: Optional[Dict] = None) -> ReasoningResult:
        return ReasoningResult(
            module_used = self.name,
            query_type  = "science",
            answer      = "Activating scientific reasoning: concept identification → principle retrieval → explanation.",
            confidence  = 0.75,
            reasoning   = ["Identified as science query", "Routing to domain-specific knowledge base"],
        )


class GeneralReasoningModule(ReasoningModule):
    """
    Fallback: semantic search over general knowledge.
    """
    name = "general"

    def can_handle(self, query_type: str) -> bool:
        return True   # always available as fallback

    def handle(self, query: str, context: Optional[Dict] = None) -> ReasoningResult:
        return ReasoningResult(
            module_used = self.name,
            query_type  = "general",
            answer      = "Activating general semantic search for: " + query,
            confidence  = 0.50,
            reasoning   = ["No specialised module matched", "Falling back to general knowledge retrieval"],
        )


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------

class ReasoningRouter:
    """
    Plug-and-play router.

    Registers modules and dispatches queries to the best match.
    Modules are tried in registration order; first `can_handle` wins.

    Design principle: Open for extension, closed for modification.
    To add a new module: router.register(MyNewModule())  ← that's it.
    """

    def __init__(self) -> None:
        self._modules  : List[ReasoningModule] = []
        self._classifier = QueryClassifier()
        self._call_log  : List[Dict] = []

    def register(self, module: ReasoningModule) -> "ReasoningRouter":
        self._modules.append(module)
        log.info("Registered module: %s", module.name)
        return self   # fluent API

    def route(self, query: str, context: Optional[Dict] = None) -> ReasoningResult:
        """Classify query → find module → execute → log."""
        classification = self._classifier.classify(query)
        qtype          = classification.query_type

        log.info("Routing query | type=%s | query='%s'", qtype, query[:60])

        # Find first module that can handle this type
        selected = None
        for m in self._modules:
            if m.can_handle(qtype):
                selected = m
                break

        if selected is None:
            # Guaranteed fallback
            selected = GeneralReasoningModule()

        result = selected.handle(query, context)

        # Annotate result with classification metadata
        result.metadata["classification_scores"] = classification.scores
        result.metadata["structural_flags"]      = classification.flags

        self._call_log.append({
            "query"      : query[:100],
            "detected"   : qtype,
            "module_used": selected.name,
            "confidence" : result.confidence,
        })

        return result

    def export_log(self, path: str) -> None:
        with open(path, "w") as fh:
            json.dump(self._call_log, fh, indent=2)
        log.info("Router log saved → %s", path)


# ---------------------------------------------------------------------------
# Factory – default fully-equipped router
# ---------------------------------------------------------------------------

def build_default_router() -> ReasoningRouter:
    """
    Factory function that builds a router with all default modules.
    The order of registration determines fallback priority.
    """
    return (
        ReasoningRouter()
        .register(MathReasoningModule())
        .register(LegalReasoningModule())
        .register(CodeReasoningModule())
        .register(ScienceReasoningModule())
        .register(GeneralReasoningModule())   # last = fallback
    )


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

def demo() -> None:
    print("\n" + "=" * 70)
    print("  VEXOO LABS – Bonus: Reasoning-Aware Plug-and-Play Adapter Demo")
    print("=" * 70)

    router = build_default_router()

    test_queries = [
        # Math
        "If Alice has 45 apples and gives 17 to Bob, how many does she have left?",
        "Calculate 128 * 7 + 256 / 4",
        "A store sells 120 items per day at $8 each. What is the weekly revenue?",
        # Legal
        "What are the GDPR compliance requirements for storing user data in the EU?",
        "Is my employer allowed to terminate my contract without severance pay?",
        # Code
        "How do I fix a KeyError in Python when accessing a dictionary?",
        "Write a SQL query to join users and orders on user_id",
        # Science
        "How does photosynthesis convert sunlight into glucose?",
        # General
        "Who wrote Romeo and Juliet?",
    ]

    for q in test_queries:
        result = router.route(q)
        print(f"\n{'─'*70}")
        print(f"❓ Query     : {q}")
        print(f"🔀 Module    : {result.module_used}")
        print(f"🎯 Confidence: {result.confidence:.0%}")
        print(f"💡 Answer    : {result.answer}")
        print(f"🔍 Reasoning :")
        for i, step in enumerate(result.reasoning, 1):
            print(f"   {i}. {step}")

    import os
    script_dir = os.path.dirname(os.path.abspath(__file__))
    logs_dir = os.path.join(script_dir, "..", "logs")
    os.makedirs(logs_dir, exist_ok=True)
    log_path = os.path.join(logs_dir, "router_log.json")
    router.export_log(log_path)
    print(f"\n✅ Bonus demo complete. Router log → logs/router_log.json\n")


if __name__ == "__main__":
    demo()
