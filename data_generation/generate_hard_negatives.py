#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


# ---------------------------------------------------------------------------
# Dataset-specific guidance
# ---------------------------------------------------------------------------

BASE_RELEVANCE_DEFINITION = """
[General Passage Retrieval Relevance Definition]

Dense retrievers should retrieve documents that satisfy the user's information
need, not merely documents that share surface-level keywords with the query.

Positive documents:
- Attempt to answer the question or satisfy the information need.
- Provide evidence, facts, or context that would help answer the query.
- Discuss the same entity/topic under the same requested attribute, relation,
  time, location, or constraint.

Valid hard negatives:
- Are topically close enough to look plausible at first glance.
- Violate at least one required condition of relevance.
- Discuss a related but different entity, attribute, relation, time, location,
  or question type.
- Do not provide correct, wrong, partial, or negated answers to the query.

Invalid negatives:
- Wrong answers that still attempt to answer the query.
- Partial answers.
- Negated answers such as "X is not Y".
- Documents that make a user believe the query has been answered.
"""


DATASET_CONFIGS: dict[str, dict[str, str]] = {
    "generic": {
        "name": "Generic Passage Retrieval",
        "relevance_definition": BASE_RELEVANCE_DEFINITION,
        "writing_style": """
Write natural passage-like documents that resemble the target corpus. Preserve
the style of candidate negatives when provided. Use concrete entities, dates,
locations, numbers, or attributes when appropriate. Keep each generated passage
self-contained and realistic.
""",
    },
    "nq": {
        "name": "Natural Questions",
        "relevance_definition": """
[Natural Questions Relevance Definition]

Dataset characteristics:
- Real Google search questions.
- Mostly factoid information needs.
- Wikipedia-style evidence passages.

Positive documents:
- Attempt to answer the specific question.
- Provide factual information directly related to the queried entity and
  requested attribute.
- Discuss the same topic under the same question type.

Valid negatives:
- Discuss related but different entities.
- Keep the entity but shift to an unrelated aspect.
- Use different seasons, versions, time periods, or media when the query has
  such constraints.
- Provide background without touching the requested attribute.

Do not generate wrong answers, partial answers, or negated answers.
""",
        "writing_style": """
Use neutral Wikipedia-style prose: third person, self-contained, fact-dense,
80-200 words when possible. Include specific names, dates, organizations, or
locations. Avoid promotional language and first-person wording.
""",
    },
    "trivia": {
        "name": "TriviaQA",
        "relevance_definition": """
[TriviaQA Relevance Definition]

Dataset characteristics:
- Trivia-style questions with clues.
- Evidence can come from Wikipedia or web pages.
- Questions often depend on names, awards, years, locations, or achievements.

Positive documents:
- Discuss the answer entity or evidence that leads to it.
- Contain the clue-bearing relation needed by the question.

Valid negatives:
- Namesake or similar-attribute confusion.
- Same award/event/category but different year, field, winner, or location.
- Related background that does not identify or support the answer.
- Right entity but wrong requested achievement or relation, without answering
  the query.

Do not generate documents that give the answer, a wrong answer, or a partial
answer.
""",
        "writing_style": """
Use mixed Wikipedia/web style: fact-dense, clue-rich, and natural. Include full
names, dates, titles, awards, places, or historical context. Avoid casual blog
language and unsupported speculation.
""",
    },
    "hotpotqa": {
        "name": "HotpotQA",
        "relevance_definition": """
[HotpotQA Relevance Definition]

Dataset characteristics:
- Multi-hop questions involving bridge entities, comparisons, or shared
  attributes.
- A relevant passage usually contributes one reasoning step.

Positive documents:
- Provide information about a key entity in the query.
- Contain the bridge attribute or relation required for multi-hop reasoning.
- Discuss the common property being asked about.

Valid negatives:
- Mention an entity but omit the bridge attribute.
- Replace one entity with a similar entity.
- Break the relationship between entities.
- Shift to a different property than the one requested.

Do not generate documents that answer either hop in a way that satisfies the
query.
""",
        "writing_style": """
Use entity-centric Wikipedia-style passages. Include concrete attributes such
as nationality, dates, creators, locations, affiliations, or categories. Keep
the passage focused enough that the missing or shifted reasoning step is clear.
""",
    },
    "mmarco": {
        "name": "mMARCO / MS MARCO",
        "relevance_definition": """
[mMARCO / MS MARCO Relevance Definition]

Dataset characteristics:
- Web-search-style queries.
- Passages can be snippets, FAQs, or web page fragments.
- User intent may be factual, procedural, comparative, or definitional.

Positive documents:
- Directly satisfy the user's search intent.
- Provide instructions, facts, definitions, comparisons, or evidence requested
  by the query.

Valid negatives:
- Stay on the same broad topic but answer a different intent.
- Discuss a related product, entity, location, method, symptom, or time period.
- Provide generic background while avoiding the requested answer.
- Preserve realistic web noise when rewriting candidate negatives.

Do not generate wrong answers, partial answers, or documents that appear to
complete the user's task.
""",
        "writing_style": """
Use realistic web passage style: snippets, FAQ-like paragraphs, lightly noisy
machine-translated prose, or short encyclopedic fragments. Prefer rewriting
candidate negatives to preserve corpus style. Keep each passage self-contained.
""",
    },
}


DISRUPTION_TYPES = """
Reference disruption types:
- entity_shift: replace a key entity with a related but incorrect entity.
- intent_drift: keep the topic but change the question type or user intent.
- constraint_violation: violate a temporal, spatial, numerical, causal, or
  categorical constraint in the query.
- attribute_shift: keep the entity but discuss a different attribute.
- relation_break: preserve entities but remove or alter the relation required
  for relevance.
- upstream_downstream: move to a prerequisite, consequence, or adjacent topic
  rather than the requested information.
"""


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GenerationConfig:
    dataset: str
    model: str
    base_url: str
    api_key: str
    temperature: float
    max_tokens: int
    max_retries: int
    retry_delay: float
    request_interval: float
    candidate_limit: int
    max_negatives_per_query: int | None
    keep_raw_responses: bool


@dataclass(frozen=True)
class InputSample:
    sample_id: str
    raw: dict[str, Any]
    query: str
    positive: str
    candidate_negatives: list[str]


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------


def get_dataset_config(dataset: str) -> dict[str, str]:
    key = dataset.lower()
    if key not in DATASET_CONFIGS:
        available = ", ".join(sorted(DATASET_CONFIGS))
        raise ValueError(f"Unknown dataset '{dataset}'. Available: {available}")
    return DATASET_CONFIGS[key]


def build_analysis_prompt(sample: InputSample, dataset: str) -> str:
    cfg = get_dataset_config(dataset)
    return f"""You are an expert in information retrieval and dense retriever training.

Your task is to analyze why a positive document satisfies a query, then design
counterfactual perturbations that would make a document no longer satisfy the
query while remaining topically close.

{cfg["relevance_definition"]}

{DISRUPTION_TYPES}

---

## Input

### Query
{sample.query}

### Positive Document
{sample.positive}

---

## Instructions

1. Identify the query's information need in one sentence.
2. Explain how the positive document satisfies that information need.
3. Define the answer boundary: what counts as answering the query, and what
   does not.
4. Decompose relevance into 4-8 information-requirement nodes. Each node should
   be concrete and falsifiable.
5. Mark a node as `critical=true` if violating it would make the document no
   longer relevant.
6. For each critical node, design 2-3 disruption strategies. A strategy must
   describe how to violate exactly one requirement while keeping the passage
   plausible and related.

Return only valid JSON matching this schema:

{{
  "query_info": {{
    "core_need": "...",
    "how_positive_answers": "...",
    "answer_boundary": "..."
  }},
  "chain_nodes": [
    {{
      "node_id": 1,
      "facet_type": "information_need | entity | attribute | constraint | relation | reasoning | style | other",
      "content": "The concrete requirement this node represents.",
      "critical": true,
      "break_strategies": [
        {{
          "id": 1,
          "type": "entity_shift | intent_drift | constraint_violation | attribute_shift | relation_break | upstream_downstream | other",
          "direction": "Specific perturbation direction.",
          "why_not_answer": "Why this perturbation prevents the document from answering the query."
        }}
      ]
    }}
  ]
}}
"""


def build_generation_prompt(
    sample: InputSample,
    query_info: dict[str, Any],
    chain_nodes: list[dict[str, Any]],
    dataset: str,
    candidate_limit: int,
    max_negatives: int | None,
) -> str:
    cfg = get_dataset_config(dataset)
    query_info_json = json.dumps(query_info, ensure_ascii=False, indent=2)
    chain_nodes_json = json.dumps(chain_nodes, ensure_ascii=False, indent=2)
    candidate_text = "\n\n".join(
        f"[Candidate Negative {i + 1}]\n{text}"
        for i, text in enumerate(sample.candidate_negatives[:candidate_limit])
    )
    if not candidate_text:
        candidate_text = "(No candidate negatives are provided. Generate from scratch while matching the target corpus style.)"

    budget = (
        f"Generate at most {max_negatives} final negatives in total. Select the most diagnostic strategies."
        if max_negatives
        else "Generate one negative for each listed disruption strategy."
    )

    return f"""You are constructing hard negative passages for dense retrieval.

{cfg["relevance_definition"]}

## Corpus Style Requirements
{cfg["writing_style"]}

---

## Input

### Query
{sample.query}

### Positive Document
{sample.positive}

### Query Analysis
{query_info_json}

### Critical Nodes and Disruption Strategies
{chain_nodes_json}

### Candidate Negative Documents From the Corpus
{candidate_text}

---

## Generation Requirements

{budget}

For every generated document:
- It must be topically close to the query and positive document.
- It must violate the selected requirement and therefore not satisfy the query.
- It must not provide a correct, wrong, partial, or negated answer.
- Prefer rewriting a candidate negative when possible; preserve its style,
  granularity, and natural noise.
- If generating from scratch, imitate the target corpus style.
- Keep the document self-contained.
- Avoid meta text such as "this is a negative sample".

Return only valid JSON matching this schema:

{{
  "negatives": [
    {{
      "node_id": 1,
      "strategy_id": 1,
      "strategy_type": "entity_shift",
      "source": "rewritten_candidate | generated_from_scratch",
      "text": "The generated hard negative passage.",
      "break_explanation": "Why this is close but does not answer the query."
    }}
  ]
}}
"""


# ---------------------------------------------------------------------------
# Input/output utilities
# ---------------------------------------------------------------------------


def read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_no}: {exc}") from exc


def first_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, list):
        for item in value:
            text = first_text(item)
            if text:
                return text
        return ""
    if isinstance(value, dict):
        if "content" in value:
            return first_text(value["content"])
        for key in ("text", "contents", "document", "passage"):
            if key in value:
                return first_text(value[key])
        return ""
    return str(value).strip()


def collect_texts(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        text = value.strip()
        return [text] if text else []
    if isinstance(value, dict):
        text = first_text(value)
        return [text] if text else []
    if isinstance(value, list):
        texts: list[str] = []
        for item in value:
            texts.extend(collect_texts(item))
        return texts
    return []


def extract_query(raw: dict[str, Any]) -> str:
    for key in ("query", "question", "q", "user_query"):
        if key in raw:
            text = first_text(raw[key])
            if text:
                return text

    messages = raw.get("messages")
    if isinstance(messages, list):
        user_messages = [
            first_text(message)
            for message in messages
            if isinstance(message, dict) and message.get("role") == "user"
        ]
        user_messages = [text for text in user_messages if text]
        if user_messages:
            return user_messages[-1]
    return ""


def extract_positive(raw: dict[str, Any]) -> str:
    for key in (
        "positive",
        "positive_doc",
        "pos_doc",
        "pos",
        "positive_passages",
        "positive_messages",
    ):
        if key in raw:
            text = first_text(raw[key])
            if text:
                return text
    return ""


def extract_candidate_negatives(raw: dict[str, Any]) -> list[str]:
    keys = (
        "candidate_negatives",
        "negative_messages",
        "negative",
        "negatives",
        "neg_doc_list",
        "neg",
        "hard_negatives",
    )
    texts: list[str] = []
    for key in keys:
        if key in raw:
            texts.extend(collect_texts(raw[key]))

    seen: set[str] = set()
    unique: list[str] = []
    for text in texts:
        normalized = re.sub(r"\s+", " ", text).strip()
        if normalized and normalized not in seen:
            seen.add(normalized)
            unique.append(normalized)
    return unique


def sample_id(raw: dict[str, Any], index: int, query: str) -> str:
    for key in ("id", "qid", "query_id", "_id"):
        if key in raw and raw[key] is not None:
            return str(raw[key])
    return f"{index}:{abs(hash(query))}"


def load_samples(path: Path, limit: int | None = None, shuffle: bool = False, seed: int = 13) -> list[InputSample]:
    samples: list[InputSample] = []
    for idx, raw in enumerate(read_jsonl(path)):
        query = extract_query(raw)
        positive = extract_positive(raw)
        if not query or not positive:
            continue
        samples.append(
            InputSample(
                sample_id=sample_id(raw, idx, query),
                raw=raw,
                query=query,
                positive=positive,
                candidate_negatives=extract_candidate_negatives(raw),
            )
        )

    if shuffle:
        random.Random(seed).shuffle(samples)
    if limit is not None:
        samples = samples[:limit]
    return samples


def load_completed_ids(output_path: Path) -> set[str]:
    if not output_path.exists():
        return set()
    completed: set[str] = set()
    for raw in read_jsonl(output_path):
        sid = raw.get("sample_id")
        if sid:
            completed.add(str(sid))
    return completed


def append_jsonl(path: Path, item: dict[str, Any], lock: threading.Lock) -> None:
    encoded = json.dumps(item, ensure_ascii=False)
    with lock:
        with path.open("a", encoding="utf-8") as f:
            f.write(encoded + "\n")


# ---------------------------------------------------------------------------
# JSON parsing and validation
# ---------------------------------------------------------------------------


def strip_json_markdown(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text)
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
    return text.strip()


def extract_json_object(text: str) -> dict[str, Any]:
    cleaned = strip_json_markdown(text)
    try:
        parsed = json.loads(cleaned)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    decoder = json.JSONDecoder()
    for match in re.finditer(r"\{", cleaned):
        try:
            parsed, _ = decoder.raw_decode(cleaned[match.start() :])
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    raise ValueError("Could not parse a JSON object from model response.")


def normalize_chain_nodes(analysis: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    query_info = analysis.get("query_info")
    if not isinstance(query_info, dict):
        query_info = {}

    raw_nodes = analysis.get("chain_nodes")
    if not isinstance(raw_nodes, list):
        raw_nodes = []

    nodes: list[dict[str, Any]] = []
    for i, node in enumerate(raw_nodes, 1):
        if not isinstance(node, dict):
            continue
        strategies = node.get("break_strategies")
        if not isinstance(strategies, list):
            strategies = []
        clean_strategies = [s for s in strategies if isinstance(s, dict) and s.get("direction")]
        if node.get("critical", False) or clean_strategies:
            clean_node = dict(node)
            clean_node["node_id"] = clean_node.get("node_id", i)
            clean_node["critical"] = bool(clean_node.get("critical", True))
            clean_node["break_strategies"] = clean_strategies
            nodes.append(clean_node)

    if not nodes:
        nodes = [
            {
                "node_id": 1,
                "facet_type": "information_need",
                "content": "The document must satisfy the query's core information need.",
                "critical": True,
                "break_strategies": [
                    {
                        "id": 1,
                        "type": "intent_drift",
                        "direction": "Discuss a closely related topic but avoid the requested answer.",
                        "why_not_answer": "The document no longer addresses the user's information need.",
                    },
                    {
                        "id": 2,
                        "type": "entity_shift",
                        "direction": "Replace the core entity with a similar but different entity.",
                        "why_not_answer": "The document concerns a different entity than the query asks about.",
                    },
                ],
            }
        ]
    return query_info, nodes


def normalize_negatives(generation: dict[str, Any], max_negatives: int | None = None) -> list[dict[str, Any]]:
    raw_negatives = generation.get("negatives", [])
    normalized: list[dict[str, Any]] = []

    if not isinstance(raw_negatives, list):
        return normalized

    for item in raw_negatives:
        if not isinstance(item, dict):
            continue

        # Preferred public schema: one object per generated negative.
        if item.get("text"):
            normalized.append(
                {
                    "node_id": item.get("node_id"),
                    "strategy_id": item.get("strategy_id", item.get("id")),
                    "strategy_type": item.get("strategy_type", item.get("type")),
                    "source": item.get("source"),
                    "text": first_text(item.get("text")),
                    "break_explanation": first_text(item.get("break_explanation")),
                }
            )
            continue

        # Backward-compatible nested schema from earlier internal scripts.
        node_id = item.get("node_id")
        strategies = item.get("break_strategies", [])
        if isinstance(strategies, list):
            for strategy in strategies:
                if not isinstance(strategy, dict):
                    continue
                text = first_text(strategy.get("text"))
                if text:
                    normalized.append(
                        {
                            "node_id": node_id,
                            "strategy_id": strategy.get("strategy_id", strategy.get("id")),
                            "strategy_type": strategy.get("strategy_type", strategy.get("type")),
                            "source": strategy.get("source"),
                            "text": text,
                            "break_explanation": first_text(strategy.get("break_explanation")),
                        }
                    )

    seen: set[str] = set()
    unique: list[dict[str, Any]] = []
    for neg in normalized:
        text = re.sub(r"\s+", " ", neg.get("text", "")).strip()
        if not text or text in seen:
            continue
        seen.add(text)
        neg["text"] = text
        unique.append(neg)
        if max_negatives and len(unique) >= max_negatives:
            break
    return unique


def basic_quality_flags(sample: InputSample, negatives: list[dict[str, Any]]) -> list[dict[str, Any]]:
    flags: list[dict[str, Any]] = []
    query_lower = sample.query.lower()
    positive_lower = sample.positive.lower()

    for idx, neg in enumerate(negatives):
        text = neg.get("text", "")
        text_lower = text.lower()
        item_flags: list[str] = []
        if len(text.split()) < 8:
            item_flags.append("very_short")
        if text_lower == positive_lower:
            item_flags.append("same_as_positive")
        if query_lower and query_lower in text_lower:
            item_flags.append("contains_full_query")
        flags.append({"index": idx, "flags": item_flags})
    return flags


# ---------------------------------------------------------------------------
# API calls and processing
# ---------------------------------------------------------------------------


def make_client(config: GenerationConfig) -> Any:
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise ImportError(
            "The OpenAI Python SDK is required for generation. "
            "Install dependencies with `pip install -r requirements.txt`, "
            "or use --dry-run to inspect prompts without API calls."
        ) from exc
    return OpenAI(api_key=config.api_key, base_url=config.base_url)


def call_chat(client: Any, prompt: str, config: GenerationConfig) -> tuple[str, dict[str, Any]]:
    last_error: Exception | None = None
    for attempt in range(config.max_retries + 1):
        try:
            response = client.chat.completions.create(
                model=config.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=config.temperature,
                max_tokens=config.max_tokens,
            )
            content = response.choices[0].message.content or ""
            usage = response.usage.model_dump() if response.usage else None
            meta = {
                "model": response.model,
                "usage": usage,
                "attempt": attempt + 1,
            }
            return content, meta
        except Exception as exc:  # Public tool: support all OpenAI-compatible errors.
            last_error = exc
            if attempt >= config.max_retries:
                break
            sleep_for = config.retry_delay * (2**attempt) + random.random()
            time.sleep(sleep_for)
    raise RuntimeError(f"Chat completion failed after retries: {last_error}") from last_error


def process_sample(sample: InputSample, config: GenerationConfig) -> dict[str, Any]:
    client = make_client(config)

    analysis_prompt = build_analysis_prompt(sample, config.dataset)
    analysis_response, analysis_meta = call_chat(client, analysis_prompt, config)
    analysis = extract_json_object(analysis_response)
    query_info, chain_nodes = normalize_chain_nodes(analysis)

    if config.request_interval > 0:
        time.sleep(config.request_interval)

    generation_prompt = build_generation_prompt(
        sample=sample,
        query_info=query_info,
        chain_nodes=chain_nodes,
        dataset=config.dataset,
        candidate_limit=config.candidate_limit,
        max_negatives=config.max_negatives_per_query,
    )
    generation_response, generation_meta = call_chat(client, generation_prompt, config)
    generation = extract_json_object(generation_response)
    negatives = normalize_negatives(generation, config.max_negatives_per_query)

    if not negatives:
        raise ValueError("Model returned no valid negatives.")

    result = {
        "sample_id": sample.sample_id,
        "query": sample.query,
        "positive": sample.positive,
        "candidate_negatives": sample.candidate_negatives,
        "causalneg_analysis": {
            "query_info": query_info,
            "chain_nodes": chain_nodes,
        },
        "causalneg_negatives": negatives,
        "quality_flags": basic_quality_flags(sample, negatives),
        "metadata": {
            "dataset": config.dataset,
            "model": config.model,
            "analysis_usage": analysis_meta.get("usage"),
            "generation_usage": generation_meta.get("usage"),
        },
    }

    if config.keep_raw_responses:
        result["raw_model_responses"] = {
            "analysis": analysis_response,
            "generation": generation_response,
        }

    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate CausalNeg hard negatives from query-positive JSONL data."
    )
    parser.add_argument("--input", type=Path, required=True, help="Input JSONL file.")
    parser.add_argument("--output", type=Path, required=True, help="Output JSONL file.")
    parser.add_argument(
        "--failures",
        type=Path,
        default=None,
        help="Optional JSONL file for failed samples. Defaults to <output>.failures.jsonl.",
    )
    parser.add_argument(
        "--dataset",
        default="generic",
        choices=sorted(DATASET_CONFIGS),
        help="Dataset-specific relevance and style profile.",
    )
    parser.add_argument("--model", default=os.environ.get("OPENAI_MODEL", "gpt-4.1"))
    parser.add_argument("--base-url", default=os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"))
    parser.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY"))
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--retry-delay", type=float, default=3.0)
    parser.add_argument(
        "--request-interval",
        type=float,
        default=0.0,
        help="Delay in seconds between the two calls for one sample.",
    )
    parser.add_argument("--candidate-limit", type=int, default=5)
    parser.add_argument("--max-items", type=int, default=None)
    parser.add_argument("--max-negatives-per-query", type=int, default=3)
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--keep-raw-responses", action="store_true")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the first sample's prompts without calling the API.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if not args.input.exists():
        raise FileNotFoundError(f"Input file not found: {args.input}")
    if not args.api_key and not args.dry_run:
        raise ValueError("OPENAI_API_KEY or --api-key is required unless --dry-run is set.")
    if args.workers < 1:
        raise ValueError("--workers must be >= 1")
    if args.max_negatives_per_query is not None and args.max_negatives_per_query < 1:
        raise ValueError("--max-negatives-per-query must be >= 1")


def run_dry_run(samples: list[InputSample], args: argparse.Namespace) -> None:
    if not samples:
        print("No valid samples found.", file=sys.stderr)
        return
    sample = samples[0]
    analysis_prompt = build_analysis_prompt(sample, args.dataset)
    _, default_nodes = normalize_chain_nodes({})
    generation_prompt = build_generation_prompt(
        sample=sample,
        query_info={"core_need": "Example core need"},
        chain_nodes=default_nodes,
        dataset=args.dataset,
        candidate_limit=args.candidate_limit,
        max_negatives=args.max_negatives_per_query,
    )
    print("\n" + "=" * 80)
    print("ANALYSIS PROMPT")
    print("=" * 80)
    print(analysis_prompt)
    print("\n" + "=" * 80)
    print("GENERATION PROMPT")
    print("=" * 80)
    print(generation_prompt)


def main() -> None:
    args = parse_args()
    validate_args(args)

    samples = load_samples(args.input, limit=args.max_items, shuffle=args.shuffle, seed=args.seed)
    if args.dry_run:
        run_dry_run(samples, args)
        return

    args.output.parent.mkdir(parents=True, exist_ok=True)
    failures_path = args.failures or args.output.with_suffix(args.output.suffix + ".failures.jsonl")
    failures_path.parent.mkdir(parents=True, exist_ok=True)

    completed = load_completed_ids(args.output) if args.resume else set()
    todo = [sample for sample in samples if sample.sample_id not in completed]

    print(
        f"Loaded {len(samples)} valid samples; completed={len(completed)}; "
        f"to_process={len(todo)}; workers={args.workers}",
        flush=True,
    )
    if not todo:
        return

    config = GenerationConfig(
        dataset=args.dataset,
        model=args.model,
        base_url=args.base_url,
        api_key=args.api_key,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        max_retries=args.max_retries,
        retry_delay=args.retry_delay,
        request_interval=args.request_interval,
        candidate_limit=args.candidate_limit,
        max_negatives_per_query=args.max_negatives_per_query,
        keep_raw_responses=args.keep_raw_responses,
    )

    output_lock = threading.Lock()
    failure_lock = threading.Lock()
    counters = {"success": 0, "failed": 0}
    started = time.time()

    def worker(sample: InputSample) -> None:
        try:
            result = process_sample(sample, config)
            append_jsonl(args.output, result, output_lock)
            counters["success"] += 1
            if counters["success"] % 10 == 0:
                elapsed = time.time() - started
                print(
                    f"[progress] success={counters['success']} failed={counters['failed']} "
                    f"elapsed={elapsed/60:.1f}min",
                    flush=True,
                )
        except Exception as exc:
            counters["failed"] += 1
            failure = {
                "sample_id": sample.sample_id,
                "query": sample.query,
                "error": str(exc),
                "raw": sample.raw,
            }
            append_jsonl(failures_path, failure, failure_lock)

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = []
        for sample in todo:
            futures.append(executor.submit(worker, sample))
            if args.request_interval > 0:
                time.sleep(args.request_interval)
        for future in as_completed(futures):
            future.result()

    elapsed = time.time() - started
    print(
        f"Done. success={counters['success']} failed={counters['failed']} "
        f"elapsed={elapsed/60:.1f}min output={args.output}",
        flush=True,
    )
    if counters["failed"]:
        print(f"Failures were written to {failures_path}", flush=True)


if __name__ == "__main__":
    main()
