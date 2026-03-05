# /// script
# dependencies = [
#   "httpx>=0.25.0",
# ]
# ///

"""Portex-eval LLM-as-a-Judge grading script for Harbor tasks.

This standalone script implements portex-eval's multi-judge, criteria-weighted
scoring as a Harbor-compatible grader.  It reads configuration from JSON files
in /tests/, calls LLM judge models via OpenRouter, and writes a reward.json
that Harbor's verifier can consume.

All logic lives in named, importable functions so that the companion unit-test
suite can exercise each piece in isolation.
"""

from __future__ import annotations

import json
import os
import random
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import httpx

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"
MAX_RETRIES = 5
BASE_DELAY = 1.0
MAX_DELAY = 60.0
JITTER = 0.5
REQUEST_TIMEOUT = 120.0
REWARD_PATH = Path("/logs/verifier/reward.json")
DETAIL_PATH = Path("/logs/verifier/portex_detail.json")
TESTS_DIR = Path("/tests")
CRITERIA_BATCH_SIZE = 5  # Max LLM criteria graded concurrently; each runs all judges in parallel

DEFAULT_JUDGE_MODELS = [
    "deepseek/deepseek-v3.2",
    "z-ai/glm-5",
    "openai/gpt-oss-120b",
    "meta-llama/llama-3.3-70b-instruct",
    "moonshotai/kimi-k2.5",
]

# Exact text from portex-eval scorer.py (lines 125-155)
GRADING_TEMPLATE = """
You are assessing a submitted answer on a given task based on a criterion. Here is the data:

[BEGIN DATA]
***
[Task]: {question}
***
[Submission]: {answer}
***
[Criterion]: {criterion}
***
[END DATA]

Does the submission meet the criterion?

{instructions}
"""

GRADING_INSTRUCTIONS = """
After assessing the submitted answer, reply with 'GRADE: $LETTER' (without quotes)
where LETTER is one of CI. Please choose ONE option for the grade: either "C" for
correct answers, or "I" for incorrect answers.

For example, after reviewing a correct answer you might write 'GRADE: C' or after
reviewing an incorrect answer you might write 'GRADE: I'.

First, write out in a step by step manner your reasoning about the criterion to be
sure that your conclusion is correct. Avoid simply stating the correct answers at
the outset. Then, end with your answer formatted as 'GRADE: $LETTER' (without
quotes) where LETTER is one of CI.
"""

_GRADING_PROMPT = GRADING_TEMPLATE.replace("{instructions}", GRADING_INSTRUCTIONS)

RE_ANSWER_LINE = re.compile(
    r"(?:^|\n)\s*(?:Final\s+)?Answer\s*:\s*(.+?)(?=\n|$)",
    re.IGNORECASE | re.DOTALL,
)


def normalize_for_compare(s: str) -> str:
    if not s or not isinstance(s, str):
        return ""
    return re.sub(r"\s+", " ", s.strip().lower())


def extract_final_answer(text: str) -> str:
    if not text or not text.strip():
        return ""
    text = text.strip()

    matches = list(RE_ANSWER_LINE.finditer(text))
    if matches:
        return matches[-1].group(1).strip()

    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if lines:
        return lines[-1]
    return text


def grade_criterion_exact_match(criterion: dict, submission: str) -> dict:
    ground_truth = (
        criterion.get("semanticPrompt")
        or criterion.get("description")
        or criterion.get("name")
        or ""
    )
    weight = float(criterion.get("weight", 0))
    extracted = extract_final_answer(submission)
    ref_norm = normalize_for_compare(ground_truth)
    pred_norm = normalize_for_compare(extracted)
    passed = bool(ref_norm and ref_norm in pred_norm)
    awarded = weight if passed else 0.0

    return {
        "criterion_id": criterion.get("id"),
        "name": criterion.get("name"),
        "semanticPrompt": ground_truth,
        "grader_type": "exactmatch",
        "weight": weight,
        "passed": passed,
        "awarded": awarded,
        "judges": [
            {
                "model": "ExactMatch",
                "grade": "C" if passed else "I",
                "passed": passed,
                "explanation": f"Reference: {ground_truth!r} | Extracted: {extracted!r} (include)",
            }
        ],
    }


def format_grading_prompt(question: str, answer: str, criterion: str) -> str:
    prompt = _GRADING_PROMPT
    prompt = prompt.replace("{question}", question)
    prompt = prompt.replace("{answer}", answer)
    prompt = prompt.replace("{criterion}", criterion)
    return prompt


def parse_grade_from_response(response_text: str) -> tuple[bool, str]:
    match = re.search(r"GRADE:\s*([CI])", response_text, re.IGNORECASE)
    if match:
        grade = match.group(1).upper()
        return grade == "C", grade
    return False, "I"


def _load_json_file(path: Path) -> object:
    """Load JSON from *path* with a tolerant fallback for control chars."""
    raw = path.read_text(encoding="utf-8")
    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        if "Invalid control character" in exc.msg:
            try:
                return json.loads(raw, strict=False)
            except json.JSONDecodeError:
                pass
        context_start = max(exc.pos - 60, 0)
        context_end = min(exc.pos + 60, len(raw))
        context = raw[context_start:context_end].encode("unicode_escape").decode("ascii")
        raise ValueError(
            f"Invalid JSON in {path}: {exc.msg} "
            f"(line {exc.lineno}, column {exc.colno}, char {exc.pos}). "
            f"Context: {context}"
        ) from exc


def load_task_config(config_path: str) -> dict:
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Task config not found: {config_path}")
    config = _load_json_file(path)
    if not isinstance(config, dict):
        raise ValueError(f"Task config must be a JSON object: {config_path}")
    config.setdefault("submission_path", "/app/answer.txt")
    config.setdefault("pass_threshold", 100)
    return config


def load_criteria(task_config: dict, criteria_path: str) -> list[dict]:
    path = Path(criteria_path)
    if path.exists():
        loaded = _load_json_file(path)
        if not isinstance(loaded, list):
            raise ValueError(f"Criteria file must be a JSON array: {criteria_path}")
        criteria = loaded
    else:
        criteria = []

    if not criteria:
        task_id = task_config.get("task_id", "unknown")
        answer = task_config.get("answer", "")
        criteria = [
            {
                "id": f"{task_id}-c1",
                "name": "",
                "description": answer,
                "type": "semantic",
                "weight": 100,
                "rationale": "",
                "examples": [],
                "semanticPrompt": answer,
            }
        ]
    return criteria


def resolve_judge_models(task_config: dict) -> list[str]:
    if task_config.get("judge_models"):
        return task_config["judge_models"]
    env_models = os.environ.get("PORTEX_JUDGE_MODELS", "")
    if env_models:
        return [m.strip() for m in env_models.split(",") if m.strip()]
    return list(DEFAULT_JUDGE_MODELS)


def calculate_delay(attempt: int, retry_after: float | None = None) -> float:
    if retry_after is not None:
        return min(retry_after, MAX_DELAY)
    delay = min(BASE_DELAY * (2**attempt), MAX_DELAY)
    jitter_amount = delay * JITTER * random.random()
    return delay + jitter_amount


def call_openrouter(model_id: str, prompt: str, api_key: str) -> str:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://portex.ai",
        "X-Title": "portex-eval",
    }
    payload = {
        "model": model_id,
        "messages": [{"role": "user", "content": prompt}],
        "provider": {
            "zdr": True,
            "data_collection": "deny",
        },
    }

    last_error: Exception | None = None
    for attempt in range(MAX_RETRIES):
        try:
            with httpx.Client(timeout=REQUEST_TIMEOUT) as client:
                response = client.post(OPENROUTER_API_URL, json=payload, headers=headers)

            if response.status_code == 429:
                retry_after = None
                if "Retry-After" in response.headers:
                    try:
                        retry_after = float(response.headers["Retry-After"])
                    except ValueError:
                        pass
                delay = calculate_delay(attempt, retry_after)
                print(
                    f"Rate limited (attempt {attempt + 1}/{MAX_RETRIES}). "
                    f"Waiting {delay:.1f}s..."
                )
                time.sleep(delay)
                continue

            response.raise_for_status()
            data = response.json()
            choices = data.get("choices", [])
            if not choices:
                raise ValueError("No choices in API response")
            return choices[0]["message"]["content"]

        except httpx.HTTPStatusError as exc:
            if exc.response.status_code == 429:
                retry_after = None
                if "Retry-After" in exc.response.headers:
                    try:
                        retry_after = float(exc.response.headers["Retry-After"])
                    except ValueError:
                        pass
                delay = calculate_delay(attempt, retry_after)
                print(
                    f"Rate limited (attempt {attempt + 1}/{MAX_RETRIES}). "
                    f"Waiting {delay:.1f}s..."
                )
                time.sleep(delay)
                continue
            raise

        except httpx.RequestError as exc:
            last_error = exc
            delay = calculate_delay(attempt)
            print(
                f"Request error (attempt {attempt + 1}/{MAX_RETRIES}): {exc}. "
                f"Waiting {delay:.1f}s..."
            )
            time.sleep(delay)
            continue

    raise RuntimeError(f"Max retries ({MAX_RETRIES}) exceeded for OpenRouter API") from last_error


def _call_one_judge(
    index: int,
    criterion: dict,
    question: str,
    submission: str,
    model: str,
    api_key: str,
) -> tuple[int, dict, dict]:
    """Call OpenRouter for one (criterion, judge). Returns (index, criterion, judge_result)."""
    criterion_prompt = (
        criterion.get("semanticPrompt")
        or criterion.get("description")
        or criterion.get("name")
        or ""
    )
    prompt = format_grading_prompt(question, submission, criterion_prompt)
    try:
        response_text = call_openrouter(model, prompt, api_key)
        passed, grade = parse_grade_from_response(response_text)
        judge_result = {
            "model": model,
            "grade": grade,
            "passed": passed,
            "explanation": response_text,
        }
    except Exception as exc:  # noqa: BLE001
        print(f"Error grading with {model}: {exc}")
        judge_result = {
            "model": model,
            "grade": "I",
            "passed": False,
            "explanation": f"Error: {exc}",
        }
    return (index, criterion, judge_result)


def grade_criterion(
    criterion: dict,
    question: str,
    submission: str,
    judges: list[str],
    api_key: str,
) -> dict:
    criterion_prompt = (
        criterion.get("semanticPrompt")
        or criterion.get("description")
        or criterion.get("name")
        or ""
    )
    weight = float(criterion.get("weight", 0))

    judge_results: list[dict] = []
    for model in judges:
        prompt = format_grading_prompt(question, submission, criterion_prompt)
        try:
            response_text = call_openrouter(model, prompt, api_key)
            passed, grade = parse_grade_from_response(response_text)
            judge_results.append(
                {
                    "model": model,
                    "grade": grade,
                    "passed": passed,
                    "explanation": response_text,
                }
            )
        except Exception as exc:  # noqa: BLE001
            print(f"Error grading with {model}: {exc}")
            judge_results.append(
                {
                    "model": model,
                    "grade": "I",
                    "passed": False,
                    "explanation": f"Error: {exc}",
                }
            )

    passed, awarded = majority_vote(judge_results, weight)
    return {
        "criterion_id": criterion.get("id"),
        "name": criterion.get("name"),
        "semanticPrompt": criterion_prompt,
        "grader_type": "llm-judge",
        "weight": weight,
        "passed": passed,
        "awarded": awarded,
        "judges": judge_results,
    }


def majority_vote(judge_results: list[dict], weight: float) -> tuple[bool, float]:
    passed_count = sum(1 for r in judge_results if r["passed"])
    majority_passed = passed_count > len(judge_results) / 2
    awarded = weight if majority_passed else 0.0
    return majority_passed, awarded


def normalize_score(total_score_raw: float) -> float:
    return total_score_raw / 100.0


def aggregate_scores(criteria_results: list[dict]) -> float:
    return sum(r["awarded"] for r in criteria_results)


def _is_exactmatch_result(result: dict) -> bool:
    judges = result.get("judges")
    if not isinstance(judges, list) or not judges:
        return False
    return all((j.get("model") == "ExactMatch") for j in judges)


def _votes_string(result: dict) -> str:
    judges = result.get("judges")
    if not isinstance(judges, list) or not judges:
        return ""
    votes: list[str] = []
    for j in judges:
        grade = (j.get("grade") or "").strip().upper()
        if grade in ("C", "I"):
            votes.append(grade)
        else:
            votes.append("C" if j.get("passed") else "I")
    return ",".join(votes)


def format_criteria_details(criteria_results: list[dict]) -> str:
    lines: list[str] = ["\n=== Criteria Details ===", ""]

    for idx, r in enumerate(criteria_results):
        semantic_prompt = str(r.get("semanticPrompt") or "").strip()
        name = r.get("name") or f"criterion_{idx + 1}"
        grader_type = "ExactMatch" if _is_exactmatch_result(r) else "llm-judge"
        awarded = r.get("awarded", 0.0)

        criteria_label = semantic_prompt or name
        lines.append(f"criteria: {criteria_label}")
        lines.append(f"grader_type: {grader_type}")
        lines.append(f"score: {awarded}")
        lines.append("details:")

        if grader_type == "ExactMatch":
            explanation = ""
            judges = r.get("judges")
            if isinstance(judges, list) and judges:
                explanation = str(judges[0].get("explanation") or "")
            lines.append(f"  explanation: {explanation}")
        else:
            lines.append(f"  votes: {_votes_string(r)}")

        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def write_reward(
    reward_path: str,
    task_id: str,
    total_score_raw: float,
    pass_threshold: int | float,
    criteria_results: list[dict],
) -> None:
    total_score = normalize_score(total_score_raw)

    reward: dict[str, float | int] = {
        # Harbor's viewer expects the primary scalar to be under the "reward" key.
        "reward": total_score,
    }

    path = Path(reward_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(reward, fh, indent=2)

    detail = {
        "task_id": task_id,
        "total_score": total_score,
        "total_score_raw": total_score_raw,
        "pass_threshold": pass_threshold,
        "criteria_results": criteria_results,
    }
    detail_path = Path(reward_path).parent / "portex_detail.json"
    with open(detail_path, "w", encoding="utf-8") as fh:
        json.dump(detail, fh, indent=2)


def write_error_reward(reward_path: str, error_message: str) -> None:
    # Keep reward.json Harbor-compatible (exactly one numeric key).
    reward = {"reward": 0.0}
    path = Path(reward_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(reward, fh, indent=2)

    # Write error detail alongside reward.json for debugging.
    detail_path = Path(reward_path).parent / "portex_detail.json"
    with open(detail_path, "w", encoding="utf-8") as fh:
        json.dump({"error": error_message}, fh, indent=2)


def run_grading(
    task_config_path: str,
    criteria_path: str,
    reward_path: str,
    api_key: str,
) -> None:
    task_config = load_task_config(task_config_path)
    criteria = load_criteria(task_config, criteria_path)

    submission_path = task_config.get("submission_path", "/app/answer.txt")
    if not Path(submission_path).exists():
        write_error_reward(reward_path, f"Submission not found at {submission_path}")
        sys.exit(1)
    submission = Path(submission_path).read_text(encoding="utf-8").strip()

    question = task_config["question"]
    criteria_results: list[dict] = []
    needs_llm = any(
        (criterion.get("grader_type") or "llm-judge").lower() != "exactmatch"
        for criterion in criteria
    )
    if needs_llm and not api_key:
        write_error_reward(reward_path, "OPENROUTER_API_KEY required for llm-judge criteria")
        sys.exit(1)
    judges = resolve_judge_models(task_config) if needs_llm else []

    # Grade each criterion: ExactMatch inline; LLM-judge in batches of CRITERIA_BATCH_SIZE
    criteria_results: list[dict | None] = [None] * len(criteria)
    llm_batch: list[tuple[int, dict]] = []

    for i, criterion in enumerate(criteria):
        grader_type = (criterion.get("grader_type") or "llm-judge").lower()
        if grader_type == "exactmatch":
            criteria_results[i] = grade_criterion_exact_match(criterion, submission)
        else:
            llm_batch.append((i, criterion))

    # Process LLM criteria: up to CRITERIA_BATCH_SIZE at a time, all judges in parallel per criterion
    for batch_start in range(0, len(llm_batch), CRITERIA_BATCH_SIZE):
        batch = llm_batch[batch_start : batch_start + CRITERIA_BATCH_SIZE]
        tasks = [
            (idx, c, question, submission, model, api_key)
            for (idx, c) in batch
            for model in judges
        ]
        judge_outputs: list[tuple[int, dict, dict]] = []
        max_workers = min(len(tasks), CRITERIA_BATCH_SIZE * max(len(judges), 1))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_task = {
                executor.submit(
                    _call_one_judge, idx, c, question, submission, model, api_key
                ): (idx, c)
                for (idx, c, _q, _s, model, _key) in tasks
            }
            for future in as_completed(future_to_task):
                idx, c = future_to_task[future]
                try:
                    out = future.result()
                    judge_outputs.append(out)
                except Exception as exc:  # noqa: BLE001
                    print(f"Error in judge task for criterion {c.get('name')}: {exc}")
                    judge_outputs.append(
                        (
                            idx,
                            c,
                            {
                                "model": "?",
                                "grade": "I",
                                "passed": False,
                                "explanation": f"Error: {exc}",
                            },
                        )
                    )
        # Group by criterion index and run majority_vote
        by_index: dict[int, list[dict]] = {}
        for idx, _c, judge_result in judge_outputs:
            by_index.setdefault(idx, []).append(judge_result)
        for idx, criterion in batch:
            weight = float(criterion.get("weight", 0))
            criterion_prompt = (
                criterion.get("semanticPrompt")
                or criterion.get("description")
                or criterion.get("name")
                or ""
            )
            judge_results = by_index.get(idx, [])
            passed, awarded = majority_vote(judge_results, weight)
            criteria_results[idx] = {
                "criterion_id": criterion.get("id"),
                "name": criterion.get("name"),
                "semanticPrompt": criterion_prompt,
                "grader_type": "llm-judge",
                "weight": weight,
                "passed": passed,
                "awarded": awarded,
                "judges": judge_results,
            }

    results: list[dict] = [r for r in criteria_results if r is not None]

    total_score_raw = aggregate_scores(results)
    pass_threshold = task_config.get("pass_threshold", 100)

    write_reward(
        reward_path=reward_path,
        task_id=task_config["task_id"],
        total_score_raw=total_score_raw,
        pass_threshold=pass_threshold,
        criteria_results=results,
    )

    print(
        f"Grading complete. Total score: {total_score_raw}/100 "
        f"(normalized: {normalize_score(total_score_raw)})"
    )

    print(format_criteria_details(results))


def main() -> None:
    task_config_path = str(TESTS_DIR / "task_config.json")
    criteria_path = str(TESTS_DIR / "criteria.json")
    reward_path = str(REWARD_PATH)
    api_key = os.environ.get("OPENROUTER_API_KEY", "")

    try:
        run_grading(
            task_config_path=task_config_path,
            criteria_path=criteria_path,
            reward_path=reward_path,
            api_key=api_key,
        )
    except SystemExit:
        raise
    except Exception as exc:  # noqa: BLE001
        write_error_reward(reward_path, str(exc))
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
