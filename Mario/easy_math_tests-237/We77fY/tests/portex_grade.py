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
TESTS_DIR = Path("/tests")

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

# Pre-combine template + instructions so user-supplied values with curly
# braces (e.g. "{x}") never interfere with Python string formatting.
_GRADING_PROMPT = GRADING_TEMPLATE.replace("{instructions}", GRADING_INSTRUCTIONS)


# ---------------------------------------------------------------------------
# Prompt formatting & response parsing
# ---------------------------------------------------------------------------

def format_grading_prompt(question: str, answer: str, criterion: str) -> str:
    """Build the full grading prompt for an LLM judge.

    Uses safe string replacement so that special characters (including curly
    braces) in *question*, *answer*, or *criterion* are preserved verbatim.
    """
    prompt = _GRADING_PROMPT
    prompt = prompt.replace("{question}", question)
    prompt = prompt.replace("{answer}", answer)
    prompt = prompt.replace("{criterion}", criterion)
    return prompt


def parse_grade_from_response(response_text: str) -> tuple[bool, str]:
    """Extract ``GRADE: C`` or ``GRADE: I`` from a judge's response.

    Returns:
        ``(passed, grade_letter)`` where *passed* is ``True`` when the grade
        letter is ``C`` (correct).  Defaults to ``(False, "I")`` when no
        grade line is found.
    """
    match = re.search(r"GRADE:\s*([CI])", response_text, re.IGNORECASE)
    if match:
        grade = match.group(1).upper()
        return grade == "C", grade
    return False, "I"


# ---------------------------------------------------------------------------
# Configuration loading
# ---------------------------------------------------------------------------

def load_task_config(config_path: str) -> dict:
    """Read and return ``task_config.json``, applying defaults for optional fields."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Task config not found: {config_path}")
    with open(path, encoding="utf-8") as fh:
        config: dict = json.load(fh)
    config.setdefault("submission_path", "/app/answer.txt")
    config.setdefault("pass_threshold", 100)
    return config


def load_criteria(task_config: dict, criteria_path: str) -> list[dict]:
    """Load criteria from *criteria_path*, generating a default if the array is empty.

    The default criterion mirrors portex-eval's ``_default_criteria()``
    behaviour: a single semantic criterion whose prompt is the reference
    answer text from the task config.
    """
    path = Path(criteria_path)
    if path.exists():
        with open(path, encoding="utf-8") as fh:
            criteria: list = json.load(fh)
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
    """Determine which judge models to use.

    Priority: ``task_config["judge_models"]`` > ``PORTEX_JUDGE_MODELS`` env
    var (comma-separated) > ``DEFAULT_JUDGE_MODELS``.
    """
    if task_config.get("judge_models"):
        return task_config["judge_models"]
    env_models = os.environ.get("PORTEX_JUDGE_MODELS", "")
    if env_models:
        return [m.strip() for m in env_models.split(",") if m.strip()]
    return list(DEFAULT_JUDGE_MODELS)


# ---------------------------------------------------------------------------
# OpenRouter HTTP client with retry
# ---------------------------------------------------------------------------

def calculate_delay(attempt: int, retry_after: float | None = None) -> float:
    """Exponential back-off with jitter (mirrors portex-eval openrouter.py)."""
    if retry_after is not None:
        return min(retry_after, MAX_DELAY)
    delay = min(BASE_DELAY * (2 ** attempt), MAX_DELAY)
    jitter_amount = delay * JITTER * random.random()
    return delay + jitter_amount


def call_openrouter(model_id: str, prompt: str, api_key: str) -> str:
    """POST a chat-completion request to OpenRouter and return the response text.

    Retries up to ``MAX_RETRIES`` times on HTTP 429 (rate-limit) using
    exponential back-off.  Raises ``RuntimeError`` if all retries are
    exhausted.
    """
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://portex.ai",
        "X-Title": "portex-eval",
    }
    payload = {
        "model": model_id,
        "messages": [{"role": "user", "content": prompt}],
    }

    last_error: Exception | None = None
    for attempt in range(MAX_RETRIES):
        try:
            with httpx.Client(timeout=REQUEST_TIMEOUT) as client:
                response = client.post(
                    OPENROUTER_API_URL,
                    json=payload,
                    headers=headers,
                )

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

    raise RuntimeError(
        f"Max retries ({MAX_RETRIES}) exceeded for OpenRouter API"
    ) from last_error


# ---------------------------------------------------------------------------
# Grading logic
# ---------------------------------------------------------------------------

def grade_criterion(
    criterion: dict,
    question: str,
    submission: str,
    judges: list[str],
    api_key: str,
) -> dict:
    """Grade a single criterion across all *judges* via OpenRouter.

    Returns a result dict suitable for inclusion in the ``criteria_results``
    list of ``reward.json``.
    """
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
            judge_results.append({
                "model": model,
                "grade": grade,
                "passed": passed,
                "explanation": response_text,
            })
        except Exception as exc:  # noqa: BLE001
            print(f"Error grading with {model}: {exc}")
            judge_results.append({
                "model": model,
                "grade": "I",
                "passed": False,
                "explanation": f"Error: {exc}",
            })

    passed, awarded = majority_vote(judge_results, weight)

    return {
        "criterion_id": criterion.get("id"),
        "name": criterion.get("name"),
        "weight": weight,
        "passed": passed,
        "awarded": awarded,
        "judges": judge_results,
    }


def majority_vote(judge_results: list[dict], weight: float) -> tuple[bool, float]:
    """Return ``(passed, awarded)`` based on strict-majority voting.

    A criterion passes only when **more than half** of the judges mark it as
    correct.  On a tie (e.g. 2/4) the criterion does **not** pass.
    """
    passed_count = sum(1 for r in judge_results if r["passed"])
    majority_passed = passed_count > len(judge_results) / 2
    awarded = weight if majority_passed else 0.0
    return majority_passed, awarded


# ---------------------------------------------------------------------------
# Scoring helpers
# ---------------------------------------------------------------------------

def normalize_score(total_score_raw: float) -> float:
    """Convert a 0-100 raw score to a 0.0-1.0 Harbor-compatible score."""
    return total_score_raw / 100.0


def check_passed(total_score_raw: float, pass_threshold: int | float) -> bool:
    """Return ``True`` when *total_score_raw* meets the *pass_threshold*."""
    return total_score_raw >= pass_threshold


def aggregate_scores(criteria_results: list[dict]) -> float:
    """Sum ``awarded`` values across all criteria results."""
    return sum(r["awarded"] for r in criteria_results)


# ---------------------------------------------------------------------------
# Reward file writers
# ---------------------------------------------------------------------------

def write_reward(
    reward_path: str,
    task_id: str,
    total_score_raw: float,
    pass_threshold: int | float,
    criteria_results: list[dict],
) -> None:
    """Write Harbor-compatible reward + detailed grading sidecar JSON."""
    total_score = normalize_score(total_score_raw)

    # Harbor verifier expects numeric-only reward map values.
    reward: dict[str, float | int] = {
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
    """Write Harbor-compatible error reward + sidecar error detail."""
    reward = {"reward": 0.0}

    path = Path(reward_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(reward, fh, indent=2)

    detail_path = Path(reward_path).parent / "portex_detail.json"
    with open(detail_path, "w", encoding="utf-8") as fh:
        json.dump({"error": error_message}, fh, indent=2)


# ---------------------------------------------------------------------------
# Top-level orchestration
# ---------------------------------------------------------------------------

def run_grading(
    task_config_path: str,
    criteria_path: str,
    reward_path: str,
    api_key: str,
) -> None:
    """Execute the full grading pipeline.

    1. Load task config and criteria
    2. Read the agent's submission
    3. Grade each criterion across all judges
    4. Aggregate scores and write ``reward.json``
    """
    # 1. Load config & criteria
    task_config = load_task_config(task_config_path)
    criteria = load_criteria(task_config, criteria_path)

    # 2. Read submission
    submission_path = task_config.get("submission_path", "/app/answer.txt")
    if not Path(submission_path).exists():
        write_error_reward(reward_path, f"Submission not found at {submission_path}")
        sys.exit(1)
    submission = Path(submission_path).read_text(encoding="utf-8").strip()

    # 3. Resolve judges
    judges = resolve_judge_models(task_config)

    # 4. Grade each criterion
    question = task_config["question"]
    criteria_results: list[dict] = []
    for criterion in criteria:
        result = grade_criterion(criterion, question, submission, judges, api_key)
        criteria_results.append(result)

    # 5. Aggregate & write
    total_score_raw = aggregate_scores(criteria_results)
    pass_threshold = task_config.get("pass_threshold", 100)

    write_reward(
        reward_path=reward_path,
        task_id=task_config["task_id"],
        total_score_raw=total_score_raw,
        pass_threshold=pass_threshold,
        criteria_results=criteria_results,
    )

    print(
        f"Grading complete. Total score: {total_score_raw}/100 "
        f"(normalized: {normalize_score(total_score_raw)})"
    )


def main() -> None:
    """Entry point when run as ``uv run portex_grade.py``."""
    task_config_path = str(TESTS_DIR / "task_config.json")
    criteria_path = str(TESTS_DIR / "criteria.json")
    reward_path = str(REWARD_PATH)

    api_key = os.environ.get("OPENROUTER_API_KEY", "")
    if not api_key:
        write_error_reward(reward_path, "OPENROUTER_API_KEY not set")
        print("ERROR: OPENROUTER_API_KEY not set", file=sys.stderr)
        sys.exit(1)

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
