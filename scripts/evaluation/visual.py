from __future__ import annotations

import base64
import io
import json
import re
from pathlib import Path
from typing import Any

import imageio.v2 as imageio
from PIL import Image

from worldsolver.task_validation import DEFAULT_VISUAL_JUDGE_FRAME_COUNT

VISUAL_JUDGE_FRAME_COUNT = DEFAULT_VISUAL_JUDGE_FRAME_COUNT


def _extract_frames(
    video_path: Path, sample_count: int = VISUAL_JUDGE_FRAME_COUNT
) -> list[bytes]:
    """Return ``sample_count`` evenly spaced JPEG frames in chronological order."""
    reader = imageio.get_reader(video_path)
    try:
        metadata = reader.get_meta_data()
        raw_frame_count = metadata.get("nframes", 0)
        frame_count = (
            int(raw_frame_count)
            if isinstance(raw_frame_count, int | float)
            and raw_frame_count > 0
            and raw_frame_count < 1_000_000
            else 0
        )
        if frame_count <= 0 or frame_count > 1_000_000:
            frame_count = sum(1 for _ in reader)
            reader.close()
            reader = imageio.get_reader(video_path)
        indices = {
            round(index * (frame_count - 1) / max(sample_count - 1, 1))
            for index in range(sample_count)
        }
        panels: list[Image.Image] = []
        for index, frame in enumerate(reader):
            if index not in indices:
                continue
            panel = Image.fromarray(frame).convert("RGB")
            panel.thumbnail((768, 768))
            panels.append(panel)
        if not panels:
            raise ValueError("video contains no decodable frames")
        encoded: list[bytes] = []
        for panel in panels:
            buffer = io.BytesIO()
            panel.save(buffer, format="JPEG", quality=85)
            encoded.append(buffer.getvalue())
        return encoded
    finally:
        reader.close()


def _data_url(data: bytes) -> str:
    return "data:image/jpeg;base64," + base64.b64encode(data).decode("ascii")


# Reasoning-capable models emit a thinking preamble (e.g. ``<think>...</think>``)
# before the JSON payload; strip it and locate the JSON object robustly so a
# preamble or surrounding prose cannot break parsing.
_THINK_PREAMBLE = re.compile(
    r"<think>.*?</think>|<reasoning>.*?</reasoning>", re.DOTALL | re.IGNORECASE
)


def _parse_json(text: str) -> dict[str, Any]:
    cleaned = _THINK_PREAMBLE.sub("", text).strip()
    fenced = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", cleaned, re.DOTALL)
    if fenced:
        cleaned = fenced.group(1).strip()
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start == -1 or end < start:
        raise ValueError(
            f"visual judge response contains no JSON object: {cleaned[:200]!r}"
        )
    value = json.loads(cleaned[start : end + 1])
    if not isinstance(value, dict):
        raise ValueError("visual judge response must be a JSON object")
    return value


def build_user_prompt(rubric: list[dict[str, Any]]) -> str:
    """Scoring instructions appended after the ordered frame images."""
    return f"""The image blocks above are video frames in chronological order, each labeled \
(frame i/N).

Evaluate every rubric item below.

For each item return criterion_id, score (number from 0 to 1, where 1 is best
and 0 is worst), and a concise evidence statement. Penalize missing
tank/subject visibility, discontinuity, and absent required events.

Rubric:
{json.dumps(rubric, indent=2)}

Required JSON shape:
{{"criteria":[{{"criterion_id":"...","score":0.0,"evidence":"..."}}]}}
"""


def _judge_frames(
    model_config: dict[str, Any],
    system: str,
    content: list[dict[str, Any]],
    *,
    timeout_seconds: int,
    max_tokens: int = 2048,
) -> str:
    """Send one-shot multimodal prompt to the configured endpoint, return raw text.

    Uses ``/v1/chat/completions`` with OpenAI ``image_url`` content blocks.
    The selected visual judge model must support this multimodal chat wire.
    """
    from openai import OpenAI

    base_url = model_config.get("base_url")
    api_key = model_config.get("api_key")
    model = model_config.get("model")
    if not base_url or not api_key or not model:
        raise ValueError("visual judging requires base_url, api_key and model in the model config")
    try:
        client = OpenAI(base_url=base_url, api_key=api_key, timeout=timeout_seconds)
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": content},
            ],
            temperature=0.0,
            max_tokens=max_tokens,
        )
    except Exception as exc:
        raise RuntimeError(f"visual judge API call failed: {exc}") from exc
    if isinstance(response, str):
        # Some OpenAI-compatible gateways return the assistant payload directly
        # instead of a ChatCompletion object.
        text = response
        try:
            payload = json.loads(response)
        except json.JSONDecodeError:
            payload = None
        if isinstance(payload, str):
            text = payload
        elif isinstance(payload, dict) and "choices" in payload:
            text = str(payload["choices"][0]["message"].get("content", ""))
    else:
        text = response.choices[0].message.content or ""
    if not text:
        raise RuntimeError("visual judge returned an empty response")
    return text


def evaluate_visual_criteria(
    video_path: Path,
    criteria: list[dict[str, Any]],
    model_config: dict[str, Any],
    *,
    frame_count: int = VISUAL_JUDGE_FRAME_COUNT,
) -> list[dict[str, Any]]:
    frames = _extract_frames(video_path, frame_count)
    frame_count = len(frames)
    rubric = [
        {
            "criterion_id": item["id"],
            "dimension": item["dimension"],
            "criterion": item["criterion"],
        }
        for item in criteria
    ]
    system = (
        "You are a strict visual benchmark judge. Inspect the attached frames in "
        "chronological reading order. Judge only visible evidence. Return JSON only."
    )
    content: list[dict[str, Any]] = []
    for index, frame in enumerate(frames):
        content.append({"type": "image_url", "image_url": {"url": _data_url(frame)}})
        content.append({"type": "text", "text": f"(frame {index + 1}/{frame_count})"})
    content.append({"type": "text", "text": build_user_prompt(rubric)})

    response_text = _judge_frames(model_config, system, content, timeout_seconds=300)
    response = _parse_json(response_text)

    by_id = {item["criterion_id"]: item for item in response.get("criteria", [])}
    results = []
    for criterion in criteria:
        item = by_id.get(criterion["id"])
        if not isinstance(item, dict):
            raise ValueError(f"visual judge omitted criterion {criterion['id']!r}")
        score = min(1.0, max(0.0, float(item["score"])))
        results.append(
            {
                "criterion_id": criterion["id"],
                "domain": "visual",
                "score": score,
                "details": {"evidence": str(item.get("evidence", ""))},
            }
        )
    return results
