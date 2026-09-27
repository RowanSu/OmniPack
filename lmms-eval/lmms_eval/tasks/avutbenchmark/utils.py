import datetime
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Union

import cv2
import numpy as np
import yaml
import ast
from loguru import logger as eval_logger

with open(Path(__file__).parent / "avutbenchmark.yaml", "r") as f:
    raw_data = f.readlines()
    safe_data = []
    for i, line in enumerate(raw_data):
        # remove function definition since yaml load cannot handle it
        if "!function" not in line:
            safe_data.append(line)
    config = yaml.safe_load("".join(safe_data))
hf_home = os.getenv("HF_HOME", "~/.cache/huggingface/")  # /mnt/sh/mmvision/data/videollm/benchmarks
cache_dir = os.path.join(hf_home, config["dataset_kwargs"]["cache_dir"])

def avut_doc_to_visual(doc):
    """
    Return the path to the video only.
    """
    video_path = os.path.join(cache_dir, doc["video_path"])
    if os.path.exists(video_path):
        pass
    else:
        sys.exit(f"video path:{video_path} does not exist, please check")
    return [video_path]


def _extract_candidates_by_label(raw_text):
    """
    Fallback parser for malformed candidate strings.
    Expected option format: "A. ...", "B. ...", "C. ...", "D. ...", "E. ...", "F. ...".
    """
    s = str(raw_text).replace("\n", " ")
    s = re.sub(r"\s+", " ", s).strip()
    marker_re = re.compile(r'(?:(?<=^)|(?<=[\s,\[\'"]))([A-F])\.\s*')
    matches = list(marker_re.finditer(s))
    if not matches:
        return []

    parsed = []
    seen = set()
    for i, m in enumerate(matches):
        label = m.group(1)
        if label in seen:
            continue
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(s)
        text = s[start:end].strip(" \t\r\n'\",]")
        text = re.sub(r"\s+", " ", text).strip()
        if text:
            parsed.append(f"{label}. {text}")
            seen.add(label)
    return parsed


def parse_candidates(raw):
    # already parsed
    if isinstance(raw, (list, tuple, np.ndarray)):
        return [str(x).strip() for x in raw if str(x).strip()]

    s = str(raw).strip()
    if not s:
        return []

    # normalize common malformed quote/comma patterns
    fixed = s.replace("\xa0", " ")
    fixed = re.sub(r'""+', '"', fixed)
    fixed = re.sub(r"''+", "'", fixed)
    fixed = re.sub(r"(['\"])\s+(['\"])", r"\1, \2", fixed)
    fixed = re.sub(r",\s*,+", ", ", fixed)

    try:
        parsed = ast.literal_eval(fixed)
        if isinstance(parsed, str):
            parsed = _extract_candidates_by_label(parsed)
        elif isinstance(parsed, (list, tuple, np.ndarray)):
            parsed = [str(x).strip() for x in parsed if str(x).strip()]
        else:
            parsed = []
        if parsed:
            return parsed
    except (ValueError, SyntaxError):
        pass

    parsed = _extract_candidates_by_label(fixed)
    if parsed:
        return parsed

    parsed = _extract_candidates_by_label(s)
    if parsed:
        return parsed
    return [s]


def get_candidates_and_answer(doc):
    candidates = []
    for opt in ["A", "B", "C", "D", "E", "F"]:
        key = f"option_{opt}"
        if key in doc and doc[key] is not None:
            candidates.append(f"{opt}. {doc[key]}")
    correct_answer = str(doc.get("answer", "")).strip().upper()
    return candidates, correct_answer


def avut_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    question = doc["question"]
    options, _ = get_candidates_and_answer(doc)
    options_text = "\n".join(options)
    prompt = (
        "You are given a video. Based on the content of the video, answer the following question:\n\n"
        f"Question:\n{question}\n\n"
        f"Options:\n{options_text}\n\n"
        "Answer with the option's letter directly(e.g., A, B, C, D, E, or F)."
        "If your access to the video content is limited, at least one option that is more likely "
        "than the others must be chosen."
        "Mustn't give any other reason for can not choose!"
    )
    return prompt

def extract_characters_regex(s):
    """Extract choice letter A, B, C, D, E, or F from the response."""
    s = str(s).strip()
    answer_prefixes = [
        "The best answer is",
        "The correct answer is",
        "The answer is",
        "The answer",
        "The best option is" "The correct option is",
        "Best answer:",
        "Best option:",
    ]
    for answer_prefix in answer_prefixes:
        s = s.replace(answer_prefix, "")

    # Long response with no A/B/C/D after prefix removal returns empty string.
    if len(s.split()) > 10 and not re.search("[ABCDEF]", s):
        return ""

    matches = re.search(r"\b[ABCDEF]\b|[ABCDEF]", s, re.IGNORECASE)
    if matches is None:
        return ""
    return matches[0].upper()


def _get_answer(doc):
    return str(doc.get("answer", "")).strip().upper()


def _get_question_id(doc):
    return (
        doc.get("question_id")
        or doc.get("qid")
        or doc.get("QA_id")
        or doc.get("id")
        or f"{doc.get('video') or doc.get('video_id') or doc.get('video_path')}::{doc.get('question')}"
    )


def _first_present(doc, keys):
    for key in keys:
        value = doc.get(key)
        if value is not None and value != "":
            return value
    return None


def avut_process_results(doc, results):
    """
    Args:
        doc: a instance of the eval dataset
        results: [pred]
    Returns:
        a dictionary with key: metric name (in this case avut score), value: metric value
    """
    pred = results[0]
    pred_ans = extract_characters_regex(pred)

    data_dict = {
        "question_id": _get_question_id(doc),
        "video_id": doc.get("video_id"),
        "QA_id": doc.get("QA_id"),
        "video_path": doc.get("video_path"),
        "question": doc.get("question"),
        "candidates": get_candidates_and_answer(doc)[0],
        "pred_answer": pred_ans,
        "answer": _get_answer(doc),
        "score": 1.0 if pred_ans == _get_answer(doc) else 0.0,
        "task_type": doc.get("task_type", "Unknown"),
    }
    return {f"avut_score": data_dict}


def convert_duration_to_seconds(time_str):
    """
    Converts a time string in 'MM:SS' or 'HH:MM:SS' format to total seconds.
        int: The total duration in seconds.
    """
    parts = time_str.split(':')
    if len(parts) == 2:  # MM:SS format
        minutes = int(parts[0])
        seconds = int(parts[1])
        total_seconds = minutes * 60 + seconds
    elif len(parts) == 3:  # HH:MM:SS format
        hours = int(parts[0])
        minutes = int(parts[1])
        seconds = int(parts[2])
        total_seconds = hours * 3600 + minutes * 60 + seconds
    else:
        raise ValueError("Invalid time format. Please use 'MM:SS' or 'HH:MM:SS'.")

    return total_seconds

def get_duration_type(duration):
    if duration is None:
        return "unknown"
    if isinstance(duration, str):
        duration = duration.strip()
        if duration == "":
            return "unknown"
        if ":" in duration:
            duration = convert_duration_to_seconds(duration)
        else:
            duration = float(duration)
    # Duration buckets: (0,1], (1,5], (5,10], (10,30] minutes.
    if duration <= 60:
        return "0-1min"
    elif duration <= 300:
        return "1-5min"
    elif duration <= 600:
        return "5-10min"
    elif duration <= 1800:
        return "10-30min"
    else:
        return ">30min"


def avut_aggregate_results(results):
    """
    Args:
        results: a list of values returned by process_results
    Returns:
        A score
    """
    category_specs = [
        ("task_type", "Task Types"),
    ]
    grouped_scores = {field: defaultdict(dict) for field, _ in category_specs}
    all_scores = {}

    for result in results:
        question_id = result["question_id"]
        score = result["score"]
        all_scores.setdefault(question_id, []).append(score)
        result = dict(result)

        for field, _ in category_specs:
            value = result.get(field)
            if value is None or value == "":
                continue
            grouped_scores[field][value].setdefault(question_id, []).append(score)

    for field, display_name in category_specs:
        if not grouped_scores[field]:
            continue
        for category_name, questions in grouped_scores[field].items():
            category_total = sum(scores[0] for scores in questions.values())
            avg_score = category_total / len(questions) * 100.0
            eval_logger.info(f"Evaluation on AVUT {display_name}: {category_name}: {avg_score:.2f}")

    total_score = sum(scores[0] for scores in all_scores.values())
    total_questions = len(all_scores)
    overall_avg_score = total_score / total_questions * 100.0 if total_questions else 0.0
    eval_logger.info(f"Overall performance on AVUT (across all questions): {overall_avg_score:.2f}")

    return overall_avg_score
