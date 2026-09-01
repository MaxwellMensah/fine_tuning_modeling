import json
import math
import re

import ollama
import pandas as pd
from tqdm import tqdm

# Configuration
BASE_MODEL = "llama3.2:latest"
FT_MODEL = "fraud-model-v7"
EMBED_MODEL = "nomic-embed-text:latest"
VAL_DATASET = "val_600.jsonl"
SAMPLE_COUNT = 600


def parse_risk_label(text: str) -> str:
    """Extracts final risk classification for exact-match scoring."""
    if not isinstance(text, str):
        return "UNKNOWN"

    patterns = [
        r"FINAL ASSESSMENT:?\s*(HIGH|MEDIUM|LOW)",
        r"RISK ASSESSMENT:?\s*(HIGH|MEDIUM|LOW)",
        r"(HIGH|MEDIUM|LOW)\s*RISK",
        r"(HIGH|MEDIUM|LOW)\s*\((FLAGGED|REVIEW|APPROVED)\)",
    ]
    for p in patterns:
        m = re.search(p, text.upper())
        if m:
            return m.group(1)
    return "UNKNOWN"


def get_embedding(text: str) -> list[float]:
    """Retrieves text embedding vector using Ollama native API."""
    try:
        res = ollama.embeddings(model=EMBED_MODEL, prompt=text)
        return res["embedding"]
    except (ollama.ResponseError, KeyError, ValueError) as e:
        print(
            f"\n[Warning] Embedding call failed for '{EMBED_MODEL}': {e}. Returning zero vector."
        )
        return []


def cosine_similarity(vec1: list[float], vec2: list[float]) -> float:
    """Computes cosine similarity score between two embedding vectors (0.0 to 1.0)."""
    if not vec1 or not vec2 or len(vec1) != len(vec2):
        return 0.0

    dot_prod = sum(a * b for a, b in zip(vec1, vec2))
    norm_a = math.sqrt(sum(a * a for a in vec1))
    norm_b = math.sqrt(sum(b * b for b in vec2))

    if norm_a == 0 or norm_b == 0:
        return 0.0

    similarity = dot_prod / (norm_a * norm_b)
    return max(0.0, float(similarity))  # Clamp negative values to 0.0


def run_inference(model_name: str, sys_msg: str, user_msg: str) -> str:
    """Invokes model inference via Ollama chat endpoint."""
    messages = []
    if sys_msg:
        messages.append({"role": "system", "content": sys_msg})
    messages.append({"role": "user", "content": user_msg})

    res = ollama.chat(model=model_name, messages=messages, options={"temperature": 0.1})
    return res["message"]["content"]


def main():
    # 1. Load evaluation dataset
    val_samples = []
    with open(VAL_DATASET, "r") as f:
        for idx, line in enumerate(f):
            if idx >= SAMPLE_COUNT:
                break
            val_samples.append(json.loads(line))

    print(f"Loaded {len(val_samples)} evaluation samples from {VAL_DATASET}.\n")

    results = []

    # 2. Evaluation Loop
    for idx, item in enumerate(
        tqdm(val_samples, desc="Evaluating Base vs Fine-Tuned Model")
    ):
        messages = item["messages"]
        sys_msg = next((m["content"] for m in messages if m["role"] == "system"), "")
        user_msg = next((m["content"] for m in messages if m["role"] == "user"), "")
        ground_truth = next(
            (m["content"] for m in messages if m["role"] == "assistant"), ""
        )

        gt_label = parse_risk_label(ground_truth)
        gt_emb = get_embedding(ground_truth)

        # Base Model Evaluation
        base_pred = run_inference(BASE_MODEL, sys_msg, user_msg)
        base_label = parse_risk_label(base_pred)
        base_exact_match = (
            1.0 if (base_label == gt_label and gt_label != "UNKNOWN") else 0.0
        )
        base_sim = cosine_similarity(get_embedding(base_pred), gt_emb)

        # Fine-Tuned Model Evaluation
        ft_pred = run_inference(FT_MODEL, sys_msg, user_msg)
        ft_label = parse_risk_label(ft_pred)
        ft_exact_match = (
            1.0 if (ft_label == gt_label and gt_label != "UNKNOWN") else 0.0
        )
        ft_sim = cosine_similarity(get_embedding(ft_pred), gt_emb)

        results.append(
            {
                "id": idx,
                "user_query": user_msg,
                "ground_truth": ground_truth,
                "gt_label": gt_label,
                "base_prediction": base_pred,
                "base_label": base_label,
                "base_exact_match": base_exact_match,
                "base_semantic_similarity": round(base_sim, 4),
                "ft_prediction": ft_pred,
                "ft_label": ft_label,
                "ft_exact_match": ft_exact_match,
                "ft_semantic_similarity": round(ft_sim, 4),
            }
        )

    # Deliverable CSV
    df = pd.DataFrame(results)
    df.to_csv("eval_results.csv", index=False)

    # Print Summary Deliverable Report
    base_acc = df["base_exact_match"].mean() * 100
    base_sim_avg = df["base_semantic_similarity"].mean()
    ft_acc = df["ft_exact_match"].mean() * 100
    ft_sim_avg = df["ft_semantic_similarity"].mean()

    print("\n" + "=" * 50)
    print("         EVALUATION SUMMARY HARNESS        ")
    print("=" * 50)
    print(
        f"Base Model ({BASE_MODEL})   | Exact Match: {base_acc:5.1f}% | Semantic Sim: {base_sim_avg:.4f}"
    )
    print(
        f"Fine-Tuned ({FT_MODEL})   | Exact Match: {ft_acc:5.1f}% | Semantic Sim: {ft_sim_avg:.4f}"
    )
    print("=" * 50)
    print("Results successfully saved to eval_results.csv and ready for commit.")


if __name__ == "__main__":
    main()
