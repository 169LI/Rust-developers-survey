import json
from pathlib import Path
from typing import List, Dict, Any
import numpy as np

def _normalize_for_embedding(s: str) -> str:
    # Preserve semantics as much as possible, but replace underscores/hyphens with spaces for SBERT to identify phrases
    return str(s).replace("_", " ").replace("-", " ").strip()

def semantic_cluster_uncategorized(
    groups_path: str,
    counts_path: str,
    output_path: str,
    model_name: str = "all-MiniLM-L6-v2",
    min_cluster_size: int = 5,
    min_samples: int = 1,
) -> None:
    """
    Perform SBERT + HDBSCAN clustering on 'Uncategorized' tags and output for manual review.
    - groups_path: Grouped JSON (containing 'Uncategorized': {tag: count})
    - counts_path: Original count file, used for fallback or supplementary information
    - output_path: Clustering result output JSON
    - model_name: SBERT model name (default all-MiniLM-L6-v2)
    - min_cluster_size/min_samples: HDBSCAN parameters, used to control cluster size and stability
    """
    try:
        from sentence_transformers import SentenceTransformer
        import hdbscan
    except ImportError as e:
        raise RuntimeError(
            "Missing dependencies, please install: pip install sentence-transformers hdbscan"
        ) from e

    gp = Path(groups_path)
    if not gp.exists():
        raise RuntimeError(f"groups_path does not exist: {groups_path}")

    # Read grouped results, extract uncategorized tags
    groups = json.loads(gp.read_text(encoding="utf-8"))
    uncategorized = groups.get("Uncategorized", {})
    if not isinstance(uncategorized, dict) or len(uncategorized) == 0:
        # No uncategorized data, generate empty output
        _write_output(
            output_path,
            {
                "model": model_name,
                "params": {"min_cluster_size": min_cluster_size, "min_samples": min_samples},
                "clusters": [],
                "noise": [],
                "review_note": "No Uncategorized tags to cluster.",
            },
        )
        return

    # Tags and counts
    tags: List[str] = list(uncategorized.keys())
    counts: Dict[str, int] = {t: int(uncategorized[t]) for t in tags}

    # Vectorization (mild normalization for phrases)
    norm_texts = [_normalize_for_embedding(t) for t in tags]
    model = SentenceTransformer(model_name)
    embeddings = model.encode(
        norm_texts,
        batch_size=64,
        convert_to_numpy=True,
        normalize_embeddings=True,  # Normalized Euclidean distance approximates Cosine distance
    )

    # HDBSCAN Clustering
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric="euclidean",
        cluster_selection_method="eom",
        prediction_data=True,
    )
    clusterer.fit(embeddings)
    labels = clusterer.labels_  # -1 indicates noise

    # Build cluster mapping
    clusters: Dict[int, List[int]] = {}
    noise_indices: List[int] = []
    for i, lbl in enumerate(labels):
        if lbl == -1:
            noise_indices.append(i)
        else:
            clusters.setdefault(lbl, []).append(i)

    # Calculate representatives (close to cluster center + high count)
    cluster_results: List[Dict[str, Any]] = []
    for cid, idxs in sorted(clusters.items()):
        member_tags = [tags[i] for i in idxs]
        member_counts = [counts[tags[i]] for i in idxs]
        member_embeds = embeddings[idxs]

        # Cluster center (mean)
        center = np.mean(member_embeds, axis=0)
        dists = np.linalg.norm(member_embeds - center, axis=1)
        # Top by center proximity
        top_center = [member_tags[i] for i in np.argsort(dists)[:5]]
        # Top by count
        top_count = [t for _, t in sorted(zip(member_counts, member_tags), key=lambda x: -x[0])[:5]]

        # Assemble output
        cluster_results.append(
            {
                "cluster_id": int(cid),
                "size": len(member_tags),
                "tags": [{"tag": t, "count": counts[t]} for t in sorted(member_tags)],
                "representatives": {
                    "top_by_center": top_center,
                    "top_by_count": top_count,
                },
                "name": "REVIEW_REQUIRED",  # Fill in name after manual review
            }
        )

    noise_tags = [tags[i] for i in noise_indices]
    output_obj = {
        "model": model_name,
        "params": {"min_cluster_size": min_cluster_size, "min_samples": min_samples},
        "clusters": cluster_results,
        "noise": [{"tag": t, "count": counts[t]} for t in sorted(noise_tags)],
        "review_note": "Please manually review the representatives of each cluster and assign theme names.",
    }
    _write_output(output_path, output_obj)

def _write_output(output_path: str, obj: Dict[str, Any]) -> None:
    op = Path(output_path)
    op.parent.mkdir(parents=True, exist_ok=True)
    with open(op, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
