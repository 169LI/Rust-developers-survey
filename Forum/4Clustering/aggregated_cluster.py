import json
from pathlib import Path
from typing import List, Dict, Any
import numpy as np


def _normalize_for_embedding(s: str) -> str:
    # Preserve semantics, replace underscores/hyphens with spaces
    return str(s).replace("_", " ").replace("-", " ").strip()


def _write_output(output_path: str, obj: Dict[str, Any]) -> None:
    op = Path(output_path)
    op.parent.mkdir(parents=True, exist_ok=True)
    with open(op, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def _sum_counts(dicts: List[Dict[str, int]]) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for d in dicts:
        for k, v in d.items():
            out[k] = out.get(k, 0) + int(v)
    return out


def cluster_from_grouped_categories(
    groups_paths: List[str],
    aggregate_def: Dict[str, List[str]],
    output_paths: Dict[str, str],
    model_name: str = "all-MiniLM-L6-v2",
    min_cluster_size: int = 5,
    min_samples: int = 1,
    min_count: int = 1,
) -> None:
    """
    Perform cross-file aggregation and semantic clustering on "categorized" tags.
    - groups_paths: Paths to two or more grouped JSON files (each file structure is {theme: {tag: count}, ...})
    - aggregate_def: Aggregation definition, e.g. {"group1": ["Core Language Concepts", ...], ...}
    - output_paths: Output file path for each aggregation group, e.g. {"group1": "...group1_clusters.json", ...}
    - model_name: SBERT model name (default all-MiniLM-L6-v2 for English data)
    - min_cluster_size/min_samples: HDBSCAN parameters
    """
    try:
        from sentence_transformers import SentenceTransformer
        import hdbscan
    except ImportError as e:
        raise RuntimeError(
            "Missing dependencies, please install: pip install sentence-transformers hdbscan"
        ) from e

    # Read all grouped files
    grouped_files: List[Dict[str, Dict[str, int]]] = []
    for p in groups_paths:
        gp = Path(p)
        if not gp.exists():
            raise RuntimeError(f"groups_path does not exist: {p}")
        grouped_files.append(json.loads(gp.read_text(encoding="utf-8")))

    # Preload model (avoid repeated loading)
    model = SentenceTransformer(model_name)

    # Process each aggregation group sequentially
    for gname, themes in aggregate_def.items():
        # Aggregate {tag: count} for specified themes in all files
        per_file_theme_counts: List[Dict[str, int]] = []
        for gf in grouped_files:
            acc: Dict[str, int] = {}
            for theme in themes:
                d = gf.get(theme, {})
                if isinstance(d, dict):
                    for k, v in d.items():
                        acc[k] = acc.get(k, 0) + int(v)
            per_file_theme_counts.append(acc)
        # Merge into a total count
        counts = _sum_counts(per_file_theme_counts)
        # Low frequency tag filtering to reduce noise (e.g. rarely occurring isolated tags)
        if min_count > 1:
            counts = {k: v for k, v in counts.items() if v >= min_count}

        tags: List[str] = list(counts.keys())
        if len(tags) == 0:
            _write_output(
                output_paths[gname],
                {
                    "model": model_name,
                    "params": {"min_cluster_size": min_cluster_size, "min_samples": min_samples},
                    "clusters": [],
                    "noise": [],
                    "aggregate_group": gname,
                    "aggregate_themes": themes,
                    "review_note": "No categorized tags for this aggregate group.",
                },
            )
            continue

        # Vectorization
        norm_texts = [_normalize_for_embedding(t) for t in tags]
        embeddings = model.encode(
            norm_texts,
            batch_size=64,
            convert_to_numpy=True,
            normalize_embeddings=True,
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
        labels = clusterer.labels_

        # Build clusters
        clusters: Dict[int, List[int]] = {}
        noise_indices: List[int] = []
        for i, lbl in enumerate(labels):
            if lbl == -1:
                noise_indices.append(i)
            else:
                clusters.setdefault(int(lbl), []).append(i)

        # Representatives and Output
        cluster_results: List[Dict[str, Any]] = []
        for cid, idxs in sorted(clusters.items()):
            member_tags = [tags[i] for i in idxs]
            member_counts = [counts[tags[i]] for i in idxs]
            member_embeds = embeddings[idxs]

            center = np.mean(member_embeds, axis=0)
            dists = np.linalg.norm(member_embeds - center, axis=1)
            top_center = [member_tags[i] for i in np.argsort(dists)[:5]]
            top_count = [t for _, t in sorted(zip(member_counts, member_tags), key=lambda x: -x[0])[:5]]

            cluster_results.append(
                {
                    "cluster_id": int(cid),
                    "size": len(member_tags),
                    "tags": [{"tag": t, "count": counts[t]} for t in sorted(member_tags)],
                    "representatives": {
                        "top_by_center": top_center,
                        "top_by_count": top_count,
                    },
                    "name": "REVIEW_REQUIRED",
                }
            )

        noise_tags = [tags[i] for i in noise_indices]
        output_obj = {
            "model": model_name,
            "params": {"min_cluster_size": min_cluster_size, "min_samples": min_samples},
            "clusters": cluster_results,
            "noise": [{"tag": t, "count": counts[t]} for t in sorted(noise_tags)],
            "aggregate_group": gname,
            "aggregate_themes": themes,
            "review_note": "Please review the representatives of each cluster and name the theme.",
        }
        _write_output(output_paths[gname], output_obj)
