import csv
import json
from pathlib import Path
from typing import List, Callable

import re
import ast
from collections import OrderedDict


# ------------ Reader ------------

def read_tsv(file_path: str) -> tuple[List[str], List[List[str]]]:
    """
    Read TSV file, return (header, rows)
    - Tab-separated
    - Use double quotes as quotechar, compatible with embedded escaped double quotes
    """
    header: List[str] = []
    rows: List[List[str]] = []
    with open(file_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f, delimiter="\t", quotechar='"')
        for i, row in enumerate(reader):
            if i == 0:
                header = row
            else:
                rows.append(row)
    return header, rows
def find_column_index(header: List[str], column_name: str) -> int:
    """
    Find the index of the specified column name in the header, raise exception if not found
    """
    try:
        return header.index(column_name)
    except ValueError as e:
        raise RuntimeError(f"Column '{column_name}' not found in header: {header}") from e


# ------------ Parser (Differences for different data sources) ------------

# ------------ Parser (Enhanced boundary cleanup rules) ------------
def _strip_boundary_noise(s: str) -> str:
    """
    Only clean up abnormal characters at the beginning and end of the tag:
    - Quotes: ' " ` and Chinese quotes “ ” ‘ ’ 「 」 『 』
    - Asterisks: * and ** (delete if present, not required to be paired)
    - Invisible characters: Zero-width characters, non-breaking spaces, etc.
    Only process the beginning and end characters, keep the middle content.
    """
    if s is None:
        return ""
    s = s.replace("\u200b", "").replace("\u200c", "").replace("\u200d", "").replace("\ufeff", "").replace("\xa0", " ")
    s = s.strip()

    noise_chars = {"\\",":","：","'", '"', "`", "“", "”", "‘", "’", "«", "»", "「", "」", "『", "』", "*"}

    for _ in range(5):
        if not s:
            break
        changed = False
        if s.startswith("**"):
            s = s[2:].strip()
            changed = True
        if s.endswith("**"):
            s = s[:-2].strip()
            changed = True
        while s and s[0] in noise_chars:
            s = s[1:].strip()
            changed = True
        while s and s[-1] in noise_chars:
            s = s[:-1].strip()
            changed = True
        if not changed:
            break
    return s

def parse_json_array_str(raw: str) -> List[str]:
    """
    Parse a JSON array formatted string into a list of strings.
    Compatible with double quote duplication caused by CSV escaping: "" -> "
    Only clean up abnormal characters at the beginning and end of each tag, do not change the middle content.
    """
    if raw is None:
        return []
    normalized = raw.replace('""', '"')
    try:
        val = json.loads(normalized)
    except json.JSONDecodeError:
        normalized = normalized.strip()
        if normalized.startswith("[") and normalized.endswith("]"):
            inner = normalized[1:-1].strip()
            if not inner:
                return []
            items = [_strip_boundary_noise(s.strip()) for s in inner.split(",")]
            return [t for t in items if t]
        return []
    if isinstance(val, list):
        cleaned = [_strip_boundary_noise(str(x).strip()) for x in val]
        return [t for t in cleaned if t]
    if isinstance(val, str):
        t = _strip_boundary_noise(val.strip())
        return [t] if t else []
    return []

# ------------ Source Parser Functions (Top-level definitions for extract_all_sources) ------------
def parse_nation_china_tags(field: str) -> List[str]:
    """
    generated_tags parsing for nation_china/forum_posts.txt:
    - Originally a JSON array string, but may have "" escaping
    - Only clean up abnormal characters at the beginning and end of tags
    """
    return parse_json_array_str(field)

def parse_rust_cc_tags(field: str) -> List[str]:
    """
    generated_tags parsing for rust_cc/rust_questions.txt:
    - Common format: [""[Tag1, Tag2, Tag3]""]
    - Step 1: JSON parse to get a single string (or list)
    - Step 2: If it is a single string and contains brackets, remove brackets and split by comma
    - Step 3: Clean up abnormal characters at the beginning and end of each item (do not change middle)
    """
    arr = parse_json_array_str(field)
    if len(arr) == 1 and isinstance(arr[0], str):
        inner = arr[0].strip()
        inner = inner.lstrip('":：').lstrip(":").strip()
        if inner.startswith("[") and inner.endswith("]"):
            inner = inner[1:-1].strip()
        if not inner:
            return []
        parts = [_strip_boundary_noise(p.strip()) for p in inner.split(",")]
        return [t for t in parts if t]
    return [t for t in arr if t]

def parse_stakeoverflow_tags(field: str) -> List[str]:
    """
    generated_tags parsing for stakeoverflow/stake_questions.txt:
    - Usually standard JSON array or '[]', can be parsed directly
    - Only clean up abnormal characters at the beginning and end of tags
    """
    return parse_json_array_str(field)

# ------------ Writer ------------

def ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

def write_tags_per_line(tags_list: List[List[str]], output_path: str) -> None:
    """
    Write the tag list of each record as a JSON array per line, ensuring Chinese is not escaped.
    """
    out_path = Path(output_path)
    ensure_parent_dir(out_path)
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        for tags in tags_list:
            f.write(json.dumps(tags, ensure_ascii=False))
            f.write("\n")

# ------------ 1. Tag Extractor ------------
def extract_and_store(
    source_path: str,
    parse_fn: Callable[[str], List[str]],
    output_path: str,
    generated_tags_column: str = "generated_tags",
) -> None:
    """
    General extraction process:
    - Read TSV
    - Find generated_tags column
    - Parse and normalize row by row
    - Output to target file (one JSON array per line)
    """
    header, rows = read_tsv(source_path)
    col_idx = find_column_index(header, generated_tags_column)
    normalized_tags_per_row: List[List[str]] = []

    for row in rows:
        # Row length may be inconsistent, defensive processing
        if col_idx >= len(row):
            normalized_tags_per_row.append([])
            continue
        raw_field = row[col_idx]
        tags = parse_fn(raw_field)
        normalized_tags_per_row.append(tags)

    write_tags_per_line(normalized_tags_per_row, output_path)


# ------------ Tag Statistics ------------

def extract_all_sources():
    """
    Extract generated_tags from three data sources in the parent directory 'form',
    and write to the sibling directory 'PY_tags' as one JSON array per line file.
    """
    base_dir = Path("e:/work-study/scientific_research/Rust_survey/论坛/after_tags")
    py_tags_dir = base_dir / "PY_tags"

    # Data source paths
    nation_src = base_dir / "form" / "nation_china" / "forum_posts.txt"
    rust_cc_src = base_dir / "form" / "rust_cc" / "rust_questions.txt"
    stake_src = base_dir / "form" / "stakeoverflow" / "stake_questions.txt"

    # Output paths (sibling directory)
    nation_out = py_tags_dir / "nation_china_generated_tags.txt"
    rust_cc_out = py_tags_dir / "rust_cc_generated_tags.txt"
    stake_out = py_tags_dir / "stakeoverflow_generated_tags.txt"

    # Execute extraction and storage
    extract_and_store(str(nation_src), parse_nation_china_tags, str(nation_out))
    extract_and_store(str(rust_cc_src), parse_rust_cc_tags, str(rust_cc_out))
    extract_and_store(str(stake_src), parse_stakeoverflow_tags, str(stake_out))


def aggregate_from_generated_tags_file(input_path: str) -> dict[str, int]:
    """
    Count tag frequencies from a file with one JSON array per line.
    - Ignore empty array ([]) lines.
    - Accurately count occurrences of each tag.
    """
    tag_counts: dict[str, int] = {}
    ip = Path(input_path)
    with open(ip, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # Parse tag array for each line
            try:
                arr = json.loads(line)
            except json.JSONDecodeError:
                # Fallback: try parsing as simple comma separation
                arr = []
                t = line.strip()
                if t.startswith("[") and t.endswith("]"):
                    inner = t[1:-1].strip()
                    if inner:
                        arr = [s.strip().strip('"').strip("'") for s in inner.split(",")]
            if not isinstance(arr, list) or len(arr) == 0:
                continue
            for tag in arr:
                tag = str(tag).strip()
                if not tag:
                    continue
                tag_counts[tag] = tag_counts.get(tag, 0) + 1
    return tag_counts


def write_tag_counts(tag_counts: dict[str, int], output_path: str) -> None:
    """
    Write tag counts as a JSON object, sorted by count descending, then tag name ascending.
    """
    # Sorting only for readability
    sorted_items = sorted(tag_counts.items(), key=lambda x: (-x[1], x[0]))
    out_obj = {k: v for k, v in sorted_items}
    op = Path(output_path)
    ensure_parent_dir(op)
    with open(op, "w", encoding="utf-8") as f:
        json.dump(out_obj, f, ensure_ascii=False, indent=2)


def aggregate_and_store(input_path: str, output_path: str) -> None:
    """
    Aggregate a generated tag file and write the corresponding tag count file.
    """
    counts = aggregate_from_generated_tags_file(input_path)
    write_tag_counts(counts, output_path)

# ------------ 2. Tag Aggregator ------------
def aggregate_all_sources():
    """
    Perform tag count statistics for the three sources separately, and write to the sibling directory 'PY_tags'.
    """
    base_dir = Path("e:/work-study/scientific_research/Rust_survey/论坛/after_tags")
    py_tags_dir = base_dir / "PY_tags"

    # Input (generated in previous stage)
    nation_in = py_tags_dir / "nation_china_generated_tags.txt"
    rust_cc_in = py_tags_dir / "rust_cc_generated_tags.txt"
    stake_in = py_tags_dir / "stakeoverflow_generated_tags.txt"

    # Output (new statistics files)
    nation_out = py_tags_dir / "nation_china_tag_counts.json"
    rust_cc_out = py_tags_dir / "rust_cc_tag_counts.json"
    stake_out = py_tags_dir / "stakeoverflow_tag_counts.json"

    aggregate_and_store(str(nation_in), str(nation_out))
    aggregate_and_store(str(rust_cc_in), str(rust_cc_out))
    aggregate_and_store(str(stake_in), str(stake_out))


# ------------ Tag Classification ------------
# ------------ Keyword Grouping Output as "Theme -> Key-Value Count", including Uncategorized ------------
def canonicalize_key(s: str) -> str:
    """
    Normalization for exact matching: lowercase, replace Chinese quotes, remove non-alphanumeric, collapse whitespace.
    """
    if s is None:
        return ""
    t = s.strip().lower()
    t = t.replace("“", '"').replace("”", '"').replace("‘", "'").replace("’", "'")
    t = re.sub(r"[^a-z0-9]+", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t

def contains_whole_word(can_tag: str, keyword: str) -> bool:
    """
    Determine if the normalized tag 'can_tag' contains the normalized 'keyword' (whole word or phrase),
    and allow common morphological variants of keywords (plural: s/es/ies, gerund: ing).
    """
    if not can_tag or not keyword:
        return False
    if can_tag == keyword:
        return True

    # First try phrase whole containment (word boundary)
    hay = f" {can_tag} "
    needle = f" {keyword} "
    if needle in hay or hay.startswith(needle) or hay.endswith(needle):
        return True

    tag_tokens = can_tag.split()
    tag_token_set = set(tag_tokens)
    kw_tokens = keyword.split()

    def variants_for_token(t: str) -> set:
        vs = {t}
        # Plural
        vs.add(t + "s")
        if t.endswith(("s", "x", "z", "ch", "sh")):
            vs.add(t + "es")
        if t.endswith("y") and len(t) > 1 and t[-2] not in "aeiou":
            vs.add(t[:-1] + "ies")
        # ing (including dropping e rule)
        vs.add(t + "ing")
        if t.endswith("e") and len(t) > 1:
            vs.add(t[:-1] + "ing")
        # Past tense (optional small tolerance)
        vs.add(t + "ed")
        if t.endswith("e") and len(t) > 1:
            vs.add(t + "d")
        if t.endswith("y") and len(t) > 1 and t[-2] not in "aeiou":
            vs.add(t[:-1] + "ied")
        return vs

    # Require each word of the keyword to appear in the tag in "original or variant" form
    for kt in kw_tokens:
        if not any(v in tag_token_set for v in variants_for_token(kt)):
            return False
    return True

def load_categories(categories_path: str) -> "OrderedDict[str, List[str]]":
    text = Path(categories_path).read_text(encoding="utf-8")
    module = ast.parse(text, filename=str(categories_path))
    categories = OrderedDict()
    for node in module.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in ("CATEGORIES", "CATEGORIES_CN"):
                    value = ast.literal_eval(node.value)
                    for k, v in value.items():
                        categories[k] = [str(x) for x in v]
                    return categories
    raise RuntimeError("CATEGORIES or CATEGORIES_CN not found in categories file")

def build_keyword_index(categories: "OrderedDict[str, List[str]]") -> "OrderedDict[str, set]":
    idx = OrderedDict()
    for theme, keywords in categories.items():
        norm_keys = set()
        for k in keywords:
            ck = canonicalize_key(k)
            if ck:
                norm_keys.add(ck)
        idx[theme] = norm_keys
    return idx

def group_tags_from_counts(counts_path: str, categories_path: str) -> "OrderedDict[str, dict]":
    """
    Read tag count file, group tags by theme into key-value counts.
    - Exact match: Classified only if tag and keyword are consistent after canonicalize_key normalization
    - Output includes eight themes (keeping categories order) and one 'Uncategorized', values are all {tag: count}
    """
    counts: dict = json.loads(Path(counts_path).read_text(encoding="utf-8"))
    categories = load_categories(categories_path)
    idx = build_keyword_index(categories)

    theme_names = list(idx.keys())
    grouped: "OrderedDict[str, dict]" = OrderedDict((name, {}) for name in theme_names)
    grouped["Uncategorized"] = {}

    for raw_tag, c in counts.items():
        can_tag = canonicalize_key(raw_tag)
        if not can_tag:
            grouped["Uncategorized"][raw_tag] = grouped["Uncategorized"].get(raw_tag, 0) + c
            continue

        hit = False
        for theme in theme_names:
            # First try strict equivalence matching; if not hit, try "whole word/phrase containment"
            for kw in idx[theme]:
                if can_tag == kw or contains_whole_word(can_tag, kw):
                    grouped[theme][raw_tag] = grouped[theme].get(raw_tag, 0) + c
                    hit = True
                    break
            if hit:
                break

        if not hit:
            grouped["Uncategorized"][raw_tag] = grouped["Uncategorized"].get(raw_tag, 0) + c

    for theme in grouped:
        grouped[theme] = {k: grouped[theme][k] for k in sorted(grouped[theme].keys())}
    return grouped

def write_theme_groups(output_path: str, groups_kv: "OrderedDict[str, dict]") -> None:
    """
    Write as JSON object:
    {
      "Core Language Concepts": { "tag": count, ... },
      ... (8 themes),
      "Uncategorized": { "tag": count, ... }
    }
    """
    op = Path(output_path)
    ensure_parent_dir(op)
    with open(op, "w", encoding="utf-8") as f:
        json.dump(groups_kv, f, ensure_ascii=False, indent=2)

def group_nation_china_by_keywords() -> None:
    base_dir = Path("e:/work-study/scientific_research/Rust_survey/论坛/after_tags")
    py_tags_dir = base_dir / "PY_tags"
    counts_path = py_tags_dir / "nation_china_tag_counts.json"
    categories_path = py_tags_dir / "categories.txt"
    output_path = py_tags_dir / "nation_china_tag_groups.json"

    groups_kv = group_tags_from_counts(str(counts_path), str(categories_path))
    write_theme_groups(str(output_path), groups_kv)
    
# New: Keyword grouping for StakeOverflow
def group_stakeoverflow_by_keywords() -> None:
    base_dir = Path("e:/work-study/scientific_research/Rust_survey/论坛/after_tags")
    py_tags_dir = base_dir / "PY_tags"
    counts_path = py_tags_dir / "stakeoverflow_tag_counts.json"
    categories_path = py_tags_dir / "categories.txt"
    output_path = py_tags_dir / "stakeoverflow_tag_groups.json"

    groups_kv = group_tags_from_counts(str(counts_path), str(categories_path))
    write_theme_groups(str(output_path), groups_kv)
    
# New: Keyword grouping for Chinese community (rust_cc), using CATEGORIES_CN.txt
def group_rust_cc_by_keywords() -> None:
    base_dir = Path("e:/work-study/scientific_research/Rust_Survey/论坛/after_tags")
    # Correct path case uniformity
    base_dir = Path("e:/work-study/scientific_research/Rust_survey/论坛/after_tags")
    py_tags_dir = base_dir / "PY_tags"
    counts_path = py_tags_dir / "rust_cc_tag_counts.json"
    categories_path = py_tags_dir / "CATEGORIES_CN.txt"
    output_path = py_tags_dir / "rust_cc_tag_groups.json"

    groups_kv = group_tags_from_counts(str(counts_path), str(categories_path))
    write_theme_groups(str(output_path), groups_kv)
    
# ... existing code ...

def cluster_uncategorized_only() -> None:
    from sbert_cluster import semantic_cluster_uncategorized
    base_dir = Path("e:/work-study/scientific_research/Rust_survey/论坛/after_tags")
    py_tags_dir = base_dir / "PY_tags"
    groups_path = py_tags_dir / "nation_china_tag_groups.json"
    counts_path = py_tags_dir / "nation_china_tag_counts.json"
    uncategorized_out = py_tags_dir / "nation_china_uncategorized_clusters.json"

    semantic_cluster_uncategorized(
        groups_path=str(groups_path),
        counts_path=str(counts_path),
        output_path=str(uncategorized_out),
        model_name="all-mpnet-base-v2",
        min_cluster_size=12,
        min_samples=3,
    )


def cluster_uncategorized_only_stakeoverflow() -> None:
    from sbert_cluster import semantic_cluster_uncategorized
    base_dir = Path("e:/work-study/scientific_research/Rust_survey/论坛/after_tags")
    py_tags_dir = base_dir / "PY_tags"
    groups_path = py_tags_dir / "stakeoverflow_tag_groups.json"
    counts_path = py_tags_dir / "stakeoverflow_tag_counts.json"
    uncategorized_out = py_tags_dir / "stakeoverflow_uncategorized_clusters.json"

    semantic_cluster_uncategorized(
        groups_path=str(groups_path),
        counts_path=str(counts_path),
        output_path=str(uncategorized_out),
        model_name="all-mpnet-base-v2",
        min_cluster_size=12,
        min_samples=3,
    )


# New: Uncategorized semantic clustering for Chinese community (rust_cc), using multilingual SBERT model
def cluster_uncategorized_only_rust_cc() -> None:
    from sbert_cluster import semantic_cluster_uncategorized
    base_dir = Path("e:/work-study/scientific_research/Rust_survey/论坛/after_tags")
    py_tags_dir = base_dir / "PY_tags"
    groups_path = py_tags_dir / "rust_cc_tag_groups.json"
    counts_path = py_tags_dir / "rust_cc_tag_counts.json"
    uncategorized_out = py_tags_dir / "rust_cc_uncategorized_clusters.json"

    semantic_cluster_uncategorized(
        groups_path=str(groups_path),
        counts_path=str(counts_path),
        output_path=str(uncategorized_out),
        model_name="paraphrase-multilingual-MiniLM-L12-v2",
        min_cluster_size=5,
        min_samples=1,
    )


# New: Perform cross-file aggregation and three sets of semantic clustering on "categorized" tags
def cluster_categorized_aggregates() -> None:
    from aggregated_cluster import cluster_from_grouped_categories

    base_dir = Path("e:/work-study/scientific_research/Rust_survey/论坛/after_tags")
    py_tags_dir = base_dir / "PY_tags"
    groups_paths = [
        str(py_tags_dir / "nation_china_tag_groups.json"),
        str(py_tags_dir / "stakeoverflow_tag_groups.json"),
    ]

    aggregate_def = {
        "group1": [
            "Core Language Concepts",
            "Compiler Interaction",
            "Learning Experience",
            "other1",
        ],
        "group2": [
            "Debugging and Analysis",
            "Testing Tools",
            "Build and Deployment",
            "other2",
        ],
        "group3": [
            "Library Ecosystem",
            "Community and Learning",
            "other3",
        ],
    }

    output_paths = {
        "group1": str(py_tags_dir / "aggregated_group1_clusters.json"),
        "group2": str(py_tags_dir / "aggregated_group2_clusters.json"),
        "group3": str(py_tags_dir / "aggregated_group3_clusters.json"),
    }

    # Tags in both files are English, use English SBERT model
    cluster_from_grouped_categories(
        groups_paths=groups_paths,
        aggregate_def=aggregate_def,
        output_paths=output_paths,
        model_name="all-mpnet-base-v2",
        min_cluster_size=12,
        min_samples=3,
        min_count=3,
    )
    
# New: Summarize the three clustering results into a simplified view, keeping only "Tag + Count", grouped by cluster
def write_simplified_aggregated_clusters() -> None:
    base_dir = Path("e:/work-study/scientific_research/Rust_survey/论坛/after_tags")
    py_tags_dir = base_dir / "PY_tags"

    group_files = {
        "group1": py_tags_dir / "aggregated_group1_clusters.json",
        "group2": py_tags_dir / "aggregated_group2_clusters.json",
        "group3": py_tags_dir / "aggregated_group3_clusters.json",
    }

    simplified: dict[str, list[list[object]]] = {}

    for gname, fpath in group_files.items():
        if not fpath.exists():
            # If not exists, give an empty array to avoid error when viewing
            simplified[gname] = []
            continue
        try:
            data = json.loads(fpath.read_text(encoding="utf-8"))
        except Exception:
            simplified[gname] = []
            continue

        clusters = data.get("clusters", [])
        out_clusters: list[list[object]] = []
        for cl in clusters:
            tags = cl.get("tags", [])
            # Only keep cluster: [{"name": "", "count": <sum>}, {"tag": count, ...}]
            clean_map: dict[str, int] = {}
            for t in tags:
                tag = t.get("tag")
                count = t.get("count")
                if tag is None or count is None:
                    continue
                try:
                    count_val = int(count)
                except Exception:
                    # Skip if not integer
                    continue
                clean_map[str(tag)] = count_val
            meta = {"name": "", "count": sum(clean_map.values())}
            out_clusters.append([meta, clean_map])
        simplified[gname] = out_clusters

    # Write merged simplified file
    output_path = py_tags_dir / "aggregated_clusters_simplified.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(simplified, f, ensure_ascii=False, indent=2)
        
def run_cross_platform_recluster_and_collect() -> None:
    """
    Perform secondary clustering and aggregation restructuring for cross-platform deployment clusters:
    1) Read "Cross-Platform Deployment" in group2 of aggregated_clusters.json and re-cluster, output cross_platform_recluster.json
    2) Restructure re-clustering results into aggregated_clusters style, output cross_platform_recluster_aggregated.json
    """
    # Execute secondary clustering first, then aggregation restructuring (merged into the same module)
    from recluster_cross_platform import (
        recluster_group2_cross_platform,
        collect_cross_platform_as_aggregated,
    )
    recluster_group2_cross_platform()
    collect_cross_platform_as_aggregated()





def write_group_name_counts_from_simplified(
    source_path: str = "e:/work-study/scientific_research/Rust_survey/论坛/after_tags/PY_tags/aggregated_clusters_simplified.json",
    output_path: str = "e:/work-study/scientific_research/Rust_survey/论坛/after_tags/PY_tags/aggregated_groups_name_counts.json",
) -> None:
    """
    Summarize name: count pairs for each cluster by group from aggregated_clusters_simplified.json,
    Output cluster-level metadata only, do not export tag details.
    """
    sp = Path(source_path)
    if not sp.exists():
        raise RuntimeError(f"Input file not found: {source_path}")
    data = json.loads(sp.read_text(encoding="utf-8"))

    out: dict[str, dict[str, int]] = {}
    for group_key, clusters in data.items():
        if not isinstance(clusters, list):
            continue
        name_counts: dict[str, int] = {}
        for entry in clusters:
            if isinstance(entry, list) and len(entry) == 2 and isinstance(entry[0], dict):
                meta = entry[0]
                name = meta.get("name")
                count = meta.get("count")
                if isinstance(name, str) and isinstance(count, (int, float)):
                    name_counts[name] = name_counts.get(name, 0) + int(count)
        if name_counts:
            out[group_key] = name_counts

    op = Path(output_path)
    op.parent.mkdir(parents=True, exist_ok=True)
    with open(op, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

def plot_group_name_counts_bar(
    source_path: str = "e:/work-study/scientific_research/Rust_survey/论坛/after_tags/PY_tags/aggregated_groups_name_counts.json",
    group_key: str = "group1",
    output_dir: str = "e:/work-study/scientific_research/Rust_survey/论坛/after_tags/PY_tags/figures",
    dpi: int = 160,
    display_group_label: str | None = "RQ1",
    title_fontsize: int = 20,
    label_fontsize: int = 20,
    tick_fontsize: int = 16,
    value_fontsize: int = 12,
    height_per_item: float = 0.33,
    horizontal: bool = False,
    width_per_item: float = 0.5,
    bar_width: float = 0.8,
    gap_ratio: float = 0.8,
    bar_height: float = 0.65,
) -> dict:
    """
    Read the specified group (default group1) from aggregated_groups_name_counts.json,
    Sort by count from high to low and draw a bar chart (horizontal bar, show all), save PNG/PDF.

    Return {"png": path, "pdf": path}.
    """
    sp = Path(source_path)
    if not sp.exists():
        raise RuntimeError(f"Input file not found: {source_path}")
    data = json.loads(sp.read_text(encoding="utf-8"))

    if group_key not in data:
        raise RuntimeError(f"Group does not exist in input file: {group_key}")

    name_counts = data[group_key]
    if not isinstance(name_counts, dict) or not name_counts:
        raise RuntimeError(f"Data for group {group_key} is empty or malformed")

    # Sort (high to low)
    items = sorted(name_counts.items(), key=lambda kv: (-kv[1], kv[0]))
    names = [k for k, _ in items]
    counts = [v for _, v in items]

    import matplotlib
    import matplotlib.pyplot as plt
    # Use fonts that support Chinese on Windows to avoid garbled Chinese (no effect if no Chinese)
    try:
        matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial"]
        matplotlib.rcParams["axes.unicode_minus"] = False
    except Exception:
        pass

    label_for_title = display_group_label if display_group_label else group_key

    if horizontal:
        height = max(6, height_per_item * len(items))
        fig, ax = plt.subplots(figsize=(12, height), dpi=dpi)
        ax.barh(names, counts, color="#4C78A8", height=bar_height)
        ax.invert_yaxis()
        ax.tick_params(axis='both', which='major', labelsize=tick_fontsize)
        max_v = max(counts)
        for p in ax.patches:
            w = p.get_width()
            y = p.get_y() + p.get_height() / 2
            ax.text(w + max_v * 0.005, y, str(int(w)), va="center", fontsize=value_fontsize)
    else:
        import numpy as np
        width = max(12, width_per_item * len(items))
        fig, ax = plt.subplots(figsize=(width, 8), dpi=dpi)
        x = np.arange(len(items)) * (1.0 + gap_ratio)
        ax.bar(x, counts, color="#4C78A8", width=bar_width)
        ax.tick_params(axis='both', which='major', labelsize=tick_fontsize)
        ax.set_xticks(x)
        ax.set_xticklabels(names)
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
        ax.margins(x=0.02)
        max_v = max(counts)
        ax.set_ylim(0, max_v * 1.25)
        for p in ax.patches:
            h = p.get_height()
            cx = p.get_x() + p.get_width() / 2
            ax.text(cx, h + max_v * 0.012, str(int(h)), ha="center", va="bottom", fontsize=value_fontsize)

    if not horizontal:
        plt.subplots_adjust(bottom=0.46)
    plt.tight_layout()

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    file_stub = (display_group_label if display_group_label else group_key)
    png_path = out_dir / f"{file_stub}_bar_sorted.png"
    pdf_path = out_dir / f"{file_stub}_bar_sorted.pdf"
    plt.savefig(png_path)
    plt.savefig(pdf_path)
    plt.close(fig)

    return {"png": str(png_path), "pdf": str(pdf_path)}

def plot_group2_name_counts_bar(
    source_path: str = "e:/work-study/scientific_research/Rust_survey/论坛/after_tags/PY_tags/aggregated_groups_name_counts.json",
    output_dir: str = "e:/work-study/scientific_research/Rust_survey/论坛/after_tags/PY_tags/figures",
    dpi: int = 160,
) -> dict:
    """Draw descending bar chart for RQ2 (group2) and save PNG/PDF."""
    return plot_group_name_counts_bar(
        source_path=source_path,
        group_key="group2",
        output_dir=output_dir,
        dpi=dpi,
        display_group_label="RQ2",
        horizontal=True,
    )

def plot_group3_name_counts_bar(
    source_path: str = "e:/work-study/scientific_research/Rust_survey/论坛/after_tags/PY_tags/aggregated_groups_name_counts.json",
    output_dir: str = "e:/work-study/scientific_research/Rust_survey/论坛/after_tags/PY_tags/figures",
    dpi: int = 160,
) -> dict:
    """Draw descending bar chart for RQ3 (group3) and save PNG/PDF."""
    return plot_group_name_counts_bar(
        source_path=source_path,
        group_key="group3",
        output_dir=output_dir,
        dpi=dpi,
        display_group_label="RQ3",
        horizontal=True,
    )

def main():
    # 1) Extraction (with boundary cleanup)
    # extract_all_sources()
    # 2) Re-aggregate counts
    # aggregate_all_sources()

    # 3) Keyword Stage (nation_china)
    # group_nation_china_by_keywords()
    # 4) Semantic Clustering Stage (nation_china Uncategorized)
    # cluster_uncategorized_only()

    # 3) Keyword Stage (stakeoverflow)
    # group_stakeoverflow_by_keywords()
    # 4) Semantic Clustering Stage (stakeoverflow Uncategorized)
    cluster_uncategorized_only_stakeoverflow()
    
    # 3) Keyword Stage (rust_cc Chinese Community)
    # group_rust_cc_by_keywords()
    # 4) Semantic Clustering Stage (rust_cc Uncategorized, Multilingual Model)
    # cluster_uncategorized_only_rust_cc()
    
    # 5) Cross-file Aggregation and Semantic Clustering of "Categorized" Tags (Three Groups)
    # cluster_categorized_aggregates()
    
    # 6) Simplify three clustering files to generate "Tag + Count" view for easy viewing
    # write_simplified_aggregated_clusters()
    
    # 6' (Choose whether to execute based on situation) Secondary clustering and aggregation restructuring for groups with poor clustering effect
    # run_cross_platform_recluster_and_collect()

    # 7) Generate "group -> {Cluster Name: Count}" Key-Value View from Simplified Clustering File
    # write_group_name_counts_from_simplified()

    # 8) Draw bar chart for group1 sorted from high to low (show all first)
    # plot_group_name_counts_bar()
    # Draw bar charts for RQ2 (group2) and RQ3 (group3)
    # plot_group2_name_counts_bar()
    # plot_group3_name_counts_bar()
    
if __name__ == "__main__":
    main()
