"""Download RefCOCO/+/g original-format annotations for text-only KD.

The REFER host is unavailable. A pinned public research mirror with published
SHA256 is used. This is NOT the unpublished SAM3-LiteText author manifest.
No COCO images, RF100 data, SAM3 weights, or user Drive data are downloaded.
"""
from __future__ import annotations
import argparse
import collections
import gzip
import hashlib
import importlib.util
import json
import pickle
from pathlib import Path
import unicodedata
import urllib.parse
import urllib.request

OFFICIAL_COMMIT = "bd0936c788fed8d51fa799437f05abd97b401b06"
MIRROR_REVISION = "9dc3ec34985f334933bb271813175428858b9e1b"
SOURCES = {
    "refcoco": ("refcoco/refs(unc).p", "d4a8dd3152f130127924f9d0f0ea30f4dc43ec19e1b47bc55af972f90614f8ba", 31742595),
    "refcoco_plus": ("refcoco+/refs(unc).p", "7de24c182449758b9d9ae87d968b1f25831236a54d38d45408e2e0fdd318e802", 31893061),
    "refcocog": ("refcocog/refs(umd).p", "0331c7533537b67c2f7ac8c8bab0da2d379d1754c6f5c110fd79f70e17e7bddb", 33853676),
}


def sha256(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def download(url, path, expected_hash=None, expected_bytes=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    req = urllib.request.Request(url, headers={"User-Agent": "Lite50-reproducible-data/2"})
    with urllib.request.urlopen(req, timeout=120) as response, temporary.open("wb") as out:
        written = 0
        while chunk := response.read(1024 * 1024):
            written += len(chunk)
            if written > 200_000_000:
                raise ValueError("Unexpected non-annotation download size")
            out.write(chunk)
    digest = sha256(temporary)
    if expected_hash and digest != expected_hash:
        raise ValueError(f"SHA256 mismatch for {path.name}: {digest}")
    if expected_bytes and written != expected_bytes:
        raise ValueError(f"Size mismatch for {path.name}: {written}")
    temporary.replace(path)
    return {"url": url, "file_name": path.name, "bytes": written, "sha256": digest}


class ContainerOnlyUnpickler(pickle.Unpickler):
    """Never execute globals or arbitrary objects from annotation pickles."""
    def find_class(self, module, name):
        raise pickle.UnpicklingError(f"Disallowed pickle global: {module}.{name}")

    def persistent_load(self, pid):
        raise pickle.UnpicklingError("Persistent IDs are not allowed")


def normalize(text):
    return " ".join(unicodedata.normalize("NFKC", text).split()).casefold()


def text_hash(text):
    return hashlib.sha256(normalize(text).encode("utf-8")).hexdigest()


def write_jsonl_gz(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as raw, gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as zipped:
        for row in rows:
            zipped.write((json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n").encode())


def load_tokenizer(output):
    assets = output / "official_text_assets"
    base = f"https://raw.githubusercontent.com/SimonZeng7108/efficientsam3/{OFFICIAL_COMMIT}"
    proofs = [download(base + "/sam3/sam3/model/tokenizer_ve.py", assets / "tokenizer_ve.py")]
    bpe = assets / "bpe_simple_vocab_16e6.txt.gz"
    try:
        proof = download(base + "/sam3/sam3/assets/bpe_simple_vocab_16e6.txt.gz", bpe)
    except Exception:
        proof = download(base + "/sam3/assets/bpe_simple_vocab_16e6.txt.gz", bpe)
    proofs.append(proof)
    spec = importlib.util.spec_from_file_location("pinned_tokenizer", assets / "tokenizer_ve.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.SimpleTokenizer(str(bpe)), proofs


def flatten_refs(dataset, refs, tokenizer):
    rows, seen, counts = [], set(), collections.Counter()
    for ref in refs:
        split = str(ref["split"])
        if split not in ("train", "val", "test", "testA", "testB"):
            raise ValueError(f"Unexpected split {dataset}/{split}")
        for sent in ref["sentences"]:
            sid = int(sent["sent_id"])
            if sid in seen:
                raise ValueError(f"Duplicate sentence ID: {dataset}/{sid}")
            seen.add(sid)
            text = sent.get("raw") or sent.get("sent")
            if not isinstance(text, str) or not text.strip():
                raise ValueError(f"Blank expression {dataset}/{sid}")
            length = len(tokenizer.encode(text))
            rows.append({"dataset": dataset, "source_split": split,
                "split": "train" if split == "train" else "val" if split == "val" else "test",
                "image_id": int(ref["image_id"]), "ann_id": int(ref["ann_id"]),
                "ref_id": int(ref["ref_id"]), "sent_id": sid, "text": text,
                "text_sha256": text_hash(text), "content_tokens": length,
                "would_truncate_ctx16": length > 14, "text_policy": "original_runtime_ctx16"})
            counts[split] += 1
            counts["over14"] += length > 14
    return rows, dict(counts)


def prepare(args):
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    tokenizer, token_sources = load_tokenizer(output)
    rows, stats, sources = [], {}, []
    for dataset, (relative, digest, size) in SOURCES.items():
        url = f"https://huggingface.co/wayneicloud/SSP-SAM/resolve/{MIRROR_REVISION}/data_seg/" + urllib.parse.quote(relative, safe="/")
        path = output / "original_annotations" / dataset / Path(relative).name
        proof = download(url, path, digest, size)
        proof.update(dataset=dataset, repository="wayneicloud/SSP-SAM", revision=MIRROR_REVISION,
                     role="third_party_public_research_mirror", split_by="umd" if dataset == "refcocog" else "unc")
        sources.append(proof)
        with path.open("rb") as f:
            refs = ContainerOnlyUnpickler(f, encoding="latin1").load()
        if not isinstance(refs, list):
            raise ValueError("Expected REFER list of dictionaries")
        records, stats[dataset] = flatten_refs(dataset, refs, tokenizer)
        write_jsonl_gz(output / "source_records" / f"{dataset}.jsonl.gz", records)
        rows.extend(records)
        print(dataset, stats[dataset], flush=True)
    # COCO image IDs are shared by the three corpora. Reserve the union of
    # their held-out images/expressions before building the mixed train set.
    held = [r for r in rows if r["split"] != "train"]
    held_images = {r["image_id"] for r in held}
    held_texts = {r["text_sha256"] for r in held}
    eligible, excluded, unique = [], collections.Counter(), {}
    for row in rows:
        if row["split"] != "train":
            continue
        if row["image_id"] in held_images:
            excluded["image_in_other_dataset_holdout"] += 1
            continue
        if row["text_sha256"] in held_texts:
            excluded["text_in_holdout"] += 1
            continue
        eligible.append(row)
        if row["text_sha256"] not in unique:
            unique[row["text_sha256"]] = {**row, "occurrences": 0}
        unique[row["text_sha256"]]["occurrences"] += 1
    train = [unique[key] for key in sorted(unique)]
    val = [r for r in held if r["split"] == "val"]
    test = [r for r in held if r["split"] == "test"]
    for split, items in [("train", train), ("val", val), ("test", test)]:
        write_jsonl_gz(output / "text" / f"refcoco_{split}.jsonl.gz", items)
    assert not ({r["image_id"] for r in train} & held_images)
    assert not ({r["text_sha256"] for r in train} & held_texts)
    report = {"status": "prepared_text_only", "source_stats": stats,
        "source_records": len(rows), "eligible_train_occurrences": len(eligible),
        "train_unique_texts": len(train), "val_records": len(val), "test_records": len(test),
        "exclusions": dict(excluded), "train_over14": sum(r["content_tokens"] > 14 for r in train),
        "train_image_overlap_holdout": 0, "train_text_overlap_holdout": 0,
        "token_context": 16, "tokenizer_commit": OFFICIAL_COMMIT,
        "text_was_rewritten": False, "images_downloaded": False,
        "policy": "Source records unchanged; train-only; cross-dataset held-out image/text exclusion; normalized-text dedup for text-only KD.",
        "limitations": ["Not the unpublished LiteText paper training manifest.",
          "Long RefCOCO strings are retained; the official runtime tokenizer handles truncation.",
          "Mirror SHA agreement is not an unavailable original-host checksum.",
          "No COCO photographs: these files are for text-only distillation."]}
    (output / "validation.json").write_text(json.dumps(report, indent=2))
    (output / "sources.json").write_text(json.dumps({"data": sources, "tokenizer": token_sources,
        "original_reference": "https://github.com/lichengunc/refer", "mirror_revision": MIRROR_REVISION}, indent=2))
    (output / "README.md").write_text(
        "# RefCOCO / RefCOCO+ / RefCOCOg text-only data\n\n"
        "Original REFER annotations are retained, from a pinned SHA-verified research mirror. "
        "RefCOCO/RefCOCO+ use UNC; RefCOCOg uses UMD, as recommended by REFER.\n\n"
        "text/refcoco_train.jsonl.gz excludes the union of held-out COCO image IDs and texts. "
        "Source records and splits remain in source_records/. Text is not paraphrased or pre-truncated; "
        "both SAM3 teacher and Lite50 student must use official ctx16.\n\n"
        "This is for the user's Recap-v3 + RefCOCO-family KD plan, not an exact LiteText paper reproduction. "
        "RF100-VL and COCO images are not included. Preserve attribution and applicable source-data terms.\n")
    files = {str(p.relative_to(output)): {"bytes": p.stat().st_size, "sha256": sha256(p)}
             for p in sorted(output.rglob("*")) if p.is_file()}
    (output / "SHA256SUMS.json").write_text(json.dumps(files, indent=2))
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    prepare(parser.parse_args())
