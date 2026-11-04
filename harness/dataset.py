"""HotpotQA dev-distractor loading and stratified sampling.

The distractor split is the right choice here because each item carries its own
10 paragraphs (2 gold + 8 distractors). That makes the tool environment local,
deterministic, and offline after a single download - no Wikipedia dump, no
live search API whose index drifts between runs.

Dataset: HotpotQA (Yang et al., 2018), CC BY-SA 4.0.
https://hotpotqa.github.io/

The downloaded JSON is untrusted input: it lands in its own directory and is
only ever parsed with json.load. Nothing is executed from it.
"""

from __future__ import annotations

import json
import os
import random
import sys
from pathlib import Path

# The canonical CMU host (curtis.ml.cmu.edu) has been unreliable for years and
# was down when this was written, so the HuggingFace mirror of the same release
# is the primary source. It carries identical content in a columnar layout,
# which _from_hf_batch converts back to the original JSON shape.
HF_PARQUET_URL = (
    "https://huggingface.co/api/datasets/hotpotqa/hotpot_qa/parquet/"
    "distractor/validation/0.parquet"
)
CMU_JSON_URL = (
    "http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_distractor_v1.json"
)
DEFAULT_PATH = Path(__file__).resolve().parent.parent / "data" / "hotpot" / "hotpot_dev_distractor_v1.json"


def _from_hf_batch(batch: dict) -> list[dict]:
    """Convert HuggingFace's columnar rows to the original HotpotQA shape.

    HF stores supporting_facts as {title: [...], sent_id: [...]} and context as
    {title: [...], sentences: [[...], ...]}; the reference format - and every
    published evaluation script - expects lists of [title, sent_id] pairs and
    [title, sentences] pairs. Normalising here means the rest of the harness
    only ever sees the canonical shape.
    """
    out = []
    for row in batch:
        ctx = row["context"]
        sf = row["supporting_facts"]
        out.append({
            "_id": row["id"],
            "question": row["question"],
            "answer": row["answer"],
            "type": row["type"],
            "level": row["level"],
            "supporting_facts": list(zip(sf["title"], sf["sent_id"])),
            "context": [
                [t, list(s)] for t, s in zip(ctx["title"], ctx["sentences"])
            ],
        })
    return out


CA_BUNDLE_PATH = Path(__file__).resolve().parent.parent / "data" / "ca-bundle.pem"


def bootstrap_ca(dest: Path = CA_BUNDLE_PATH) -> Path | None:
    """Export the macOS trust store to a PEM bundle.

    On a corporate laptop TLS is intercepted by a proxy whose root lives in the
    system keychain. curl reads that keychain; Python's ssl module does not, so
    httpx fails with "self-signed certificate in certificate chain" on hosts
    that work fine in a browser. Exporting both the system roots and the
    admin-installed roots gives Python the same trust anchors.

    Contains only public CA certificates - no keys, nothing secret.
    """
    import subprocess

    keychains = [
        "/System/Library/Keychains/SystemRootCertificates.keychain",
        "/Library/Keychains/System.keychain",
    ]
    pems = []
    for kc in keychains:
        if not Path(kc).exists():
            continue
        try:
            out = subprocess.run(
                ["security", "find-certificate", "-a", "-p", kc],
                capture_output=True, text=True, timeout=60, check=False,
            )
            if out.stdout:
                pems.append(out.stdout)
        except (OSError, subprocess.SubprocessError):
            continue
    if not pems:
        return None
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text("\n".join(pems), encoding="utf-8")
    n = dest.read_text(encoding="utf-8").count("BEGIN CERTIFICATE")
    print(f"wrote {n} certificates to {dest}")
    return dest


def _verify():
    """httpx `verify=` value. Explicit env vars win, then a bundle we exported
    earlier, then - on macOS - export one now, then the default trust store."""
    for var in ("HOTPOT_CA_BUNDLE", "SSL_CERT_FILE", "REQUESTS_CA_BUNDLE"):
        v = os.getenv(var)
        if v and Path(v).exists():
            return v
    if CA_BUNDLE_PATH.exists():
        return str(CA_BUNDLE_PATH)
    if sys.platform == "darwin" and bootstrap_ca():
        return str(CA_BUNDLE_PATH)
    return True


def download(dest: Path = DEFAULT_PATH, url: str = HF_PARQUET_URL) -> Path:
    """Fetch the dev-distractor split and write it as canonical JSON."""
    import httpx

    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 1_000_000:
        print(f"already present: {dest} ({dest.stat().st_size / 1e6:.1f} MB)")
        return dest

    print(f"downloading {url}")
    raw = dest.parent / "distractor_validation.parquet"
    with httpx.stream("GET", url, follow_redirects=True, timeout=180,
                      verify=_verify()) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        done = next_tick = 0
        with open(raw, "wb") as fh:
            for chunk in r.iter_bytes(chunk_size=1 << 16):
                fh.write(chunk)
                done += len(chunk)
                if done < next_tick:  # keep the log readable when piped
                    continue
                next_tick = done + 4_000_000
                print(f"  {done / 1e6:6.1f} MB"
                      + (f" / {total / 1e6:.1f} MB" if total else ""),
                      flush=True)

    import pyarrow.parquet as pq

    items = _from_hf_batch(pq.read_table(raw).to_pylist())
    with open(dest, "w", encoding="utf-8") as fh:
        json.dump(items, fh)
    raw.unlink()
    print(f"saved {len(items)} items to {dest} ({dest.stat().st_size / 1e6:.1f} MB)")
    return dest


def load_items(path: Path = DEFAULT_PATH) -> list[dict]:
    if not path.exists():
        raise SystemExit(
            f"dataset not found at {path}\n"
            f"Run:  python -m harness.dataset download"
        )
    with open(path, encoding="utf-8") as fh:
        items = json.load(fh)
    return items


def gold_titles(item: dict) -> set:
    return {t for t, _sent_id in item.get("supporting_facts", [])}


def sample(items: list[dict], n: int, types: tuple = (), levels: tuple = (),
           seed: int = 0) -> list[dict]:
    """Stratified sample.

    When a condition names more than one `type`, draw equally from each so the
    mix is fixed by the condition rather than by the dev set's own proportions
    (which are ~80% bridge). Sampling is seeded, so every arm sees exactly the
    same questions in the same order - the comparison is paired, not
    independent, which is what makes small-n differences meaningful.
    """
    pool = items
    if levels:
        pool = [it for it in pool if it.get("level") in levels]

    rng = random.Random(seed)

    if not types:
        pool = list(pool)
        rng.shuffle(pool)
        return pool[:n]

    per_type = {t: [it for it in pool if it.get("type") == t] for t in types}
    for t, bucket in per_type.items():
        if not bucket:
            raise SystemExit(f"no questions of type {t!r} in the dataset")
        rng.shuffle(bucket)

    out: list[dict] = []
    i = 0
    # Round-robin across types so a truncated sample stays balanced.
    while len(out) < n:
        progressed = False
        for t in types:
            if i < len(per_type[t]) and len(out) < n:
                out.append(per_type[t][i])
                progressed = True
        if not progressed:
            break
        i += 1
    return out


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "download":
        download()
    elif len(sys.argv) > 1 and sys.argv[1] == "bootstrap-ca":
        bootstrap_ca()
    else:
        items = load_items()
        from collections import Counter

        print(f"{len(items)} items")
        print("type: ", Counter(it.get("type") for it in items))
        print("level:", Counter(it.get("level") for it in items))
