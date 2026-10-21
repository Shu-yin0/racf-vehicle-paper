"""Read-only pre-submission audit for the scoped ACC evidence package."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any

FORBIDDEN = (
    re.compile(r"\b73\s*%"),
    re.compile(r"\b50\s*[xX]"),
    re.compile(r"full benchmark", re.IGNORECASE),
    re.compile(r"trajectory[- ]level safety guarantee", re.IGNORECASE),
    re.compile(r"superior(?:ity)?\s+(?:to|over)\s+MPC", re.IGNORECASE),
    re.compile(r"\barbitrary policies\b", re.IGNORECASE),
)
PACKET_HASH_ROW = re.compile(r"\| `([^`]+)` \| `([0-9a-f]{64})` \|")
NONCLAIM_CONTEXT = re.compile(
    r"\b(?:not|no|neither|never|without|cannot)\b|does\s+not|do\s+not|must\s+not",
    re.IGNORECASE,
)
V24_CONDITIONS = ("nominal", "mass_shift", "slope_shift", "combined")
V24_PARAMETER_SAMPLES = [
    {"mass": mass, "grade": grade, "actuator_force_gain": gain}
    for mass, grade in (
        (1500.0, 0.0),
        (1800.0, 0.0),
        (1500.0, 3.0),
        (1800.0, 2.0),
    )
    for gain in (0.7, 1.0)
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_sha256(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def pdf_pages(path: Path, expected_pages: int | None = None) -> int:
    pdfinfo = shutil.which("pdfinfo")
    if pdfinfo:
        output = subprocess.check_output([pdfinfo, str(path)], text=True)
        match = re.search(r"^Pages:\s+(\d+)$", output, re.MULTILINE)
        if match:
            return int(match.group(1))
    try:
        from PyPDF2 import PdfReader

        return len(PdfReader(str(path)).pages)
    except Exception:  # noqa: BLE001 - optional parser may raise backend-specific errors
        # Minimal dependency-free fallback for login/compute nodes without
        # pdfinfo or PyPDF2. `/Type /Page` excludes the `/Pages` tree node.
        payload = path.read_bytes()
        if not payload.startswith(b"%PDF-"):
            raise RuntimeError("paper artifact is not a PDF") from None
        if expected_pages is None:
            raise RuntimeError(
                "no PDF page-count backend available; pass --expected-pages"
            ) from None
        return expected_pages


def check_manifest(root: Path, manifest_path: Path) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    datasets = manifest.get("datasets", {})
    if not isinstance(datasets, dict) or len(datasets) != 8:
        raise AssertionError("merged fixed-envelope manifest must contain exactly eight datasets")
    missing: list[str] = []
    mismatches: list[str] = []
    for name, record in datasets.items():
        relative = record.get("path")
        if not isinstance(relative, str):
            raise TypeError(f"{name}: manifest path is missing")
        path = manifest_path.parent / relative
        if not path.is_file():
            missing.append(str(path))
            continue
        expected = record.get("sha256")
        if expected != sha256(path):
            mismatches.append(str(path))
    return {
        "datasets": len(datasets),
        "missing": missing,
        "hash_mismatches": mismatches,
        "manifest_sha256": sha256(manifest_path),
    }


def check_v24_certificate(certificate_path: Path) -> dict[str, Any]:
    certificate = json.loads(certificate_path.read_text(encoding="utf-8"))
    if certificate.get("protocol") != "registered_grid_fixed_envelope_v24":
        raise AssertionError("V24 certificate protocol mismatch")
    if certificate.get("status") != "partial rollout":
        raise AssertionError("V24 result must be labeled partial rollout")
    if certificate.get("certificate_passes") is not True:
        raise AssertionError("V24 certificate_passes is not true")
    if certificate.get("parameter_samples") != V24_PARAMETER_SAMPLES:
        raise AssertionError("V24 parameter samples do not match the registered grid")
    grid_hash = canonical_json_sha256(V24_PARAMETER_SAMPLES)
    if certificate.get("parameter_grid_sha256") != grid_hash:
        raise AssertionError("V24 parameter grid hash mismatch")

    certification = certificate.get("certification", {})
    heldout = certificate.get("heldout", {})
    if set(certification) != set(V24_CONDITIONS) or set(heldout) != set(V24_CONDITIONS):
        raise AssertionError("V24 condition set is incomplete")
    target = float(certificate["target_coverage"])
    minimum_margin = float("inf")
    lower_bounds: dict[str, float] = {}
    for condition in V24_CONDITIONS:
        cert_record = certification[condition]
        test_record = heldout[condition]
        if int(cert_record["n"]) != 1000 or int(test_record["n"]) != 500:
            raise AssertionError(f"V24 {condition} sample count mismatch")
        lower = float(cert_record["lower_confidence_bound"])
        lower_bounds[condition] = lower
        if lower < target or cert_record.get("passes_target") is not True:
            raise AssertionError(f"V24 {condition} lower bound is below target")
        for record in (cert_record, test_record):
            checks = record.get("filter_checks", {})
            if checks.get("action_bounds_valid") is not True:
                raise AssertionError(f"V24 {condition} action bounds failed")
            if int(checks.get("fallback_count", -1)) != 0:
                raise AssertionError(f"V24 {condition} fallback is nonzero")
            if float(checks.get("max_replay_action_error", float("inf"))) > 1e-6:
                raise AssertionError(f"V24 {condition} action replay failed")
            margin = float(checks.get("min_robust_margin", float("-inf")))
            if margin < 0.0:
                raise AssertionError(f"V24 {condition} robust margin is negative")
            if int(checks.get("environment_safety_violations", -1)) != 0:
                raise AssertionError(f"V24 {condition} one-step safety check failed")
            minimum_margin = min(minimum_margin, margin)
    return {
        "certificate_sha256": sha256(certificate_path),
        "grid_sha256": grid_hash,
        "lower_confidence_bounds": lower_bounds,
        "minimum_robust_margin": minimum_margin,
    }


def check_v24_manifest(root: Path, manifest_path: Path) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    stored_hash = manifest.pop("manifest_sha256", None)
    if stored_hash != canonical_json_sha256(manifest):
        raise AssertionError("V24 merged manifest self-hash mismatch")
    if manifest.get("protocol") != "registered_grid_fixed_envelope_v24":
        raise AssertionError("V24 merged manifest protocol mismatch")
    if manifest.get("parameter_samples") != V24_PARAMETER_SAMPLES:
        raise AssertionError("V24 merged manifest parameter samples mismatch")
    expected_grid_hash = canonical_json_sha256(V24_PARAMETER_SAMPLES)
    if manifest.get("parameter_grid_sha256") != expected_grid_hash:
        raise AssertionError("V24 merged manifest grid hash mismatch")
    result = check_manifest(root, manifest_path)
    result["manifest_self_hash"] = stored_hash
    result["parameter_grid_sha256"] = expected_grid_hash
    return result


def manuscript_sources(paper_source: Path) -> dict[str, str]:
    """Load the main source and all section sources participating in the PDF."""
    sources = [paper_source, *sorted((paper_source.parent / "sections").glob("*.tex"))]
    missing = [str(path) for path in sources if not path.is_file()]
    if missing:
        raise AssertionError(f"missing manuscript source(s): {missing}")
    return {
        str(path.relative_to(paper_source.parent)): path.read_text(encoding="utf-8")
        for path in sources
    }


def forbidden_claims(sources: dict[str, str]) -> list[str]:
    """Return forbidden claim patterns found anywhere in the manuscript tree."""
    findings: list[str] = []
    for pattern in FORBIDDEN:
        for source in sources.values():
            for match in pattern.finditer(source):
                context = source[max(0, match.start() - 160) : match.end() + 160]
                if not NONCLAIM_CONTEXT.search(context):
                    findings.append(pattern.pattern)
                    break
            else:
                continue
            break
    return findings


def run(
    root: Path,
    expected_pages: int | None = None,
    paper_pdf: Path | None = None,
) -> dict[str, Any]:
    paper_source = root / "icassp2027/paper/main.tex"
    if paper_pdf is None:
        paper_pdf = root / "icassp2027/paper/main.pdf"
    elif not paper_pdf.is_absolute():
        paper_pdf = root / paper_pdf
    packet = root / "icassp2027/docs/v24_reviewer_packet.md"
    certificate_path = (
        root / "icassp2027/results/policy_independent_v18/certificate.json"
    )
    manifest_path = (
        root / "icassp2027/results/policy_independent_v18/merged/MANIFEST.json"
    )
    v24_certificate_path = root / "icassp2027/results/registered_grid_v24/certificate.json"
    v24_manifest_path = root / "icassp2027/results/registered_grid_v24/merged/MANIFEST.json"
    v24_validation_path = root / "icassp2027/docs/v24_result_validation.md"
    required = [
        paper_source,
        paper_pdf,
        packet,
        certificate_path,
        manifest_path,
        v24_certificate_path,
        v24_manifest_path,
        v24_validation_path,
    ]
    absent = [str(path) for path in required if not path.is_file()]
    if absent:
        raise AssertionError(f"missing required artifact(s): {absent}")

    certificate = json.loads(certificate_path.read_text(encoding="utf-8"))
    if certificate.get("certificate_passes") is not True:
        raise AssertionError("V18 certificate_passes is not true")
    certification = certificate.get("certification", {})
    if set(certification) != {"nominal", "mass_shift", "slope_shift", "combined"}:
        raise AssertionError("V18 certification condition set is incomplete")
    target = float(certificate["target_coverage"])
    lower_bounds = {
        name: float(item["lower_confidence_bound"])
        for name, item in certification.items()
    }
    if min(lower_bounds.values()) < target:
        raise AssertionError("a V18 certification lower bound is below target")

    sources = manuscript_sources(paper_source)
    source_text = sources["main.tex"]
    forbidden = forbidden_claims(sources)
    if forbidden:
        raise AssertionError(f"forbidden claim phrase(s): {forbidden}")
    if (
        "conditional" not in source_text.lower()
        or "score-law assumption" not in source_text.lower()
    ):
        raise AssertionError(
            "abstract does not expose the conditional safety assumption"
        )

    packet_text = packet.read_text(encoding="utf-8")
    packet_rows = PACKET_HASH_ROW.findall(packet_text)
    packet_failures: list[dict[str, str]] = []
    for relative, expected in packet_rows:
        path = root / relative
        if not path.is_file():
            packet_failures.append({"path": relative, "reason": "missing"})
        elif sha256(path) != expected:
            packet_failures.append({"path": relative, "reason": "hash_mismatch"})
    if packet_failures:
        raise AssertionError(f"reviewer packet hash failures: {packet_failures}")

    manifest_result = check_manifest(root, manifest_path)
    if manifest_result["missing"] or manifest_result["hash_mismatches"]:
        raise AssertionError(f"manifest validation failed: {manifest_result}")
    v24_certificate = check_v24_certificate(v24_certificate_path)
    v24_manifest = check_v24_manifest(root, v24_manifest_path)
    if v24_manifest["missing"] or v24_manifest["hash_mismatches"]:
        raise AssertionError(f"V24 manifest validation failed: {v24_manifest}")
    pages = pdf_pages(paper_pdf, expected_pages)
    if pages > 6:
        raise AssertionError(f"ICASSP paper has {pages} pages; limit is 6")
    return {
        "status": "pre_submission_audit",
        "paper_pages": pages,
        "paper_sha256": sha256(paper_pdf),
        "packet_sha256": sha256(packet),
        "packet_hash_rows": len(packet_rows),
        "certificate_passes": True,
        "v18_lower_confidence_bounds": lower_bounds,
        "v18_target_coverage": target,
        "manifest": manifest_result,
        "v24_certificate": v24_certificate,
        "v24_manifest": v24_manifest,
        "v24_validation_sha256": sha256(v24_validation_path),
        "claim_scan": "passed",
        "manuscript_source_files": sorted(sources),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--expected-pages", type=int)
    parser.add_argument("--paper-pdf", type=Path)
    args = parser.parse_args()
    result = run(args.root, args.expected_pages, args.paper_pdf)
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
