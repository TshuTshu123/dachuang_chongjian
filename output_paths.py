"""Shared discovery for reports relocated under outputs/."""


def latest_report(root, stem="export_00000"):
    """Find the newest full-cloud report; support old root-level outputs too."""
    candidates = list(root.glob(f"{stem}*_report.json"))
    output_root = root / "outputs"
    if output_root.is_dir():
        candidates.extend(output_root.rglob(f"{stem}*_report.json"))
    candidates = [p for p in candidates if p.is_file() and not p.name.startswith(f"{stem}_roi")]
    return max(candidates, key=lambda p: p.stat().st_mtime_ns) if candidates else None
