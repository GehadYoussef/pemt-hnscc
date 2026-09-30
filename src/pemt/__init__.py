"""Shared helpers for the HNSCC pEMT pipeline.

Each script in src/<stage>/ puts src/ on sys.path and imports the package:

    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from pemt import load_config, load_signatures, load_registry, nan_guard
    cfg = load_config()
"""

from pathlib import Path
import sys
import yaml


_HERE = Path(__file__).resolve().parent
_DEFAULT_ROOT = _HERE.parent.parent


def project_root(start=None) -> Path:
    """Return the directory containing config/config.yaml."""
    if start is None:
        if (_DEFAULT_ROOT / "config" / "config.yaml").exists():
            return _DEFAULT_ROOT
        start = sys.argv[0] or str(Path.cwd())
    p = Path(start).resolve()
    if p.is_file():
        p = p.parent
    for parent in [p, *p.parents]:
        if (parent / "config" / "config.yaml").exists():
            return parent
    return Path.cwd()


def load_config() -> dict:
    with open(project_root() / "config" / "config.yaml") as fh:
        return yaml.safe_load(fh)


def _read_signature_file(path: Path) -> list[str]:
    """Read a one-gene-per-line signature file. Skip blanks and # comments."""
    genes = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        genes.append(line)
    return genes


def load_signatures() -> dict[str, list[str]]:
    """Load gene signatures.

    Sources merged in this order (later wins):
        1. config/signatures.yaml          (lineage markers)
        2. config/signatures/*.txt         (one gene per line)
    """
    root = project_root()
    sigs: dict[str, list[str]] = {}

    yaml_path = root / "config" / "signatures.yaml"
    if yaml_path.exists():
        loaded = yaml.safe_load(yaml_path.read_text()) or {}
        for k, v in loaded.items():
            if isinstance(v, list):
                sigs[k] = list(v)

    sig_dir = root / "config" / "signatures"
    if sig_dir.is_dir():
        for txt in sorted(sig_dir.glob("*.txt")):
            sigs[txt.stem] = _read_signature_file(txt)

    return sigs


def load_registry():
    import pandas as pd
    return pd.read_csv(project_root() / "config" / "dataset_registry.tsv", sep="\t")


def nan_guard(arr):
    """NaN/inf scrubber for use inside a sklearn FunctionTransformer.

    Defined here so the fitted classifier pipeline stores a stable, fully
    qualified module path (pemt.nan_guard). Scripts that load the saved model
    need src/ on sys.path and nothing else.
    """
    import numpy as np
    return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)


__all__ = ["project_root", "load_config", "load_signatures", "load_registry", "nan_guard"]
