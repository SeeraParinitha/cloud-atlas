"""Download pinned source and make explicit import/device compatibility changes."""
import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REVISION = "8d59013b7f9ecaf3e048b1d9dc7eac85805b0b38"
source = ROOT / "third_party" / "HSFormer-CR"
if not source.exists():
    subprocess.run(["git", "clone", "https://github.com/qi657/HSFormer-CR.git", str(source)], check=True)
    subprocess.run(["git", "-C", str(source), "checkout", "--detach", REVISION], check=True)
actual = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
if actual != REVISION:
    raise RuntimeError(f"Expected upstream commit {REVISION}, got {actual}; inspect before changing pin")
target = ROOT / "third_party" / "hsformer_runtime"
target.mkdir(parents=True, exist_ok=True)
(target / "__init__.py").write_text("", encoding="utf-8")
hashes = {}
for name in ("HSFormer.py", "base_net_snow.py", "condconv.py"):
    code = (source / "models" / name).read_text(encoding="utf-8")
    if name == "HSFormer.py":
        code = code.split("from ptflops import get_model_complexity_info")[0]
        code = code.replace('if torch.cuda.is_available():\n    device = torch.device("cuda:0")',
                            'device = torch.device("cpu")')
    (target / name).write_text(code, encoding="utf-8")
    hashes[name] = hashlib.sha256(code.encode()).hexdigest()
(target / "provenance.json").write_text(json.dumps({"revision": REVISION,
    "sha256": hashes, "changes": ["Remove import-time GPU benchmark", "Construct coordinate tensors on CPU; upstream forward moves them to input device"]}, indent=2))
print("Prepared pinned HSFormer RGB runtime:", target)
