"""Regression test — ``import custody.detection`` works without numba.cuda.

VLM-backend users do not need CUDA; the annular-percentile CUDA kernel lives
in its own submodule so ``custody.detection.__init__`` stays importable on
machines without numba-cuda installed.

This test runs in a subprocess because we need to simulate the "numba-cuda
not installed" environment.  We do that by installing an ImportError-raising
meta-path finder for ``numba.cuda*`` *before* any custody import.  Subprocess
isolation guarantees the rigged sys.modules state does not leak back into
the parent test session.
"""
from __future__ import annotations

import subprocess
import sys
import textwrap
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC = _REPO_ROOT / "src"


def _run_child(script: str) -> subprocess.CompletedProcess:
    """Run ``script`` in a fresh interpreter with src/ on the path."""
    return subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        env={**_child_env()},
        cwd=str(_REPO_ROOT),
    )


def _child_env() -> dict:
    import os

    env = dict(os.environ)
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = (
        str(_SRC) + (os.pathsep + existing if existing else "")
    )
    return env


_BLOCK_CUDA_SHIM = textwrap.dedent(
    """
    import sys
    from importlib.abc import MetaPathFinder

    class _BlockNumbaCuda(MetaPathFinder):
        def find_spec(self, fullname, path=None, target=None):
            if fullname == "numba.cuda" or fullname.startswith("numba.cuda."):
                raise ImportError(
                    f"simulated: {fullname} unavailable for regression test"
                )
            return None

    sys.meta_path.insert(0, _BlockNumbaCuda())
    """
)


def test_custody_detection_imports_without_numba_cuda():
    """import custody.detection must succeed when numba.cuda is unavailable."""
    script = _BLOCK_CUDA_SHIM + textwrap.dedent(
        """
        import custody.detection  # must not raise

        # Also verify the CPU annular-percentile module is importable + callable.
        import numpy as np
        from custody.detection.annular_percentile import annular_percentile_filter
        img = np.zeros((32, 32), dtype=np.float32)
        _ = annular_percentile_filter(img, guard=3, reference=5, percentile=75.0)

        # And the GPU module's import SHOULD fail informatively when numba.cuda
        # is absent — this is the contract in annular_percentile_gpu.
        try:
            import custody.detection.annular_percentile_gpu  # noqa: F401
        except ImportError as e:
            msg = str(e)
            assert "numba-cuda" in msg or "CUDA runtime" in msg, (
                f"ImportError message should name the missing dep; got: {msg}"
            )
        else:
            raise AssertionError(
                "annular_percentile_gpu imported despite numba.cuda being blocked"
            )

        print("OK")
        """
    )
    result = _run_child(script)
    assert result.returncode == 0, (
        f"child failed: returncode={result.returncode}\n"
        f"stdout: {result.stdout}\nstderr: {result.stderr}"
    )
    assert "OK" in result.stdout, f"unexpected stdout: {result.stdout!r}"
