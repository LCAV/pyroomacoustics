"""
Tests that ``open_sofa_file`` releases the underlying SOFA/netCDF4 handle.

Regression test for https://github.com/LCAV/pyroomacoustics/issues/410
"""

import numpy as np
import pytest

sofa = pytest.importorskip("sofa")

from pyroomacoustics.datasets.sofa import DEFAULT_SOFA_PATH
from pyroomacoustics.directivities import sofa as pra_sofa

# one DIRPAT file and one file following the SOFA conventions
SOFA_FILES = [
    "AKG_c480_c414_CUBE",
    "mit_kemar_normal_pinna",
]


@pytest.fixture
def opened_databases(monkeypatch):
    """Record every database opened by ``open_sofa_file``."""
    opened = []
    original_open = sofa.Database.open

    def recording_open(*args, **kwargs):
        db = original_open(*args, **kwargs)
        opened.append(db)
        return db

    monkeypatch.setattr(sofa.Database, "open", recording_open)
    return opened


def _sofa_path(name):
    path = DEFAULT_SOFA_PATH / f"{name}.sofa"
    if not path.exists():
        pytest.skip(f"SOFA file {path} is not available")
    return path


@pytest.mark.parametrize("name", SOFA_FILES)
def test_open_sofa_file_closes_handle(name, opened_databases):
    path = _sofa_path(name)

    ir, fs, src_loc, mic_loc, _, _ = pra_sofa.open_sofa_file(path)

    assert len(opened_databases) == 1
    db = opened_databases[0]
    assert db.dataset is None

    # the returned data must remain usable after the file is closed
    assert ir.ndim == 3
    assert fs > 0
    assert ir.shape[0] in src_loc.shape
    assert ir.shape[1] in mic_loc.shape
    assert np.all(np.isfinite(ir))
    assert np.any(ir != 0.0)


def test_open_sofa_file_closes_handle_on_error(opened_databases, monkeypatch):
    path = _sofa_path("mit_kemar_normal_pinna")

    def failing_reader(*args, **kwargs):
        raise RuntimeError("reader failed")

    monkeypatch.setattr(pra_sofa, "_read_simple_free_field_hrir", failing_reader)

    with pytest.raises(RuntimeError, match="reader failed"):
        pra_sofa.open_sofa_file(path)

    assert len(opened_databases) == 1
    assert opened_databases[0].dataset is None
