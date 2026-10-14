"""Tests for receptor_service.py - Increment 020, Checkpoint 3.

Covers T-081-090 (preserved/rewritten) and T-131-135 (new) from the frozen
final specification's normative matrix (Section 35.6). Each test is
labeled with its exact scenario id, and every test calls real production
code: a real `ProteinInput` is registered via `molecular.registry.
register_protein`, a real minimal mmCIF/PDB fixture is parsed by the real
`gemmi`-backed `receptor_parsing.py`, and `inspect_receptor`/
`declare_receptor_selection`/`resolve_receptor_selection` are the actual
Application Service entry points under test - never a test-local stand-in
for any of the three.
"""

from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from molecular.atomic import compute_sha256
from molecular.receptor_inspection import ReceptorInspectionReport
from molecular.receptor_registry import (
    ReceptorInspectionNotFoundError,
    ReceptorInspectionRecordInvalidError,
    _receptor_inspections_dir,
    _resolved_receptor_selections_dir,
)
from molecular.receptor_selection import ChainSelector
from molecular.receptor_service import (
    AmbiguousModelSelectionError,
    ChainNotFoundError,
    InspectionSelectionUnsupportedError,
    ReceptorInputNotFoundError,
    ReceptorPersistenceError,
    ReceptorSelectionConflictError,
    declare_receptor_selection,
    inspect_receptor,
    load_receptor_inspection,
    load_receptor_selection,
    load_resolved_receptor_selection,
    resolve_receptor_selection,
)
from molecular.receptor_parsing import RawChecksumMismatchError
from molecular.registry import register_protein

# --- fixture helpers -------------------------------------------------------

_ATOM_SITE_HEADER = """\
loop_
_atom_site.group_PDB
_atom_site.id
_atom_site.type_symbol
_atom_site.label_atom_id
_atom_site.label_alt_id
_atom_site.label_comp_id
_atom_site.label_asym_id
_atom_site.label_entity_id
_atom_site.label_seq_id
_atom_site.pdbx_PDB_ins_code
_atom_site.Cartn_x
_atom_site.Cartn_y
_atom_site.Cartn_z
_atom_site.occupancy
_atom_site.B_iso_or_equiv
_atom_site.auth_seq_id
_atom_site.auth_asym_id
_atom_site.pdbx_PDB_model_num
"""


def _minimal_cif() -> str:
    return f"""\
data_TEST
#
loop_
_entity.id
_entity.type
_entity.pdbx_description
1 polymer 'Test protein'
#
{_ATOM_SITE_HEADER}\
ATOM 1 N N . ALA A 1 1 ? 1.0 1.0 1.0 1.00 10.0 1 A 1
ATOM 2 C CA . ALA A 1 1 ? 2.0 1.0 1.0 1.00 10.0 1 A 1
#
"""


def _two_model_cif() -> str:
    return f"""\
data_TEST
#
{_ATOM_SITE_HEADER}\
ATOM 1 N N . ALA A 1 1 ? 1.0 1.0 1.0 1.00 10.0 1 A 1
ATOM 2 N N . ALA A 1 1 ? 1.0 1.0 1.0 1.00 10.0 1 A 2
#
"""


def _pdb_atom(record, serial, name, resname, chain, resseq, occ=1.00, elem=None, icode=" "):
    elem = elem or name.strip()[0]
    name_field = (" " + name).ljust(4) if len(name) < 4 else name
    return (
        f"{record:<6}{serial:>5} {name_field:<4} {resname:<3} {chain:<1}{resseq:>4}{icode:<1}   "
        f"{1.0:>8.3f}{1.0:>8.3f}{1.0:>8.3f}{occ:>6.2f}{10.0:>6.2f}          {elem:>2}"
    )


def _minimal_pdb() -> str:
    lines = [
        _pdb_atom("ATOM", 1, "N", "ALA", "A", 1),
        _pdb_atom("ATOM", 2, "CA", "ALA", "A", 1),
        "END",
    ]
    return "\n".join(lines) + "\n"


def _register_demo_protein(
    project_dir: Path, *, raw_text: str, input_id: str = "demo-protein", raw_name: str = "demo.cif"
):
    raw_file = project_dir / raw_name
    raw_file.write_text(raw_text, encoding="utf-8")
    return register_protein(
        project_dir,
        input_id=input_id,
        name="Demo protein",
        method="uploaded",
        raw_file=raw_file,
    )


def _declare_minimal_selection(project_dir: Path, *, selection_id="sel-001", **overrides):
    defaults = dict(
        protein_input_id="demo-protein",
        selected_model=1,
        selected_assembly=None,
        selected_chains=(ChainSelector(label_asym_id="A"),),
        water_disposition=None,
        metal_disposition=None,
        component_dispositions=(),
        declared_by="Dr. Jane Doe",
        label=None,
        rationale=None,
    )
    defaults.update(overrides)
    return declare_receptor_selection(project_dir, selection_id=selection_id, **defaults)


# --- T-081-082: inspect_receptor ------------------------------------------


def test_inspect_receptor_missing_protein_input_raises_not_found(tmp_path):  # T-081
    with pytest.raises(ReceptorInputNotFoundError):
        inspect_receptor(tmp_path, protein_input_id="does-not-exist")


def test_inspect_receptor_mutated_raw_raises_checksum_mismatch_no_persistence(tmp_path):  # T-082
    protein = _register_demo_protein(tmp_path, raw_text=_minimal_cif())
    raw_path = tmp_path / protein.acquisition.raw_artifact.path
    raw_path.write_text(_minimal_cif() + "# mutated\n", encoding="utf-8")

    with pytest.raises(RawChecksumMismatchError):
        inspect_receptor(tmp_path, protein_input_id="demo-protein")

    assert not _receptor_inspections_dir(tmp_path).exists() or not any(
        _receptor_inspections_dir(tmp_path).iterdir()
    )


# --- T-083-084: declare_receptor_selection ---------------------------------


def test_declare_receptor_selection_identical_retry_is_reused(tmp_path):  # T-083
    _register_demo_protein(tmp_path, raw_text=_minimal_cif())

    first = _declare_minimal_selection(tmp_path)
    second = _declare_minimal_selection(tmp_path)

    assert first.reused is False
    assert second.reused is True
    assert second.policy == first.policy


def test_declare_receptor_selection_divergent_retry_raises_conflict(tmp_path):  # T-084
    _register_demo_protein(tmp_path, raw_text=_minimal_cif())

    _declare_minimal_selection(tmp_path, declared_by="Dr. Jane Doe")

    with pytest.raises(ReceptorSelectionConflictError):
        _declare_minimal_selection(tmp_path, declared_by="Dr. John Roe")


# --- T-085-086: resolve_receptor_selection edge cases ----------------------


def test_resolve_receptor_selection_ambiguous_model_raises(tmp_path):  # T-085
    _register_demo_protein(tmp_path, raw_text=_two_model_cif())
    _declare_minimal_selection(tmp_path, selected_model=None, selected_chains=())

    with pytest.raises(AmbiguousModelSelectionError):
        resolve_receptor_selection(tmp_path, selection_id="sel-001")

    assert not any(_resolved_receptor_selections_dir(tmp_path).glob("*.yaml"))


def test_resolve_receptor_selection_missing_chain_raises_chain_not_found(tmp_path):  # T-086
    _register_demo_protein(tmp_path, raw_text=_minimal_cif())
    _declare_minimal_selection(
        tmp_path, selected_chains=(ChainSelector(label_asym_id="ZZZ"),)
    )

    with pytest.raises(ChainNotFoundError):
        resolve_receptor_selection(tmp_path, selection_id="sel-001")

    assert not any(_resolved_receptor_selections_dir(tmp_path).glob("*.yaml"))


# --- T-087-088: resolve_receptor_selection retry/revalidation --------------


def test_resolve_receptor_selection_identical_retry_is_reused_no_rewrite(tmp_path):  # T-087
    _register_demo_protein(tmp_path, raw_text=_minimal_cif())
    _declare_minimal_selection(tmp_path)

    first = resolve_receptor_selection(tmp_path, selection_id="sel-001")
    resolved_file = _resolved_receptor_selections_dir(tmp_path) / "sel-001.yaml"
    content_before = resolved_file.read_text(encoding="utf-8")
    mtime_before = resolved_file.stat().st_mtime_ns

    second = resolve_receptor_selection(tmp_path, selection_id="sel-001")

    assert first.reused is False
    assert second.reused is True
    assert resolved_file.read_text(encoding="utf-8") == content_before
    assert resolved_file.stat().st_mtime_ns == mtime_before


def test_resolve_receptor_selection_raw_altered_since_inspection_raises_checksum_mismatch(
    tmp_path,
):  # T-088
    protein = _register_demo_protein(tmp_path, raw_text=_minimal_cif())
    _declare_minimal_selection(tmp_path)

    inspection = load_receptor_inspection(
        tmp_path, load_receptor_selection(tmp_path, "sel-001").inspection_id
    )
    inspection_file = _receptor_inspections_dir(tmp_path) / f"{inspection.inspection_id}.yaml"
    content_before = inspection_file.read_text(encoding="utf-8")

    raw_path = tmp_path / protein.acquisition.raw_artifact.path
    raw_path.write_text(_minimal_cif() + "# altered after inspection\n", encoding="utf-8")

    with pytest.raises(RawChecksumMismatchError):
        resolve_receptor_selection(tmp_path, selection_id="sel-001")

    assert inspection_file.read_text(encoding="utf-8") == content_before


# --- T-089: inspect_receptor under a different parser version -------------


def test_inspect_receptor_different_parser_version_yields_new_id_no_conflict_class(
    tmp_path, monkeypatch
):  # T-089
    _register_demo_protein(tmp_path, raw_text=_minimal_cif())

    first = inspect_receptor(tmp_path, protein_input_id="demo-protein")
    assert first.created is True

    import gemmi

    monkeypatch.setattr(gemmi, "__version__", "9.9.9")

    second = inspect_receptor(tmp_path, protein_input_id="demo-protein")
    assert second.created is True
    assert second.report.inspection_id != first.report.inspection_id

    # the old inspection, under its own id, remains intact and loadable
    reloaded_first = load_receptor_inspection(tmp_path, first.report.inspection_id)
    assert reloaded_first == first.report

    # the eliminated class (Section 25.4) textually never reappears anywhere
    # in this checkpoint's own source files
    for module_name in ("receptor_service", "receptor_registry", "receptor_resolution"):
        source = (Path(__file__).parent.parent / "src" / "molecular" / f"{module_name}.py").read_text(
            encoding="utf-8"
        )
        assert "ReceptorResolutionParserVersionConflictError" not in source


# --- T-090: activity logging shape -----------------------------------------


def test_each_service_call_logs_exactly_one_started_and_one_terminal_entry(tmp_path):  # T-090
    _register_demo_protein(tmp_path, raw_text=_minimal_cif())

    inspect_receptor(tmp_path, protein_input_id="demo-protein")
    log_lines = (tmp_path / "logs" / "activity.log").read_text(encoding="utf-8").splitlines()
    inspect_lines = [line for line in log_lines if "command=receptor-inspect" in line]
    assert sum(1 for line in inspect_lines if "outcome=started" in line) == 1
    assert sum(1 for line in inspect_lines if "outcome=success" in line or "outcome=error" in line) == 1

    declare_receptor_selection(
        tmp_path,
        selection_id="sel-001",
        protein_input_id="demo-protein",
        selected_model=1,
        selected_assembly=None,
        selected_chains=(ChainSelector(label_asym_id="A"),),
        water_disposition=None,
        metal_disposition=None,
        component_dispositions=(),
        declared_by="Dr. Jane Doe",
        label=None,
        rationale=None,
    )
    log_lines = (tmp_path / "logs" / "activity.log").read_text(encoding="utf-8").splitlines()
    declare_lines = [line for line in log_lines if "command=receptor-selection-declare" in line]
    assert sum(1 for line in declare_lines if "outcome=started" in line) == 1
    assert sum(1 for line in declare_lines if "outcome=success" in line or "outcome=error" in line) == 1

    resolve_receptor_selection(tmp_path, selection_id="sel-001")
    log_lines = (tmp_path / "logs" / "activity.log").read_text(encoding="utf-8").splitlines()
    resolve_lines = [line for line in log_lines if "command=receptor-selection-resolve" in line]
    assert sum(1 for line in resolve_lines if "outcome=started" in line) == 1
    assert sum(1 for line in resolve_lines if "outcome=success" in line or "outcome=error" in line) == 1


# --- T-131-135 (new) --------------------------------------------------------


def test_declare_receptor_selection_against_inspection_only_raises_unsupported(tmp_path):  # T-131
    _register_demo_protein(tmp_path, raw_text=_minimal_pdb(), raw_name="demo.pdb")

    with pytest.raises(InspectionSelectionUnsupportedError):
        _declare_minimal_selection(tmp_path)

    policy_file = tmp_path / "scientific" / "receptor_selections" / "sel-001.yaml"
    assert not policy_file.exists()


def test_declare_receptor_selection_copies_inspection_id_and_checksum_from_inspect(tmp_path):  # T-132
    _register_demo_protein(tmp_path, raw_text=_minimal_cif())

    inspect_outcome = inspect_receptor(tmp_path, protein_input_id="demo-protein")
    declare_outcome = _declare_minimal_selection(tmp_path)

    assert declare_outcome.policy.inspection_id == inspect_outcome.report.inspection_id
    assert declare_outcome.policy.raw_checksum_observed == inspect_outcome.report.raw_checksum


def test_resolve_receptor_selection_missing_pinned_inspection_raises_not_found(tmp_path):  # T-133
    _register_demo_protein(tmp_path, raw_text=_minimal_cif())
    _declare_minimal_selection(tmp_path)

    policy = load_receptor_selection(tmp_path, "sel-001")
    inspection_file = _receptor_inspections_dir(tmp_path) / f"{policy.inspection_id}.yaml"
    inspection_file.unlink()

    with pytest.raises(ReceptorInspectionNotFoundError):
        resolve_receptor_selection(tmp_path, selection_id="sel-001")

    assert not any(_resolved_receptor_selections_dir(tmp_path).glob("*.yaml"))


def test_resolve_receptor_selection_corrupted_pinned_inspection_raises_record_invalid(
    tmp_path,
):  # T-134
    _register_demo_protein(tmp_path, raw_text=_minimal_cif())
    _declare_minimal_selection(tmp_path)

    policy = load_receptor_selection(tmp_path, "sel-001")
    inspection_file = _receptor_inspections_dir(tmp_path) / f"{policy.inspection_id}.yaml"
    inspection_file.write_text("not: [valid, yaml", encoding="utf-8")

    with pytest.raises(ReceptorInspectionRecordInvalidError) as excinfo:
        resolve_receptor_selection(tmp_path, selection_id="sel-001")

    assert excinfo.value.__cause__ is not None
    assert not any(_resolved_receptor_selections_dir(tmp_path).glob("*.yaml"))


def test_resolve_receptor_selection_revalidates_raw_checksum_against_pinned_report(
    tmp_path,
):  # T-135
    protein = _register_demo_protein(tmp_path, raw_text=_minimal_cif())
    _declare_minimal_selection(tmp_path)

    policy = load_receptor_selection(tmp_path, "sel-001")
    inspection_before = load_receptor_inspection(tmp_path, policy.inspection_id)

    raw_path = tmp_path / protein.acquisition.raw_artifact.path
    raw_path.write_text(_minimal_cif() + "# tampered\n", encoding="utf-8")

    with pytest.raises(RawChecksumMismatchError):
        resolve_receptor_selection(tmp_path, selection_id="sel-001")

    assert not any(_resolved_receptor_selections_dir(tmp_path).glob("*.yaml"))
    inspection_after = load_receptor_inspection(tmp_path, policy.inspection_id)
    assert inspection_after == inspection_before
