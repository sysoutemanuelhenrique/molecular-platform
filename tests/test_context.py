from pathlib import Path

import pytest

from molecular.context import (
    AmbiguousExperimentError,
    ExperimentNotFoundError,
    resolve_experiment,
    scan_experiments,
)


def _write_experiment_yaml(project: Path, experiment_id: str, schema_version: str = "1.1") -> None:
    project.mkdir(parents=True, exist_ok=True)
    (project / "experiment.yaml").write_text(
        f'schema_version: "{schema_version}"\n'
        f'experiment_id: "{experiment_id}"\n',
        encoding="utf-8",
    )


def test_resolve_finds_experiment_when_directory_name_equals_id(tmp_path):
    _write_experiment_yaml(tmp_path / "exp001", experiment_id="exp001")

    context = resolve_experiment("exp001", base_dir=tmp_path)

    assert context.experiment_id == "exp001"
    assert context.project_dir == tmp_path / "exp001"


def test_resolve_finds_experiment_when_directory_name_differs_from_id(tmp_path):
    _write_experiment_yaml(tmp_path / "some_folder", experiment_id="exp001")

    context = resolve_experiment("exp001", base_dir=tmp_path)

    assert context.experiment_id == "exp001"
    assert context.project_dir == tmp_path / "some_folder"


def test_resolve_raises_not_found_for_unknown_id(tmp_path):
    with pytest.raises(ExperimentNotFoundError):
        resolve_experiment("nao_existe", base_dir=tmp_path)


def test_resolve_directory_name_alone_never_satisfies_lookup(tmp_path):
    """A directory literally named after the identifier, but declaring a
    different experiment_id, must not resolve - identity comes only from
    experiment_id, never from the directory name."""

    _write_experiment_yaml(tmp_path / "exp001", experiment_id="totally-different-id")

    with pytest.raises(ExperimentNotFoundError):
        resolve_experiment("exp001", base_dir=tmp_path)


def test_resolve_is_case_sensitive(tmp_path):
    _write_experiment_yaml(tmp_path / "exp001", experiment_id="exp001")

    with pytest.raises(ExperimentNotFoundError):
        resolve_experiment("EXP001", base_dir=tmp_path)


def test_resolve_raises_ambiguous_for_duplicate_ids(tmp_path):
    _write_experiment_yaml(tmp_path / "folder_a", experiment_id="dup001")
    _write_experiment_yaml(tmp_path / "folder_b", experiment_id="dup001")

    with pytest.raises(AmbiguousExperimentError) as excinfo:
        resolve_experiment("dup001", base_dir=tmp_path)

    assert sorted(excinfo.value.directories) == ["folder_a", "folder_b"]


def test_invalid_candidate_does_not_block_discovery_of_valid_ones(tmp_path):
    _write_experiment_yaml(tmp_path / "good_exp", experiment_id="good")

    broken = tmp_path / "broken_exp"
    broken.mkdir()
    (broken / "experiment.yaml").write_text("experiment_id: [unclosed\n", encoding="utf-8")

    valid, skipped = scan_experiments(base_dir=tmp_path)

    assert [context.experiment_id for context in valid] == ["good"]
    assert skipped == ["broken_exp"]

    context = resolve_experiment("good", base_dir=tmp_path)
    assert context.project_dir == tmp_path / "good_exp"


def test_invalid_schema_candidate_does_not_block_discovery_of_valid_ones(tmp_path):
    _write_experiment_yaml(tmp_path / "good_exp", experiment_id="good")
    _write_experiment_yaml(tmp_path / "invalid_exp", experiment_id="bad", schema_version="2.0")

    valid, skipped = scan_experiments(base_dir=tmp_path)

    assert [context.experiment_id for context in valid] == ["good"]
    assert skipped == ["invalid_exp"]


def test_scan_ignores_directories_without_experiment_yaml(tmp_path):
    _write_experiment_yaml(tmp_path / "exp001", experiment_id="exp001")
    (tmp_path / "not_an_experiment").mkdir()

    valid, skipped = scan_experiments(base_dir=tmp_path)

    assert [context.experiment_id for context in valid] == ["exp001"]
    assert skipped == []


def test_scan_and_resolve_perform_no_writes(tmp_path):
    """Reading experiment identity must never have scientific side effects:
    no files created, no experiment.yaml touched, no logs written."""

    _write_experiment_yaml(tmp_path / "exp001", experiment_id="exp001")
    before = {
        path: path.stat().st_mtime_ns
        for path in tmp_path.rglob("*")
        if path.is_file()
    }

    scan_experiments(base_dir=tmp_path)
    resolve_experiment("exp001", base_dir=tmp_path)

    after_paths = {path for path in tmp_path.rglob("*") if path.is_file()}
    assert after_paths == set(before)
    for path, mtime in before.items():
        assert path.stat().st_mtime_ns == mtime

    assert not (tmp_path / "exp001" / "logs").exists()
