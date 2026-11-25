import json
from pathlib import Path

from src.prepare_dataset import split_groups, prepare
from src.utils import ROOT, sha256


def test_scene_split_reproducible_and_disjoint():
    groups = json.loads((ROOT / 'configs/scene_groups.json').read_text(encoding='utf-8'))
    first = split_groups(groups, 42)
    assert first == split_groups(groups, 42)
    assert first != split_groups(groups, 7)
    assert sorted(sum(first.values(), [])) == sorted(groups)
    seen = {}
    for split, names in first.items():
        assert names
        for name in names:
            group = groups[name]
            assert group not in seen or seen[group] == split
            seen[group] = split


def test_full_preparation_is_reproducible(tmp_path):
    source = ROOT / 'test_data/traffic_v5_mini'
    before = {p: sha256(p) for p in source.rglob('*') if p.is_file()}
    for index in (1, 2):
        prepare(source, tmp_path / f'data{index}', 42, tmp_path / f'reports{index}')
    assert (tmp_path / 'reports1/manifest.sha256').read_bytes() == (tmp_path / 'reports2/manifest.sha256').read_bytes()
    assert before == {p: sha256(p) for p in before}
    for line in (tmp_path / 'reports1/manifest.sha256').read_text(encoding='utf-8').splitlines():
        digest, filename = line.split('  ', 1)
        assert sha256(tmp_path / 'data1' / filename) == digest
