import pytest

from src.validate_labels import parse_labels, inspect_dataset


@pytest.mark.parametrize('text', [
    '11 .5 .5 .2 .2', '-1 .5 .5 .2 .2', '1.5 .5 .5 .2 .2',
    '1 nan .5 .2 .2', '1 .5 .5 0 .2', '1 .1 .5 .4 .2',
    '1 .5 .5 .2 .2 extra', '1 .5 .5 .2', '1 inf .5 .2 .2',
])
def test_bad_boxes_rejected(text):
    assert parse_labels(text, 11)[1]


def test_valid_boundary_and_empty():
    assert not parse_labels('1 .5 .5 1 1', 11)[1]
    assert parse_labels('', 11) == ([], [])


def test_duplicates_rejected():
    assert parse_labels('1 .5 .5 .2 .2\n1 .5 .5 .2 .2', 11)[1]


def test_broken_image_missing_and_orphan(tmp_path):
    (tmp_path / 'images').mkdir()
    (tmp_path / 'labels').mkdir()
    (tmp_path / 'images/bad.jpg').write_bytes(b'broken')
    (tmp_path / 'labels/orphan.txt').write_text('')
    records, orphans = inspect_dataset(tmp_path, ['car'])
    assert 'missing annotation' in records[0]['errors']
    assert any('broken image' in e for e in records[0]['errors'])
    assert orphans == ['orphan.txt']
