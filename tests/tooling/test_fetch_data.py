"""Tests for scripts/fetch_data.py on a dataset of its own: the data folder and the downloads
replaced, so nothing leaves the machine. A present file is checked against its pinned sum, a
missing archive member stops with the member's name, a download that fails while it is written
leaves no file behind, and `verify` reports every file whose sum differs."""

import hashlib
import io
import zipfile

import pytest
from conftest import _fetch_data

fetch_data = _fetch_data()                                # scripts/fetch_data.py, the module conftest loads on first use


def _sha(b):
    return hashlib.sha256(b).hexdigest()


@pytest.fixture
def toy(tmp_path, monkeypatch):
    """A dataset 'toy' of two files, one plain and one from an archive."""
    blob = io.BytesIO()
    with zipfile.ZipFile(blob, "w") as z:
        z.writestr("inner/b.txt", b"bee")
    blob = blob.getvalue()
    monkeypatch.setattr(fetch_data, "DATA", tmp_path)
    monkeypatch.setitem(fetch_data.DATASETS, "toy", ("toy", [("a.txt", "http://x/a.txt", _sha(b"ay")),
                                                             ("b.txt", "zip:inner/b.txt", _sha(b"bee"))]))
    monkeypatch.setitem(fetch_data.ARCHIVES, "toy", ("http://x/t.zip", _sha(blob)))
    bodies = {"http://x/a.txt": b"ay", "http://x/t.zip": blob}
    monkeypatch.setattr(fetch_data, "download", lambda url: bodies[url])
    return tmp_path / "toy", bodies


def test_fetch_writes_the_files_and_a_second_fetch_checks_them(toy):
    folder, _ = toy
    fetch_data.fetch("toy")
    assert (folder / "a.txt").read_bytes() == b"ay" and (folder / "b.txt").read_bytes() == b"bee"
    assert fetch_data.present("toy") and fetch_data.verify("toy") == []
    (folder / "a.txt").write_bytes(b"changed")
    assert fetch_data.verify("toy") == ["a.txt"]
    with pytest.raises(SystemExit, match="SHA-256 differs"):
        fetch_data.fetch("toy")


def test_a_missing_archive_member_stops_with_its_name(toy, monkeypatch):
    _folder, _ = toy
    monkeypatch.setitem(fetch_data.DATASETS, "toy", ("toy", [("c.txt", "zip:inner/c.txt", _sha(b"c"))]))
    with pytest.raises(SystemExit, match="inner/c.txt"):
        fetch_data.fetch("toy")


def test_a_write_that_fails_leaves_no_file(toy, monkeypatch):
    folder, _ = toy

    def broken(self, data):
        self.open("wb").write(data[:1])
        raise OSError("disk full")
    monkeypatch.setattr(fetch_data.Path, "write_bytes", broken)
    with pytest.raises(OSError):
        fetch_data.fetch("toy")
    assert not (folder / "a.txt").exists()
