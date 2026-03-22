"""Unit tests for LocalFileLoader."""

import pytest

from rag.infra.loading.local_file_loader import LocalFileLoader, RawArtifact


@pytest.fixture
def loader():
    return LocalFileLoader()


@pytest.fixture
def sample_txt(tmp_path):
    f = tmp_path / "sample.txt"
    f.write_text("Hello, RAG framework!\nLine two.", encoding="utf-8")
    return f


@pytest.mark.unit
class TestLocalFileLoader:
    def test_load_txt_file_returns_raw_artifact(self, loader, sample_txt):
        artifact = loader.load(sample_txt)
        assert isinstance(artifact, RawArtifact)

    def test_load_returns_correct_bytes(self, loader, sample_txt):
        artifact = loader.load(sample_txt)
        assert artifact.raw_bytes == b"Hello, RAG framework!\nLine two."

    def test_load_text_property(self, loader, sample_txt):
        artifact = loader.load(sample_txt)
        assert "Hello, RAG framework!" in artifact.text
        assert "Line two." in artifact.text

    def test_load_metadata_file_name(self, loader, sample_txt):
        artifact = loader.load(sample_txt)
        assert artifact.metadata["file_name"] == "sample.txt"

    def test_load_metadata_extension(self, loader, sample_txt):
        artifact = loader.load(sample_txt)
        assert artifact.metadata["extension"] == ".txt"

    def test_load_metadata_file_size(self, loader, sample_txt):
        artifact = loader.load(sample_txt)
        assert artifact.metadata["file_size_bytes"] == len(b"Hello, RAG framework!\nLine two.")
        assert artifact.size_bytes == artifact.metadata["file_size_bytes"]

    def test_load_metadata_modified_at(self, loader, sample_txt):
        artifact = loader.load(sample_txt)
        assert "modified_at" in artifact.metadata
        assert "T" in artifact.metadata["modified_at"]  # ISO 8601

    def test_load_source_path_is_absolute(self, loader, sample_txt):
        artifact = loader.load(sample_txt)
        assert artifact.source_path == str(sample_txt.resolve())

    def test_load_missing_file_raises(self, loader, tmp_path):
        with pytest.raises(FileNotFoundError):
            loader.load(tmp_path / "nonexistent.txt")

    def test_load_directory_raises(self, loader, tmp_path):
        with pytest.raises(IsADirectoryError):
            loader.load(tmp_path)

    def test_load_binary_file(self, loader, tmp_path):
        binary_file = tmp_path / "data.bin"
        binary_file.write_bytes(b"\x00\x01\x02\x03")
        artifact = loader.load(binary_file)
        assert artifact.raw_bytes == b"\x00\x01\x02\x03"
        assert artifact.metadata["extension"] == ".bin"
