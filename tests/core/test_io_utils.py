import gzip

from deepspacedb_xenium_pipeline import io_utils


def test_gzip_roundtrip(tmp_path):
    src = tmp_path / "a.txt"
    src.write_bytes(b"hello world")
    gz = tmp_path / "a.txt.gz"
    with gzip.open(gz, "wb") as fh:
        fh.write(b"hello world")
    assert io_utils.is_gzip_file(gz)
    assert not io_utils.is_gzip_file(src)
    out = tmp_path / "out.txt"
    io_utils.gzip_decompress(gz, out)
    assert out.read_bytes() == b"hello world"


def test_unique_path(tmp_path):
    p = tmp_path / "x.txt"
    assert io_utils.unique_path(p) == p
    p.write_text("1")
    p2 = io_utils.unique_path(p)
    assert p2.name == "x_1.txt"


def test_junk_filenames():
    assert io_utils.is_junk_filename("._foo")
    assert io_utils.is_junk_filename(".DS_Store")
    assert not io_utils.is_junk_filename("cells.parquet")


def test_clean_appledouble_non_appledouble_is_noop(tmp_path):
    p = tmp_path / "real.bin"
    p.write_bytes(b"not appledouble content")
    assert io_utils.clean_appledouble_file(p) is True
    assert p.read_bytes() == b"not appledouble content"


def test_clean_appledouble_extracts_data_fork(tmp_path):
    # Build a minimal AppleDouble file with a single data-fork (id=1) entry.
    magic = b"\x00\x05\x16\x07"
    header = magic + b"\x00" * 22  # up to offset 26
    num_entries = (1).to_bytes(2, "big")
    payload = b"DATA-FORK-PAYLOAD"
    # entry table is 12 bytes (id, offset, length); data starts right after.
    data_offset = len(header) + len(num_entries) + 12
    entry = (1).to_bytes(4, "big") + data_offset.to_bytes(4, "big") + len(payload).to_bytes(4, "big")
    p = tmp_path / "._file"
    p.write_bytes(header + num_entries + entry + payload)
    assert io_utils.clean_appledouble_file(p) is True
    assert p.read_bytes() == payload
