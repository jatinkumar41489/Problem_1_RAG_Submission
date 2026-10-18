"""Idempotency tests - the mandatory 'no duplicate vectors' requirement, exercised for real."""
from pathlib import Path

from tests.conftest import SAMPLE_DIR


def test_ingest_new_then_identical_reingest_keeps_count(services):
    r1 = services.pipeline.ingest_path(SAMPLE_DIR)
    assert r1.files_discovered == 12 and r1.files_new == 12 and r1.files_failed == 0
    n = services.store.count()
    assert n == r1.chunks_upserted > 0
    ids_before = sorted(c["chunk_id"] for c in services.store.all_chunks())
    for _ in range(2):
        r = services.pipeline.ingest_path(SAMPLE_DIR)
        assert r.files_unchanged == 12 and r.chunks_upserted == 0 and r.count_before == r.count_after == n
    assert sorted(c["chunk_id"] for c in services.store.all_chunks()) == ids_before   # same ids too


def test_unchanged_files_are_not_re_embedded(services, monkeypatch):
    services.pipeline.ingest_path(SAMPLE_DIR)
    calls = []
    orig = services.encoder.encode
    monkeypatch.setattr(services.encoder, "encode", lambda t: calls.append(len(t)) or orig(t))
    services.pipeline.ingest_path(SAMPLE_DIR)
    assert calls == []


def test_modified_document_replaces_only_its_own_chunks(services, corpus_copy):
    services.pipeline.ingest_path(corpus_copy)
    before = {c["chunk_id"]: c for c in services.store.all_chunks()}
    target = corpus_copy / "hr" / "leave_policy.md"
    target.write_text(target.read_text().replace("25 days of paid annual leave", "30 days of paid annual leave"))
    r = services.pipeline.ingest_path(corpus_copy)
    assert r.files_updated == 1 and r.files_unchanged == 11 and r.chunks_deleted > 0
    after = {c["chunk_id"]: c for c in services.store.all_chunks()}
    assert len(after) == len(before)                                   # no growth: stale chunks removed
    old_leave = {i for i, c in before.items() if c["metadata"]["source_id"] == "hr/leave_policy.md"}
    assert not (old_leave & set(after))                               # old leave chunks gone
    assert {i for i in before if i not in old_leave} <= set(after)    # every other chunk untouched
    assert any("30 days of paid annual leave" in c["text"] for c in after.values())
    assert not any("25 days of paid annual leave" in c["text"] for c in after.values())


def test_whitespace_only_edit_is_not_a_change(services, corpus_copy):
    services.pipeline.ingest_path(corpus_copy)
    t = corpus_copy / "hr" / "leave_policy.md"
    t.write_text(t.read_text().replace("\n\n", "\n\n\n"))
    assert services.pipeline.ingest_path(corpus_copy).files_unchanged == 12


def test_deleted_source_is_pruned_only_with_prune_flag(services, corpus_copy):
    services.pipeline.ingest_path(corpus_copy)
    n = services.store.count()
    (corpus_copy / "product" / "README.md").unlink()
    assert services.pipeline.ingest_path(corpus_copy).count_after == n            # default: no deletion
    r = services.pipeline.ingest_path(corpus_copy, prune=True)
    assert r.sources_pruned == ["product/README.md"] and r.count_after == n - 1
    assert "product/README.md" not in {s["source_id"] for s in services.store.source_summary()}


def test_same_basename_different_directories_do_not_collide(services):
    services.pipeline.ingest_path(SAMPLE_DIR)
    ids = {s["source_id"] for s in services.store.source_summary()}
    assert {"hr/README.md", "product/README.md"} <= ids
    chunks = [c for c in services.store.all_chunks() if c["metadata"]["source_name"] == "README.md"]
    assert len(chunks) == 2 and len({c["chunk_id"] for c in chunks}) == 2


def test_identical_content_under_two_paths_warns_and_stores_both(services, tmp_path):
    root = tmp_path / "c"; (root / "a").mkdir(parents=True); (root / "b").mkdir()
    (root / "a" / "x.md").write_text("# T\nSame content here for both files.")
    (root / "b" / "x.md").write_text("# T\nSame content here for both files.")
    r = services.pipeline.ingest_path(root)
    assert services.store.count() == 2
    assert any(f.warning and "identical content" in f.warning for f in r.results)


def test_bad_and_unsupported_files_reported_not_fatal(services, tmp_path):
    root = tmp_path / "c"; root.mkdir()
    (root / "ok.md").write_text("# Ok\nsome useful content")
    (root / "bad.pdf").write_bytes(b"not a pdf")
    (root / "empty.md").write_text("")
    (root / "notes.docx").write_bytes(b"x")
    r = services.pipeline.ingest_path(root)
    st = {f.source_id: f.status for f in r.results}
    assert st == {"ok.md": "new", "bad.pdf": "failed", "empty.md": "failed", "notes.docx": "skipped"}
    assert r.files_failed == 2 and r.files_skipped == 1 and services.store.count() == 1


def test_chunk_config_change_reprocesses_and_cleans_up(services, make_services):
    services.pipeline.ingest_path(SAMPLE_DIR)
    import dataclasses
    from app.ingestion.chunking import Chunker
    services.pipeline.chunker = Chunker(60, 10)
    r = services.pipeline.ingest_path(SAMPLE_DIR)
    assert r.files_updated > 0
    expected = sum(len(Chunker(60, 10).chunk_document(__import__("app.ingestion.loaders", fromlist=["x"]).load_document(p, p.relative_to(SAMPLE_DIR).as_posix())))
                   for p in SAMPLE_DIR.rglob("*") if p.is_file())
    assert services.store.count() == expected                         # no leftovers from the 180/30 chunking
