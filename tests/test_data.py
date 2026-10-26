"""Data-layer tests. Label-level checks run only on open splits (tune, dev, eda); blind
splits are checked through their ids and types alone."""
import collections

import pytest

import data

OPEN = ("tune", "dev", "dev_smoke")


@pytest.fixture(scope="module")
def splits():
    return {name: data.load_split_ids(name) for name in data.EXPECTED_SPLIT_HASHES}


@pytest.fixture(scope="module")
def types():
    val = data._raw("validation").select_columns(["id", "type"])
    return dict(zip(val["id"], val["type"]))


def test_revision_pinned():
    assert len(data.REVISION) == 40


def test_corpus_counts_and_fingerprint():
    c = data.corpus()
    titles = collections.Counter(d.title for d in c.values())
    assert len(titles) == 66_581
    assert sum(n > 1 for n in titles.values()) == 54
    assert max(titles.values()) == 2
    assert len(c) == 66_581 + 54
    assert data.verify_corpus() == data.EXPECTED_CORPUS_FINGERPRINT


def test_duplicate_title_ids_do_not_depend_on_order():
    c = data.corpus()
    for d in c.values():
        if d.doc_id != d.title:
            assert d.doc_id == f"{d.title} #{d.text_hash[:8]}"
    assert len(data.doc_id_of()) == len(c)


def test_text_hash_includes_segmentation():
    assert data.text_hash(["A b.", " C."]) != data.text_hash(["A b. C."])


def test_splits_are_deterministic_and_match_pinned_hashes(splits):
    assert data.make_splits() == splits
    for name, ids in splits.items():
        assert data.ids_hash(ids) == data.EXPECTED_SPLIT_HASHES[name], name


def test_split_sizes_and_types(splits, types):
    want = {"tune": (400, 100), "dev": (80, 20), "test_retrieval": (799, 201),
            "dev_smoke": (16, 4), "test_rag": (160, 40), "test_agents": (50, 50)}
    for name, (b, c) in want.items():
        cnt = collections.Counter(types[q] for q in splits[name])
        assert (cnt["bridge"], cnt["comparison"]) == (b, c), name
        assert len(set(splits[name])) == b + c, name
    assert len(set(splits["eda"])) == data.EDA_SIZE
    assert not set(splits["eda"]) & set(types)  # train ids, not validation


def test_split_relationships(splits):
    s = {k: set(v) for k, v in splits.items()}
    assert not s["tune"] & s["dev"]
    assert not s["tune"] & s["test_retrieval"]
    assert not s["dev"] & s["test_retrieval"]
    assert s["dev_smoke"] <= s["dev"]
    assert s["test_rag"] <= s["test_retrieval"]
    assert s["test_agents"] <= s["test_retrieval"]


@pytest.mark.parametrize("name", sorted(data.BLIND_SPLITS))
def test_blind_splits_refuse_to_load(name):
    with pytest.raises(data.BlindSplitError):
        data.load_split(name)


@pytest.mark.parametrize("name", OPEN)
def test_open_split_gold_is_exact_and_in_corpus(name):
    ids = data.doc_id_of()
    for q in data.load_split(name):
        assert len(q.gold) == 2
        titles = [t for t, _ in q.context]
        assert len(titles) == len(set(titles))  # a title occurs once in its own context
        for g in q.gold:
            assert g in q.context and g in ids


@pytest.mark.parametrize("name", sorted(data.BLIND_SPLITS))
def test_blind_contexts_refuse_to_load(name):
    with pytest.raises(data.BlindSplitError):
        data.load_contexts(name)


def test_contexts_match_question_identities():
    qs = data.load_split("dev_smoke")
    ctx = data.load_contexts("dev_smoke")
    for q in qs:
        assert [(t, data.text_hash(s)) for t, s in ctx[q.qid].items()] == list(q.context)


@pytest.fixture(scope="module")
def dev_ids():
    return data.load_split_ids("dev")


def test_validate_accepts_exact_splits_including_blind_ids(splits):
    for name, ids in splits.items():         # ids only; no questions are loaded
        data.validate_split_ids(name, ids)
        data.validate_split_ids(name, list(reversed(ids)))   # order does not matter


@pytest.mark.parametrize("change", ["altered", "missing", "duplicated", "appended_duplicate", "extra"])
def test_validate_rejects_changed_ids(dev_ids, change):
    ids = list(dev_ids)
    if change == "altered":
        ids[0] = "not-a-real-question"
    elif change == "missing":
        ids = ids[1:]
    elif change == "duplicated":
        ids[0] = ids[1]
    elif change == "appended_duplicate":
        ids = ids + [ids[0]]
    elif change == "extra":
        ids = ids + ["not-a-real-question"]
    with pytest.raises(data.SplitMismatch):
        data.validate_split_ids("dev", ids)


def test_validate_rejects_a_subset_under_a_real_name_but_allows_adhoc(dev_ids):
    with pytest.raises(data.SplitMismatch):
        data.validate_split_ids("test_agents", dev_ids)         # right size elsewhere, wrong ids
    with pytest.raises(data.SplitMismatch, match="unknown split"):
        data.validate_split_ids("my_subset", dev_ids[:5])
    data.validate_split_ids("adhoc:debug", dev_ids[:5])
    with pytest.raises(data.SplitMismatch, match="duplicate"):
        data.validate_split_ids("adhoc:debug", [dev_ids[0], dev_ids[0]])
    with pytest.raises(data.SplitMismatch):
        data.validate_split_ids("adhoc:empty", [])
