"""Tests for core.database.collections — collection profile registry."""

import pytest

from core.database.collections import CollectionProfile, PROFILES, get_profile


class TestCollectionProfile:
    def test_frozen(self):
        profile = CollectionProfile(metadata="m", chunks="c", embeddings="e")
        with pytest.raises(AttributeError):
            profile.metadata = "other"

    def test_fields(self):
        profile = CollectionProfile(metadata="m", chunks="c", embeddings="e")
        assert profile.metadata == "m"
        assert profile.chunks == "c"
        assert profile.embeddings == "e"


class TestBuiltinProfiles:
    def test_arxiv_profile(self):
        p = get_profile("arxiv")
        assert p.metadata == "arxiv_metadata"
        assert p.chunks == "arxiv_abstract_chunks"
        assert p.embeddings == "arxiv_abstract_embeddings"

    def test_sync_profile(self):
        p = get_profile("sync")
        assert p.metadata == "arxiv_papers"
        assert p.chunks == "arxiv_abstracts"
        assert p.embeddings == "arxiv_embeddings"

    def test_default_profile(self):
        p = get_profile("default")
        assert p.metadata == "documents"
        assert p.chunks == "chunks"
        assert p.embeddings == "embeddings"

    def test_unknown_profile_raises(self):
        with pytest.raises(KeyError, match="Unknown collection profile"):
            get_profile("nonexistent")

    def test_profiles_dict_has_all_builtins(self):
        assert set(PROFILES.keys()) == {"arxiv", "sync", "default"}
