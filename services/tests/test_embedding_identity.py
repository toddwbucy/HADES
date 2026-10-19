"""CPU-only cross-language artifact/receipt contracts; no model or database."""

import copy
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from embedding.identity import (
    ArtifactFile, ArtifactManifest, MAX_MANIFEST_BYTES, MAX_MANIFEST_FILES,
    ProviderReceipt,
)

ROOT = Path(__file__).resolve().parents[2]
FIXTURE = ROOT / "crates/hades-core/tests/fixtures/embedding_profile.json"


class EmbeddingIdentityTests(unittest.TestCase):
    """Exercise exact bytes, failure cases, and snapshot receipt comparisons."""

    def test_shared_manifest_vectors(self):
        """Both implementations must reproduce the design's independently fixed bytes."""
        vectors = json.loads((ROOT / "docs/specs/database-embedding-manifest-vectors.json").read_text())
        for case in vectors["cases"]:
            with self.subTest(case=case["name"]):
                files = tuple(ArtifactFile.from_stream(file["path"], io.BytesIO(bytes.fromhex(file["content_hex"])))
                              for file in case["source_files"])
                manifest = ArtifactManifest(files)
                self.assertEqual(manifest.canonical_bytes(), case["canonical_json"].encode())
                self.assertEqual(manifest.canonical_bytes().hex(), case["canonical_utf8_hex"])
                self.assertEqual(manifest.digest(), case["manifest_md5"])
                self.assertEqual(ArtifactManifest(tuple(reversed(files))).digest(), manifest.digest())
                self.assertEqual(ArtifactManifest.from_json(manifest.canonical_bytes()).digest(), manifest.digest())

    def test_literal_controls_unicode_and_backslash(self):
        """Preserve scalar sequences and escape controls exactly as the Rust encoder."""
        file = ArtifactFile.from_stream('x/\x01\b\t\n\f\r"\\\u2028\u2029', io.BytesIO())
        text = ArtifactManifest((file,)).canonical_bytes().decode()
        self.assertIn('x/\\u0001\\b\\t\\n\\f\\r\\"\\\\\u2028\u2029', text)
        composed = ArtifactManifest((ArtifactFile.from_stream("caf\u00e9", io.BytesIO()),))
        decomposed = ArtifactManifest((ArtifactFile.from_stream("cafe\u0301", io.BytesIO()),))
        self.assertNotEqual(composed.digest(), decomposed.digest())

    def test_rejects_invalid_paths_and_digest_encodings(self):
        """Do not normalize ambiguous paths or accept malformed checksums."""
        for path in ["", "/absolute", "./model", "model/../weights", "model//weights",
                     "model/", "nul\0x", "a" * 4097, "\ud800"]:
            with self.subTest(path=repr(path)), self.assertRaises(ValueError):
                ArtifactFile.from_stream(path, io.BytesIO())
        for digest in ["", "D41D8CD98F00B204E9800998ECF8427E", "g41d8cd98f00b204e9800998ecf8427e"]:
            with self.subTest(digest=digest), self.assertRaises(ValueError):
                ArtifactFile("weights", 0, digest)
        for size in [-1, 1.0, True, "3", None, 2**64]:
            with self.subTest(size=size), self.assertRaises(ValueError):
                ArtifactFile("weights", size, "0" * 32)

    def test_rejects_missing_duplicate_and_aliased_assets(self):
        """An asset set cannot be empty or contain conflicting logical paths."""
        for paths in [[], ["a", "a"], ["a", "a/b"], ["a/b", "a"]]:
            with self.subTest(paths=paths), self.assertRaises(ValueError):
                ArtifactManifest(tuple(ArtifactFile.from_stream(path, io.BytesIO()) for path in paths))
        with self.assertRaises(ValueError):
            ArtifactManifest(tuple(ArtifactFile(str(i), 0, "0" * 32) for i in range(MAX_MANIFEST_FILES + 1)))

    def test_strict_wire_parsing(self):
        """Unknown schemas, duplicate keys, malformed Unicode, and excess bytes fail closed."""
        original = json.loads(FIXTURE.read_text())["manifest"]
        cases = []
        bad = copy.deepcopy(original); bad["format"] = "hades-model-manifest-v2"; cases.append(bad)
        bad = copy.deepcopy(original); bad["extra"] = True; cases.append(bad)
        bad = copy.deepcopy(original); bad["files"][0]["extra"] = True; cases.append(bad)
        bad = copy.deepcopy(original); bad["files"] = {}; cases.append(bad)
        for size in [-1, 1.0, True, "3", None, 2**64]:
            bad = copy.deepcopy(original); bad["files"][0]["bytes"] = size; cases.append(bad)
        encoded_cases = [json.dumps(case).encode() for case in cases]
        encoded_cases.extend([
            b'{"format":"hades-model-manifest-v1","format":"hades-model-manifest-v1","files":[]}',
            b'{"format":"hades-model-manifest-v1","files":[{"path":"\\ud800","bytes":0,"md5":"d41d8cd98f00b204e9800998ecf8427e"}]}',
            b"\xff", b" " * (MAX_MANIFEST_BYTES + 1),
        ])
        for index, encoded in enumerate(encoded_cases):
            with self.subTest(case=index), self.assertRaises(ValueError):
                ArtifactManifest.from_json(encoded)

    def test_relocation_and_asset_drift(self):
        """Installation roots do not affect identity; every required dependency does."""
        names = ("weights.bin", "tokenizer.json", "config.json", "adapter.bin")
        digests = []
        for _ in range(2):
            with tempfile.TemporaryDirectory() as root:
                files = []
                for name in names:
                    path = Path(root) / name
                    path.write_bytes(b"original")
                    with path.open("rb") as stream:
                        files.append(ArtifactFile.from_stream(name, stream))
                digests.append(ArtifactManifest(tuple(files)).digest())
        self.assertEqual(*digests)
        for changed in names:
            manifest = ArtifactManifest(tuple(ArtifactFile.from_stream(
                name, io.BytesIO(b"modified" if name == changed else b"original")) for name in names))
            self.assertNotEqual(manifest.digest(), digests[0])
            with self.assertRaises(ValueError):
                manifest.check_digest(digests[0])

    def test_stream_hash_and_read_failure(self):
        """Multi-block hashing counts bytes and never returns a partial identity on failure."""
        content = b"x" * 150000
        file = ArtifactFile.from_stream("weights.bin", io.BytesIO(content))
        self.assertEqual(file.bytes, len(content))
        self.assertEqual(file.md5, hashlib.md5(content, usedforsecurity=False).hexdigest())

        class Broken:
            def read(self, size):
                """Simulate an asset read failure without returning partial source bytes."""
                raise OSError("fixture read error")

        with self.assertRaises(OSError):
            ArtifactFile.from_stream("weights.bin", Broken())

    def test_shared_space_and_receipt_vector(self):
        """Pin the receipt wire shape and independently confirm Rust's space SHA-256."""
        fixture = json.loads(FIXTURE.read_text())
        space = fixture["profile"]["workloads"]["documents"]["space"]
        canonical = json.dumps(space, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
        self.assertEqual(canonical.hex(), fixture["space_canonical_utf8_hex"])
        receipt = ProviderReceipt.from_dict(fixture["receipt"])
        self.assertEqual(hashlib.sha256(canonical).hexdigest(), receipt.space_id)
        manifest = ArtifactManifest.from_json(json.dumps(fixture["manifest"]).encode())
        manifest.check_digest(receipt.artifact_md5)
        self.assertEqual(receipt.to_dict(), fixture["receipt"])
        receipt.check_matches(ProviderReceipt.from_dict(receipt.to_dict()))

    def test_receipt_rejects_reload_and_identity_changes(self):
        """A provider reload invalidates a request even when weights and space match."""
        original = json.loads(FIXTURE.read_text())["receipt"]
        expected = ProviderReceipt.from_dict(original)
        for key, value in [("serving_generation", "load-fixture-2"), ("artifact_md5", "0" * 32),
                           ("space_id", "0" * 64)]:
            changed = dict(original, **{key: value})
            with self.subTest(key=key), self.assertRaises(ValueError):
                ProviderReceipt.from_dict(changed).check_matches(expected)
        for key, value in [("format", "hades-provider-receipt-v2"), ("serving_generation", ""),
                           ("serving_generation", "x\n"), ("space_id", "garbage"), ("extra", True)]:
            with self.subTest(key=key), self.assertRaises(ValueError):
                ProviderReceipt.from_dict(dict(original, **{key: value}))
        for key in original:
            missing = dict(original)
            del missing[key]
            with self.subTest(missing=key), self.assertRaises(ValueError):
                ProviderReceipt.from_dict(missing)


if __name__ == "__main__":
    unittest.main()
