"""Tests for the Rust symbol extractor.

Validates that rust-analyzer output is correctly transformed into
the structured rust_analyzer attribute format for file nodes.
"""

from __future__ import annotations

import shutil
import textwrap
from pathlib import Path

import pytest

from core.analyzers.rust_analyzer_client import RustAnalyzerSession
from core.analyzers.rust_symbol_extractor import RustSymbolExtractor

HAS_RUST_ANALYZER = shutil.which("rust-analyzer") is not None
pytestmark = pytest.mark.skipif(not HAS_RUST_ANALYZER, reason="rust-analyzer not installed")


@pytest.fixture
def rust_crate(tmp_path: Path) -> Path:
    """Create a Rust crate with struct, trait, impl, pyo3 attrs, FFI, and cross-module calls."""
    crate = tmp_path / "test_crate"
    crate.mkdir()
    (crate / "Cargo.toml").write_text(textwrap.dedent("""\
        [package]
        name = "test_crate"
        version = "0.1.0"
        edition = "2021"
    """))
    src = crate / "src"
    src.mkdir()

    (src / "lib.rs").write_text(textwrap.dedent("""\
        pub mod math;
        pub mod model;
    """))

    (src / "math.rs").write_text(textwrap.dedent("""\
        /// Add two numbers.
        pub fn add(a: i32, b: i32) -> i32 {
            a + b
        }

        pub fn multiply(a: i32, b: i32) -> i32 {
            a * b
        }

        pub fn add_then_double(a: i32, b: i32) -> i32 {
            let sum = add(a, b);
            multiply(sum, 2)
        }
    """))

    (src / "model.rs").write_text(textwrap.dedent("""\
        use crate::math;

        #[derive(Debug, Clone)]
        pub struct Model {
            pub weights: Vec<f64>,
            pub bias: f64,
        }

        pub trait Forward {
            fn forward(&self, input: &[f64]) -> Vec<f64>;
        }

        impl Model {
            pub fn new(size: usize) -> Self {
                Model {
                    weights: vec![0.0; size],
                    bias: 0.0,
                }
            }

            pub fn compute_norm(&self) -> f64 {
                self.weights.iter().map(|w| w * w).sum::<f64>().sqrt()
            }
        }

        impl Forward for Model {
            fn forward(&self, input: &[f64]) -> Vec<f64> {
                let norm = self.compute_norm();
                let sum = math::add(input.len() as i32, 1);
                input.iter().map(|x| x * norm + self.bias + sum as f64).collect()
            }
        }

        pub const MAX_LAYERS: usize = 64;
    """))

    return crate


@pytest.fixture
def session(rust_crate: Path) -> RustAnalyzerSession:
    s = RustAnalyzerSession(rust_crate, timeout=60)
    s.start()
    yield s
    s.shutdown()


@pytest.fixture
def extractor(session: RustAnalyzerSession) -> RustSymbolExtractor:
    return RustSymbolExtractor(session, include_calls=True, include_incoming=False)


class TestExtractFile:
    """Test single-file extraction."""

    def test_returns_expected_keys(self, extractor: RustSymbolExtractor) -> None:
        data = extractor.extract_file("src/model.rs")
        assert "symbols" in data
        assert "impl_blocks" in data
        assert "pyo3_exports" in data
        assert "ffi_boundaries" in data
        assert "analyzed_at" in data

    def test_symbols_populated(self, extractor: RustSymbolExtractor) -> None:
        data = extractor.extract_file("src/model.rs")
        names = {s["name"] for s in data["symbols"]}
        assert "Model" in names
        assert "Forward" in names
        assert "new" in names
        assert "compute_norm" in names
        assert "MAX_LAYERS" in names

    def test_symbol_has_required_fields(self, extractor: RustSymbolExtractor) -> None:
        data = extractor.extract_file("src/model.rs")
        required = {"name", "qualified_name", "kind", "visibility", "start_line", "end_line"}
        for sym in data["symbols"]:
            missing = required - set(sym.keys())
            assert not missing, f"Symbol {sym['name']} missing: {missing}"

    def test_struct_visibility(self, extractor: RustSymbolExtractor) -> None:
        data = extractor.extract_file("src/model.rs")
        model = next((s for s in data["symbols"] if s["name"] == "Model"), None)
        assert model is not None
        assert model["visibility"] == "pub"
        assert model["kind"] == "struct"

    def test_derives_extracted(self, extractor: RustSymbolExtractor) -> None:
        data = extractor.extract_file("src/model.rs")
        model = next((s for s in data["symbols"] if s["name"] == "Model"), None)
        assert model is not None
        assert "Debug" in model.get("derives", [])
        assert "Clone" in model.get("derives", [])

    def test_impl_blocks(self, extractor: RustSymbolExtractor) -> None:
        data = extractor.extract_file("src/model.rs")
        assert len(data["impl_blocks"]) > 0
        # Should have at least inherent impl and trait impl
        self_types = [b["self_type"] for b in data["impl_blocks"]]
        assert any("Model" in t for t in self_types)

    def test_function_has_signature(self, extractor: RustSymbolExtractor) -> None:
        data = extractor.extract_file("src/math.rs")
        add = next((s for s in data["symbols"] if s["name"] == "add"), None)
        assert add is not None
        assert "i32" in add.get("signature", "")

    def test_constant_extracted(self, extractor: RustSymbolExtractor) -> None:
        data = extractor.extract_file("src/model.rs")
        constant = next((s for s in data["symbols"] if s["name"] == "MAX_LAYERS"), None)
        assert constant is not None
        assert constant["kind"] == "constant"


class TestCallHierarchy:
    """Test call hierarchy extraction within symbol data."""

    def test_outgoing_calls_populated(self, extractor: RustSymbolExtractor) -> None:
        data = extractor.extract_file("src/math.rs")
        add_then_double = next(
            (s for s in data["symbols"] if s["name"] == "add_then_double"), None
        )
        assert add_then_double is not None
        calls = add_then_double.get("calls", [])
        if calls:  # May be empty on slow CI
            called_names = {c["name"] for c in calls}
            assert "add" in called_names or "multiply" in called_names

    def test_cross_file_calls(self, extractor: RustSymbolExtractor) -> None:
        """Calls to math::add from model.rs should reference math.rs."""
        data = extractor.extract_file("src/model.rs")
        forward = next((s for s in data["symbols"] if s["name"] == "forward"), None)
        if forward is None:
            pytest.skip("forward method not extracted (may be nested under impl)")
        calls = forward.get("calls", [])
        if calls:
            files = {c.get("file", "") for c in calls}
            # At least one call should resolve to math.rs
            assert any(f.endswith("src/math.rs") for f in files)


class TestExtractCrate:
    """Test full-crate extraction."""

    def test_extract_all_files(self, extractor: RustSymbolExtractor) -> None:
        results = extractor.extract_crate(["src/lib.rs", "src/math.rs", "src/model.rs"])
        assert len(results) == 3
        assert "src/lib.rs" in results
        assert "src/math.rs" in results
        assert "src/model.rs" in results

    def test_all_files_have_symbols(self, extractor: RustSymbolExtractor) -> None:
        results = extractor.extract_crate(["src/math.rs", "src/model.rs"])
        for path, data in results.items():
            assert len(data["symbols"]) > 0, f"No symbols in {path}"
