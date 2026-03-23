"""Tests for Rust codebase ingest integration.

Validates:
- _git_rust_files: Rust file discovery via git
- _find_crate_roots: grouping .rs files by Cargo.toml
- _analyze_rust_crate: end-to-end flow (mocked LSP + DB)
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

from core.cli.commands.codebase import _find_crate_roots, _git_rust_files

# ── _git_rust_files ──────────────────────────────────────────────


class TestGitRustFiles:
    def test_finds_rs_files(self, tmp_path: Path) -> None:
        """Should find .rs files in a git repo."""
        # Set up a git repo with .rs files
        (tmp_path / "src").mkdir()
        (tmp_path / "src" / "lib.rs").write_text("pub fn foo() {}")
        (tmp_path / "src" / "main.rs").write_text("fn main() {}")
        (tmp_path / "build.py").write_text("# not rust")

        import subprocess

        subprocess.run(["git", "init"], cwd=tmp_path, capture_output=True)
        subprocess.run(["git", "add", "."], cwd=tmp_path, capture_output=True)

        files = _git_rust_files(str(tmp_path))
        assert "src/lib.rs" in files
        assert "src/main.rs" in files
        assert "build.py" not in files

    def test_excludes_target_dir(self, tmp_path: Path) -> None:
        """Should skip files in the 'target' directory."""
        (tmp_path / "src").mkdir()
        (tmp_path / "src" / "lib.rs").write_text("pub fn foo() {}")
        (tmp_path / "target" / "debug").mkdir(parents=True)
        (tmp_path / "target" / "debug" / "build.rs").write_text("// generated")

        import subprocess

        subprocess.run(["git", "init"], cwd=tmp_path, capture_output=True)
        subprocess.run(["git", "add", "src/"], cwd=tmp_path, capture_output=True)

        files = _git_rust_files(str(tmp_path))
        assert "src/lib.rs" in files
        assert not any("target" in f for f in files)


# ── _find_crate_roots ────────────────────────────────────────────


class TestFindCrateRoots:
    def test_single_crate(self, tmp_path: Path) -> None:
        """Single Cargo.toml at root groups all .rs files."""
        (tmp_path / "Cargo.toml").write_text("[package]\nname = \"test\"")
        (tmp_path / "src").mkdir()
        (tmp_path / "src" / "lib.rs").write_text("")
        (tmp_path / "src" / "main.rs").write_text("")

        rs_files = ["src/lib.rs", "src/main.rs"]
        result = _find_crate_roots(str(tmp_path), rs_files)

        assert len(result) == 1
        crate_root = list(result.keys())[0]
        assert crate_root == str(tmp_path)
        assert set(result[crate_root]) == {"src/lib.rs", "src/main.rs"}

    def test_workspace_multiple_crates(self, tmp_path: Path) -> None:
        """Workspace with multiple crates groups files by nearest Cargo.toml."""
        # Root workspace
        (tmp_path / "Cargo.toml").write_text("[workspace]\nmembers = [\"crate_a\", \"crate_b\"]")

        # Crate A
        crate_a = tmp_path / "crate_a"
        crate_a.mkdir()
        (crate_a / "Cargo.toml").write_text("[package]\nname = \"crate_a\"")
        (crate_a / "src").mkdir()
        (crate_a / "src" / "lib.rs").write_text("")

        # Crate B
        crate_b = tmp_path / "crate_b"
        crate_b.mkdir()
        (crate_b / "Cargo.toml").write_text("[package]\nname = \"crate_b\"")
        (crate_b / "src").mkdir()
        (crate_b / "src" / "lib.rs").write_text("")

        rs_files = ["crate_a/src/lib.rs", "crate_b/src/lib.rs"]
        result = _find_crate_roots(str(tmp_path), rs_files)

        assert len(result) == 2
        assert any("crate_a" in k for k in result)
        assert any("crate_b" in k for k in result)

    def test_no_cargo_toml(self, tmp_path: Path) -> None:
        """Files without a Cargo.toml ancestor are skipped."""
        (tmp_path / "src").mkdir()
        (tmp_path / "src" / "orphan.rs").write_text("")

        rs_files = ["src/orphan.rs"]
        result = _find_crate_roots(str(tmp_path), rs_files)
        assert len(result) == 0


# ── _analyze_rust_crate (mocked) ─────────────────────────────────


class TestAnalyzeRustCrate:
    def test_skips_when_no_rust_analyzer(self) -> None:
        """Should return zeros when rust-analyzer is not installed."""
        from core.cli.commands.codebase import _analyze_rust_crate

        with patch("shutil.which", return_value=None):
            result = _analyze_rust_crate(
                "/fake/crate", ["src/lib.rs"], "/fake/repo",
                MagicMock(), "test_db", MagicMock(),
            )
        assert result["rust_symbols_created"] == 0
        assert result["rust_edges_created"] == 0
