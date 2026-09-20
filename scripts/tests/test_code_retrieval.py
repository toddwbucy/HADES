"""Git snapshots must bind source bytes and exclude live/untracked inputs."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from prepare_code_retrieval import freeze


class CodeSnapshotTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.repo = self.root / 'repo'
        self.repo.mkdir()
        self.git('init', '-q')
        self.git('config', 'user.name', 'Private Fixture')
        self.git('config', 'user.email', 'fixture@example.invalid')
        (self.repo / 'crates').mkdir()
        (self.repo / 'crates/main.rs').write_text('fn original() {}\n')
        (self.repo / 'private.md').write_text('not in the code corpus\n')
        self.commit()
        self.queries = [{'id': 'C1', 'text': 'Where is original?', 'relevance': {}}]

    def git(self, *args):
        return subprocess.run(['git', '-c', 'core.hooksPath=/dev/null', '-C', str(self.repo), *args],
                              check=True, capture_output=True).stdout

    def commit(self):
        self.git('add', '.')
        self.git('-c', 'commit.gpgsign=false', 'commit', '-qm', 'fixture')

    def test_snapshot_uses_pinned_blobs_and_is_reproducible(self):
        revision = self.git('rev-parse', 'HEAD').decode().strip()
        (self.repo / 'crates/main.rs').write_text('fn changed() {}\n')
        (self.repo / 'crates/untracked.rs').write_text('fn untracked() {}\n')
        first = self.root / 'first'
        a = freeze(self.repo, revision, first, self.queries)
        b = freeze(self.repo, revision, self.root / 'second', self.queries)
        self.assertEqual(a, b)
        self.assertEqual((a['files'], a['queries']), (1, 1))
        data = json.loads((first / 'code-candidates-v1.json').read_text())
        self.assertEqual(data['documents'][0]['text'], 'fn original() {}\n')
        self.assertEqual(data['source_commit'], revision)
        self.assertEqual(data['scoring_policy'], 'complete_top10')
        self.assertEqual(data['queries'][0]['relevance'], {})
        self.assertEqual(first.stat().st_mode & 0o777, 0o700)
        self.assertEqual((first / 'code-candidates-v1.json').stat().st_mode & 0o777, 0o600)
        self.assertEqual(a['dataset_sha256'], hashlib.sha256((first / 'code-candidates-v1.json').read_bytes()).hexdigest())
        with self.assertRaises(FileExistsError):
            freeze(self.repo, revision, first, self.queries)

    def test_selected_symlink_is_rejected_without_reading_target(self):
        os.symlink('../private.md', self.repo / 'crates/linked.rs')
        self.commit()
        out = self.root / 'out'
        with self.assertRaisesRegex(ValueError, 'regular Git blob'):
            freeze(self.repo, 'HEAD', out, self.queries)
        self.assertFalse(out.exists())

    def test_duplicate_or_prejudged_queries_rejected(self):
        for queries in [self.queries * 2, [dict(self.queries[0], relevance={'invented': 3})]]:
            out = self.root / 'out'
            with self.assertRaises(ValueError):
                freeze(self.repo, 'HEAD', out, queries)
            self.assertFalse(out.exists())


if __name__ == '__main__':
    unittest.main()
