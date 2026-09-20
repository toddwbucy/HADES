"""Exercise installer source options against a private file-only APT repository."""
import getpass
import hashlib
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]


class InstallerSignatures(unittest.TestCase):
    def test_repository_authentication(self):
        required = ('apt-get', 'gpg', 'gpgconf')
        missing = [name for name in required if not shutil.which(name)]
        if missing:
            if os.environ.get('CI'):
                self.fail(f'Missing required fixture tools: {missing}')
            self.skipTest(f'Requires disposable APT environment: {missing}')
        with tempfile.TemporaryDirectory(prefix='hades-apt-auth-') as folder:
            root = Path(folder)
            home = root / 'gnupg'
            home.mkdir(mode=0o700)
            env = {**os.environ, 'GNUPGHOME': str(home)}

            def run(args, **kwargs):
                return subprocess.run(args, env=env, capture_output=True,
                                      text=True, timeout=30, **kwargs)

            try:
                run(['gpg', '--batch', '--pinentry-mode', 'loopback', '--passphrase', '',
                     '--quick-generate-key', 'HADES fixture <fixture@example.invalid>',
                     'ed25519', 'sign', '1d'], check=True)
                key = root / 'vendor.gpg'
                run(['gpg', '--batch', '--output', str(key), '--export'], check=True)
                repo = root / 'repository'
                repo.mkdir()
                (repo / 'Packages').write_bytes(b'')
                release = ('Origin: HADES private fixture\nLabel: HADES fixture\n'
                           'Suite: stable\nCodename: stable\nArchitectures: amd64\n'
                           'Date: Sun, 20 Sep 2026 00:00:00 UTC\n'
                           f'SHA256:\n {hashlib.sha256(b"").hexdigest()} 0 Packages\n')
                # Avoid a future-date dependency on runner wall time.
                from email.utils import formatdate
                release = re.sub(r'Date: .*', 'Date: ' + formatdate(usegmt=True), release)
                (repo / 'Release').write_text(release)
                run(['gpg', '--batch', '--yes', '--clearsign', '--output',
                     str(repo / 'InRelease'), str(repo / 'Release')], check=True)
                signed = (repo / 'InRelease').read_bytes()
                config = root / 'apt.conf'
                config.write_text(f'''Dir::Etc::main "-";
Dir::Etc::parts "-";
Dir::Etc::sourcelist "{root / 'sources.list'}";
Dir::Etc::sourceparts "-";
Dir::Etc::trusted "{root / 'no-global-keys'}";
Dir::Etc::trustedparts "-";
Dir::State "{root / 'state'}";
Dir::State::status "{root / 'status'}";
Dir::Cache "{root / 'cache'}";
Dir::Log "{root / 'log'}";
APT::Sandbox::User "{getpass.getuser()}";
APT::Update::Error-Mode "any";
Acquire::Languages "none";
''')
                env['APT_CONFIG'] = str(config)
                (root / 'status').write_text('')
                for document in ('README.md', 'scripts/install/test/Dockerfile.fresh-vps'):
                    with self.subTest(document=document):
                        source = (ROOT / document).read_text()
                        match = re.search(r'deb (\[[^\]]+\] )?https://download.arangodb.com/arangodb312/DEBIAN/ /', source)
                        self.assertIsNotNone(match)
                        options = match.group(1) or ''
                        self.assertIn('signed-by=', options)
                        options = re.sub(r'signed-by=[^\] ]+', f'signed-by={key}', options)
                        (root / 'sources.list').write_text(f'deb {options}file:{repo} /\n')
                        for case in ('valid', 'unsigned', 'tampered', 'missing-key'):
                            with self.subTest(case=case):
                                shutil.rmtree(root / 'state', ignore_errors=True)
                                (root / 'state/lists/partial').mkdir(parents=True)
                                (repo / 'InRelease').write_bytes(signed)
                                if case == 'unsigned':
                                    (repo / 'InRelease').unlink()
                                if case == 'tampered':
                                    (repo / 'InRelease').write_bytes(signed.replace(b'HADES private fixture', b'TAMPERED fixture'))
                                if case == 'missing-key':
                                    key.rename(root / 'saved-key')
                                result = run(['apt-get', 'update'])
                                if case == 'missing-key':
                                    (root / 'saved-key').rename(key)
                                output = result.stdout + result.stderr
                                if case == 'valid':
                                    self.assertEqual(result.returncode, 0, output)
                                else:
                                    self.assertNotEqual(result.returncode, 0, output)
                                    self.assertRegex(output, r'(?i)(signature|not signed|public key|NO_PUBKEY)')
            finally:
                run(['gpgconf', '--kill', 'gpg-agent'])


if __name__ == '__main__':
    unittest.main()
