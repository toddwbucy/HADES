"""Exercise the real smoke script with fake CLI/curl; never contact a database."""
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

SCRIPT = Path(__file__).resolve().parents[1] / 'bident_burn_smoke.sh'
MOCK = r'''#!/usr/bin/env python3
import json, os, sys
from pathlib import Path
p=Path(os.environ['SMOKE_STATE']); s=json.loads(p.read_text()); a=sys.argv[1:]
if a==['--version']: print('fixture'); sys.exit(0)
assert a[:2]==['--db','bident_burn_smoke'],a
a=a[2:]; s['calls'].append(a); data={}; rc=0
if a[:2]==['db','collections']: rc=0 if s.get('exists') else 1
elif a[:2]==['db','create-database']: s['exists']=True
elif a[:2]==['db','count']: data={'count':s.get(a[2],0)}
elif a[:2]==['db','truncate']:
 assert a[3:]==['--force'],a
 s[a[2]]=0
elif a[:2]==['task','create']: data={'_key':'task'}
elif a[:2]==['task','show']: data={'status':'closed'}
elif a[0]=='ingest':
 for c in ['documents','chunks','embeddings']: s[c]=1
 data={'completed':1}
 if '--batch' in a: rc=1
elif a[:2]==['codebase','ingest']:
 for c in ['codebase_files','codebase_chunks','codebase_embeddings']: s[c]=1
elif a[:2]==['db','aql']: data={'results':[0]}
elif a[:2]==['codebase','drift']:
 missing=not (Path(a[2])/'src/lib.rs').exists()
 data={'stale':{'count':int(missing),'keys':['file'] if missing else []}}
elif a[:2]==['codebase','retire']: s['codebase_files']=0
elif a[:2]==['codebase','prune-orphans']: s['codebase_chunks']=0
elif a[:2]==['db','query']: data={'results':[{'text':'fixture'}]}
p.write_text(json.dumps(s)); print(json.dumps({'success':rc==0,'data':data})); sys.exit(rc)
'''


class SmokeScript(unittest.TestCase):
    def test_created_and_existing_database_each_count_37_and_force_cleanup(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            mock=root/'hades'; mock.write_text(MOCK); mock.chmod(0o755)
            state=root/'state.json'; state.write_text(json.dumps({'calls':[]}))
            hooks=root/'hooks'
            # Override only fixture placement and the network preflight. The
            # script itself, command dispatch, assertions and counters run intact.
            hooks.write_text('curl() { return 0; }\nmktemp() { command mktemp -d "$SMOKE_ROOT/work.XXXXXX"; }\n')
            env={**os.environ,'HADES_BIN':str(mock),'SMOKE_STATE':str(state),
                 'SMOKE_ROOT':str(root),'BASH_ENV':str(hooks)}
            for _ in range(2):
                run=subprocess.run(['bash',str(SCRIPT)],env=env,capture_output=True,text=True,timeout=30)
                self.assertEqual(run.returncode,0,run.stdout+run.stderr)
                self.assertIn('37 passed, 0 failed',run.stdout)
            calls=json.loads(state.read_text())['calls']
            self.assertEqual(sum(a[:2]==['db','create-database'] for a in calls),1)
            truncates=[a for a in calls if a[:2]==['db','truncate'] and a[2].startswith('codebase_')]
            self.assertEqual(len(truncates),16)
            self.assertTrue(all(a[-1]=='--force' for a in truncates))
