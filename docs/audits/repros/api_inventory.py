"""Print a static wire-command/tier/MCP inventory; never execute application code.

Run from the repository root. This extractor deliberately fails if the reviewed
source layout changes. Counts describe declarations, not behavioral coverage.
"""
import hashlib
import json
from pathlib import Path
import re
import subprocess

paths = [
    'crates/hades-core/src/dispatch.rs',
    'crates/hades-core/src/service.rs',
    'crates/hades-cli/src/commands/mcp_server.rs',
    'crates/hades-cli/src/commands/daemon.rs',
    'crates/hades-cli/src/commands/output.rs',
    'crates/hades-cli/src/main.rs',
    'crates/hades-frontend/src/assemble.rs',
    'crates/hades-frontend/src/backend_process.rs',
    'crates/hades-frontend/src/server.rs',
]
sources = {p: Path(p).read_text() for p in paths}
dispatch = sources[paths[0]]
enum = dispatch.split('pub enum DaemonCommand {', 1)[1].split('\n}\n', 1)[0]
commands = re.findall(r'#\[serde\(rename = "([^"]+)"\)\]\s+(\w+)', enum)
assert len(commands) == enum.count('#[serde(rename = ')
assert len({v for _, v in commands}) == len(commands)
tier_block = dispatch.split('pub fn access_tier(&self)', 1)[1].split('/// Whether this command', 1)[0]
tiers = dict(re.findall(r'Self::(\w+)[^\n]*?=> AccessTier::(\w+)', tier_block))
assert set(tiers) == {variant for _, variant in commands}
mcp = sources[paths[2]].split('#[tool_router]', 1)[1].split('#[tool_handler', 1)[0]
functions = re.split(r'    async fn (\w+)\(', mcp)
tools = {}
for i in range(1, len(functions), 2):
    variants = re.findall(r'DaemonCommand::(\w+)', functions[i+1])
    assert len(variants) == 1, (functions[i], variants)
    tools[functions[i]] = variants[0]
assert len(tools) == mcp.count('#[tool(')
assert set(tools.values()) <= set(tiers)
result = {
    'source_revision': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
    'scope': 'Static exhaustive DaemonCommand tier mapping and curated MCP methods only; not the complete CLI or runtime parity certification',
    'source_sha256': {p: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in paths},
    'wire_command_count': len(commands),
    'mcp_tool_count': len(tools),
    'commands': [{'wire': wire, 'variant': variant, 'tier': tiers[variant],
                  'mcp_tools': sorted(k for k, v in tools.items() if v == variant)}
                 for wire, variant in commands],
}
print(json.dumps(result, indent=2))
