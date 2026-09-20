"""Real CLI/Rust-client RPC exercise with explicit CPU device adaptation."""
import asyncio
import http.client
import json
from pathlib import Path
import shutil
import socket

import grpc
import torch

from hades.training import training_pb2_grpc as rpc
from training.config import TrainingConfig
from training.server import TrainingServicer
from training.session import SessionTrainingServicer


async def exercise(root, manifest, expected):
    # The CLI only accepts GPU indices. Adapt that declaration at the private
    # provider boundary; all actual model/checkpoint/graph/export work stays CPU.
    declared_devices = []
    class CpuFixtureConfig(TrainingConfig):
        def resolve_device(self, requested):
            if requested != "cuda:0":
                raise ValueError(f"unexpected fixture device {requested!r}")
            declared_devices.append(requested)
            return "cpu"

    directory = root / 'rpc'
    directory.mkdir()
    service = SessionTrainingServicer(lambda: TrainingServicer(CpuFixtureConfig()))
    server = grpc.aio.server()
    rpc.add_TrainingServiceServicer_to_server(service, server)
    assert server.add_insecure_port(f'unix:{directory}/training.sock')
    await server.start()
    shutil.copyfile(root / 'after.pt', root / 'best.pt')

    def document(ident, value=None):
        # Only the explicitly passed fixture socket/database; no discovery.
        db_socket = Path(manifest['database_socket'])
        if not str(db_socket).startswith('/tmp/hades-tests-') or db_socket.name != 'arango.sock':
            raise ValueError('expected the disposable runner socket')
        connection = http.client.HTTPConnection('localhost', timeout=3)
        connection.sock = socket.socket(socket.AF_UNIX)
        connection.sock.settimeout(3)
        connection.sock.connect(str(db_socket))
        try:
            connection.request('GET' if value is None else 'PATCH',
                               f'/_db/{manifest["database"]}/_api/document/{ident}',
                               body=None if value is None else json.dumps(value),
                               headers={'Content-Type':'application/json'})
            response = connection.getresponse()
            result = json.loads(response.read())
            assert response.status < 300 and result.get('error') is not True, result
            return result
        finally:
            connection.close()

    async def cli(missing_only=False):
        command = ['bwrap', '--ro-bind', '/', '/', '--tmpfs', '/run',
                   '--dir', '/run/hades', '--ro-bind', str(directory), '/run/hades',
                   '--bind', str(root), str(root), '--dev', '/dev', '--unshare-net', '--',
                   manifest['cli_binary'], '--db', manifest['database'], '--gpu', '0',
                   'graph-embed', 'update', '--checkpoint-dir', str(root)]
        if missing_only:
            command.append('--new-nodes')
        child = await asyncio.create_subprocess_exec(
            'timeout', '--kill-after=5', '30', *command,
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
        stdout, stderr = await child.communicate()
        assert child.returncode == 0, stderr.decode()
        result = json.loads(stdout)
        assert result['success'] is True
        return result['data']

    try:
        full = await cli()
        assert full['export']['count'] == 3
        before = {ident: document(ident) for ident in manifest['ids']}
        for ident, row in before.items():
            torch.testing.assert_close(torch.tensor(row['structural_embedding']),
                                       expected[manifest['ids'].index(ident)], rtol=0, atol=0)
        document('papers/same', {'structural_embedding': None})
        subset = await cli(True)
        assert subset['export']['count'] == 1
        for ident in manifest['ids']:
            row = document(ident)
            if ident != 'papers/same':
                assert row == before[ident], 'unselected row or revision changed'
            torch.testing.assert_close(torch.tensor(row['structural_embedding']),
                                       expected[manifest['ids'].index(ident)], rtol=0, atol=0)
        assert declared_devices == ['cuda:0', 'cuda:0']
        print(json.dumps({'actual_cli_full_export':3, 'actual_cli_missing_only_export':1,
                          'unselected_rows_and_revisions_preserved':True,
                          'device_adaptation':'private provider maps declared cuda:0 to CPU',
                          'live_training_socket_and_gpu_devices_hidden':True}), flush=True)
    finally:
        await server.stop(0)
        await service.close()
