"""Opt-in actual CLI CPU training and checkpoint/export contract."""
import asyncio
import http.client
import json
from pathlib import Path
import socket
import sys

import grpc
import torch
from test_training_rpc_validation import invoke, pb
from hades.training import training_pb2_grpc as rpc
from training.config import TrainingConfig
from training.server import TrainingServicer
from training.session import SessionTrainingServicer


async def main(root):
    manifest = json.loads((root / 'manifest.json').read_text())
    torch.set_num_threads(1)
    torch.manual_seed(1729)
    events = []

    class CpuConfig(TrainingConfig):
        def resolve_device(self, requested):
            assert requested == 'cuda:0'
            return 'cpu'

    class Sessions(SessionTrainingServicer):
        async def AcquireSession(self, request, context):
            result = await super().AcquireSession(request, context)
            events.append('acquired')
            return result

        async def ReleaseSession(self, request, context):
            result = await super().ReleaseSession(request, context)
            events.append('released')
            return result

    directory = root / 'rpc'
    directory.mkdir()
    service = Sessions(lambda: TrainingServicer(CpuConfig()))
    server = grpc.aio.server()
    rpc.add_TrainingServiceServicer_to_server(service, server)
    assert server.add_insecure_port(f'unix:{directory}/training.sock')
    await server.start()

    def rows():
        db_socket = Path(manifest['database_socket'])
        assert str(db_socket).startswith('/tmp/hades-tests-') and db_socket.name == 'arango.sock'
        result = {}
        for ident in manifest['ids']:
            connection = http.client.HTTPConnection('localhost', timeout=3)
            connection.sock = socket.socket(socket.AF_UNIX)
            connection.sock.settimeout(3)
            connection.sock.connect(str(db_socket))
            try:
                connection.request('GET', f'/_db/{manifest["database"]}/_api/document/{ident}')
                response = connection.getresponse()
                assert response.status == 200
                result[ident] = json.loads(response.read())
            finally:
                connection.close()
        return result

    async def cli(action, extra=()):
        command = ['timeout', '--kill-after=5', '30', 'bwrap', '--ro-bind', '/', '/',
                   '--tmpfs', '/run', '--dir', '/run/hades', '--ro-bind', str(directory), '/run/hades',
                   '--bind', str(root), str(root), '--dev', '/dev', '--unshare-net', '--',
                   manifest['cli_binary'], '--db', manifest['database'], '--gpu', '0',
                   'graph-embed', action, '--checkpoint-dir', str(root), *extra]
        child = await asyncio.create_subprocess_exec(*command, stdout=asyncio.subprocess.PIPE,
                                                     stderr=asyncio.subprocess.PIPE)
        stdout, stderr = await child.communicate()
        assert child.returncode == 0, stderr.decode()
        result = json.loads(stdout)
        assert result['success'] is True
        return result['data']

    try:
        trained = await cli('train', ['--epochs', '2', '--dimension', '4', '--hidden-dim', '8',
                                     '--num-bases', '1', '--dropout', '0', '--val-ratio', '0.25',
                                     '--test-ratio', '0.25', '--seed', '1729'])
        assert trained['training']['total_epochs'] == 2
        assert trained['export']['count'] == len(manifest['ids'])
        assert Path(trained['checkpoint_path']).resolve() == (root / 'best.pt').resolve()
        for value in [trained['training']['best_val_loss'], *trained['training']['test'].values()]:
            assert isinstance(value, (int, float)) and torch.isfinite(torch.tensor(value))
        before = rows()
        # Load the saved best checkpoint into a separate CPU backend, and use
        # the CLI-produced graph to verify the persisted node/vector mapping.
        verifier = TrainingServicer(TrainingConfig())
        await asyncio.to_thread(invoke, verifier, 'LoadCheckpoint', pb.LoadCheckpointRequest(path=str(root / 'best.pt'), device='cpu'))
        await asyncio.to_thread(invoke, verifier, 'LoadGraph', pb.LoadGraphRequest(safetensors_path=str(root / 'graph.safetensors')))
        output = await asyncio.to_thread(invoke, verifier, 'GetEmbeddings', pb.GetEmbeddingsRequest(output_path=str(root / 'verify.bin')))
        assert output.num_nodes == len(manifest['ids']) and output.embed_dim == 4
        expected = torch.frombuffer(bytearray((root / 'verify.bin').read_bytes()), dtype=torch.float32).reshape(-1, 4)
        for index, ident in enumerate(manifest['ids']):
            torch.testing.assert_close(torch.tensor(before[ident]['structural_embedding']), expected[index], rtol=0, atol=0)
        updated = await cli('update')
        assert updated['export']['count'] == len(manifest['ids'])
        await asyncio.to_thread(invoke, verifier, 'LoadGraph', pb.LoadGraphRequest(
            safetensors_path=str(root / 'graph_inference.safetensors')))
        await asyncio.to_thread(invoke, verifier, 'GetEmbeddings', pb.GetEmbeddingsRequest(
            output_path=str(root / 'verify-inference.bin')))
        inference = torch.frombuffer(bytearray((root / 'verify-inference.bin').read_bytes()),
                                     dtype=torch.float32).reshape(-1, 4)
        changed = []
        for index, (ident, row) in enumerate(rows().items()):
            torch.testing.assert_close(torch.tensor(row['structural_embedding']), inference[index], rtol=0, atol=0)
            if row['structural_embedding'] != before[ident]['structural_embedding']:
                changed.append(ident)
        assert changed, 'fixture must distinguish training and full-graph adjacency'
        assert events == ['acquired', 'released'] * 2
        print(json.dumps({'train_epochs':2, 'exported_nodes':len(manifest['ids']),
                          'best_checkpoint_vectors_exact':True, 'immediate_update_matches_full_graph_checkpoint':True,
                          'changed_with_full_graph_context':changed,
                          'sessions_released':2, 'device_adaptation':'declared cuda:0 mapped to CPU',
                          'training':trained['training']}), flush=True)
    finally:
        await server.stop(0)
        await service.close()


if __name__ == '__main__':
    asyncio.run(main(Path(sys.argv[1])))
