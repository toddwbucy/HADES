"""Measure installed gRPC defaults using a private extraction servicer/socket.

Run from the repository root after generating private stubs. Synthetic payloads
only, fake extraction route, no production endpoint/model/GPU.
"""
import asyncio
import hashlib
import json
from pathlib import Path
import sys
import tempfile

import grpc
sys.path.insert(0, str(Path('services').resolve()))
from extraction.server import ExtractionServicer, ExtractionConfig, ExtractionResult
from generated.persephone.extraction import extraction_pb2 as pb
from generated.persephone.extraction import extraction_pb2_grpc as rpc

async def main():
    with tempfile.TemporaryDirectory(prefix='hades-grpc-boundary-') as directory:
        calls = []
        servicer = ExtractionServicer(ExtractionConfig())
        async def route(path, source_type, request):
            calls.append(len(request.content))
            text = 'x'*(4*1024*1024+1024) if request.file_path == 'large-response.txt' else 'ok'
            return ExtractionResult(text=text)
        servicer._route_extraction = route
        server = grpc.aio.server()  # same construction as the inspected entry point
        rpc.add_ExtractionServiceServicer_to_server(servicer, server)
        endpoint = 'unix:'+str(Path(directory)/'fixture.sock')
        assert server.add_insecure_port(endpoint)
        await server.start()
        rows = []
        try:
            async with grpc.aio.insecure_channel(endpoint) as channel:
                stub = rpc.ExtractionServiceStub(channel)
                for name, content, path, expected in [
                    ('one_mib_upload', b'x'*(1024*1024), 'fixture.txt', 'OK'),
                    ('over_four_mib_upload', b'x'*(4*1024*1024+1024), 'fixture.txt', 'RESOURCE_EXHAUSTED'),
                    ('over_four_mib_response', b'', 'large-response.txt', 'RESOURCE_EXHAUSTED'),
                ]:
                    before = len(calls)
                    response_size = None
                    try:
                        response = await stub.Extract(pb.ExtractRequest(file_path=path, content=content, source_type=pb.SOURCE_TYPE_TEXT), timeout=5)
                        code = 'OK'
                        response_size = len(response.full_text)
                    except grpc.aio.AioRpcError as error:
                        code = error.code().name
                    assert code == expected, (name,code)
                    rows.append({'case':name,'status':code,'backend_calls':len(calls)-before,'returned_text_bytes':response_size})
                assert rows[1]['backend_calls'] == 0
                assert rows[2]['backend_calls'] == 1
        finally:
            servicer.stop_admission()
            await server.stop(0)
            await servicer.close()
        paths = [Path('services/extraction/server.py'),Path(__file__).resolve().relative_to(Path.cwd())]
        print(json.dumps({'grpcio_version':grpc.__version__,'scope':'Actual private gRPC transport using current CI environment defaults; fake extraction. Not live installation or Rust client capacity verification.', 'source_sha256':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},'cases':rows},indent=2))

asyncio.run(main())
