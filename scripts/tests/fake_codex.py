#!/usr/bin/env python3
"""Deterministic App Server fixture. Only reads/writes its explicit CODEX_HOME."""
import fcntl
import json
import os
from pathlib import Path
import sys
import threading
import time
import uuid

if '--version' in sys.argv:
    print('codex-cli 0.163.0')
    raise SystemExit
root = Path(os.environ['CODEX_HOME'])
assert (root / 'hades-test-fixture').is_file(), 'This fake requires an isolated fixture directory'
output_lock = threading.Lock()
owned = {}
active = {}
pending = {}


def emit(value):
    with output_lock:
        print(json.dumps(value), flush=True)


def event(method, params):
    emit({'method': method, 'params': params})


def load(tid):
    return json.loads((root / (tid + '.json')).read_text())


def save(tid, history):
    path = root / (tid + '.json')
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(history))
    tmp.replace(path)


def finish(tid, turn, text='Fixture response', status='completed'):
    if active.get(tid) != turn:
        return
    item = {'id': 'assistant_' + turn, 'type': 'agentMessage', 'text': text}
    history = load(tid)
    for row in history:
        if row['id'] == turn:
            row['status'] = status
            row['items'].append(item)
    save(tid, history)
    event('item/completed', {'threadId': tid, 'turnId': turn, 'item': item})
    event('turn/completed', {'threadId': tid, 'turn': {'id': turn, 'status': status}})
    active.pop(tid, None)


def respond(tid, turn, prompt):
    time.sleep(.2)
    if '/approval' in prompt:
        pending[900] = (tid, turn)
        emit({'id': 900, 'method': 'item/tool/requestUserInput', 'params': {
            'threadId': tid, 'turnId': turn, 'questions': [
                {'id': 'scope', 'question': 'Which passage scope?', 'options': [
                    {'label': 'Selected passage'}, {'label': 'Whole document'}]}]}})
        return
    if '/slow' in prompt:
        return
    event('item/agentMessage/delta', {'threadId': tid, 'turnId': turn,
                                    'itemId': 'assistant_' + turn, 'delta': 'Discussing the frozen passage…'})
    time.sleep(.2)
    finish(tid, turn, 'The frozen passage is the evidence for this discussion.')


for line in sys.stdin:
    message = json.loads(line)
    if 'method' not in message:
        pair = pending.pop(message.get('id'), None)
        if pair:
            event('serverRequest/resolved', {'threadId': pair[0], 'requestId': message['id']})
            finish(*pair, 'Answer received; discussion continued.')
        continue
    method, params = message['method'], message.get('params', {})
    if 'id' not in message:
        continue
    try:
        result = {}
        if method == 'config/read':
            result = {'config': {'mcp_servers': {'unsafe': {'enabled': True}}, 'plugins': {}}}
        elif method == 'mcpServerStatus/list':
            result = {'data': [{'name': 'hades_reading', 'runtimeStatus': 'connected', 'tools': {
                name: {'name': name} for name in ('reading_context', 'reading_exchange_context',
                'reading_propose', 'db_query', 'db_get', 'db_lookup', 'graph_neighbors')}}], 'nextCursor': None}
        elif method == 'thread/list':
            result = {'data': [{'id': p.stem, 'name': 'Fixture conversation', 'source': 'exec'}
                               for p in root.glob('*.json')], 'nextCursor': None}
        elif method in ('thread/start', 'thread/resume'):
            tid = params.get('threadId') or str(uuid.uuid4())
            if method == 'thread/resume' and not (root / (tid + '.json')).exists():
                raise ValueError('no rollout found for thread id ' + tid)
            policy = params['config']
            assert policy['features']['shell_tool'] is False
            assert policy['mcp_servers']['unsafe']['enabled'] is False
            assert 'kg_add_node' not in policy['mcp_servers']['hades_reading']['enabled_tools']
            lock = (root / (tid + '.lock')).open('a')
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                lock.close()
                raise ValueError('thread ' + tid + ' already has an active writer')
            owned[tid] = lock
            if method == 'thread/start':
                save(tid, [])
            result = {'thread': {'id': tid}}
        elif method == 'thread/turns/list':
            result = {'data': load(params['threadId']), 'nextCursor': None}
        elif method == 'turn/start':
            tid, turn = params['threadId'], str(uuid.uuid4())
            assert tid in owned
            history = load(tid)
            prompt = params['input'][0]['text']
            history.append({'id': turn, 'status': 'inProgress', 'items': [{
                'id': str(uuid.uuid4()), 'type': 'userMessage', 'clientId': params['clientUserMessageId'],
                'content': [{'type': 'text', 'text': prompt}]}]})
            save(tid, history)
            active[tid] = turn
            if '/disconnect' in prompt:
                finish(tid, turn, 'Completed before acknowledgement was lost.')
                os._exit(0)
            result = {'turn': {'id': turn, 'status': 'inProgress'}}
            event('turn/started', {'threadId': tid, 'turn': {'id': turn, 'status': 'inProgress'}})
            threading.Thread(target=respond, args=(tid, turn, prompt), daemon=True).start()
        elif method == 'turn/interrupt':
            finish(params['threadId'], params['turnId'], 'Interrupted', 'interrupted')
        elif method == 'initialize':
            result = {'userAgent': 'codex/0.163.0'}
        else:
            raise ValueError('Unsupported fake method ' + method)
        emit({'id': message['id'], 'result': result})
    except Exception as error:
        emit({'id': message['id'], 'error': {'code': -32600, 'message': str(error)}})
