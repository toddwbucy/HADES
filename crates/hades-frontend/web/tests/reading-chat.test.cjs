const {test} = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs'), path = require('node:path'), vm = require('node:vm');
const source = fs.readFileSync(path.join(__dirname, '../reading-chat.js'), 'utf8').replaceAll('export ', '');
function fixture(api) {
  const saved = new Map();
  const context = {localStorage:{getItem:k => saved.get(k) || null, setItem:(k,v) => saved.set(k,v), removeItem:k => saved.delete(k)}, crypto:{randomUUID:() => 'request_one'}};
  const {ReadingChat, makeSubmission, pendingChatKey} = vm.runInNewContext(source + ';({ReadingChat, makeSubmission, pendingChatKey})', context);
  const chat = Object.create(ReadingChat.prototype);
  Object.assign(chat, {binding:{_key:'binding'}, epoch:1, references:[], api, getDatabase:() => 'test', getState:() => ({revision:3}),
    $:id => id === 'chat-message' ? {value:'Question'} : {selectedOptions:[]}, note:()=>{}, renderReferences:()=>{}, render:()=>{throw Error('stale render');}});
  return {chat, saved, makeSubmission, pendingChatKey};
}
test('submission freezes selections and keeps database/binding recovery separate', () => {
  const {makeSubmission, pendingChatKey} = fixture();
  const selected = ['h1'], references = [{thread_id:'t1',turn_id:'t2',item_id:'i1'}];
  const payload = makeSubmission('b1', {revision:4}, 'message', selected, references, 'r1');
  selected[0] = 'h2'; references[0].item_id = 'changed';
  assert.equal(payload.highlights[0], 'h1'); assert.equal(payload.references[0].item_id, 'i1');
  assert.notEqual(pendingChatKey('a','b'), pendingChatKey('b','a'));
});
test('lost acknowledgement preserves exact request without automatic retry', async () => {
  let calls = 0;
  const f = fixture(async () => {calls++; throw Error('connection lost');}); f.chat.poll = async () => {};
  await f.chat.send(); assert.equal(calls, 1); assert.equal(f.saved.size, 1);
  const raw = [...f.saved.values()][0];
  await assert.rejects(f.chat.send(), /unresolved/);
  assert.equal(calls, 1); assert.equal([...f.saved.values()][0], raw);
});
test('definitive revision rejection clears recovery and allows a corrected submission', async () => {
  const calls = [];
  const f = fixture(async payload => {
    calls.push(payload);
    return {rejected: true, error: 'Reading changed before dispatch; review context again'};
  });
  await f.chat.send(); assert.equal(f.saved.size, 0);
  f.chat.getState = () => ({revision:4});
  await f.chat.send(); assert.equal(f.saved.size, 0);
  assert.deepEqual(calls.map(p => p.expected_revision), [3, 4]);
});
test('late poll cannot clear another conversation submission or render stale state', async () => {
  let resolve;
  const f = fixture(() => new Promise(r => {resolve = r;}));
  const key = f.pendingChatKey('test', 'binding');
  f.saved.set(key, JSON.stringify({request_id:'request_one'}));
  const pending = f.chat.poll(); f.chat.epoch++; f.chat.binding = {_key:'different'};
  resolve({exchanges:[{request_id:'request_one'}]}); await pending;
  assert(f.saved.has(key)); assert.equal(f.chat.polling, false);
});
