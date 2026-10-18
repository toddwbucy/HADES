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

function transcriptFixture(api) {
  const f = fixture(api), elements = new Map();
  const element = (tag, text) => ({tag, textContent:text, children:[],
    append(...children){this.children.push(...children);}, replaceChildren(...children){this.children = children;},
    scrollHeight:0, scrollTop:0, clientHeight:0});
  delete f.chat.render; // Exercise the real renderer, with only DOM operations stubbed.
  Object.assign(f.chat, {exchanges:new Map(), node:element, renderApprovals:()=>{},
    $:id => {if (!elements.has(id)) elements.set(id, element()); return elements.get(id);}});
  return f;
}
const exchangeKey = n => `exchange_${String(n).padStart(3, '0')}`;
function historyPage(end) {
  const start = Math.max(1, end - 19);
  return {exchanges:Array.from({length:end - start + 1}, (_, i) => ({_key:exchangeKey(start + i),
    created_at:'2026-10-18T00:00:00Z', status:'completed', message:`Message ${start + i}`})),
    next_before:start > 1 ? exchangeKey(start) : null, pending:{unresolved:false, active:false}, live:{status:'disconnected', items:[], approvals:[]}};
}
const beforeNumber = before => Number(before.split('_')[1]) - 1;
test('remembered read permission is explicit, revocable and reflects saved state', async () => {
  const calls = [], {chat} = transcriptFixture(async value => calls.push(value));
  delete chat.renderApprovals;
  chat.act = fn => fn(); chat.poll = async () => {};
  const live = {generation:'g1', approvals:[]};
  chat.renderApprovals(live, false);
  const root = chat.$('chat-approvals');
  let button = root.children[0].children.find(n => n.tag === 'button');
  assert.equal(button.textContent, 'Always allow reads in this conversation');
  assert.equal(calls.length, 0, 'rendering does not grant permission');
  await button.onclick();
  assert.equal(JSON.stringify(calls[0]), JSON.stringify({action:'read_permission',binding:'binding',allowed:true}));
  chat.renderApprovals(live, true);
  button = root.children[0].children.find(n => n.tag === 'button');
  assert.equal(button.textContent, 'Revoke remembered read permission');
  await button.onclick(); assert.equal(calls[1].allowed, false);
  chat.epoch++; chat.binding = {_key:'other'};
  await button.onclick(); assert.equal(calls.length, 2, 'stale control cannot change another conversation');
});

async function loadAll(chat, count) {
  for (let i = 0; chat.olderCursor && i < 12; i++) await chat.loadOlder();
  assert.equal(chat.olderCursor, null);
  assert.deepEqual([...chat.exchanges.keys()].sort(), Array.from({length:count}, (_, i) => exchangeKey(i + 1)));
  assert.equal(chat.$('chat-older').hidden, true);
}
test('polling after an idle tab exposes and fills a gap in previously complete history', async () => {
  let tip = 5;
  const {chat} = transcriptFixture(async ({before}) => historyPage(before ? beforeNumber(before) : tip));
  await chat.poll(); assert.equal(chat.olderCursor, null);
  tip = 65; await chat.poll();
  assert.equal(chat.olderCursor, exchangeKey(46)); assert.equal(chat.$('chat-older').hidden, false);
  await chat.loadOlder(); assert.equal(chat.olderCursor, exchangeKey(26));
  await chat.poll(); assert.equal(chat.olderCursor, exchangeKey(26), 'overlapping polls preserve pagination progress');
  await loadAll(chat, 65);
});
test('a late older page cannot overwrite a newer gap discovered by polling', async () => {
  let tip = 65, resolve;
  const {chat} = transcriptFixture(async ({before}) => {
    if (!before) return historyPage(tip);
    if (!resolve) return new Promise(r => {resolve = () => r(historyPage(beforeNumber(before)));});
    return historyPage(beforeNumber(before));
  });
  await chat.poll();
  const older = chat.loadOlder();
  tip = 105; await chat.poll(); assert.equal(chat.olderCursor, exchangeKey(86));
  resolve(); await older;
  assert.equal(chat.olderCursor, exchangeKey(86), 'older response must retain the newer gap cursor');
  await loadAll(chat, 105);
});
test('a late older page cannot populate a newly selected binding', async () => {
  let resolve;
  const {chat} = transcriptFixture(() => new Promise(r => {resolve = r;}));
  chat.render(historyPage(40));
  const older = chat.loadOlder();
  chat.epoch++; chat.binding = {_key:'different'}; chat.exchanges.clear(); chat.olderCursor = undefined;
  resolve(historyPage(20)); await older;
  assert.equal(chat.exchanges.size, 0); assert.equal(chat.olderCursor, undefined);
});
test('an older receipt clears only its matching saved submission and permits another send', async () => {
  let sends = 0;
  const f = transcriptFixture(async request => {
    if (request.action === 'send') {sends++; return {};}
    const page = historyPage(request.before ? beforeNumber(request.before) : 45);
    if (request.before) page.exchanges[0].request_id = 'recovered_request';
    return page;
  });
  const key = f.pendingChatKey('test', 'binding'), other = f.pendingChatKey('test', 'other');
  f.saved.set(key, JSON.stringify({request_id:'recovered_request'}));
  f.saved.set(other, JSON.stringify({request_id:'recovered_request'}));
  await f.chat.poll(); assert(f.saved.has(key), 'newest page has no receipt');
  await f.chat.loadOlder(); assert(!f.saved.has(key)); assert(f.saved.has(other));
  f.chat.$('chat-message').value = 'Continue after recovery';
  f.chat.$('chat-passages').selectedOptions = []; f.chat.$('chat-knowledge').selectedOptions = [];
  await f.chat.send(); assert.equal(sends, 1);
});
test('an older page cannot clear a different request saved while it was loading', async () => {
  let resolve;
  const f = transcriptFixture(request => request.before ? new Promise(r => {resolve = r;}) : Promise.resolve(historyPage(45)));
  const key = f.pendingChatKey('test', 'binding');
  f.saved.set(key, JSON.stringify({request_id:'old_request'}));
  await f.chat.poll(); const older = f.chat.loadOlder();
  const replacement = JSON.stringify({request_id:'new_request'}); f.saved.set(key, replacement);
  const page = historyPage(25); page.exchanges[0].request_id = 'old_request';
  resolve(page); await older;
  assert.equal(f.saved.get(key), replacement);
});
test('a late older receipt cannot clear recovery for a newly selected binding', async () => {
  let resolve;
  const f = transcriptFixture(() => new Promise(r => {resolve = r;}));
  f.chat.render(historyPage(45));
  const key = f.pendingChatKey('test', 'different');
  f.saved.set(key, JSON.stringify({request_id:'recovered_request'}));
  const older = f.chat.loadOlder(); f.chat.epoch++; f.chat.binding = {_key:'different'};
  const page = historyPage(25); page.exchanges[0].request_id = 'recovered_request';
  resolve(page); await older;
  assert(f.saved.has(key));
});
test('stale cached acknowledgements cannot block controls after the server completes them', async () => {
  let tip = 1;
  const {chat} = transcriptFixture(async () => {
    const page = historyPage(tip); page.live.status = 'completed';
    if (tip === 1) {page.exchanges[0].status = 'acknowledged'; page.pending = {unresolved:true, active:true};}
    return page;
  });
  await chat.poll();
  assert.equal(chat.$('chat-send').disabled, true); assert.equal(chat.$('chat-release').disabled, true);
  tip = 25; await chat.poll();
  assert.equal(chat.exchanges.get(exchangeKey(1)).status, 'acknowledged', 'old record remains outside the latest page');
  assert.equal(chat.$('chat-send').disabled, false); assert.equal(chat.$('chat-release').disabled, false);
});
test('authoritative pending state still blocks unresolved work outside the displayed page', async () => {
  const f = transcriptFixture();
  const page = historyPage(25); page.live.status = 'completed';
  page.pending = {unresolved:true, active:true}; f.chat.render(page);
  assert.equal(f.chat.$('chat-send').disabled, true); assert.equal(f.chat.$('chat-release').disabled, true);
  page.pending.active = false; f.chat.render(page);
  assert.equal(f.chat.$('chat-send').disabled, true); assert.equal(f.chat.$('chat-release').disabled, false);
  page.pending.unresolved = false;
  f.saved.set(f.pendingChatKey('test', 'binding'), JSON.stringify({request_id:'unknown_submission'}));
  f.chat.render(page); assert.equal(f.chat.$('chat-send').disabled, true, 'an uncertain browser submission still requires recovery');
});
function renderedText(element) {
  return [element.textContent, ...element.children.flatMap(renderedText)].filter(text => text !== undefined);
}
test('recovered terminal answers replace matching disconnected or unresolved live projections', () => {
  for (const status of ['completed', 'failed', 'interrupted']) for (const closed of [true, false]) {
    const {chat} = transcriptFixture();
    const page = historyPage(1), exchange = page.exchanges[0];
    exchange.provider_turn = 'turn_one'; exchange.status = status;
    exchange.projection = {items:[{type:'agentMessage', text:'Recovered final answer'}]};
    page.live = {turn_id:'turn_one', status:'unknown', closed, items:[{type:'agentMessage', text:'Partial answer'}]};
    chat.render(page);
    const text = renderedText(chat.$('chat-messages'));
    assert(text.includes('Recovered final answer')); assert(text.includes(status));
    assert(!text.includes('Partial answer')); assert(!text.includes('unknown'));
  }
});
test('unsettled exchanges still render matching live output before the monitor saves it', () => {
  for (const status of ['inProgress', 'completed']) {
    const {chat} = transcriptFixture();
    const page = historyPage(1), exchange = page.exchanges[0];
    exchange.provider_turn = 'turn_one'; exchange.status = 'acknowledged';
    page.pending = {unresolved:true, active:true};
    page.live = {turn_id:'turn_one', status, items:[{type:'agentMessage', text:'Current streamed answer'}]};
    chat.render(page);
    const text = renderedText(chat.$('chat-messages'));
    assert(text.includes('Current streamed answer')); assert(text.includes(status));
  }
});
