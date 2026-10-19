const {test} = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs'), path = require('node:path'), vm = require('node:vm');
const source = fs.readFileSync(path.join(__dirname, '../workspace-chat.js'), 'utf8').replace(/^import .*;\n/, '');
const tick = () => new Promise(resolve => setImmediate(resolve));
function fixture() {
  let database = 'reading_db', chat;
  const requests = [], notes = [];
  const window = {workspace: {database:() => database, initialConversation:() => ({db:database, binding:'continued'}), references:() => []}};
  class ReadingChat {
    constructor(options) {Object.assign(this, options); chat = this;}
    async selectReading() {this.current = this.getState();}
    note(text) {notes.push(text);}
  }
  vm.runInNewContext(source, {window, ReadingChat, fetch:() => new Promise(resolve => requests.push(resolve))});
  return {window, requests, chat, notes, change:value => {database = value;}};
}
const config = database => ({ok:true, json:async () => ({enabled:true, database})});
test('workspace chat disables on another database and reconnects only to its configured database', async () => {
  const f = fixture(); await tick(); f.requests.shift()(config('reading_db')); await tick();
  assert(f.chat.getState()); assert.equal(f.chat.preferredBinding, 'continued');
  f.change('other_db'); const other = f.window.workspaceChat.sync(); await tick();
  assert.equal(f.chat.getState(), null);
  f.requests.shift()(config('reading_db')); await other;
  assert.equal(f.chat.getState(), null); assert.match(f.notes.at(-1), /Select that database/);
  assert.equal(f.chat.preferredBinding, null);
  f.change('reading_db'); const back = f.window.workspaceChat.sync(); await tick();
  f.requests.shift()(config('reading_db')); await back;
  assert.equal(f.chat.getState().id, 'workspace:reading_db');
});
test('a late configuration response cannot enable chat in a different database', async () => {
  const f = fixture(); await tick(); const old = f.requests.shift();
  f.change('other_db'); const next = f.window.workspaceChat.sync(); await tick();
  f.requests.shift()(config('reading_db')); await next;
  old(config('reading_db')); await tick();
  assert.equal(f.chat.getDatabase(), 'other_db'); assert.equal(f.chat.getState(), null);
});
