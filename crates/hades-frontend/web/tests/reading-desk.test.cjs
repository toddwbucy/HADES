const {test} = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs'), path = require('node:path'), vm = require('node:vm');
const source = fs.readFileSync(path.join(__dirname, '../reading.js'), 'utf8');
test('reading URL and workspace link retain the selected conversation and clear it on deselection', () => {
  const code = source.slice(source.indexOf('function syncReadingConversation('), source.indexOf('const readingChat ='));
  const link = {}, urls = [];
  const sync = vm.runInNewContext(code + ';syncReadingConversation', {URLSearchParams,
    state:{id:'reading', graph:'knowledge'}, database:'test', $:() => link,
    history:{replaceState:(_state, _title, url) => urls.push(url)}});
  sync({_key:'older'});
  assert.equal(new URL('http://fixture'+urls.at(-1)).searchParams.get('conversation'), 'older');
  assert.equal(new URL('http://fixture'+link.href).searchParams.get('conversation'), 'older');
  sync(null);
  assert.equal(urls.at(-1), '/reading?session=reading'); assert.equal(link.hidden, true);
});
const guard = source.slice(source.indexOf('function checkRemovalDrafts('), source.indexOf('async function setArchived('));
function fixture(state, drafts, message = '', editingPassage = null) {
  return vm.runInNewContext(guard + ';checkRemovalDrafts', {
    state, boundaryDrafts:new Map(drafts.map(id => [id, {is:'Unsaved boundary'}])),
    editingPassage, $:() => ({value:message})
  });
}
test('a target boundary draft blocks removal even after changing or clearing the selection', () => {
  for (const state of [{id:'target'}, {id:'another'}, null]) {
    assert.throws(() => fixture(state, ['target'])('target'), /unfinished edits/);
  }
});
test('passage edits and unsent chat only block removal of their selected reading', () => {
  for (const [message, edit] of [['Question', null], ['', {text:'Draft'}]]) {
    const check = fixture({id:'selected'}, [], message, edit);
    assert.throws(() => check('selected'), /unfinished edits/);
    assert.doesNotThrow(() => check('another'));
  }
  assert.doesNotThrow(() => fixture({id:'another'}, ['another'])('target'));
});

test('passage edits block source creation, resumption and direct restoration before requests', async () => {
  const switchAndOpen = source.slice(source.indexOf('function checkReadingSwitch('), source.indexOf('async function load('));
  const archive = source.slice(source.indexOf('async function setArchived('), source.indexOf('function checkReadingSwitch('));
  let calls = 0;
  const {openSource, setArchived} = vm.runInNewContext(switchAndOpen + archive + ';({openSource,setArchived})', {
    editingPassage:{text:'Unsaved correction'}, api:() => {calls++; throw Error('Unexpected request');}
  });
  await assert.rejects(openSource('hash', 'title'), /Save or cancel the passage edit/);
  await assert.rejects(openSource('hash', 'title', true), /Save or cancel the passage edit/);
  await assert.rejects(setArchived('removed', false), /Save or cancel the passage edit/);
  assert.equal(calls, 0);
});
