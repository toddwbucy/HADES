const {test} = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs'), path = require('node:path'), vm = require('node:vm');
const source = fs.readFileSync(path.join(__dirname, '../reading.js'), 'utf8');
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
