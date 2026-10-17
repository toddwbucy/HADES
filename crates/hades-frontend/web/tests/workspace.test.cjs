const test = require('node:test');
const assert = require('node:assert/strict');
const M = require('../workspace-model.js');
const n = (id,type,label,position) => ({id,label,collection:'notes',attrs:{type,source_location:{body_start:position},content:'An exact source passage.'}});
const e = (source,target) => ({source,target,attrs:{relation:'contains'}});
test('outline follows declared hierarchy and source order without looping on cycles', () => {
  const nodes=[n('notes/b','section','B',20),n('notes/a','section','A',10),n('notes/root','document','Document',0),n('notes/cycle','section','Cycle',30)];
  const rows=M.library({nodes,edges:[e('notes/root','notes/b'),e('notes/root','notes/a'),e('notes/cycle','notes/cycle')]},'outline');
  assert.deepEqual(rows.map(r=>r.node.id),['notes/root','notes/a','notes/b','notes/cycle']);
  assert.deepEqual(rows.map(r=>r.depth),[0,1,1,0]);
});
test('concepts remain document-specific even when labels match', () => {
  const a=n('notes/c1','concept','State',0),b=n('notes/c2','concept','State',1);
  assert.equal(M.library({nodes:[a,b],edges:[]},'concepts','state').length,2);
});
test('reference packets retain scope, exact IDs, source version, offsets and quotation bounds', () => {
  const node=n('notes/a','section','A',10);
  node.attrs.source_version='0.14';node.attrs.source_node_id='doc/a';
  const ref=M.reference({db:'test',graph:'workspace'},node,'node','x'.repeat(2000));
  const packet=JSON.parse(M.contextPacket([ref]));
  assert.equal(packet.references[0].id,'notes/a');
  assert.equal(ref.source_version,'0.14');assert.equal(ref.source_location.body_start,10);
  assert.equal(ref.excerpt.length,1200);assert.equal(ref.excerpt_truncated,true);
  assert.throws(()=>M.contextPacket(Array(7).fill(ref)));
});
test('edge references retain their endpoints and relation, never just a visual label', () => {
  const ref=M.reference({db:'test',graph:'g'},{id:'edges/1',source:'notes/a',target:'notes/b',attrs:{relation:'supports',assertion_origin:'source_explicit'}},'edge');
  assert.equal(ref.from,'notes/a');assert.equal(ref.to,'notes/b');assert.equal(ref.relation,'supports');
  assert.equal(ref.assertion_origin,'source_explicit');
});
test('reader preserves source text, including markup, without rewriting it', () => {
  assert.equal(M.text({attrs:{content:'<script>example</script>\n$$x$$'}}),'<script>example</script>\n$$x$$');
});
