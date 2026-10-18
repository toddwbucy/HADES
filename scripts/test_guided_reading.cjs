/* Real browser + MCP flow against the private runner. No API interception. */
const fs=require('node:fs');const path=require('node:path');const assert=require('node:assert/strict');const crypto=require('node:crypto');
const {chromium}=require(process.env.PLAYWRIGHT_MODULE||'playwright');
const [root,url,mcp]=process.argv.slice(2);let mcpSession=null,requestId=0;
async function rpc(method,params){const response=await fetch(mcp,{method:'POST',headers:{Authorization:'Bearer isolated-reading-fixture-token','Content-Type':'application/json',Accept:'application/json, text/event-stream',...(mcpSession?{'Mcp-Session-Id':mcpSession,'MCP-Protocol-Version':'2024-11-05'}:{})},body:JSON.stringify({jsonrpc:'2.0',id:++requestId,method,params})});mcpSession=response.headers.get('mcp-session-id')||mcpSession;const text=await response.text();if(!response.ok)throw Error(`MCP HTTP ${response.status}: ${text}`);const parsed=text.startsWith('data:')||text.includes('\ndata:')?text.split('\n').filter(l=>l.startsWith('data:')&&l.slice(5).trim()).map(l=>JSON.parse(l.slice(5))).find(v=>v.id===requestId):JSON.parse(text);if(parsed.error)throw Error(JSON.stringify(parsed.error));return parsed.result}
async function tool(name,args){const result=await rpc('tools/call',{name,arguments:args});if(result.isError)throw Error(JSON.stringify(result));const value=JSON.parse(result.content.find(c=>c.type==='text').text);assert.equal(value.success,true,JSON.stringify(value));return value.data}
async function main(){await rpc('initialize',{protocolVersion:'2024-11-05',capabilities:{},clientInfo:{name:'hades-reading-e2e',version:'1'}});await fetch(mcp,{method:'POST',headers:{Authorization:'Bearer isolated-reading-fixture-token','Content-Type':'application/json',Accept:'application/json, text/event-stream','Mcp-Session-Id':mcpSession},body:JSON.stringify({jsonrpc:'2.0',method:'notifications/initialized'})});const tools=await rpc('tools/list',{});assert(tools.tools.some(t=>t.name==='reading_propose'));const denied=await rpc('tools/call',{name:'kg_create',arguments:{db:'reading_e2e',name:'must_not_create'}});assert.equal(denied.isError,true);assert(JSON.stringify(denied).includes('guided reading'));
const browser=await chromium.launch({executablePath:process.env.CHROMIUM_PATH,headless:true,args:['--no-sandbox']});const page=await browser.newPage({viewport:{width:1800,height:1200}});const errors=[];page.on('pageerror',e=>errors.push(e.message));
async function state(){return page.evaluate(async()=>{const id=new URLSearchParams(location.search).get('session');return (await(await fetch('/api/reading/state?session='+id)).json()).state})}
async function revision(n){
 const deadline=Date.now()+30000;
 while(Date.now()<deadline){if((await state())?.revision>=n){await page.waitForTimeout(150);return;}await page.waitForTimeout(100);}
 throw Error(`Expected revision ${n}: ${await page.locator('#status').textContent()}`);
}
try{await page.goto(url+'/reading');await page.waitForFunction(()=>document.querySelector('#formats').textContent.includes('advertised'));
for(const extension of ['docx','pdf']){
 await page.locator('#upload').setInputFiles(path.join(root,'fixture.'+extension));await page.waitForFunction(ext=>document.querySelector('#title').textContent==='fixture.'+ext,extension,{timeout:180000});await page.waitForTimeout(400);
 assert.equal(await page.locator('[data-label="node"]').isDisabled(),true);
 await page.locator('#bound-is').fill('A synthetic fixture for guided-reading validation.');
 await page.locator('#advance').click();await page.waitForFunction(()=>document.querySelector('#status').textContent.includes('Fill in both IS and IS NOT'));
 assert.equal(await page.locator('#bound-is').inputValue(),'A synthetic fixture for guided-reading validation.');
 assert.equal((await state()).revision,0);
 await page.locator('#bound-not').fill('An accepted claim about a research paper.');
 // A rejected save must preserve both fields after the server refresh, even after blur.
 await page.locator('#bound-is').fill('x'.repeat(4001));await page.locator('#status').evaluate(el=>el.classList.remove('error'));await page.locator('#save-bound').click();
 await page.waitForFunction(()=>document.querySelector('#status').classList.contains('error'));
 assert.equal(await page.locator('#bound-is').inputValue(),'x'.repeat(4001));
 assert.equal(await page.locator('#bound-not').inputValue(),'An accepted claim about a research paper.');
 await page.locator('#bound-is').fill('A synthetic fixture for guided-reading validation.');
 if(extension==='pdf'){await page.locator('#save-bound').click();await revision(1);}
 await page.locator('#advance').click();await revision(2);
 assert.equal((await state()).stage,'read');
 const surface='[data-anchor-text]';assert.equal(await page.locator('.pdf-page:visible').count(),0);assert.equal(await page.locator('#surface-title').textContent(),'Converted document');await page.locator(surface).first().waitFor({timeout:30000});
 // Use a real pointer drag. Programmatic Selection bypasses text-layer hit testing.
 assert.equal(await page.locator('[data-label="node"]').isEnabled(),true);
 const passage=page.locator(surface).filter({hasText:'The human selects a source passage.'}).first();
 await passage.scrollIntoViewIfNeeded();
 const rect=await passage.evaluate(el=>{const range=document.createRange();range.selectNodeContents(el);return range.getBoundingClientRect().toJSON()});
 await page.mouse.move(rect.x+.5,rect.y+rect.height/2);await page.mouse.down();await page.mouse.move(rect.right-.5,rect.y+rect.height/2,{steps:15});await page.mouse.up();
 assert((await page.evaluate(()=>getSelection().toString())).includes('human selects'));
 await page.locator('[data-label="node"]').click();await revision(4);
 let s=await state();const h=Object.values(s.highlights)[0];assert(h,JSON.stringify({state:s,status:await page.locator('#status').textContent()}));assert(h.anchor.exact.includes('human selects'));assert.equal(h.anchor.representation,'docling-blocks-v1');assert(h.anchor.block);
 assert(await page.evaluate(()=>CSS.highlights.get('hades-node')?.size>0),'Saved classification must paint the source');
 if(extension==='pdf'){
  await page.locator('#view-split').click();await page.locator('#pdf-pane .textLayer span').first().waitFor();
  assert(await page.locator('#converted-pane').isVisible());assert(await page.locator('#pdf-pane').isVisible());
  assert.equal(await page.locator('#view-split').getAttribute('aria-pressed'),'true');
  assert(await page.evaluate(()=>document.body.classList.contains('wide')));
  const panes=await page.evaluate(()=>['converted-pane','pdf-pane'].map(id=>document.getElementById(id).getBoundingClientRect().toJSON()));
  assert(panes[0].right<=panes[1].left,'Comparison panes must sit beside one another');
  assert(await page.evaluate(()=>CSS.highlights.get('hades-node')?.size>0));
  await page.locator('#reader').scrollIntoViewIfNeeded();assert(await page.locator('#view-converted').isVisible());
  await page.screenshot({path:path.join(root,'reading-comparison.png'),fullPage:true});
  await page.locator('#view-pdf').click();assert(await page.locator('#converted-pane').isHidden());
  await page.locator('#view-converted').click();assert(await page.locator('#pdf-pane').isHidden());
 }
 const context=await tool('reading_context',{db:'reading_e2e',session:s.id,view:'context'});assert.equal(context.references[0].id,h.id);
 const draft={title:'Selected passage',note:'A proposed reading note.',is:'A fixture observation',is_not:null,standing:'fixture only',highlight:h.id,basis:'It addresses the bounded test task.',from:null,to:null,relation:null,inferred:false,rationale:null};
 await tool('reading_propose',{db:'reading_e2e',session:s.id,request_id:'mcp_proposal',expected_revision:s.revision,actor:'model:fixture',action:{action:'propose',id:'proposal1',draft}});
 await page.locator('[data-tab="proposals"]').click();await page.getByRole('button',{name:'Review exact revision'}).waitFor({timeout:10000});await page.getByRole('button',{name:'Review exact revision'}).click();await page.locator('#draft-note').fill('The human revised the proposed note.');await page.locator('#save-draft').click();await revision(6);await page.locator('#accept').click();await revision(7);s=await state();assert.equal(s.proposals.proposal1.status,'accepted');assert.equal(s.proposals.proposal1.origin,'model:fixture');assert.equal(s.proposals.proposal1.editor,'human');assert.equal(s.proposals.proposal1.draft.note,'The human revised the proposed note.');assert.equal(s.proposals.proposal1.created_ids.length,2);
 await page.locator('[data-tab="highlights"]').click();await page.getByRole('button',{name:'Reveal',exact:true}).click();await page.waitForTimeout(200);assert((await page.evaluate(()=>window.getSelection().toString())).includes('human selects'));
 await tool('reading_propose',{db:'reading_e2e',session:s.id,request_id:'mcp_pins',expected_revision:s.revision,actor:'model:fixture',action:{action:'pins',name:'review',pins:[{reference:h.id,note:'Return to the selected evidence',seen:false}]}});
 await page.getByRole('button',{name:/Return to the selected evidence/}).waitFor({timeout:10000});await page.getByRole('button',{name:/Return to the selected evidence/}).click();await revision(9);
 let finalRevision=11;
 if(extension==='pdf'){
  // Compare the source page without converting or moving the saved passage anchor.
  await page.locator('[data-block]').filter({hasText:'The human selects a source passage.'}).getByRole('button',{name:'Check page 1'}).click();
  await page.locator('.textLayer span').first().waitFor();
  const sizing=await page.locator('.textLayer span').filter({hasText:'The human selects a source passage.'}).evaluate(el=>({font:parseFloat(getComputedStyle(el).fontSize),scale:parseFloat(el.closest('.pdf-page').style.getPropertyValue('--scale-factor'))}));
  assert(Math.abs(sizing.font-12*sizing.scale)<.1,'Original PDF text remains aligned');
  await page.locator('#view-converted').click();await page.locator('[data-anchor-text]').first().waitFor();
  assert(await page.evaluate(()=>CSS.highlights.get('hades-node')?.size>0));
  await page.locator('#view-pdf').click();await page.locator('.pdf-page').waitFor();
  await page.locator('#figure').click();await page.locator('#figure-caption').fill('Synthetic selected area');
  const box=await page.locator('.pdf-page').boundingBox();await page.mouse.move(box.x+40,box.y+40);await page.mouse.down();await page.mouse.move(box.x+220,box.y+120);await page.mouse.up();await page.locator('[data-label="question"]').click();await revision(11);finalRevision=13;
  const figures=Object.values((await state()).highlights).filter(h=>h.anchor.figure_box);assert.equal(figures.length,1);assert.equal(figures[0].anchor.exact,'Synthetic selected area');
  await page.locator('#view-converted').click();await page.locator('[data-anchor-text]').first().waitFor();
  assert.equal(await page.locator('#reader').evaluate(el=>el.classList.contains('figure-mode')),false);
  await page.locator('#highlights .card').filter({hasText:'Synthetic selected area'}).getByRole('button',{name:'Reveal',exact:true}).click();
  await page.locator('.figure-mark').waitFor();
  assert.equal(await page.locator('#surface-title').textContent(),'Original PDF reference');
 }
 await page.locator('#advance').click();await revision(finalRevision-1);await page.locator('#advance').click();await revision(finalRevision);
 const record=await page.evaluate(async()=>{const id=new URLSearchParams(location.search).get('session');return await(await fetch('/api/reading/state?session='+id+'&view=export')).json()});assert.equal(record.chain_valid,true);assert.equal(record.events.length,finalRevision);assert.equal(record.session.stage,'outline');assert(record.outline.includes('Selected passage'));assert.equal(record.session.pins.review[0].seen,true);
 const original=await(await fetch(url+'/api/reading/original?source_hash='+s.source_hash)).arrayBuffer();assert.equal(crypto.createHash('sha256').update(Buffer.from(original)).digest('hex'),s.source_hash);
 await page.reload();await page.waitForFunction(()=>document.querySelector('#stages .current')?.textContent==='outline');await page.locator('[data-anchor-text]').first().waitFor();assert.equal(await page.locator('.pdf-page:visible').count(),0);assert(await page.evaluate(()=>CSS.highlights.get('hades-node')?.size>0));await page.screenshot({path:path.join(root,'reading-'+extension+'.png'),fullPage:true});
 fs.writeFileSync(path.join(root,'reading-'+extension+'-record.json'),JSON.stringify(record,null,2));
 // Correct only the isolated fixture. Existing accepted evidence must retain its wording.
 const block=page.locator('[data-block]').filter({has:page.locator(`[data-anchor-text]`)}).filter({hasText:'The human selects a source passage.'}).first();
 const originalText=await block.locator('[data-anchor-text]').textContent();
 await block.getByRole('button',{name:'Edit passage',exact:true}).click();
 await page.getByRole('textbox',{name:'Correct passage text'}).fill('Discard this draft');
 await page.getByRole('button',{name:'Cancel edit',exact:true}).click();assert.equal((await state()).revision,finalRevision);
 await block.getByRole('button',{name:'Edit passage',exact:true}).click();
 const correctedText=originalText.replace('human','reader');
 await page.getByRole('textbox',{name:'Correct passage text'}).fill(correctedText);
 await page.getByRole('button',{name:'Save correction',exact:true}).click();await revision(++finalRevision);
 s=await state();assert.equal(s.corrections[h.anchor.block][0].text,correctedText);assert.equal(s.corrections[h.anchor.block][0].actor,'human');
 const corrected=page.locator('[data-block]').filter({hasText:correctedText}).first();
 await page.locator('[data-tab="highlights"]').click();
 const originalCard=page.locator('#highlights .card').filter({hasText:h.anchor.exact}).first();
 assert((await originalCard.textContent()).includes('Passage corrected'));
 await originalCard.getByRole('button',{name:'Reveal',exact:true}).click();
 assert.equal(await block.locator('[data-anchor-text]').textContent(),originalText);
 assert((await block.textContent()).includes('Historical text: revision 0'));
 await block.getByRole('button',{name:'Return to latest',exact:true}).click();
 assert.equal(await corrected.locator('[data-anchor-text]').textContent(),correctedText);
 await corrected.locator('[data-anchor-text]').scrollIntoViewIfNeeded();
 const correctionRect=await corrected.locator('[data-anchor-text]').evaluate(el=>{const r=document.createRange();r.selectNodeContents(el);return r.getBoundingClientRect().toJSON()});
 await page.mouse.move(correctionRect.x+.5,correctionRect.y+correctionRect.height/2);await page.mouse.down();await page.mouse.move(correctionRect.right-.5,correctionRect.y+correctionRect.height/2,{steps:15});await page.mouse.up();
 await page.locator('[data-label="question"]').click();finalRevision+=2;await revision(finalRevision);
 s=await state();const correctedHighlight=Object.values(s.highlights).find(v=>v.anchor.passage_revision===1);
 assert(correctedHighlight);assert.equal(correctedHighlight.anchor.correction_session,s.id);
 const correctedContext=await tool('reading_context',{db:'reading_e2e',session:s.id,view:'context'});
 assert.equal(correctedContext.references[0].highlight.anchor.passage_revision,1);assert.equal(correctedContext.references[0].passage_changed,false);
 await corrected.getByRole('button',{name:'History / changes',exact:true}).click();
 assert.equal(await page.locator('#passage-before').textContent(),originalText);assert.equal(await page.locator('#passage-after').textContent(),correctedText);
 assert((await page.locator('#passage-diff ins').textContent()).length>0);
 await page.screenshot({path:path.join(root,'reading-'+extension+'-corrections.png'),fullPage:true});
 await page.locator('#passage-history-list').getByRole('button',{name:/Revision 0/}).click();
 await page.getByRole('button',{name:'Restore this revision',exact:true}).click();await revision(++finalRevision);
 s=await state();assert.equal(s.corrections[h.anchor.block].length,2);assert.equal(s.corrections[h.anchor.block][1].restored_from,0);
 assert.equal(await block.locator('[data-anchor-text]').textContent(),originalText);
 await page.reload();await page.locator('[data-anchor-text]').first().waitFor();
 assert.equal(await block.locator('[data-anchor-text]').textContent(),originalText);
 const revisedRecord=await page.evaluate(async()=>{const id=new URLSearchParams(location.search).get('session');return await(await fetch('/api/reading/state?session='+id+'&view=export')).json()});
 assert.equal(revisedRecord.chain_valid,true);assert.equal(revisedRecord.session.corrections[h.anchor.block].length,2);
 assert.equal(revisedRecord.session.proposals.proposal1.draft.note,'The human revised the proposed note.');
 assert(revisedRecord.reference_warnings.some(w=>w.highlight===correctedHighlight.id));
 fs.writeFileSync(path.join(root,'reading-'+extension+'-correction-record.json'),JSON.stringify(revisedRecord,null,2));

}
 await page.setViewportSize({width:3840,height:2400});await page.locator('#wide').click();await page.screenshot({path:path.join(root,'reading-wide.png'),fullPage:true});
 await page.setViewportSize({width:600,height:900});await page.screenshot({path:path.join(root,'reading-compact.png'),fullPage:true});
 assert.deepEqual(errors,[]);fs.writeFileSync(path.join(root,'reading-e2e.json'),JSON.stringify({passed:true,formats:['docx','pdf'],checks:['real Docling extraction','original hash','persistent view controls','wide side-by-side comparison','converted PDF/DOCX default','stable passage anchors','original PDF comparison','original figure reference compatibility','real pointer selection','Bound highlight gate','saved highlight painting','human classification','MCP exact focus','MCP proposal','human edit and acceptance','persisted attribution','pin reveal and seen','stage progression','export','reload','correction save/cancel','historical evidence reveal','corrected MCP anchors','revision diff','append-only restore','wide/compact rendering']}));console.log('Guided reading PDF + DOCX browser/MCP flow PASS');
}finally{await browser.close()}}
main().catch(e=>{console.error(e);process.exit(1)});
