/* Run against a local hades-viewer. API fixtures are intercepted; no database writes.
 * PLAYWRIGHT_MODULE and CHROMIUM_PATH may point to an existing local installation. */
const assert=require('node:assert/strict');
const {chromium}=require(process.env.PLAYWRIGHT_MODULE || 'playwright');
(async()=>{
 const browser=await chromium.launch({executablePath:process.env.CHROMIUM_PATH || undefined,headless:true,args:['--no-sandbox','--use-angle=swiftshader','--enable-unsafe-swiftshader']});
 try {
 const httpCredentials=process.env.WORKSPACE_PASSWORD_FILE ? {username:process.env.WORKSPACE_USERNAME || 'viewer',password:require('node:fs').readFileSync(process.env.WORKSPACE_PASSWORD_FILE,'utf8').trim()} : undefined;
 const page=await browser.newPage({viewport:{width:1440,height:1000},httpCredentials}),errors=[];
 page.on('pageerror',e=>errors.push(e.message));
 const node=(id,type,label,content)=>({id,collection:'notes',label,degree:1,attrs:{type,content,source_version:'1',source_node_id:id,source_location:{body_start:1,start_byte:0,end_byte:40}}});
 const nodes=[node('notes/doc','document','Test document','An exact source passage. <img src=x onerror=alert(1)>'),node('notes/section','section','Source section','Section evidence.'),node('notes/concept','concept','State','A scoped definition.')];
 const edges=[{id:'edges/contains',source:'notes/doc',target:'notes/section',collection:'edges',attrs:{relation:'contains'}},{id:'edges/keyword',source:'notes/section',target:'notes/concept',collection:'edges',attrs:{relation:'uses_keyword'}}];
 const chunk=node('chunks/real','chunk','Stored search passage','Section evidence.');
 const overlap={id:'links/overlap',source:chunk.id,target:'notes/section',collection:'links',attrs:{relation:'overlaps_source'}};
 let failSearch=false,delaySearch=false,delayExpand=false,searchStarted,expandStarted,expandCalls=0;
 await page.route('**/api/**',async route=>{
  const u=new URL(route.request().url()),db=u.searchParams.get('db'); let value;
  switch(u.pathname){
   case '/api/databases': value={databases:['fixture','other','empty']};break;
   case '/api/graphs': value={graphs:db==='empty'?[]:['workspace']};break;
   case '/api/graph-data':value={meta:{database:db,named_graph:'workspace',node_count:3,node_total:3,edge_count:2,edge_total:2,partial:false,node_collections:['notes'],edge_collections:['edges'],attribute_index:{notes:['type'],edges:['relation']}},nodes:nodes.map(n=>({...n,label:db==='other'?'Other '+n.label:n.label})),edges};break;
   case '/api/node':value=chunk;break;
   case '/api/expand':expandCalls++;if(delayExpand){expandStarted?.();await new Promise(r=>setTimeout(r,400));}value={nodes:[nodes[1],chunk],edges:[overlap]};break;
   case '/api/search':
    if(delaySearch){searchStarted?.();await new Promise(r=>setTimeout(r,400));}
    if(failSearch){await route.fulfill({status:502,body:'unavailable'});return;}
    value={profile:'default',results:[{title:'Evidence',text:'Section evidence.',chunk_id:chunk.id,parent_id:'notes/doc'}]};break;
   default: value={};
  }
  await route.fulfill({json:value});
 });
 await page.goto(process.env.WORKSPACE_URL || 'http://127.0.0.1:8090/');
 await page.waitForFunction(()=>document.querySelector('#reader-title').textContent==='Test document');
 assert.equal(await page.locator('#library button').count(),2);
 assert.equal(await page.locator('#reader-body img').count(),0);
 assert.ok(await page.evaluate(()=>document.documentElement.scrollHeight<=innerHeight+1),'desktop stays within viewport');
 await page.evaluate(()=>{window.originalLibraryButton=document.querySelector('#library button');});
 await page.locator('#library button').filter({hasText:'Source section'}).click();
 assert.ok(await page.evaluate(()=>window.originalLibraryButton===document.querySelector('#library button')),'selection preserves library elements');
 await page.locator('#library button').filter({hasText:'Test document'}).click();
 await page.evaluate(()=>{const t=document.querySelector('#reader-body').firstChild;const r=document.createRange();r.setStart(t,0);r.setEnd(t,23);getSelection().removeAllRanges();getSelection().addRange(r);});
 await page.locator('#add-context').click();
 await page.evaluate(()=>{Object.defineProperty(navigator,'clipboard',{configurable:true,value:{writeText:async text=>{window.copied=text;}}});});
 await page.evaluate(()=>{document.execCommand=command=>{if(command!=='copy')return false;window.copied=[...document.querySelectorAll('textarea')].find(t=>t.style.opacity==='0').value;return true;};});
 await page.locator('#copy-context').click();
 let packet=JSON.parse(await page.evaluate(()=>window.copied));
 assert.equal(packet.references[0].excerpt,'An exact source passage');
 assert.equal(packet.references[0].id,'notes/doc');
 await page.locator('#reader-links button[aria-label="Inspect contains relationship"]').click();
 assert.ok(await page.evaluate(()=>window.originalLibraryButton===document.querySelector('#library button')),'edge selection preserves library elements');
 assert.equal(await page.locator('#library button[aria-current=true]').count(),0);
 await page.locator('#add-context').click();await page.locator('#copy-context').click();
 packet=JSON.parse(await page.evaluate(()=>window.copied));assert.equal(packet.references[1].kind,'edge');assert.equal(packet.references[1].to,'notes/section');
 await page.locator('#share-view').click();const edgeUrl=await page.evaluate(()=>window.copied);assert.ok(edgeUrl.includes('edge=edges%2Fcontains'));
 await page.locator('#search-query').fill('evidence');await page.locator('#search-submit').click();await page.waitForSelector('.search-hit');const beforeExpand=expandCalls;await page.locator('.search-hit').click();
 await page.waitForFunction(()=>document.querySelector('#reader-title').textContent==='Stored search passage');
 assert.equal(expandCalls-beforeExpand,1,'unloaded search hit expands once');
 await page.locator('#reader-links button').filter({hasText:'Source section'}).click();assert.equal(await page.locator('#reader-body').innerText(),'Section evidence.');
 await page.locator('#view-tabs button[data-view=reading]').click();assert.equal(await page.locator('#canvas').isVisible(),false);
 await page.locator('#view-tabs button[data-view=graph]').click();assert.equal(await page.locator('#reader').isVisible(),false);
 await page.locator('#view-tabs button[data-view=split]').click();
 failSearch=true;await page.locator('#search-submit').click();await page.waitForFunction(()=>document.querySelector('#search-status').textContent.includes('Search unavailable'));
 failSearch=false;delaySearch=true;const waiting=new Promise(r=>searchStarted=r);await page.locator('#search-submit').click();await waiting;
 await page.locator('#db-picker').selectOption('other');await page.waitForFunction(()=>document.querySelector('#reader-title').textContent==='Other Test document');await page.waitForTimeout(500);assert.equal(await page.locator('.search-hit').count(),0,'stale search cannot cross database scope');
 delaySearch=false;delayExpand=true;const expanding=new Promise(r=>expandStarted=r);
 await page.locator('#search-submit').click();await page.waitForSelector('.search-hit');await page.locator('.search-hit').click();await expanding;
 await page.locator('#db-picker').selectOption('fixture');await page.waitForFunction(()=>document.querySelector('#reader-title').textContent==='Test document');await page.waitForTimeout(500);assert.equal(await page.evaluate(()=>graph.hasNode('chunks/real')),false,'stale expansion cannot enter another database');
 await page.locator('#db-picker').selectOption('empty');await page.waitForFunction(()=>document.querySelector('#reader-title').textContent==='Workspace unavailable');assert.equal(await page.locator('#add-context').isDisabled(),true);
 await page.goto(edgeUrl);await page.waitForFunction(()=>document.querySelector('#reader-kind').textContent==='Relationship');
 await page.setViewportSize({width:3840,height:2400});
 await page.locator('#workspace-layout').selectOption('wide');
 await page.waitForTimeout(100);
 const panes=await page.evaluate(()=>['controls','reader','canvas','inspector'].map(id=>{const r=document.getElementById(id).getBoundingClientRect();return {left:r.left,right:r.right,width:r.width};}));
 for(let i=1;i<panes.length;i++) assert.ok(panes[i].left>=panes[i-1].right-1,'wide panes sit side by side');
 assert.ok(panes[3].width>=500,'wide context pane uses available monitor space');
 assert.ok(await page.locator('.brand-logo').evaluate(img=>img.complete && img.naturalWidth>0),'supplied logo loads');
 await page.reload();await page.waitForFunction(()=>document.querySelector('#reader-kind').textContent==='Relationship');
 assert.equal(await page.locator('#workspace-layout').inputValue(),'wide','layout preference survives reload');
 await page.locator('#workspace-layout').selectOption('compact');
 await page.waitForTimeout(100);
 assert.ok(await page.evaluate(()=>document.querySelector('#canvas').getBoundingClientRect().top>=document.querySelector('#reader').getBoundingClientRect().bottom-1),'compact stacks the graph below the reader');
 await page.setViewportSize({width:390,height:844});await page.waitForTimeout(100);assert.ok(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth+1),'mobile has no horizontal overflow');
 assert.deepEqual(errors,[]);console.log('Browser checks passed: outline, source, XSS, quotes, node/edge references, deep links, semantic navigation, failure, stale requests, empty database, responsive layout.');
 } finally {await browser.close();}
})().catch(e=>{console.error(e);process.exitCode=1;});
