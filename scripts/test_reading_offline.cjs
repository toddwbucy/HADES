/* Real saved-document recovery while the runner's extractor is stopped. */
const fs=require('node:fs'),path=require('node:path'),assert=require('node:assert/strict');
const {chromium}=require(process.env.PLAYWRIGHT_MODULE||'playwright');
const [root,url]=process.argv.slice(2);
const pendingKey=db=>'hades.reading.pending.'+encodeURIComponent(db);
(async()=>{
 const browser=await chromium.launch({executablePath:process.env.CHROMIUM_PATH,headless:true,args:['--no-sandbox']});
 try{
  const errors=[];
  const list=await(await fetch(url+'/api/reading/state')).json();
  assert.equal(list.sessions.length,2);
  for(const session of list.sessions){
   const page=await browser.newPage({viewport:{width:1800,height:1200}});
   page.on('pageerror',e=>errors.push(e.message));
   let mutations=0;page.on('request',r=>{if(r.method()==='POST'&&r.url().endsWith('/api/reading/mutate'))mutations++;});
   const record=await(await fetch(url+'/api/reading/state?session='+session.id+'&view=export')).json();
   const payload=record.events.at(-1).request;
   const current={database:list.database,payload},other={database:'other_reading_database',payload};
   const currentKey=pendingKey(list.database),otherKey=pendingKey(other.database);
   async function ready(){await page.locator('[data-anchor-text]').first().waitFor();await page.waitForFunction(()=>document.querySelector('#formats').textContent.includes('Extractor unavailable'));}
   await page.goto(url+'/reading?session='+session.id);await ready();
   // A valid same-ID action from another database, and unscoped old receipts,
   // must never be offered for replay against the current database.
   for(const [key,value]of [['hades.reading.pending',payload],['hades.reading.pending',other],[otherKey,other]]){
    await page.evaluate(({key,value})=>{
     for(const k of Object.keys(localStorage))if(k.startsWith('hades.reading.pending'))localStorage.removeItem(k);
     localStorage.setItem(key,JSON.stringify(value));
    },{key,value});
    await page.reload();await ready();
    assert.equal(await page.getByRole('button',{name:'Reconcile interrupted action',exact:true}).count(),0);
    assert.deepEqual(await page.evaluate(key=>JSON.parse(localStorage.getItem(key)),key),value);
    assert.equal(mutations,0);
   }
   // Returning to the matching database restores retry, without deleting the
   // other database's pending action. The committed request is an exact no-op.
   await page.evaluate(({key,value})=>localStorage.setItem(key,JSON.stringify(value)),{key:currentKey,value:current});
   await page.reload();await ready();
   assert.equal(await page.locator('#title').textContent(),session.title);
   assert(await page.locator('#upload').isDisabled());assert.equal(await page.locator('#sessions .session').count(),2);
   await page.getByRole('button',{name:'Reconcile interrupted action',exact:true}).click();
   await page.getByRole('button',{name:'Reconcile interrupted action',exact:true}).waitFor({state:'detached'});
   assert.equal(mutations,1);
   assert.equal(await page.evaluate(key=>localStorage.getItem(key),currentKey),null);
   assert.deepEqual(await page.evaluate(key=>JSON.parse(localStorage.getItem(key)),otherKey),other);
   // A tab left open across a server reconfiguration is also rejected at HTTP.
   const rejected=await page.request.post(url+'/api/reading/mutate',{headers:{Origin:url,'X-Hades-Workspace':'1','X-Hades-Database':other.database},data:payload});
   assert.equal(rejected.status(),409);assert((await rejected.json()).error.includes('database changed'));
   const download=page.waitForEvent('download');await page.locator('#export').click();
   const file=await download;const exported=JSON.parse(fs.readFileSync(await file.path(),'utf8'));
   assert.equal(exported.chain_valid,true);assert.equal(exported.session.revision,record.session.revision);
   assert.deepEqual(exported.events,record.events);
   await page.screenshot({path:path.join(root,'offline-'+session.title+'.png'),fullPage:true});await page.close();
  }
  assert.deepEqual(errors,[]);
  fs.writeFileSync(path.join(root,'reading-offline.json'),JSON.stringify({passed:true,checks:['real extractor stopped','saved PDF/DOCX load','uploads disabled','database-bound retries','other database receipts retained','unscoped legacy receipts withheld','stale-tab writes rejected','exact retry available','export valid and unchanged']}));
  console.log('Saved PDF/DOCX reading and database-scoped recovery with extractor offline PASS');
 }finally{await browser.close();}
})().catch(e=>{console.error(e);process.exit(1)});
