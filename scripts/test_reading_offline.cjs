/* Real saved-document recovery while the runner's extractor is stopped. */
const fs=require('node:fs'),path=require('node:path'),assert=require('node:assert/strict');
const {chromium}=require(process.env.PLAYWRIGHT_MODULE||'playwright');
const [root,url]=process.argv.slice(2);
(async()=>{
 const browser=await chromium.launch({executablePath:process.env.CHROMIUM_PATH,headless:true,args:['--no-sandbox']});
 try{
  const errors=[];
  const list=await(await fetch(url+'/api/reading/state')).json();
  assert.equal(list.sessions.length,2);
  for(const session of list.sessions){
   const page=await browser.newPage({viewport:{width:1800,height:1200}});
   page.on('pageerror',e=>errors.push(e.message));
   const record=await(await fetch(url+'/api/reading/state?session='+session.id+'&view=export')).json();
   const pending=record.events.at(-1).request;
   // Simulate a committed response that the browser lost. Retrying is a no-op.
   await page.addInitScript(payload=>localStorage.setItem('hades.reading.pending',JSON.stringify(payload)),pending);
   await page.goto(url+'/reading?session='+session.id);
   await page.locator('[data-anchor-text]').first().waitFor();
   await page.waitForFunction(()=>document.querySelector('#formats').textContent.includes('Extractor unavailable'));
   assert.equal(await page.locator('#title').textContent(),session.title);
   assert(await page.locator('#upload').isDisabled());
   assert.equal(await page.locator('#sessions .session').count(),2);
   await page.getByRole('button',{name:'Reconcile interrupted action',exact:true}).click();
   await page.getByRole('button',{name:'Reconcile interrupted action',exact:true}).waitFor({state:'detached'});
   const download=page.waitForEvent('download');await page.locator('#export').click();
   const file=await download;const exported=JSON.parse(fs.readFileSync(await file.path(),'utf8'));
   assert.equal(exported.chain_valid,true);assert.equal(exported.session.revision,record.session.revision);
   assert.deepEqual(exported.events,record.events);
   await page.screenshot({path:path.join(root,'offline-'+session.title+'.png'),fullPage:true});
   await page.close();
  }
  assert.deepEqual(errors,[]);
  fs.writeFileSync(path.join(root,'reading-offline.json'),JSON.stringify({passed:true,checks:['real extractor stopped','saved PDF/DOCX load','uploads disabled','exact retry available','export valid and unchanged']}));
  console.log('Saved PDF/DOCX reading, reconciliation, and export with extractor offline PASS');
 }finally{await browser.close();}
})().catch(e=>{console.error(e);process.exit(1)});
