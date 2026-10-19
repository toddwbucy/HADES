import { ReadingChat } from '/reading-chat.js';
import * as pdfjs from '/vendor/pdf.mjs';
pdfjs.GlobalWorkerOptions.workerSrc='/vendor/pdf.worker.mjs';
const $=id=>document.getElementById(id);
let state=null,source=null,pdf=null,page=1,structured=true,busy=false,selectedAnchor=null,reviewing=null,database='',generation=0,figureMode=false,returnBlock=null,sideBySide=false,viewScroll={},splitScroll={},editingPassage=null,passageViews={};
const boundaryDrafts=new Map();
// Loading can temporarily clear the binding without deselecting its URL target.
function syncReadingConversation(binding, {restoring=false}={}) {
 const link=$('chat-workspace-link');link.hidden=!binding;
 const previous=new URLSearchParams(location.search);
 const conversation=binding?._key||(restoring&&previous.get('session')===state?.id?previous.get('conversation'):null);
 const params=new URLSearchParams();
 if(conversation)params.set('conversation',conversation);
 if(state)params.set('session',state.id);
 if(binding){
  link.href='/?'+new URLSearchParams({db:database,graph:state.graph,conversation:binding._key});
 }
 history.replaceState(null,'', '/reading'+(params.size?'?'+params:''));
}
const readingChat = new ReadingChat({onBinding: syncReadingConversation, api: body => api('chat', body), getState: () => state, getDatabase: () => database,
 getPosition: () => {
  const pane=sideBySide?$('converted-pane'):document.querySelector('.document');
  const top=Math.max(0,pane.getBoundingClientRect().top);
  const block=[...$('reader').querySelectorAll('[data-block]')].find(el=>el.getBoundingClientRect().bottom>top)?.dataset.block;
  return {page,block:block||returnBlock};
 }, restorePosition: async position => {
  if (!position || !source) return; page=position.page||1;returnBlock=position.block||null;await renderSource();
  if(returnBlock)[...$('reader').querySelectorAll('[data-block]')].find(el=>el.dataset.block===returnBlock)?.scrollIntoView({block:'center'});
 }});

readingChat.preferredBinding=new URLSearchParams(location.search).get('conversation');

const pendingKey=db=>'hades.reading.pending.'+encodeURIComponent(db);
const requestKey=(db,p)=>pendingKey(db)+':'+encodeURIComponent(p.session)+':'+encodeURIComponent(p.request_id);
const actor=()=> $('actor').value.trim()||'human';
const uid=()=>crypto.randomUUID ? crypto.randomUUID().replaceAll('-','_') : `r_${Date.now()}_${Math.random().toString(36).slice(2)}`;
function status(text,error=false){$('status').textContent=text;$('status').classList.toggle('error',error)}
function element(tag,text,cls){const el=document.createElement(tag);if(text!==undefined)el.textContent=text;if(cls)el.className=cls;return el}
function button(text,fn){const b=element('button',text);b.onclick=()=>run(fn);return b}
async function api(path,body){const response=await fetch('/api/reading/'+path,{method:body?'POST':'GET',headers:body?{'Content-Type':'application/json','X-Hades-Workspace':'1','X-Hades-Database':database}:{},body:body?JSON.stringify(body):undefined});const value=await response.json();if(!response.ok)throw Error(value.error||`Request failed (${response.status})`);return value}
async function run(fn){if(busy)return;busy=true;try{await fn()}catch(e){status(e.message,true)}finally{busy=false}}
async function mutate(action){if(!state)throw Error('Choose a reading session first');const payload={session:state.id,request_id:uid(),expected_revision:state.revision,actor:actor(),action};
// Keep a failed request available for reconciliation; never blindly invent a new request ID after a lost response.
const key=requestKey(database,payload),raw=JSON.stringify({database,payload});
localStorage.setItem(key,raw);
try{const result=await api('mutate',payload);if(localStorage.getItem(key)===raw)localStorage.removeItem(key);await refresh();status(`Saved revision ${result.revision}${result.created_ids.length?' · accepted graph objects created':''}`);return result}catch(e){await refresh();throw e}finally{restorePendingAction()}}
async function sessions(){
 const [active,archived]=await Promise.all([api('state'),api('state?view=archived')]);
 for(const [data,target,removed] of [[active,'sessions',false],[archived,'archived-sessions',true]]){
  $(target).replaceChildren();
  for(const item of data.sessions){
   const card=element('section',undefined,'card');card.dataset.session=item.id;
   if(removed){card.append(element('strong',item.title),button('Restore to desk',()=>setArchived(item.id,false)));}
   else {const open=button(item.title,()=>load(item.id));open.className='session';open.append(element('div',`${item.stage} · revision ${item.revision}`,'muted'));card.append(open,button('Remove from desk',()=>setArchived(item.id,true)));}
   $(target).append(card);
  }
  if(data.partial)$(target).append(element('p','Showing the most recent 100 readings. Add a document to find its saved reading.','muted'));
 }
 await archives();restorePendingAction();
}
function clearReading(){
 generation++;state=null;source=null;pdf=null;selectedAnchor=null;reviewing=null;editingPassage=null;
 passageViews={};viewScroll={};splitScroll={};returnBlock=null;sideBySide=false;figureMode=false;figureStart=null;
 $('figure').classList.remove('active');$('figure-caption').hidden=true;$('reader').classList.remove('side-by-side','figure-mode');
 for(const id of ['view-converted','view-pdf','view-split'])$(id).disabled=true;
 for(const id of ['stages','warnings','highlights','proposals','knowledge','pins'])$(id).replaceChildren();
 $('reader').replaceChildren(element('div','Add a document or restore a saved reading.','empty'));
 $('title').textContent='Choose a document to begin';$('original').hidden=true;$('bound-panel').hidden=true;
 for(const id of ['save-bound','advance','prev','next','figure','copy-context','export','outline'])$(id).disabled=true;
 for(const b of document.querySelectorAll('[data-label]'))b.disabled=true;
 $('selection-help').textContent='Choose a document to begin.';$('page-label').textContent='';
 history.replaceState(null,'','/reading');void readingChat.selectReading();
}
function checkRemovalDrafts(id){
 if(boundaryDrafts.has(id)||(state?.id===id&&(editingPassage||$('chat-message').value.trim())))throw Error('Save or clear your unfinished edits and chat message before removing this reading.');
}
async function setArchived(id,archived){
 if(archived)checkRemovalDrafts(id);else checkReadingSwitch();
 const current=(await api(`state?session=${encodeURIComponent(id)}`)).state;
 if(!!current.archived!==archived){
  const payload={session:id,request_id:uid(),expected_revision:current.revision,actor:actor(),action:{action:'archive',archived}};
  const key=requestKey(database,payload),raw=JSON.stringify({database,payload});localStorage.setItem(key,raw);
  try {await api('mutate',payload);if(localStorage.getItem(key)===raw)localStorage.removeItem(key);}
  finally {restorePendingAction();}
 }
 if(archived&&state?.id===id)clearReading();
 await sessions();
 if(!archived)await load(id);
 status(archived?'Removed from desk. Restore it from Removed from desk; notes and history are saved.':'Restored the saved reading.');
}
function checkReadingSwitch(){if(editingPassage)throw Error('Save or cancel the passage edit before switching readings.');}
async function openSource(sourceHash,title,separate=false){
 checkReadingSwitch();
 const saved=await api(separate?'open':'open-or-resume',{id:'s_'+uid().slice(0,28),title,source_hash:sourceHash,actor:actor()});
 if(saved.archived){await setArchived(saved.id,false);return;}
 await sessions();await load(saved.id);
 status(separate?'Started a separate reading.':'Opened the saved reading. Adding the same document resumes this reading.');
}
async function load(id){checkReadingSwitch();generation++;for(const key of ['save-bound','advance','prev','next','figure','copy-context','export','outline'])$(key).disabled=false;$('bound-panel').hidden=false;const loaded=await api(`state?session=${encodeURIComponent(id)}`);state=loaded.state;state.connectivity=loaded.connectivity;source=await api(`state?source_hash=${state.source_hash}`);pdf=null;page=1;selectedAnchor=null;structured=true;sideBySide=false;returnBlock=null;viewScroll={};splitScroll={};editingPassage=null;passageViews={};$('reader').replaceChildren();$('title').textContent=state.title;$('original').hidden=false;$('original').href=`/api/reading/original?source_hash=${state.source_hash}`;syncReadingConversation(readingChat.session===id?readingChat.binding:null,{restoring:true});renderState();await renderSource();void readingChat.selectReading();status(state.archived?'This reading was removed from the desk. Restore it from Removed from desk.':state.stage==='bound'?'Enter IS and IS NOT, then choose Save boundaries and start reading to enable highlighting.':'Drag over source text, then choose a highlight meaning.');}
async function refresh(){if(!state)return;const id=state.id;const next=await api(`state?session=${encodeURIComponent(id)}`);if(state?.id!==id)return;state=next.state;state.connectivity=next.connectivity;renderState()}
function acceptedLink(proposal,id){
 const edge=id.startsWith(state.graph+'_edges/');
 const node=edge?(proposal.draft.from||proposal.created_ids.find(v=>v.startsWith(state.graph+'_nodes/'))):id;
 const params=new URLSearchParams({db:database,graph:state.graph,node,expand:'1'});
 if(edge)params.set('edge',id);
 return '/?'+params;
}
function renderState(){
$('stages').replaceChildren(...['bound','read','connect','outline'].map((s,i)=>{const b=button(s,()=>mutate({action:'revisit',stage:s}));b.classList.toggle('current',s===state.stage);b.disabled=i>=['bound','read','connect','outline'].indexOf(state.stage);return b}));
const boundaries=boundaryDrafts.get(state.id)||state.bound;
if(document.activeElement!==$('bound-is'))$('bound-is').value=boundaries.is;
if(document.activeElement!==$('bound-not'))$('bound-not').value=boundaries.is_not;
$('advance').textContent=state.stage==='bound'?'Save boundaries and start reading':'Complete stage and continue';
$('advance').disabled=state.stage==='outline';
for(const b of document.querySelectorAll('[data-label]'))b.disabled=state.stage==='bound';
$('selection-help').textContent=state.stage==='bound'?'Highlighting starts in Read. Enter IS and IS NOT above, then choose Save boundaries and start reading.':'Drag over the source text, then choose New node, IS, Explicit IS NOT, or Question above. Saved highlights appear on the right.';
$('warnings').replaceChildren(...(source.structured?.warnings||['No structured content available.']).map(w=>element('p',w)));
$('highlights').replaceChildren();for(const h of Object.values(state.highlights)){if(h.removed)continue;const c=element('section',undefined,'card');c.append(element('span',h.label.replaceAll('_',' ').toUpperCase(),'badge'),element('p',h.anchor.exact||(h.anchor.figure_box?'Figure selection (no caption supplied)':'')));if(changedReference(h))c.append(element('p',`Passage corrected: this reference uses revision ${h.anchor.passage_revision||0}. Reveal to review its original text.`,'passage-warning'));const controls=element('div',undefined,'controls');controls.append(button('Reveal',()=>reveal(h.id)),button('Share focus',()=>mutate({action:'focus',references:[h.id]})),button('Write a note',()=>newNote(h)),button('Remove',()=>mutate({action:'remove_highlight',id:h.id})));const select=element('select');select.setAttribute('aria-label','Highlight meaning');for(const l of ['node','is','is_not','question']){const o=element('option',l);o.value=l;o.selected=l===h.label;select.append(o)}select.onchange=()=>run(()=>mutate({action:'recolor',id:h.id,label:select.value}));controls.append(select);c.append(controls);$('highlights').append(c)}
$('proposals').replaceChildren();$('knowledge').replaceChildren();for(const p of Object.values(state.proposals).sort((a,b)=>state.outline_order.indexOf(a.id)-state.outline_order.indexOf(b.id))){const c=element('section',undefined,'card');c.append(element('strong',p.draft.title),element('p',`${p.status} · revision ${p.revision} · origin: ${p.origin}`,'muted'));if(changedReference(state.highlights[p.draft.highlight]))c.append(element('p','Evidence passage corrected. Review the referenced revision; this note has not been moved.','passage-warning'));if(p.status==='pending'){c.append(element('p',p.draft.note),button('Review exact revision',()=>review(p.id)));$('proposals').append(c)}else if(p.status==='accepted'){c.append(element('p',p.draft.note));if(state.connectivity?.some(v=>v.proposal===p.id&&!v.connected))c.append(element('p','Suspect: the note or its basis path is missing from the graph.','error'));if(state.stage==='outline'&&!p.draft.from){for(const [name,direction]of [['Move up',-1],['Move down',1]])c.append(button(name,async()=>{const proposals=[...state.outline_order],i=proposals.indexOf(p.id),j=i+direction;if(j<0||j>=proposals.length)return;[proposals[i],proposals[j]]=[proposals[j],proposals[i]];await mutate({action:'order_outline',proposals})}))}if(!p.draft.is||!p.draft.is_not)c.append(element('p','One or more node boundaries remain unresolved.','muted'));for(const id of p.created_ids){const a=element('a',id);a.href=acceptedLink(p,id);a.target='_blank';c.append(a,element('br'))}c.append(button('Reveal evidence',()=>reveal(p.draft.highlight)));$('knowledge').append(c)}else{$('proposals').append(c)}}
$('pins').replaceChildren();for(const [name,pins]of Object.entries(state.pins)){const section=element('section');section.append(element('h3',name));section.append(button('Next unread pin',async()=>{const i=pins.findIndex(p=>!p.seen);if(i<0)throw Error('Every pin in this set has been seen');await reveal(pins[i].reference);await mutate({action:'seen',name,index:i})}));pins.forEach((p,i)=>section.append(button(`${p.seen?'✓':'○'} ${i+1}. ${p.note}`,async()=>{await reveal(p.reference);await mutate({action:'seen',name,index:i})})));$('pins').append(section)}
syncConverted();paintHighlights();readingChat.updateContext();
}
const editableKinds=new Set(['text','title','section_header','caption','list_item','footnote','page_header','page_footer','reference']);
function passageRevision(id){return state?.corrections?.[id]?.at(-1)?.revision||0;}
function passageText(block,revision){if(revision===0)return block.text||'';const v=state.corrections?.[block.id]?.find(v=>v.revision===revision);if(!v)throw Error('Passage revision is unavailable');return v.text;}
function changedReference(h){return h?.anchor.representation==='docling-blocks-v1'&&passageRevision(h.anchor.block)!==(h.anchor.passage_revision||0);}
function syncConverted(){
 const root=$('converted-pane');if(!root||!source)return;
 // Do not erase a human's in-progress edit when another participant changes state.
 if(editingPassage&&root.querySelector('.passage-edit'))return;
 const key=JSON.stringify([state.corrections||{},passageViews]);if(root.dataset.revisions===key&&root.hasChildNodes())return;
 const scroll=root.scrollTop;root.replaceChildren();renderConverted(root);root.dataset.revisions=key;root.scrollTop=scroll;
}
function editPassage(block){
 if(editingPassage)throw Error('Save or cancel the current passage edit first.');
 delete passageViews[block.id];syncConverted();
 const revision=passageRevision(block.id);editingPassage={block,revision,text:passageText(block,revision)};
 const section=[...$('converted-pane').querySelectorAll('[data-block]')].find(el=>el.dataset.block===block.id);
 const editor=element('div',undefined,'passage-edit');editor.append(element('p',`Correcting revision ${revision} in this reading session. Use this for transcription errors, not interpretive notes.`));
 const input=element('textarea');input.value=editingPassage.text;input.rows=8;input.setAttribute('aria-label','Correct passage text');editor.append(input);
 editor.append(button('Save correction',async()=>{
  const pending=editingPassage;
  await mutate({action:'correct_passage',block:block.id,passage_revision:pending.revision,text:input.value,restore_revision:null});
  editingPassage=null;delete passageViews[block.id];$('converted-pane').dataset.revisions='';syncConverted();paintHighlights();status('Correction saved as a new passage revision. Existing references keep their earlier text.');
 }),button('Cancel edit',()=>{editingPassage=null;editor.remove();syncConverted();paintHighlights();}));
 section.append(editor);input.focus();
}
function showPassageHistory(block){
 if(editingPassage)throw Error('Save or cancel the passage edit before opening history.');
 const current=passageRevision(block.id),versions=[{revision:0,text:block.text||'',actor:'Original Docling extraction',at:source.archived_at||'',restored_from:null},...(state.corrections?.[block.id]||[])];
 $('passage-history-title').textContent='Passage history';$('passage-history-note').textContent='Corrections belong to this reading session. Restoring adds a revision; it does not erase history.';
 const list=$('passage-history-list');list.replaceChildren();
 const show=revision=>{
  const v=versions.find(v=>v.revision===revision),previous=versions.find(p=>p.revision===revision-1)||versions[0];
  $('passage-change-title').textContent=`Revision ${revision} · ${v.actor}${v.at?' · '+v.at:''}${v.restored_from!==null?' · restored from revision '+v.restored_from:''}`;
  $('passage-before').textContent=previous.text;$('passage-after').textContent=v.text;
  const a=Array.from(previous.text),b=Array.from(v.text);let start=0,end=0;
  while(start<a.length&&start<b.length&&a[start]===b[start])start++;
  while(end<a.length-start&&end<b.length-start&&a[a.length-1-end]===b[b.length-1-end])end++;
  const diff=$('passage-diff');diff.replaceChildren(element('span',b.slice(0,start).join('')));
  if(a.length-start-end)diff.append(element('del',a.slice(start,a.length-end).join('')));
  if(b.length-start-end)diff.append(element('ins',b.slice(start,b.length-end).join('')));
  diff.append(element('span',end?b.slice(-end).join(''):''));
  const actions=$('passage-history-actions');actions.replaceChildren();
  actions.append(button('View this revision',async()=>{passageViews[block.id]=revision;await setView(sideBySide?'split':'converted');$('passage-history').close();[...$('reader').querySelectorAll('[data-block]')].find(el=>el.dataset.block===block.id)?.scrollIntoView({block:'center'});}));
  if(revision!==current&&v.text!==passageText(block,current))actions.append(button('Restore this revision',async()=>{
   await mutate({action:'correct_passage',block:block.id,passage_revision:current,text:v.text,restore_revision:revision});
   delete passageViews[block.id];syncConverted();paintHighlights();$('passage-history').close();status(`Restored revision ${revision} as a new correction.`);
  }));
 };
 for(const v of [...versions].reverse())list.append(button(`Revision ${v.revision}${v.revision===current?' (current)':''} · ${v.actor}`,()=>show(v.revision)));
 show(current);$('passage-history').showModal();
}

function currentView(){return sideBySide?'split':structured?'converted':'pdf';}
function updateViewControls(){
 const isPdf=!!source?.filename.toLowerCase().endsWith('.pdf');
 for(const mode of ['converted','pdf','split']){
  const control=$('view-'+mode);control.disabled=!state||(mode!=='converted'&&!isPdf);
  control.classList.toggle('active',!!state&&mode===currentView());control.setAttribute('aria-pressed',String(!!state&&mode===currentView()));
 }
 $('wide').setAttribute('aria-pressed',String(document.body.classList.contains('wide')));
}
async function setView(mode){
 if(!source)return;
 if(mode!=='converted'&&!source.filename.toLowerCase().endsWith('.pdf'))return;
 const before=currentView();viewScroll[before]=document.querySelector('.document').scrollTop;
 if(sideBySide)splitScroll={converted:$('converted-pane').scrollTop,pdf:$('pdf-pane').scrollTop,page:Number($('pdf-pane').dataset.page)};
 sideBySide=mode==='split';structured=mode!=='pdf';
 if(sideBySide){document.body.classList.add('wide');localStorage.setItem('hades.reading.wide','true');}
 await renderSource();
 if(sideBySide){$('converted-pane').scrollTop=splitScroll.converted||0;if(splitScroll.page===page)$('pdf-pane').scrollTop=splitScroll.pdf||0;}
 if(before!==mode){
  if(viewScroll[mode]!==undefined)document.querySelector('.document').scrollTop=viewScroll[mode];
  else if(mode==='converted'&&returnBlock)[...$('reader').querySelectorAll('[data-block]')].find(b=>b.dataset.block===returnBlock)?.scrollIntoView({block:'center'});
  else $('reader').scrollIntoView({block:'start'});
 }
}
async function renderSource(){
 selectedAnchor=null;figureMode=false;figureStart=null;
 $('figure').classList.remove('active');$('reader').classList.remove('figure-mode');$('figure-caption').hidden=true;
 const gen=generation,reader=$('reader');
 let converted=$('converted-pane'),original=$('pdf-pane');
 if(!converted){converted=element('div',undefined,'converted reading-pane');converted.id='converted-pane';converted.setAttribute('aria-label','Converted document');reader.append(converted);}
 if(!original){original=element('div',undefined,'reading-pane');original.id='pdf-pane';original.setAttribute('aria-label','Original PDF');reader.append(original);}
 const showConverted=structured||sideBySide,showPdf=!structured||sideBySide;
 converted.hidden=!showConverted;original.hidden=!showPdf;
 reader.classList.toggle('side-by-side',sideBySide);reader.classList.toggle('converted',structured&&!sideBySide);
 $('prev').hidden=$('next').hidden=$('figure').hidden=!showPdf;
 if(!showPdf)$('page-label').textContent='';
 $('surface-title').textContent=sideBySide?'Converted document + original PDF':structured?'Converted document':'Original PDF reference';
 $('surface-description').textContent=sideBySide?'Read and highlight on the left; compare the original on the right. Each pane scrolls independently. Check page opens the matching source page.':structured?'This saved conversion is the shared text for reading, highlighting, and discussion. Use the view buttons above to compare the original.':'Use Converted above to return to the reading copy, or Side by side to compare both.';
 updateViewControls();
 syncConverted();
 if(showPdf)await renderPdf(original,gen);
 paintHighlights();
}
async function renderPdf(root,gen){
  if(!pdf)pdf=await pdfjs.getDocument({url:`/api/reading/original?source_hash=${state.source_hash}`,isEvalSupported:false}).promise;
  if(gen!==generation)return;
  $('page-label').textContent=`Page ${page} of ${pdf.numPages}`;
  const key=`${state.source_hash}/${page}/${root.clientWidth}/${sideBySide}`;if(root.dataset.renderKey===key)return;
  const previousPage=Number(root.dataset.page),scroll=root.scrollTop;
  const p=await pdf.getPage(page),scale=Math.min(2.2,Math.max(.6,(root.clientWidth-32)/p.getViewport({scale:1}).width)),viewport=p.getViewport({scale});
  root.replaceChildren();
  if(sideBySide){
   const controls=element('nav',undefined,'pdf-controls');controls.setAttribute('aria-label','Original PDF pages');
   const previous=button('Previous page',async()=>{if(page>1){page--;await renderSource();}}),next=button('Next page',async()=>{if(page<pdf.numPages){page++;await renderSource();}});
   previous.disabled=page===1;next.disabled=page===pdf.numPages;controls.append(previous,element('span',`Original PDF · Page ${page} of ${pdf.numPages}`),next);root.append(controls);
  }
  const container=element('div',undefined,'pdf-page');container.style.setProperty('--scale-factor',scale);container.style.setProperty('--user-unit',viewport.userUnit);container.style.width=viewport.width+'px';container.style.height=viewport.height+'px';container.dataset.page=page;
  const canvas=element('canvas');canvas.width=viewport.width;canvas.height=viewport.height;container.append(canvas);
  const layer=element('div',undefined,'textLayer');container.append(layer);root.append(container);
  await p.render({canvasContext:canvas.getContext('2d'),viewport}).promise;
  await new pdfjs.TextLayer({textContentSource:await p.getTextContent(),container:layer,viewport}).render();
  root.dataset.page=page;root.dataset.renderKey=key;root.scrollTop=previousPage===page?scroll:0;
}
function renderConverted(root){
  const blocks=source.structured?.blocks;
  if(!Array.isArray(blocks)||!blocks.length){root.append(element('p','No converted passages are available. Open the original to inspect the source.','empty'));return;}
  for(const [i,b]of blocks.entries()){
   const revision=passageViews[b.id]??passageRevision(b.id),body=passageText(b,revision);
   const block=element('section',undefined,`block ${b.kind}`);block.dataset.block=b.id;block.id=`converted-${i+1}`;
   const meta=element('div',undefined,'passage-meta');meta.append(element('span',`Passage ${i+1} · ${b.kind.replaceAll('_',' ')}`));
   const pages=[...new Set((b.locations||[]).map(l=>l.page).filter(p=>Number.isInteger(p)&&p>0))];
   for(const sourcePage of pages){
    if(source.filename.toLowerCase().endsWith('.pdf'))meta.append(button(`Check page ${sourcePage}`,async()=>{returnBlock=b.id;page=sourcePage;await setView(sideBySide?'split':'pdf');}));
    else meta.append(element('span',`Page ${sourcePage}`));
   }
   meta.append(element('span',`Revision ${revision}${revision===0?' · extracted':' · corrected'}`));
   if(editableKinds.has(b.kind)){
    meta.append(button('Edit passage',()=>editPassage(b)),button('History / changes',()=>showPassageHistory(b)));
   }
   block.append(meta);
   if(revision!==passageRevision(b.id)){
    const warning=element('div',undefined,'passage-warning');warning.append(element('span',`Historical text: revision ${revision}. Current revision is ${passageRevision(b.id)}. `),button('Return to latest',()=>{delete passageViews[b.id];syncConverted();paintHighlights();}));block.append(warning);
   }

   if(b.kind==='table'&&b.rows){
    const table=element('table');for(const row of b.rows){const tr=element('tr');for(const cell of row)tr.append(element('td',cell));table.append(tr);}block.append(table);
    const details=element('details');details.append(element('summary','Select text from the converted table'));const text=element('pre',body);text.dataset.anchorText='1';text.dataset.passageRevision=revision;details.append(text);block.append(details);
   }else{
    if(b.kind==='picture')block.append(element('p','Figure pixels are not included in this conversion. Check the original; the available caption follows.','conversion-gap'));
    if(body){const text=element(b.kind==='section_header'?'h3':b.kind==='title'?'h2':'div',body);text.dataset.anchorText='1';text.dataset.passageRevision=revision;block.append(text);}
    else block.append(element('p',revision>0?'This passage is empty at this revision.':b.kind==='picture'?'No caption was extracted.':'No readable text was extracted for this passage.','conversion-gap'));
   }
   root.append(block);
  }
}
function capture(){
 const selection=window.getSelection();selectedAnchor=null;
 if(!state||!selection?.rangeCount||selection.isCollapsed)return;
 const selected=selection.getRangeAt(0),matches=[];
 // A native drag can end on the paragraph's parent boundary. Clip to actual
 // passage text, excluding labels/controls; never silently span two passages.
 for(const root of $('reader').querySelectorAll('[data-anchor-text],.textLayer')){
  if(root.closest('.reading-pane').hidden||(figureMode&&root.classList.contains('textLayer')))continue;
  if(!selected.intersectsNode(root))continue;
  const content=document.createRange();content.selectNodeContents(root);
  const range=selected.cloneRange();
  if(range.compareBoundaryPoints(Range.START_TO_START,content)<0)range.setStart(content.startContainer,content.startOffset);
  if(range.compareBoundaryPoints(Range.END_TO_END,content)>0)range.setEnd(content.endContainer,content.endOffset);
  if(range.toString().trim())matches.push({root,range});
 }
 if(matches.length!==1)return;
 const {root,range}=matches[0],before=range.cloneRange();before.selectNodeContents(root);before.setEnd(range.startContainer,range.startOffset);
 const exact=range.toString(),start=before.toString().length,all=root.textContent,isConverted=root.hasAttribute('data-anchor-text');
 const passageRevision=isConverted?Number(root.dataset.passageRevision||0):0;
 selectedAnchor={source_hash:state.source_hash,...(passageRevision?{passage_revision:passageRevision,correction_session:state.id}:{}),representation:isConverted?'docling-blocks-v1':'pdfjs-text-v1',page:isConverted?null:page,block:isConverted?root.closest('[data-block]').dataset.block:null,exact,prefix:all.slice(Math.max(0,start-100),start),suffix:all.slice(start+exact.length,start+exact.length+100)};
}
$('reader').addEventListener('mouseup',capture);$('reader').addEventListener('keyup',capture);
for(const b of document.querySelectorAll('[data-label]')){b.onmousedown=e=>e.preventDefault();b.onclick=()=>run(async()=>{if(!selectedAnchor)throw Error('Select text within one paragraph or PDF page first. For tables, open the extracted text selector.');const anchor=selectedAnchor;const id=uid();await mutate({action:'highlight',id,anchor,label:b.dataset.label});await mutate({action:'focus',references:[id]});selectedAnchor=null;window.getSelection().removeAllRanges()})}
async function reveal(id){let h=state.highlights[id];if(!h){const p=state.proposals[id]||Object.values(state.proposals).find(p=>p.created_ids.includes(id));h=p&&state.highlights[p.draft.highlight]}if(!h)throw Error('Reference is unavailable');if(h.anchor.representation==='docling-blocks-v1')passageViews[h.anchor.block]=h.anchor.passage_revision||0;if(!sideBySide)structured=h.anchor.representation==='docling-blocks-v1';if(h.anchor.page)page=h.anchor.page;await renderSource();if(h.anchor.figure_box){$('reader').querySelector('.figure-mark')?.scrollIntoView({block:'center'});status('Revealed source page and figure box.');return;}const root=anchorRoot(h.anchor);if(!root)throw Error('Anchor representation unavailable');const all=root.textContent,needle=h.anchor.prefix+h.anchor.exact+h.anchor.suffix;const start=all.indexOf(needle);if(start<0||all.indexOf(needle,start+1)>=0)throw Error('Anchor is missing or ambiguous. It has not been moved.');const offset=start+h.anchor.prefix.length;const walker=document.createTreeWalker(root,NodeFilter.SHOW_TEXT);let n,pos=0,a,b;while(n=walker.nextNode()){const end=pos+n.textContent.length;if(!a&&offset>=pos&&offset<end)a=[n,offset-pos];if(offset+h.anchor.exact.length>pos&&offset+h.anchor.exact.length<=end){b=[n,offset+h.anchor.exact.length-pos];break}pos=end}if(a&&b){const r=document.createRange();r.setStart(...a);r.setEnd(...b);const sel=window.getSelection();sel.removeAllRanges();sel.addRange(r);root.closest('details')?.setAttribute('open','');root.scrollIntoView({block:'center'});status(`Revealed exact reference ${h.id}`)}else throw Error('Anchor text cannot be resolved in this rendering')}
async function newNote(h){const draft={title:'New reading note',note:h.anchor.exact||'Draft a note from the selected figure.',is:null,is_not:null,standing:'unreviewed',highlight:h.id,basis:'Explain how this note relates to the reading boundaries.',from:null,to:null,relation:null,inferred:false,rationale:null};const id=uid();await mutate({action:'propose',id,draft});review(id)}
function review(id){reviewing=structuredClone(state.proposals[id]);const d=reviewing.draft;$('proposal-origin').textContent=`Origin: ${reviewing.origin}. Last editor: ${reviewing.editor}. Reviewing revision ${reviewing.revision}.`;$('proposal-evidence').textContent=state.highlights[d.highlight]?.anchor.exact||'Evidence unavailable';for(const [field,value]of Object.entries({title:d.title,note:d.note,is:d.is||'',not:d.is_not||'',standing:d.standing,basis:d.basis}))$('draft-'+field).value=value;$('relation-details').textContent=d.from?`Connection: ${d.from} → ${d.relation} → ${d.to}. ${d.inferred?'Inferred: '+d.rationale:'Source-stated.'}`:d.inferred?'Inferred: '+d.rationale:'';$('review-warning').textContent='Save edits before accepting. Acceptance retains the original draft author.'+(changedReference(state.highlights[d.highlight])?' The source passage has been corrected. Acceptance will keep the earlier evidence revision.':'');$('editor').showModal()}
function edited(){return {...reviewing.draft,title:$('draft-title').value,note:$('draft-note').value,is:$('draft-is').value.trim()||null,is_not:$('draft-not').value.trim()||null,standing:$('draft-standing').value,basis:$('draft-basis').value}}
$('save-draft').onclick=()=>run(async()=>{await mutate({action:'edit',id:reviewing.id,proposal_revision:reviewing.revision,draft:edited()});$('editor').close();review(reviewing.id)});
$('accept').onclick=()=>run(async()=>{if(JSON.stringify(edited())!==JSON.stringify(reviewing.draft))throw Error('Save your edits, then review the new revision before accepting.');await mutate({action:'accept',id:reviewing.id,proposal_revision:reviewing.revision});$('editor').close();tab('knowledge')});
$('reject').onclick=()=>run(async()=>{await mutate({action:'reject',id:reviewing.id,proposal_revision:reviewing.revision});$('editor').close()});
function tab(name){for(const b of document.querySelectorAll('[data-tab]'))b.classList.toggle('active',b.dataset.tab===name);for(const n of ['highlights','proposals','knowledge','chat'])$(n).hidden=n!==name}
for(const b of document.querySelectorAll('[data-tab]'))b.onclick=()=>tab(b.dataset.tab);
for(const field of ['bound-is','bound-not'])$(field).oninput=()=>{if(state)boundaryDrafts.set(state.id,{is:$('bound-is').value,is_not:$('bound-not').value});};
async function saveBoundaries(){
 if(!state)throw Error('Choose a reading first.');
 const id=state.id,draft={is:$('bound-is').value,is_not:$('bound-not').value};
 boundaryDrafts.set(id,draft);
 if(!draft.is.trim()||!draft.is_not.trim())throw Error('Fill in both IS and IS NOT before starting to read. Your draft has been kept.');
 if(!state.bound.revision||draft.is!==state.bound.is||draft.is_not!==state.bound.is_not)await mutate({action:'bound',...draft});
 const current=boundaryDrafts.get(id);
 if(current?.is===draft.is&&current?.is_not===draft.is_not)boundaryDrafts.delete(id);
 else throw Error('The previous wording was saved. Save your latest boundary edits before continuing.');
 renderState();
}
$('save-bound').onclick=()=>run(async()=>{await saveBoundaries();status('Boundaries saved. Choose Save boundaries and start reading to enable highlighting.');});
$('advance').onclick=()=>run(async()=>{if(state?.stage==='bound')await saveBoundaries();await mutate({action:'advance'});if(state.stage==='read')status('Ready to read. Select text in a passage, then choose a highlight color.');});
$('prev').onclick=()=>run(async()=>{if(page>1){page--;await renderSource()}});$('next').onclick=()=>run(async()=>{if(pdf&&page<pdf.numPages){page++;await renderSource()}});
for(const mode of ['converted','pdf','split'])$('view-'+mode).onclick=()=>run(()=>setView(mode));
$('wide').onclick=()=>run(async()=>{
 document.body.classList.toggle('wide');localStorage.setItem('hades.reading.wide',document.body.classList.contains('wide'));
 if(sideBySide&&!document.body.classList.contains('wide'))await setView('converted');
 else if(source)await renderSource();
 updateViewControls();
});
document.body.classList.toggle('wide',localStorage.getItem('hades.reading.wide')==='true');updateViewControls();
$('actor').value=localStorage.getItem('hades.reading.actor')||'human';$('actor').onchange=()=>localStorage.setItem('hades.reading.actor',actor());
$('upload').onchange=()=>run(async()=>{const file=$('upload').files[0];if(!file)return;if(file.size>8*1024*1024)throw Error('This reading slice accepts originals up to 8 MiB. The file was not uploaded.');status('Archiving original bytes…');const bytes=new Uint8Array(await file.arrayBuffer());let binary='';for(let i=0;i<bytes.length;i+=16384)binary+=String.fromCharCode(...bytes.subarray(i,i+16384));const archive=await api('upload',{filename:file.name,content_base64:btoa(binary)});localStorage.setItem('hades.reading.archived',archive.source_hash);status('Original archived. Docling is preparing the reading view…');const extracted=await api('extract',{source_hash:archive.source_hash});await archives();if(extracted.status!=='ready')throw Error(`Original retained. Extraction failed: ${extracted.error}`);await openSource(archive.source_hash,file.name);$('upload').value='';});
function download(name,body,type){const url=URL.createObjectURL(new Blob([body],{type}));const a=element('a');a.href=url;a.download=name;a.click();setTimeout(()=>URL.revokeObjectURL(url),10000)}
$('export').onclick=()=>run(async()=>{if(!state)throw Error('Choose a session');download(state.id+'-authorship.json',JSON.stringify(await api(`state?session=${state.id}&view=export`),null,2),'application/json')});
$('outline').onclick=()=>run(async()=>{if(!state||state.stage!=='outline')throw Error('Complete Connect to export the outline.');const v=await api(`state?session=${state.id}`);download(state.id+'-outline.md',v.outline,'text/markdown')});
$('copy-context').onclick=()=>run(async()=>{if(!state)throw Error('Choose a session');const v=await api(`state?session=${state.id}&view=context`);$('context-text').value=JSON.stringify(v,null,2);$('context-dialog').showModal();$('context-text').select()});$('close-context').onclick=()=>$('context-dialog').close();
async function discoverCapabilities(){
 $('upload').disabled=true;
 try{
  const caps=await api('capabilities');$('upload').accept=caps.extensions.join(',');
  $('formats').textContent=`${caps.extensions.length} advertised formats · 8 MiB per source. Some formats require optional converter dependencies.`;
  $('upload').disabled=false;
 }catch(e){$('formats').textContent='Extractor unavailable. Saved readings, review, and exports remain available. Refresh to reconnect for new uploads.';}
}
function restorePendingAction(){
 if(!database)return;
 let panel=$('pending-actions');if(!panel){panel=element('div');panel.id='pending-actions';$('sessions').prepend(panel);}panel.replaceChildren();
 const legacyKeys=new Set(['hades.reading.pending',pendingKey(database)]),prefix=pendingKey(database)+':';
 for(const originalKey of Object.keys(localStorage).filter(k=>legacyKeys.has(k)||k.startsWith(prefix))){
  const raw=localStorage.getItem(originalKey);if(raw===null)continue;
  let saved;try{saved=JSON.parse(raw);}catch(e){panel.append(element('p','An unreadable interrupted action was kept for manual recovery.','error'));continue;}
  if(saved?.database!==database||typeof saved.payload?.session!=='string'||typeof saved.payload?.request_id!=='string'){
   panel.append(element('p','An interrupted action belongs to another database or has no database identity. It was kept for recovery and will not be replayed here.','muted'));continue;
  }
  const p=saved.payload,key=requestKey(database,p);
  if(originalKey!==key){
   const existing=localStorage.getItem(key);
   if(existing!==null&&existing!==raw){panel.append(element('p','Conflicting retry records were kept for manual recovery.','error'));continue;}
   // Migrate known database-scoped receipts without replacing another request.
   localStorage.setItem(key,raw);if(localStorage.getItem(originalKey)===raw)localStorage.removeItem(originalKey);
  }
  if(panel.querySelectorAll('[data-request]').length&&[...panel.querySelectorAll('[data-request]')].some(el=>el.dataset.request===key))continue;
  const card=element('section',undefined,'card');card.dataset.request=key;
  card.append(element('p',`Interrupted action in ${p.session}: ${p.action?.action||'save'} (${p.request_id})`,'muted'));
  card.append(button('Reconcile interrupted action',async()=>{
   if(localStorage.getItem(key)!==raw)throw Error('The pending action changed. Reload before reconciling it.');
   if(p.action?.action==='archive'&&p.action.archived)checkRemovalDrafts(p.session);else checkReadingSwitch();
   await api('mutate',p);if(localStorage.getItem(key)===raw)localStorage.removeItem(key);
   restorePendingAction();if(p.action?.action==='archive'&&p.action.archived){if(state?.id===p.session)clearReading();await sessions();status('Removed from desk. Notes and history are saved.');}else{await sessions();await load(p.session);}
  }));panel.append(card);
 }
}
window.addEventListener('storage',e=>{if(e.key===null||e.key.startsWith('hades.reading.pending'))restorePendingAction()});

async function start(){const config=await api('config');database=config.database;$('database').textContent=database||'Reading is not configured';if(!config.enabled)throw Error('Start the viewer with a reading daemon to enable this workflow.');await sessions();const session=new URLSearchParams(location.search).get('session');if(session)await load(session);else {clearReading();status('Add a document or resume a reading.');}restorePendingAction();void discoverCapabilities();}
run(start);
async function pollReading(){if(!state||busy||document.hidden)return;try{const id=state.id;const next=await api(`state?session=${encodeURIComponent(id)}`);if(!busy&&state?.id===id&&next.state.revision>state.revision){const archiveChanged=!!state.archived!==!!next.state.archived;state=next.state;state.connectivity=next.connectivity;renderState();if(archiveChanged)await sessions();if(state?.id!==id)return;status(state.archived?'This reading was removed from the desk. Restore it from Removed from desk.':'Shared reading updated. New proposals and pins are ready.')}}catch(e){if(!busy)status(`Connection interrupted; polling will retry. ${e.message}`,true)}}
setInterval(pollReading,3000);

function anchorRoot(a){
 if(a.representation==='docling-blocks-v1'){const root=$('converted-pane')&&!$('converted-pane').hidden?[...$('converted-pane').querySelectorAll('[data-block]')].find(n=>n.dataset.block===a.block)?.querySelector('[data-anchor-text]'):null;return root&&Number(root.dataset.passageRevision||0)===(a.passage_revision||0)?root:null;}
 if(a.page===page&&$('pdf-pane')&&!$('pdf-pane').hidden)return $('pdf-pane').querySelector('.textLayer');
 return null;
}
function anchorRange(h){
 const a=h.anchor;if(a.figure_box)return null;
 const root=anchorRoot(a);if(!root)return null;
 const text=root.textContent,needle=a.prefix+a.exact+a.suffix,at=text.indexOf(needle);if(at<0||text.indexOf(needle,at+1)>=0)return null;
 const start=at+a.prefix.length,end=start+a.exact.length,walker=document.createTreeWalker(root,NodeFilter.SHOW_TEXT);let n,pos=0,first,last;
 while(n=walker.nextNode()){const next=pos+n.length;if(!first&&start>=pos&&start<next)first=[n,start-pos];if(end>pos&&end<=next){last=[n,end-pos];break}pos=next;}
 if(!first||!last)return null;const range=document.createRange();range.setStart(...first);range.setEnd(...last);return range;
}
function paintHighlights(){
 if(!state)return;for(const el of document.querySelectorAll('.figure-mark'))el.remove();
 const sets={node:[],is:[],is_not:[],question:[]};for(const h of Object.values(state.highlights)){if(h.removed)continue;if(h.anchor.figure_box&&$('pdf-pane')&&!$('pdf-pane').hidden&&h.anchor.page===page){const mark=element('div',undefined,'figure-mark '+h.label);const [x,y,w,height]=h.anchor.figure_box;Object.assign(mark.style,{left:x*100+'%',top:y*100+'%',width:w*100+'%',height:height*100+'%'});$('reader').querySelector('.pdf-page')?.append(mark);continue;}const r=anchorRange(h);if(r)sets[h.label].push(r);}
 if(globalThis.CSS?.highlights&&globalThis.Highlight)for(const [label,ranges]of Object.entries(sets))CSS.highlights.set('hades-'+label,new Highlight(...ranges));
}
$('figure').onclick=()=>{figureMode=!figureMode;$('figure').classList.toggle('active',figureMode);$('figure-caption').hidden=!figureMode;$('reader').classList.toggle('figure-mode',figureMode);status(figureMode?'Drag a rectangle around the figure, then choose its meaning.':'Text selection enabled.');};
let figureStart=null;
$('reader').addEventListener('pointerdown',e=>{if(!figureMode)return;const sheet=e.target.closest('.pdf-page');if(!sheet)return;const rect=sheet.getBoundingClientRect();figureStart={sheet,x:Math.max(0,Math.min(1,(e.clientX-rect.left)/rect.width)),y:Math.max(0,Math.min(1,(e.clientY-rect.top)/rect.height))};e.preventDefault();});
$('reader').addEventListener('pointerup',e=>{if(!figureStart)return;const start=figureStart;figureStart=null;const rect=start.sheet.getBoundingClientRect();const x=Math.max(0,Math.min(1,(e.clientX-rect.left)/rect.width)),y=Math.max(0,Math.min(1,(e.clientY-rect.top)/rect.height));const box=[Math.min(x,start.x),Math.min(y,start.y),Math.abs(x-start.x),Math.abs(y-start.y)];if(box[2]<.01||box[3]<.01)return;selectedAnchor={source_hash:state.source_hash,figure_box:box,representation:'pdf-figure-v1',page,block:null,exact:$('figure-caption').value,prefix:'',suffix:''};status('Figure area selected. Choose a classification. A caption can be preserved by selecting it separately.');});

async function archives(){
 const data=await api('state?view=sources');$('sources').replaceChildren();
 for(const saved of data.sources){
  const card=element('section',undefined,'card');card.append(element('strong',saved.filename),element('p',saved.status,'muted'));
  if(saved.error)card.append(element('p',saved.error,'error'));
  card.append(button(saved.status==='ready'?'Open saved reading':'Retry preparation',async()=>{
   if(saved.status!=='ready'){status('Preparing the archived source…');const result=await api('extract',{source_hash:saved.source_hash});await archives();if(result.status!=='ready')throw Error(result.error||'Preparation failed');}
   await openSource(saved.source_hash,saved.filename);
  }));
  if(saved.status==='ready')card.append(button('Start a separate reading',()=>openSource(saved.source_hash,saved.filename,true)));
  $('sources').append(card);
 }
}
