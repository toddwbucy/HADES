/* Browser -> viewer -> daemon -> App Server, using a with_temp_db fixture. */
const fs = require('node:fs'), path = require('node:path'), assert = require('node:assert/strict');
const {spawn} = require('node:child_process'), net = require('node:net');
const {chromium} = require(process.env.PLAYWRIGHT_MODULE);
const database = process.argv[2], root = process.env.HADES_CHAT_TEST_ROOT;
assert(database.startsWith('hades_test_chat_browser_') && root);
assert.equal(process.env.ARANGO_SOCKET, path.join(root, 'arango.sock'));
const repo = path.resolve(__dirname, '..'), target = path.join(process.env.CARGO_TARGET_DIR || path.join(repo, 'target'), 'debug');
const real = !!process.env.HADES_CHAT_TEST_AUTH, home = path.join(root, 'codex');
const children = [], logs = [];
let browser;
const sleep = ms => new Promise(resolve => setTimeout(resolve, ms));
async function port() {const s = net.createServer(); await new Promise(r => s.listen(0, '127.0.0.1', r)); const p = s.address().port; await new Promise(r => s.close(r)); return p;}
async function until(fn, timeout = 30000) {const deadline = Date.now() + timeout; let error; while (Date.now() < deadline) {try {const v = await fn(); if(v) return v;} catch(e) {error = e;} await sleep(100);} throw error || Error('Timed out waiting for fixture');}
function child(name, argv, env) {const fd = fs.openSync(path.join(root, name + '.log'), 'w'); logs.push(fd); const p = spawn(argv[0], argv.slice(1), {cwd: repo, env, stdio: ['ignore', fd, fd]}); children.push(p); return p;}
async function main() {
  fs.mkdirSync(home, {mode: 0o700});
  fs.writeFileSync(path.join(home, 'hades-test-fixture'), '');
  // Authentication only: never copy user history, plugins, or project settings.
  if (real) fs.copyFileSync(process.env.HADES_CHAT_TEST_AUTH, path.join(home, 'auth.json'));
  if (real) fs.chmodSync(path.join(home, 'auth.json'), 0o600);
  fs.writeFileSync(path.join(home, 'config.toml'), 'model = "gpt-6-astra"\nmodel_reasoning_effort = "low"\n[mcp_servers.unwanted]\ncommand = "/bin/false"\nenabled = true\n');
  const mcpPort = await port(), viewPort = await port(), token = 'private-conversation-fixture';
  const mcp = `http://127.0.0.1:${mcpPort}/mcp`, url = `http://127.0.0.1:${viewPort}`;
  fs.writeFileSync(path.join(root, 'mcp-token'), token, {mode: 0o600});
  const config = JSON.parse(fs.readFileSync(path.join(root, 'hades.yaml')));
  config.reading_chat = {enabled:true, executable:real ? process.env.HADES_CHAT_CODEX : path.join(repo, 'scripts/tests/fake_codex.py'), home, cwd:home, mcp_url:mcp, mcp_token_env:'HADES_READING_MCP_TOKEN'};
  fs.writeFileSync(path.join(root, 'hades.yaml'), JSON.stringify(config));
  const env = {...process.env, HADES_READING_MCP_TOKEN:token, ARANGO_USERNAME:'root'};
  const daemon = child('chat-daemon', [path.join(target, 'hades'), 'daemon', '--database', database, '--socket', path.join(root, 'hades.sock'), '--mcp-bind', `127.0.0.1:${mcpPort}`, '--mcp-token-file', path.join(root, 'mcp-token'), '--mcp-guided-reading', '--mcp-dbs', database, '--mcp-db-prefix', 'hades_test_', '--mcp-ingest-root', root], env);
  await until(() => {assert.equal(daemon.exitCode, null); return fs.existsSync(path.join(root, 'hades.sock'));});
  child('chat-viewer', [path.join(target, 'hades-viewer'), 'serve', '--db', database, '--hades-bin', path.join(target, 'hades'), '--reading-db', database, '--reading-socket', path.join(root, 'hades.sock'), '--bind', `127.0.0.1:${viewPort}`], env);
  await until(async () => (await fetch(url + '/api/reading/config')).ok);
  let session, rpcId = 0;
  async function rpc(method, params) {
    const id = ++rpcId, response = await fetch(mcp, {method:'POST', headers:{Authorization:`Bearer ${token}`, 'Content-Type':'application/json', Accept:'application/json, text/event-stream', ...(session ? {'Mcp-Session-Id':session} : {})}, body:JSON.stringify({jsonrpc:'2.0', id, method, params})});
    session = response.headers.get('mcp-session-id') || session;
    const raw = await response.text(); assert(response.ok, raw);
    const result = raw.includes('data:') ? raw.split('\n').filter(l => l.startsWith('data:') && l.slice(5).trim()).map(l => JSON.parse(l.slice(5))).find(v => v.id === id) : JSON.parse(raw);
    assert(!result.error, JSON.stringify(result)); return result.result;
  }
  async function tool(name, args) {const result = await rpc('tools/call', {name, arguments:{db:database,...args}}); assert(!result.isError, JSON.stringify(result)); const data = JSON.parse(result.content.find(c => c.type === 'text').text); assert(data.success, JSON.stringify(data)); return data.data;}
  await rpc('initialize', {protocolVersion:'2024-11-05', capabilities:{}, clientInfo:{name:'reading-chat-contract', version:'1'}});
  await fetch(mcp, {method:'POST', headers:{Authorization:`Bearer ${token}`, 'Content-Type':'application/json', Accept:'application/json, text/event-stream', 'Mcp-Session-Id':session}, body:JSON.stringify({jsonrpc:'2.0', method:'notifications/initialized'})});
  assert((await rpc('tools/list', {})).tools.some(t => t.name === 'reading_exchange_context'));
  const denied = await rpc('tools/call', {name:'kg_create', arguments:{db:database, name:'must_not_create'}});
  assert(denied.isError && JSON.stringify(denied).includes('guided reading'));
  browser = await chromium.launch({executablePath:process.env.CHROMIUM_PATH, headless:true, args:['--no-sandbox']});
  const context = await browser.newContext({viewport:{width:1800,height:1200}}), page = await context.newPage(), errors = [];
  page.on('pageerror', e => errors.push(e.message));
  async function api(action, body) {const response = await fetch(url + '/api/reading/' + action, {method:body ? 'POST' : 'GET', headers:body ? {'Content-Type':'application/json', 'X-Hades-Workspace':'1', 'X-Hades-Database':database, Origin:url} : {}, body:body && JSON.stringify(body)}); const result = await response.json(); assert(response.ok, JSON.stringify(result)); return result;}
  const state = async () => (await api('state?session=browser_fixture')).state;
  await page.goto(url + '/reading?session=browser_fixture');
  await page.waitForFunction(() => document.querySelector('#title').textContent === 'Discussing a source passage' && document.querySelector('[data-anchor-text]'));
  await page.locator('#bound-is').fill('A synthetic passage for testing document conversation.');
  await page.locator('#bound-not').fill('A factual research claim.');
  await page.locator('#advance').click();
  await until(async () => (await state()).stage === 'read');
  const passage = page.locator('[data-anchor-text]').first(); await passage.scrollIntoViewIfNeeded();
  const rect = await passage.evaluate(el => {const r = document.createRange(); r.selectNodeContents(el); return r.getBoundingClientRect().toJSON();});
  await page.mouse.move(rect.x + 1, rect.y + rect.height/2); await page.mouse.down(); await page.mouse.move(rect.right - 1, rect.y + rect.height/2, {steps:15}); await page.mouse.up();
  await page.locator('[data-label="node"]').click();
  const highlight = await until(async () => Object.values((await state()).highlights)[0]);
  await page.locator('[data-tab="chat"]').click(); await page.locator('#chat-new').click();
  const binding = await until(async () => {
    const status = await page.locator('#chat-status').textContent();
    if (await page.locator('#chat-status').evaluate(el => el.classList.contains('error'))) throw Error(status);
    const bindings = await api('chat', {action:'bindings', session:'browser_fixture'});
    return bindings.bindings.find(b => b.thread_id);
  }, 60000);
  const chatState = () => api('chat', {action:'state', binding:binding._key});
  await until(async () => !(await page.locator('#chat-send').isDisabled()));
  await page.getByRole('button', {name:'Always allow reads in this conversation', exact:true}).click();
  await until(async () => (await chatState()).binding.allow_reads === true);
  await page.locator('#chat-passages').selectOption(highlight.id);
  await page.locator('#chat-message').fill(real ? 'Retrieve the frozen passage and propose a brief sourced note for my review (proposal ID codex_proposal, actor model:codex). Use the reading MCP tools and leave acceptance to me.' : 'Discuss this selected passage.');
  await page.locator('#chat-send').click();
  await until(async () => {
    const allow = page.locator('#chat-approvals [data-reading-tool]').first();
    if (real && await allow.count()) await allow.click();
    const current = await chatState();
    if (current.exchanges[0]?.status === 'failed') throw Error(JSON.stringify(current.live));
    return current.exchanges[0]?.status === 'completed';
  }, real ? 180000 : 30000);
  const exchange = (await chatState()).exchanges[0];
  assert.equal(exchange.context.highlights[0].id, highlight.id);
  assert.deepEqual((await tool('reading_exchange_context', {exchange_id:exchange._key})).context, exchange.context);
  if (real) {
    const items = exchange.projection.items;
    assert(items.some(i => i.type === 'mcpToolCall' && i.tool === 'reading_exchange_context'), JSON.stringify(items));
    assert(items.some(i => i.type === 'mcpToolCall' && i.tool === 'reading_propose' && i.status === 'completed'), JSON.stringify(items));
    assert(!items.some(i => i.type === 'mcpToolCall' && i.status === 'failed'), 'Published tool schemas must allow this fixture without rejected calls');
  } else {
    const s = await state();
    await tool('reading_propose', {session:s.id, request_id:'proposal_request', expected_revision:s.revision, actor:'model:fixture', action:{action:'propose', id:'codex_proposal', draft:{title:'Checking the source', note:'A claim needs support from the selected passage.', is:'A test observation', is_not:null, standing:'fixture only', highlight:highlight.id, basis:'The selected passage describes checking evidence.', from:null,to:null,relation:null,inferred:false,rationale:null}}});
  }
  const proposal = (await state()).proposals.codex_proposal;
  assert.equal(proposal.status, 'pending'); assert.equal(proposal.created_ids.length, 0);
  await page.reload(); await page.locator('[data-tab="chat"]').click();
  await page.locator('#chat-messages').getByText(exchange.message, {exact:true}).waitFor();
  await page.getByRole('button', {name:'Revoke remembered read permission', exact:true}).waitFor();
  assert.equal((await chatState()).exchanges.length, 1, 'reload cannot replay a message');
  await page.locator('#chat-release').click();
  await until(async () => (await chatState()).live.status === 'disconnected');
  await page.locator('#chat-resume').click();
  await until(async () => !(await page.locator('#chat-send').isDisabled()));
  if (!real) {
    await page.locator('#chat-message').fill('/approval'); await page.locator('#chat-send').click();
    await page.getByRole('button', {name:'Send answers', exact:true}).click();
    await until(async () => (await chatState()).exchanges[1]?.status === 'completed');
    await page.locator('#chat-message').fill('/slow'); await page.locator('#chat-send').click();
    await until(async () => (await chatState()).live.status === 'inProgress');
    await page.locator('#chat-stop').click();
    await until(async () => (await chatState()).exchanges[2]?.status === 'interrupted');
  }
  if (!real) {
    const exited = new Promise(resolve => daemon.once('exit', resolve));
    daemon.kill('SIGKILL'); await exited;
    const replacement = child('chat-daemon-restarted', [path.join(target, 'hades'), ...daemon.spawnargs.slice(1)], env);
    await until(async () => {assert.equal(replacement.exitCode, null); return (await chatState()).live.status === 'disconnected';});
    await api('chat', {action:'open',id:binding._key,session:binding.session,thread_id:binding.thread_id});
    assert.equal((await chatState()).binding.thread_id, binding.thread_id);
    assert.equal((await chatState()).binding.allow_reads, true);
  }
  await page.getByRole('button', {name:'Revoke remembered read permission', exact:true}).click();
  await until(async () => (await chatState()).binding.allow_reads === false);
  if (!real) {
    const count = (await chatState()).exchanges.length;
    await page.locator('#chat-message').fill('/read-tool'); await page.locator('#chat-send').click();
    await page.getByRole('button', {name:'Allow db_get once', exact:true}).waitFor();
    await page.getByRole('button', {name:'Always allow reads in this conversation', exact:true}).click();
    await until(async () => (await chatState()).exchanges[count]?.status === 'completed');
    await page.locator('#chat-message').fill('/read-tool'); await page.locator('#chat-send').click();
    await until(async () => (await chatState()).exchanges[count + 1]?.status === 'completed');
  }
  await page.locator('[data-tab="proposals"]').click();
  await page.getByRole('button', {name:'Review exact revision'}).first().click();
  await page.getByRole('button', {name:'Accept reviewed revision', exact:true}).click();
  await until(async () => (await state()).proposals.codex_proposal.status === 'accepted');
  assert((await state()).proposals.codex_proposal.created_ids.length > 0);
  await page.locator('[data-tab="chat"]').click();
  await until(async () => (await page.locator('#chat-status').textContent()) === 'Connected to Codex.');
  await page.locator('#chat-messages').evaluate(el => {el.scrollTop = el.scrollHeight;});
  await page.locator('.document').evaluate(el => {el.scrollTop = el.scrollHeight;});
  await page.screenshot({path:path.join(root, 'codex-reading.png'), fullPage:true});
  if (!real) {
    await page.locator('#chat-history').click();
    await page.getByRole('button', {name:'Include in next message', exact:true}).first().click();
    await page.locator('#chat-new').click();
    const next = await until(async () => (await api('chat', {action:'bindings',session:'browser_fixture'})).bindings.find(b => b.thread_id && b._key !== binding._key));
    assert.notEqual(next.allow_reads, true);
    await page.getByRole('button', {name:'Always allow reads in this conversation', exact:true}).waitFor();
    await page.locator('#chat-knowledge').selectOption('codex_proposal');
    await page.locator('#chat-message').fill('Compare the earlier message and the selected accepted note.');
    await page.locator('#chat-send').click();
    const exchange = await until(async () => {
      const e = (await api('chat', {action:'state',binding:next._key})).exchanges[0];
      return e?.status === 'completed' && e;
    });
    assert.equal(exchange.context.conversation_references.length, 1);
    assert.equal(exchange.context.knowledge[0].id, 'codex_proposal');
    assert.equal(exchange.context.knowledge[0].graph_ids.length, 2);
    await api('chat', {action:'release',binding:next._key});
  }
  assert.deepEqual(errors, []);
  await api('chat', {action:'release', binding:binding._key});
  fs.writeFileSync(path.join(root, 'chat-evidence.json'), JSON.stringify({real_codex:real, runtime:'0.163.0', passed:true, checks:['browser passage selection','frozen MCP exchange context','restricted tools','pending proposal','human acceptance','reload without replay','release and resume', ...(!real ? ['question response','interrupt','daemon crash and ownership recovery','history reference selection','accepted knowledge snapshot'] : ['model MCP retrieval','model MCP proposal'])]}, null, 2));
  console.log('Reading chat browser/MCP contract passed:', root);
}
main().catch(async e => {console.error(e); const page=browser?.contexts()[0]?.pages()[0]; if(page) console.error(await page.locator('#status, #chat-status').allTextContents()); process.exitCode = 1;}).finally(async () => {
  if (browser) await browser.close();
  for (const p of children.reverse()) {p.kill('SIGTERM'); await Promise.race([new Promise(r => p.once('exit', r)), sleep(5000)]); if(p.exitCode === null) p.kill('SIGKILL');}
  for (const fd of logs) fs.closeSync(fd);
  fs.rmSync(path.join(home, 'auth.json'), {force:true});
});
