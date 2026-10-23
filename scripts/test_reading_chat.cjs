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
    if (real && await allow.count()) {
      assert.equal(await allow.getAttribute('data-reading-tool'), 'reading_propose', 'remembered reads must not require approval');
      await allow.click(); // Proposal approval remains an explicit human action.
    }
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
  if (!real) {
    const original = await state(), savedBindings = (await api('chat', {action:'bindings',session:'browser_fixture'})).bindings;
    const active = async () => (await api('state')).sessions;
    const removed = async () => (await api('state?view=archived')).sessions;
    // The source is already prepared; only capability discovery is stubbed.
    await page.route('**/api/reading/capabilities', route => route.fulfill({json:{extensions:['.txt']}}));
    await page.reload();
    await page.waitForFunction(() => !document.querySelector('#upload').disabled);
    for (const name of ['renamed-fixture.txt', 'another-name.txt']) {
      await page.locator('#upload').setInputFiles({name, mimeType:'text/plain', buffer:Buffer.from('The reader checks a claim against the selected passage.')});
      await until(async () => (await page.locator('#status').textContent()).includes('Opened the saved reading'));
      assert.equal((await active()).length, 1);
      assert.equal(new URL(page.url()).searchParams.get('session'), 'browser_fixture');
      assert.equal((await state()).title, original.title);
    }
    await page.getByRole('button', {name:'Edit passage', exact:true}).first().click();
    await page.getByText('Archived sources and recovery', {exact:true}).click();
    for (const name of ['Start a separate reading', 'Open saved reading']) {
      await page.getByRole('button', {name, exact:true}).click();
      await until(async () => (await page.locator('#status').textContent()).includes('Save or cancel the passage edit'));
      assert.equal((await active()).length, 1);
      assert.equal(new URL(page.url()).searchParams.get('session'), 'browser_fixture');
    }
    await page.getByRole('button', {name:'Cancel edit', exact:true}).click();
    await page.getByRole('button', {name:'Start a separate reading', exact:true}).click();
    const separate = await until(async () => (await active()).find(s => s.id !== 'browser_fixture'));
    await until(async () => new URL(page.url()).searchParams.get('session') === separate.id);
    let separateRemovalRequests = 0;
    await page.route('**/api/reading/mutate', async route => {
      const payload = route.request().postDataJSON();
      if (payload.session === separate.id && payload.action.action === 'archive') {
        separateRemovalRequests++;
        if (separateRemovalRequests === 1) {await route.fetch(); await route.abort('failed'); return;}
      }
      await route.continue();
    });
    await page.locator(`#sessions [data-session="${separate.id}"]`).getByRole('button', {name:'Remove from desk', exact:true}).click();
    await until(async () => (await removed()).some(s => s.id === separate.id));
    // A draft written after the lost response must block retry even from another reading.
    await page.locator('#bound-is').fill('An unfinished boundary for the separate reading');
    await page.locator('#sessions [data-session="browser_fixture"] .session').click();
    await until(async () => new URL(page.url()).searchParams.get('session') === 'browser_fixture');
    await page.getByRole('button', {name:'Reconcile interrupted action', exact:true}).click();
    await until(async () => (await page.locator('#status').textContent()).includes('unfinished edits'));
    assert.equal(separateRemovalRequests, 1, 'draft guard prevents the retry request');
    assert.equal(new URL(page.url()).searchParams.get('session'), 'browser_fixture');
    // Reset the browser's unsaved fixture draft before testing successful exact recovery.
    await page.goto(url + '/reading?session=' + separate.id);
    await page.waitForFunction(id => document.querySelector('#title').textContent === 'conversation-fixture.txt' && location.search.includes(id), separate.id);
    await page.getByRole('button', {name:'Reconcile interrupted action', exact:true}).click();
    await page.waitForFunction(() => !location.search && document.querySelector('#chat-controls').hidden);
    assert.equal(separateRemovalRequests, 2);
    await page.unroute('**/api/reading/mutate');
    await page.reload();
    await page.waitForFunction(() => document.querySelector('#sessions .session'));
    assert.equal((await active()).length, 1);
    await page.locator('#sessions .session').click();
    await until(async () => new URL(page.url()).searchParams.get('session') === 'browser_fixture');
    // A direct restore must also reject before mutating while a passage edit is active.
    await page.getByRole('button', {name:'Edit passage', exact:true}).first().click();
    await page.locator('#archived-readings summary').click();
    await page.locator(`#archived-sessions [data-session="${separate.id}"]`).getByRole('button', {name:'Restore to desk', exact:true}).click();
    await until(async () => (await page.locator('#status').textContent()).includes('Save or cancel the passage edit'));
    assert((await removed()).some(s => s.id === separate.id));
    assert.equal((await active()).length, 1);
    await page.getByRole('button', {name:'Cancel edit', exact:true}).click();
    // Do not discard unsent chat text while clearing the selected reading.
    await page.locator('[data-tab="chat"]').click();
    await page.locator('#chat-message').fill('An unfinished question');
    const removeOriginal = () => page.locator('#sessions [data-session="browser_fixture"]').getByRole('button', {name:'Remove from desk', exact:true}).click();
    await removeOriginal();
    await until(async () => (await page.locator('#status').textContent()).includes('unfinished edits'));
    assert.equal((await active()).length, 1);
    await page.locator('#chat-message').fill('');
    // Lose the acknowledgement after committing removal, then reconcile exactly.
    let lost = false;
    await page.route('**/api/reading/mutate', async route => {
      if (!lost && route.request().postDataJSON().action.action === 'archive') {
        lost = true; await route.fetch(); await route.abort('failed');
      } else await route.continue();
    });
    await removeOriginal();
    await until(async () => (await state()).archived);
    await page.locator('#chat-message').fill('Keep this draft during retry');
    await page.getByRole('button', {name:'Reconcile interrupted action', exact:true}).click();
    await until(async () => (await page.locator('#status').textContent()).includes('unfinished edits'));
    assert.equal(await page.locator('#chat-message').inputValue(), 'Keep this draft during retry');
    await page.locator('#chat-message').fill('');
    await page.getByRole('button', {name:'Reconcile interrupted action', exact:true}).click();
    await page.waitForFunction(() => !location.search && document.querySelector('#chat-controls').hidden);
    assert.equal((await active()).length, 0);
    assert.equal((await state()).revision, original.revision + 1);
    await page.reload();
    await page.locator('#archived-readings summary').click();
    const restore = () => page.locator('#archived-sessions [data-session="browser_fixture"]').getByRole('button', {name:'Restore to desk', exact:true}).click();
    await restore();
    await until(async () => !(await state()).archived);
    await page.waitForFunction(() => location.search.includes('browser_fixture') && document.querySelector('[data-anchor-text]'));
    const restored = await state();
    for (const field of ['graph','bound','highlights','proposals','corrections']) assert.deepEqual(restored[field], original[field]);
    assert.deepEqual((await api('chat', {action:'bindings',session:'browser_fixture'})).bindings.map(b => b._key).sort(), savedBindings.map(b => b._key).sort());
    assert.equal((await api('state?session=browser_fixture&view=export')).chain_valid, true);
    // Restore the saved transcript and measure real CSS, including the chat override.
    await page.locator('[data-tab="chat"]').click();
    await page.locator('#chat-bindings').selectOption(binding._key);
    await until(async () => (await page.locator('#chat-messages').textContent()).length > 0);
    const widths = () => page.evaluate(() => ['.library','.document','.review'].map(s => document.querySelector(s).getBoundingClientRect().width));
    const layoutChecks = [];
    for (const width of [1280,1800,2560]) {
      await page.setViewportSize({width,height:1200});
      if (await page.locator('body').evaluate(el => el.classList.contains('wide'))) await page.locator('#wide').click();
      const compact = await widths();
      await page.locator('#wide').click();
      const wide = await widths();
      assert(wide[0] > compact[0], `left panel grows at ${width}`);
      assert(wide[2] > compact[2], `chat panel grows at ${width}`);
      assert(wide[1] >= 400, `document remains usable at ${width}`);
      layoutChecks.push({viewport:width,compact,wide});
    }
    await page.setViewportSize({width:1000,height:1200});
    assert.equal((await page.locator('main').evaluate(el => getComputedStyle(el).gridTemplateColumns)).split(' ').length, 2);
    await page.setViewportSize({width:600,height:1200});
    assert.equal(await page.locator('main').evaluate(el => getComputedStyle(el).display), 'block');
    await page.setViewportSize({width:1800,height:1200});
    await page.locator('#archived-readings').evaluate(el => {el.open = true;});
    await page.screenshot({path:path.join(root, 'reading-desk-wide.png'), fullPage:true});
    fs.writeFileSync(path.join(root, 'desk-evidence.json'), JSON.stringify({passed:true, checks:['renamed repeated upload resumes saved reading','intentional separate reading','passage edit blocks creation and restoration before mutation','remove and reload','unsent message retained','boundary draft blocks retry from another reading','lost removal response exact retry','restore notes, graph and chat bindings','valid audit chain','wider side panels and responsive fallback'],layoutChecks}, null, 2));
    assert.deepEqual(errors, []);
  }
  if (!real) {
    // Continue the original saved conversation in the graph workspace, without
    // releasing ownership or cloning its transcript into another binding.
    await page.goto(url + '/reading?' + new URLSearchParams({session:'browser_fixture', conversation:binding._key}));
    await page.locator('[data-tab="chat"]').click();
    await until(async () => (await page.locator('#chat-bindings').inputValue()) === binding._key);
    const before = await chatState();
    let releaseWorkspace, workspaceArrived;
    const holdWorkspace = new Promise(resolve => {releaseWorkspace=resolve;});
    const workspacePending = new Promise(resolve => {workspaceArrived=resolve;});
    const delayWorkspace = async route => {
      const body=route.request().postDataJSON();
      if (body?.action !== 'bindings' || body.session) return route.continue();
      workspaceArrived(); await holdWorkspace; await route.continue();
    };
    await page.route('**/api/reading/chat', delayWorkspace);
    await page.locator('#chat-workspace-link').click();
    await page.waitForURL(u => u.pathname === '/');
    await workspacePending;
    await page.locator('.library-item').first().waitFor();
    assert.equal(new URL(page.url()).searchParams.get('conversation'), binding._key, 'graph navigation preserves a pending workspace conversation');
    releaseWorkspace();
    await until(async () => (await page.locator('#chat-bindings').inputValue()) === binding._key);
    await page.unroute('**/api/reading/chat', delayWorkspace);
    await until(async () => (await page.locator('#chat-messages').textContent()).includes('Discuss this selected passage.'));
    assert.equal(new URL(page.url()).searchParams.get('conversation'), binding._key);
    if (await page.locator('#chat-resume').isEnabled()) await page.locator('#chat-resume').click();
    await until(async () => await page.locator('#chat-send').isEnabled());
    await page.locator('[data-library="all"]').click();
    await page.locator('.library-item').first().click();
    await page.locator('#add-context').click();
    await page.locator('#chat-message').fill('Continue our reading discussion using this knowledge object.');
    await page.locator('#chat-send').click();
    const continued = await until(async () => {const v=await chatState(); return v.exchanges.length === before.exchanges.length + 1 && v.exchanges.at(-1).status === 'completed' && v;});
    assert.equal(continued.binding.thread_id, before.binding.thread_id);
    assert.equal(continued.binding.allow_reads, before.binding.allow_reads);
    assert.equal(continued.exchanges.at(-1).context.format, 'hades-workspace-context/v1');
    assert.equal(continued.exchanges.at(-1).context.objects.length, 1);
    assert.equal(continued.exchanges.at(-1).context.origin_reading, 'browser_fixture');
    await page.reload();
    await until(async () => (await page.locator('#chat-bindings').inputValue()) === binding._key);
    await until(async () => (await page.locator('#chat-messages').textContent()).includes('Continue our reading discussion'));
    assert.equal((await chatState()).exchanges.length, continued.exchanges.length, 'reload cannot replay a message');
    await page.locator('#workspace-layout').selectOption('wide');
    await page.screenshot({path:path.join(root, 'workspace-codex-chat.png'), fullPage:true});
    const newest = (await api('chat',{action:'bindings',session:'browser_fixture'})).bindings[0];
    assert.notEqual(newest._key, binding._key, 'exercise returning to an older conversation');
    let releaseRestore, restorationArrived;
    const holdRestore = new Promise(resolve => {releaseRestore=resolve;});
    const restoring = new Promise(resolve => {restorationArrived=resolve;});
    const failRestoration = async route => {
      if (route.request().postDataJSON()?.action !== 'bindings') return route.continue();
      restorationArrived(); await holdRestore;
      await route.fulfill({status:503, contentType:'application/json', body:JSON.stringify({error:'Fixture history temporarily unavailable'})});
    };
    await page.route('**/api/reading/chat', failRestoration);
    await page.locator('#chat-reading-link').click();
    await restoring;
    assert.equal(new URL(page.url()).searchParams.get('conversation'), binding._key, 'keep the requested link while history is pending');
    releaseRestore();
    await until(async () => (await page.locator('#chat-status').textContent()).includes('Fixture history temporarily unavailable'));
    assert.equal(new URL(page.url()).searchParams.get('conversation'), binding._key, 'failed restoration cannot erase the direct link');
    await page.unroute('**/api/reading/chat', failRestoration);
    await page.reload();
    await until(async () => (await page.locator('#chat-bindings').inputValue()) === binding._key);
    assert.equal(new URL(page.url()).searchParams.get('conversation'), binding._key);
    await page.locator('[data-tab="chat"]').click();
    await until(async () => (await page.locator('#chat-messages').textContent()).includes('Continue our reading discussion'));
    await page.locator('#chat-workspace-link').click();
    await until(async () => (await page.locator('#chat-bindings').inputValue()) === binding._key);
    await page.locator('#chat-new').click();
    const free = await until(async () => {const v=await api('chat',{action:'bindings'});return v.bindings.find(b => b.session === null && b.thread_id);});
    await until(async () => (await page.locator('#chat-bindings').inputValue()) === free._key);
    await until(async () => await page.locator('#chat-send').isEnabled());
    await page.locator('[data-library="all"]').click();
    await page.locator('.library-item').first().click();
    await page.locator('#reader-links button[aria-label^="Inspect "]').first().click();
    const edgeId = await page.evaluate(() => state.selectedEdge);
    assert(edgeId, 'fixture must select an edge');
    // The accepted basis edge may contain stored text. Exercise the alternative
    // projection too: no text field, with a rendered source-evidence block.
    await page.evaluate(id => {
      const attrs={...graph.getEdgeAttribute(id, 'attrs')};
      for(const field of ['content','body','definition','text','statement'])delete attrs[field];
      attrs.evidence={quote:'Synthetic source evidence'};
      graph.setEdgeAttribute(id, 'attrs', attrs); inspectEdge(id);
    }, edgeId);
    assert.equal(await page.locator('#reader-body').evaluate(el => getComputedStyle(el).userSelect), 'none');
    assert.equal(await page.locator('#reader-body p').evaluate(el => getComputedStyle(el).userSelect), 'none');
    // A programmatic selection can bypass CSS; the tray must still reject a
    // generated summary locally instead of submitting an invalid excerpt.
    await page.locator('#reader-body').evaluate(el => {
      el.style.userSelect='text'; for(const child of el.children)child.style.userSelect='text';
      const range=document.createRange(); range.selectNodeContents(el);
      const selection=window.getSelection(); selection.removeAllRanges(); selection.addRange(range);
    });
    await page.locator('#add-context').click();
    assert.match(await page.locator('#context-status').textContent(), /Select only stored source text/);
    assert.equal(await page.locator('#context-items .context-card').count(), 0);
    await page.evaluate(() => window.getSelection().removeAllRanges());
    await page.locator('#add-context').click();
    assert.equal(await page.locator('#context-items .context-card').count(), 1);
    await page.locator('#chat-message').fill('A new workspace conversation without a reading.');
    await page.locator('#chat-send').click();
    await until(async () => (await api('chat',{action:'state',binding:free._key})).exchanges[0]?.status === 'completed');
    const edgeExchange = (await api('chat',{action:'state',binding:free._key})).exchanges[0];
    assert.equal(edgeExchange.context.objects[0].selection.id, edgeId);
    assert.equal(edgeExchange.context.objects[0].selection.excerpt, '');
    assert.deepEqual(errors, []);
    fs.writeFileSync(path.join(root, 'workspace-chat-evidence.json'), JSON.stringify({passed:true, same_thread:true, same_binding:true, remembered_reads_preserved:true, frozen_graph_context:true, reload_without_replay:true, return_to_reading:true, standalone_workspace_chat:true}, null, 2));
  }
  fs.writeFileSync(path.join(root, 'chat-evidence.json'), JSON.stringify({real_codex:real, runtime:'0.163.0', passed:true, checks:['browser passage selection','frozen MCP exchange context','restricted tools','pending proposal','human acceptance','reload without replay','release and resume', ...(!real ? ['question response','interrupt','daemon crash and ownership recovery','history reference selection','accepted knowledge snapshot'] : ['model MCP retrieval','model MCP proposal'])]}, null, 2));
  console.log('Reading chat browser/MCP contract passed:', root);
}
main().catch(async e => {console.error(e); const page=browser?.contexts()[0]?.pages()[0]; if(page) console.error(await page.locator('#status, #chat-status').allTextContents()); process.exitCode = 1;}).finally(async () => {
  if (browser) await browser.close();
  for (const p of children.reverse()) {p.kill('SIGTERM'); await Promise.race([new Promise(r => p.once('exit', r)), sleep(5000)]); if(p.exitCode === null) p.kill('SIGKILL');}
  for (const fd of logs) fs.closeSync(fd);
  fs.rmSync(path.join(home, 'auth.json'), {force:true});
});
