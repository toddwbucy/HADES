// The daemon owns execution. This view never replays an uncertain submission.
export const pendingChatKey = (database, binding) => `hades.chat.pending.${encodeURIComponent(database)}.${encodeURIComponent(binding)}`;
export function makeSubmission(binding, state, message, highlights, references, requestId, knowledge = []) {
  return JSON.parse(JSON.stringify({action: 'send', binding, request_id: requestId,
    expected_revision: state.revision, message, highlights, knowledge, references}));
}
export class ReadingChat {
  constructor({api, getState, getDatabase, getPosition, restorePosition}) {
    Object.assign(this, {api, getState, getDatabase, getPosition, restorePosition});
    this.binding = null; this.session = null; this.references = []; this.epoch = 0;
    this.exchanges = new Map(); this.olderCursor = undefined; this.lastError = null;
    this.polling = false; this.acting = false; this.lastSignature = ''; this.approvalSignature = '';
    this.$ = id => document.getElementById(id);
    this.$('chat-new').onclick = () => this.act(() => this.open());
    this.$('chat-find').onclick = () => this.act(() => this.findThreads());
    this.$('chat-history').onclick = () => this.act(async () => {
      if (!this.binding?.thread_id) throw Error('Choose a saved conversation first.');
      await this.history(this.binding.thread_id);
    });
    this.$('chat-resume').onclick = () => this.act(() => this.resume());
    this.$('chat-release').onclick = () => this.act(async () => {
      await this.api({action: 'release', binding: this.binding._key}); await this.poll();
    });
    this.$('chat-stop').onclick = () => this.act(async () => {
      await this.api({action: 'interrupt', binding: this.binding._key}); await this.poll();
    });
    this.$('chat-reconcile').onclick = () => this.act(async () => {
      const result = await this.api({action: 'reconcile', binding: this.binding._key});
      this.note(result.unresolved ? 'Some submissions remain unresolved. Nothing was resent.' : 'History reconciled. Nothing was resent.');
      await this.poll();
    });
    this.$('chat-older').onclick = () => this.act(() => this.loadOlder());
    this.$('chat-send').onclick = () => this.act(() => this.send());
    this.$('chat-recover').onclick = () => this.act(async () => {
      if (!this.binding) return;
      const key = pendingChatKey(this.getDatabase(), this.binding._key), raw = localStorage.getItem(key);
      if (!raw) {this.note('No browser submission is awaiting recovery.'); return;}
      const payload = JSON.parse(raw);
      const result = await this.api(payload); // Exact HADES request; its durable journal prevents a second Codex send.
      if (localStorage.getItem(key) === raw) localStorage.removeItem(key);
      await this.poll();
      if (result.rejected) this.note(result.error, true);
    });
    this.$('chat-bindings').onchange = () => this.act(() => this.chooseBinding(this.$('chat-bindings').value));
    this.$('chat-reference-clear').onclick = () => {this.references = []; this.renderReferences();};
    setInterval(() => {if (!document.hidden) void this.poll();}, 750);
    setInterval(() => {if (this.binding && !document.hidden) void this.savePosition();}, 5000);
  }
  node(tag, text, cls) {
    const el = document.createElement(tag); if (text !== undefined) el.textContent = text;
    if (cls) el.className = cls; return el;
  }
  button(text, action) {
    const el = this.node('button', text); el.type = 'button'; el.onclick = () => this.act(action); return el;
  }
  note(message, error = false) {
    if (error) this.lastError = message;
    this.$('chat-status').textContent = message; this.$('chat-status').classList.toggle('error', error);
  }
  async act(fn) {
    if (this.acting) return; this.acting = true; this.lastError = null;
    try {await fn();} catch (e) {this.note(e.message, true);} finally {this.acting = false;}
  }
  async selectReading() {
    const state = this.getState(); if (!state || this.session === state.id) return;
    const epoch = ++this.epoch; this.session = state.id; this.binding = null; this.references = [];
    this.lastSignature = ''; this.approvalSignature = ''; this.$('chat-messages').replaceChildren();
    this.$('chat-approvals').replaceChildren(); this.$('chat-browser').replaceChildren();
    this.exchanges.clear(); this.olderCursor = undefined; this.lastError = null;
    this.$('chat-passages').replaceChildren(); this.$('chat-knowledge').replaceChildren();
    this.updateContext(); this.renderReferences();
    try {
      const config = await this.api({action: 'config'}); if (epoch !== this.epoch) return;
      this.$('chat-controls').hidden = !config.enabled;
      this.note(config.enabled ? 'Start a conversation or choose one to resume.' : 'Codex chat is not configured on the reading daemon.');
      if (!config.enabled) return;
      await this.bindings(epoch);
    } catch (e) {if (epoch === this.epoch) this.note(e.message, true);}
  }
  updateContext() {
    const state = this.getState(); if (!state) return;
    const select = this.$('chat-passages');
    const selected = new Set([...select.selectedOptions].map(o => o.value));
    select.replaceChildren();
    for (const h of Object.values(state.highlights).filter(h => !h.removed)) {
      const o = this.node('option', `${h.label}: ${h.anchor.exact.slice(0, 160) || 'Figure selection'}`);
      o.value = h.id; o.selected = selected.has(h.id); select.append(o);
    }
    const knowledge = this.$('chat-knowledge'), chosen = new Set([...knowledge.selectedOptions].map(o => o.value));
    knowledge.replaceChildren();
    for (const proposal of Object.values(state.proposals).filter(p => p.status === 'accepted')) {
      const o = this.node('option', proposal.draft.title); o.value = proposal.id; o.selected = chosen.has(o.value); knowledge.append(o);
    }
    this.$('chat-context-note').textContent = `Source ${state.source_hash.slice(0, 12)} · reading revision ${state.revision}. Select saved highlights to include; each sent exchange keeps its exact text.`;
  }
  async bindings(epoch = this.epoch) {
    const result = await this.api({action: 'bindings', session: this.session}); if (epoch !== this.epoch) return;
    const select = this.$('chat-bindings'); select.replaceChildren(this.node('option', 'Choose a saved conversation'));
    select.firstChild.value = '';
    this.savedBindings = result.bindings;
    for (const b of result.bindings) {const o = this.node('option', `${b.provider} · ${b.thread_id || 'Creation unresolved'}`); o.value = b._key; select.append(o);}
    if (this.binding) select.value = this.binding._key;
    else if (result.bindings.length) await this.chooseBinding(result.bindings[0]._key);
  }
  async chooseBinding(id) {
    this.exchanges.clear(); this.olderCursor = undefined; this.lastError = null;
    this.epoch++; this.binding = this.savedBindings.find(b => b._key === id) || null;
    this.$('chat-bindings').value = this.binding?._key || '';
    this.$('chat-messages').replaceChildren(); this.$('chat-approvals').replaceChildren();
    this.lastSignature = ''; this.approvalSignature = '';
    if (!this.binding) return;
    const binding = this.binding;
    await this.restorePosition(binding.position);
    if (binding !== this.binding) return;
    await this.poll();
  }
  async open(threadId = null) {
    if (!this.getState()) throw Error('Choose a reading first.');
    const epoch = this.epoch;
    const id = 'chat_' + crypto.randomUUID().replaceAll('-', '');
    const result = await this.api({action: 'open', id, session: this.session, thread_id: threadId});
    if (epoch !== this.epoch) return;
    this.exchanges.clear(); this.olderCursor = undefined;
    this.binding = result; this.$('chat-browser').replaceChildren(); await this.bindings(); await this.poll();
  }
  async resume() {
    if (!this.binding) throw Error('Choose a saved conversation first.');
    const b = this.binding;
    await this.api({action: 'open', id: b._key, session: b.session, thread_id: b.thread_id});
    await this.poll();
  }
  async findThreads(cursor = null) {
    const epoch = this.epoch;
    const result = await this.api({action: 'threads', cursor}); if (epoch !== this.epoch) return;
    const root = this.$('chat-browser'); if (!cursor) root.replaceChildren();
    root.querySelector('[data-more]')?.remove();
    for (const thread of result.data) {
      const row = this.node('section', undefined, 'card');
      row.append(this.node('strong', thread.name || thread.preview || thread.id));
      row.append(this.node('p', `${thread.id} · ${thread.source}`, 'muted'));
      row.append(this.button('Resume here', () => this.open(thread.id)),
        this.button('Reference messages', () => this.history(thread.id)));
      root.append(row);
    }
    if (result.nextCursor) {const more = this.button('More conversations', () => this.findThreads(result.nextCursor)); more.dataset.more = '1'; root.append(more);}
  }
  async history(threadId, cursor = null) {
    const epoch = this.epoch;
    const result = await this.api({action: 'history', thread_id: threadId, cursor}); if (epoch !== this.epoch) return;
    const root = this.$('chat-browser'); if (!cursor) root.replaceChildren(this.node('p', 'Choose messages to copy into the next exchange. This does not take over the source conversation.'));
    root.querySelector('[data-more]')?.remove();
    for (const turn of result.data) for (const item of turn.items || []) {
      if (!['userMessage', 'agentMessage'].includes(item.type)) continue;
      const card = this.node('section', undefined, 'card');
      const text = item.text || (item.content || []).filter(c => c.type === 'text').map(c => c.text).join('\n');
      card.append(this.node('strong', item.type === 'userMessage' ? 'User' : 'Codex'), this.node('p', text));
      card.append(this.button('Include in next message', () => {
        if (this.references.length >= 6) throw Error('Select at most six reference messages.');
        if (!this.references.some(r => r.item_id === item.id && r.thread_id === threadId))
          this.references.push({thread_id: threadId, turn_id: turn.id, item_id: item.id});
        this.renderReferences();
      })); root.append(card);
    }
    if (result.nextCursor) {const more = this.button('More messages', () => this.history(threadId, result.nextCursor)); more.dataset.more = '1'; root.append(more);}
  }
  renderReferences() {
    this.$('chat-reference-note').textContent = `${this.references.length} earlier conversation messages selected for the next exchange.`;
  }
  async send() {
    if (!this.binding) throw Error('Start or resume a conversation first.');
    const database = this.getDatabase(), binding = this.binding._key;
    const key = pendingChatKey(database, binding);
    if (localStorage.getItem(key)) throw Error('A submission is unresolved. Reconcile history before sending another.');
    const payload = makeSubmission(binding, this.getState(), this.$('chat-message').value,
      [...this.$('chat-passages').selectedOptions].map(o => o.value), this.references, crypto.randomUUID(),
      [...this.$('chat-knowledge').selectedOptions].map(o => o.value));
    if (!payload.message.trim()) throw Error('Write a message first.');
    const raw = JSON.stringify(payload); localStorage.setItem(key, raw);
    try {
      const result = await this.api(payload);
      if (localStorage.getItem(key) === raw) localStorage.removeItem(key);
      if (result.rejected) {this.note(result.error, true); return;}
      if (this.binding?._key === binding) {this.$('chat-message').value = ''; this.references = []; this.renderReferences();}
    } catch (e) {
      this.note(`Submission outcome may be unknown. Reconcile history; do not resend. ${e.message}`, true);
    }
    await this.poll();
  }
  async poll() {
    if (this.polling || !this.binding) return;
    const binding = this.binding._key, epoch = this.epoch; this.polling = true;
    try {
      const data = await this.api({action: 'state', binding});
      if (epoch !== this.epoch || this.binding?._key !== binding) return;
      this.recoverFromHistory(binding, data.exchanges);
      this.render(data);
    } catch (e) {if (epoch === this.epoch) this.note(`Connection interrupted: ${e.message}`, true);}
    finally {this.polling = false;}
  }
  recoverFromHistory(binding, exchanges) {
    const key = pendingChatKey(this.getDatabase(), binding), raw = localStorage.getItem(key);
    if (!raw) return;
    let pending; try {pending = JSON.parse(raw);} catch {return;}
    if (pending && exchanges.some(e => e.request_id === pending.request_id)) localStorage.removeItem(key);
  }
  async loadOlder() {
    if (!this.binding || !this.olderCursor) return;
    const binding = this.binding._key, epoch = this.epoch, before = this.olderCursor;
    const data = await this.api({action: 'state', binding, before});
    if (epoch !== this.epoch || this.binding?._key !== binding) return;
    this.recoverFromHistory(binding, data.exchanges);
    for (const exchange of data.exchanges) this.exchanges.set(exchange._key, exchange);
    // A concurrent poll may have discovered a newer gap while this page loaded.
    if (this.olderCursor === before) this.olderCursor = data.next_before;
    await this.poll();
  }
  render(data) {
    // A disjoint latest page can hide exchanges added while this tab was idle.
    // Restart pagination at that gap; overlapping polls preserve older progress.
    const overlaps = data.exchanges.some(exchange => this.exchanges.has(exchange._key));
    if (this.olderCursor === undefined || (data.next_before && !overlaps)) this.olderCursor = data.next_before;
    for (const exchange of data.exchanges) this.exchanges.set(exchange._key, exchange);
    data = {...data, exchanges: [...this.exchanges.values()].sort((a, b) =>
      a.created_at.localeCompare(b.created_at) || a._key.localeCompare(b._key))};
    this.$('chat-older').hidden = !this.olderCursor;
    const live = data.live, root = this.$('chat-messages');
    const signature = JSON.stringify([data.exchanges, live.items, live.status, live.error]);
    if (signature !== this.lastSignature) {
      const nearBottom = root.scrollHeight - root.scrollTop - root.clientHeight < 60;
      root.replaceChildren();
      for (const exchange of data.exchanges) {
        const card = this.node('article', undefined, 'chat-exchange');
        card.append(this.node('strong', 'You'), this.node('p', exchange.message));
        const context = this.node('details'), summary = this.node('summary', 'Context sent with this message');
        context.append(summary, this.node('pre', JSON.stringify(exchange.context, null, 2))); card.append(context);
        const isLive = exchange.provider_turn && exchange.provider_turn === live.turn_id;
        const projection = isLive ? live : exchange.projection;
        this.renderItems(card, projection?.items || []);
        card.append(this.node('p', isLive ? live.status : exchange.status, 'muted'));
        if (exchange.error) card.append(this.node('p', exchange.error, 'error'));
        root.append(card);
      }
      if (live.items?.length && !data.exchanges.some(e => e.provider_turn === live.turn_id)) {
        const card = this.node('article', undefined, 'chat-exchange'); this.renderItems(card, live.items); root.append(card);
      }
      if (live.truncated) root.append(this.node('p', 'Display shortened. Full history remains in Codex.', 'muted'));
      if (nearBottom) root.scrollTop = root.scrollHeight;
      this.lastSignature = signature;
    }
    // Cached transcript rows can predate reconciliation; use the whole journal's
    // current pending state for controls, independently of pagination.
    const pending = data.pending.unresolved;
    this.$('chat-send').disabled = pending || live.closed || live.status === 'disconnected' || !!localStorage.getItem(pendingChatKey(this.getDatabase(), this.binding._key));
    this.$('chat-stop').disabled = live.status !== 'inProgress';
    this.$('chat-release').disabled = live.status === 'inProgress' || (!live.closed && data.pending.active);
    this.$('chat-resume').disabled = !live.closed && live.status !== 'disconnected';
    this.note(this.lastError || live.error || (pending ? 'Response active or submission awaiting reconciliation.' : (live.closed || live.status === 'disconnected') ? 'Conversation saved. Resume to continue.' : 'Connected to Codex.'), !!this.lastError || !!live.error);
    this.renderApprovals(live);
  }
  renderItems(root, items) {
    for (const item of items) {
      if (item.type === 'agentMessage') root.append(this.node('strong', 'Codex'), this.node('p', item.text || ''));
      else if (item.type !== 'userMessage' && item.type !== 'reasoning') {
        const details = this.node('details'); details.append(this.node('summary', `${item.type} · ${item.status || ''}`), this.node('pre', JSON.stringify(item, null, 2))); root.append(details);
      }
    }
  }
  renderApprovals(live) {
    const signature = JSON.stringify([live.generation, live.approvals]); if (signature === this.approvalSignature) return;
    this.approvalSignature = signature; const root = this.$('chat-approvals'); root.replaceChildren();
    for (const request of live.approvals || []) {
      const card = this.node('section', undefined, 'card');
      const binding = this.binding._key, generation = live.generation;
      const reply = result => this.act(async () => {
        await this.api({action: 'approval', binding, generation, request_id: request.id, result});
        await this.poll();
      });
      if (request.method === 'item/tool/requestUserInput') {
        const fields = [];
        for (const question of request.params.questions) {
          const label = this.node('label', question.question);
          const input = question.options?.length ? this.node('select') : this.node('input');
          if (question.options?.length) for (const option of question.options) {const o = this.node('option', option.label); o.value = option.label; input.append(o);}
          label.append(input); card.append(label); fields.push([question.id, input]);
        }
        const submit = this.node('button', 'Send answers'); submit.onclick = () => reply({answers: Object.fromEntries(fields.map(([id, input]) => [id, {answers: [input.value]}]))}); card.append(submit);
      } else {
        card.append(this.node('p', request.reading_tool ? `Codex requests permission to use ${request.reading_tool}. This does not accept a graph proposal.` : 'Codex requested permission outside the reading tools. Decline it or stop the response.'), this.node('pre', JSON.stringify(request.params, null, 2)));
        if (request.reading_tool) {
          const allow = this.node('button', `Allow ${request.reading_tool} once`);
          allow.dataset.readingTool = request.reading_tool;
          allow.onclick = () => reply({action: 'accept', content: {}});
          card.append(allow);
        }
        const deny = this.node('button', 'Decline');
        deny.onclick = () => reply(request.method === 'mcpServer/elicitation/request' ? {action: 'decline'} : request.method === 'item/permissions/requestApproval' ? {permissions: {}, scope: 'turn'} : {decision: 'decline'});
        card.append(deny);
      }
      root.append(card);
    }
  }
  async savePosition() {
    const binding = this.binding?._key; if (!binding) return;
    const position = this.getPosition(); const signature = JSON.stringify([binding, position]);
    if (signature === this.positionSignature) return;
    try {await this.api({action: 'position', binding, ...position}); this.positionSignature = signature;} catch { /* Retry the same position later. */ }
  }
}
