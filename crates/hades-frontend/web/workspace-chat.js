import {ReadingChat} from '/reading-chat.js';

const requested = window.workspace.initialConversation();
let database = null, connected = false, sequence = 0;
const chat = new ReadingChat({
  workspace: true,
  getDatabase: () => database,
  getState: () => connected ? {id: `workspace:${database}`, revision: 0} : null,
  getWorkspaceContext: () => window.workspace.references(),
  api: async body => {
    const response = await fetch('/api/reading/chat', {method: 'POST',
      headers: {'Content-Type': 'application/json', 'X-Hades-Workspace': '1', 'X-Hades-Database': database},
      body: JSON.stringify(body)});
    const value = await response.json();
    if (!response.ok) throw Error(value.error || `Request failed (${response.status})`);
    return value;
  },
  onBinding: binding => {
    const link = document.getElementById('chat-reading-link'); link.hidden = !binding?.session;
    if (binding?.session) link.href = '/reading?' + new URLSearchParams({session: binding.session, conversation: binding._key});
    const url = new URL(location.href);
    if (binding) url.searchParams.set('conversation', binding._key);
    else url.searchParams.delete('conversation');
    history.replaceState(null, '', url);
  },
});
// Capture this before graph navigation rewrites the address bar.
chat.preferredBinding = requested.binding;
async function sync() {
  const next = window.workspace.database();
  if (next === database) return;
  const ticket = ++sequence;
  database = next; connected = false;
  await chat.selectReading();
  if (requested.db && requested.db !== database) chat.preferredBinding = null;
  try {
    const response = await fetch('/api/reading/config');
    if (!response.ok) throw Error('Chat configuration is unavailable.');
    const config = await response.json();
    if (ticket !== sequence) return;
    if (!config.enabled || config.database !== database) {
      chat.note(config.enabled ? `Chat is configured for ${config.database}. Select that database to continue.` : 'Chat is not configured on this viewer.');
      return;
    }
    connected = true; await chat.selectReading();
  } catch (error) { if (ticket === sequence) chat.note(error.message, true); }
}
window.workspaceChat = {sync, binding: () => chat.binding?._key};
void sync();
