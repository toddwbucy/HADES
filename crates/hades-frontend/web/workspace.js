/* Reading, source navigation, and bounded conversation references over viewer state. */
(() => {
  const M = window.HadesWorkspaceModel;
  const $ = id => document.getElementById(id);
  let mode = "outline", selected = null, refs = [], epoch = 0, searchSequence = 0, selectionSequence = 0;
  const el = (tag, text, className) => {
    const node = document.createElement(tag); if (text !== undefined) node.textContent = text;
    if (className) node.className = className; return node;
  };
  function snapshot() {
    if (!graph || !state.snapshot) return {nodes:[],edges:[]};
    const nodes = graph.nodes().map(id => ({id, ...graph.getNodeAttributes(id)}));
    const edges = graph.edges().map(id => ({id, source:graph.source(id), target:graph.target(id), ...graph.getEdgeAttributes(id)}));
    return {nodes, edges};
  }
  const scope = () => ({db:state.db,graph:state.snapshot?.meta.named_graph});
  function renderLibrary() {
    const rows = M.library(snapshot(), mode, $("library-filter").value);
    $("library-count").textContent = rows.length;
    $("library").replaceChildren();
    for (const {node,depth} of rows) {
      const button = el("button", node.label, "library-item");
      button.dataset.depth = Math.min(depth, 3); button.dataset.id = node.id;
      button.classList.toggle("selected", selected?.id === node.id);
      button.setAttribute("aria-current", selected?.id === node.id ? "true" : "false");
      if (mode !== "outline") button.append(el("small", M.kind(node)));
      button.onclick = () => open(node.id);
      $("library").append(button);
    }
    if (!rows.length) $("library").append(el("p", state.snapshot?.meta.partial
      ? "No matches in this loaded slice. Expand the graph or increase its limit."
      : "No matching objects. Try another library tab or graph.", "empty-state"));
  }
  async function open(id, expand = false) {
    const ticket = epoch, selection = ++selectionSequence;
    if (expand || !graph?.hasNode(id)) await expandNode(id);
    if (ticket !== epoch || selection !== selectionSequence) return;
    if (graph?.hasNode(id)) { focusNode(id); $("reader").scrollTop = 0; }
    else $("search-status").textContent = "This reference could not be loaded. Refresh the graph and try again.";
  }
  function show(kind, item) {
    selected = {kind, id:item.id, item};
    $("add-context").disabled = false;
    $("reader-kind").textContent = kind === "edge" ? "Relationship" : M.kind(item).replaceAll("_", " ");
    $("reader-title").textContent = item.label || M.relation(item);
    $("view-title").textContent = item.label || M.relation(item);
    const a = item.attrs || {};
    $("reader-meta").replaceChildren();
    for (const value of [a.source_version && `Version ${a.source_version}`, a.source_node_id, a.assertion_origin,
      a.non_normative && "Illustration · non-normative"]) {
      if (value) $("reader-meta").append(el("span", value, "badge"));
    }
    if (a.standing_text) $("reader-meta").append(el("span", a.standing_text, "badge standing"));
    const body = M.text(item);
    $("reader-body").textContent = body || (kind === "edge"
      ? `${item.source}\n${M.relation(item)}\n${item.target}`
      : "This object has no inline source text. Explore its connections or open Object details for its metadata.");
    if (kind === "edge" && a.evidence) {
      $("reader-body").append(el("p", "Source evidence\n" + JSON.stringify(a.evidence, null, 2)));
    }
    $("reader-links").replaceChildren(el("div", "Connected objects · in this view", "eyebrow"));
    const connections = [];
    if (kind === "node") {
      graph.forEachEdge(item.id, (id, attrs, source, target) => {
        connections.push({id, other:source === item.id ? target : source, relation:M.relation(attrs), outgoing:source === item.id});
      });
    } else {
      connections.push({other:item.source, relation:"From"}, {other:item.target, relation:"To"});
    }
    connections.sort((x,y) => x.relation.localeCompare(y.relation));
    for (const connection of connections.slice(0, 60)) {
      const row = el("div", undefined, "relation-row");
      const label = el("span", `${connection.outgoing === false ? "← " : ""}${connection.relation}`, "relation-name");
      const button = el("button", graph.getNodeAttribute(connection.other, "label") || connection.other);
      button.onclick = () => open(connection.other); row.append(label,button);
      if (connection.id) {
        const inspect = el("button", "↗"); inspect.style.flex = "0 0 30px";
        inspect.setAttribute("aria-label", `Inspect ${connection.relation} relationship`);
        inspect.onclick = () => inspectEdge(connection.id); row.append(inspect);
      }
      $("reader-links").append(row);
    }
    if (connections.length > 60) $("reader-links").append(el("p", `Showing 60 of ${connections.length} connections. Use the graph to explore further.`, "hint"));
    if (!connections.length) $("reader-links").append(el("p", "No connections in the loaded view.", "hint"));
    for (const button of $("library").querySelectorAll(".library-item")) {
      const current = kind === "node" && button.dataset.id === item.id;
      button.classList.toggle("selected", current);
      button.setAttribute("aria-current", current ? "true" : "false");
    }
  }
  function renderContext() {
    $("context-items").replaceChildren();
    refs.forEach((ref,i) => {
      const card = el("div", undefined, "context-card");
      card.append(el("strong",ref.title), el("small",`${ref.kind} · ${ref.db}`,"muted"));
      if (ref.excerpt) card.append(el("p", ref.excerpt.slice(0,160) + (ref.excerpt.length > 160 ? "…" : "")));
      const remove = el("button","Remove","remove"); remove.onclick = () => { refs.splice(i,1); renderContext(); };
      card.append(remove); $("context-items").append(card);
    });
    $("copy-context").disabled = $("clear-context").disabled = !refs.length;
    $("copy-context").textContent = `Copy ${refs.length || ""} reference${refs.length === 1 ? "" : "s"} for agent`;
  }
  $("add-context").onclick = () => {
    if (!selected || refs.length >= 6) { $("context-status").textContent = "The tray holds up to six references. Remove one to add another."; return; }
    const selection = window.getSelection();
    const excerpt = selection?.rangeCount && $("reader-body").contains(selection.getRangeAt(0).commonAncestorContainer) ? selection.toString() : "";
    const ref = M.reference(scope(),selected.item,selected.kind,excerpt);
    if (refs.some(r => r.db === ref.db && r.graph === ref.graph && r.id === ref.id && r.excerpt === ref.excerpt)) {
      $("context-status").textContent = "That reference is already in the tray."; return;
    }
    try { M.contextPacket([...refs,ref]); refs.push(ref); renderContext(); $("context-status").textContent = "Added. This reference will be included with your next chat message."; }
    catch (error) { $("context-status").textContent = error.message; }
  };
  $("add-context").onmousedown = event => event.preventDefault();
  $("copy-context").onclick = () => { try { copyText(M.contextPacket(refs), "Conversation context"); } catch(error) { $("context-status").textContent = error.message; } };
  $("clear-context").onclick = () => {refs=[];renderContext();};
  $("library-filter").oninput = renderLibrary;
  $("library-tabs").onclick = event => {
    if (!event.target.dataset.library) return;
    mode = event.target.dataset.library;
    for (const button of $("library-tabs").children) button.classList.toggle("active",button === event.target);
    renderLibrary();
  };
  $("view-tabs").onclick = event => {
    if (!event.target.dataset.view) return;
    $("stage").className = `${event.target.dataset.view}-mode`;
    for (const button of $("view-tabs").children) button.classList.toggle("active",button === event.target);
    requestAnimationFrame(() => renderer?.resize());
  };
  const layoutPicker = $("workspace-layout");
  function applyLayout(value) {
    const layout = ["auto", "wide", "compact"].includes(value) ? value : "auto";
    document.body.dataset.layout = layout;
    layoutPicker.value = layout;
    requestAnimationFrame(() => renderer?.resize());
  }
  try { applyLayout(localStorage.getItem("hades-workspace-layout")); }
  catch (_) { applyLayout("auto"); }
  layoutPicker.onchange = () => {
    applyLayout(layoutPicker.value);
    try { localStorage.setItem("hades-workspace-layout", layoutPicker.value); } catch (_) {}
  };
  $("search-form").onsubmit = async event => {
    event.preventDefault(); const ticket=epoch, request=++searchSequence;
    const query = $("search-query").value.trim(); if (!query) return;
    $("search-status").textContent = "Searching the embedded corpus…";
    $("search-results").replaceChildren();
    const params = new URLSearchParams({db:state.db,text:query,profile:$("search-profile").value});
    try {
      const response = await fetch(`/api/search?${params}`);
      if (!response.ok) throw new Error("Search unavailable. Check that this corpus is embedded and the embedder is running.");
      const data = await response.json(); if (ticket !== epoch || request !== searchSequence) return;
      $("search-status").textContent = `${data.results.length} matches in ${data.profile === "default" ? "documents" : "code"}. Results cover the database, not just this graph.`;
      for (const hit of data.results) {
        const button = el("button",hit.title || hit.parent_key,"search-hit");
        button.append(el("small",String(hit.text || "").slice(0,180) + "…"));
        button.onclick = async () => {
          if (ticket !== epoch) return;
          const id = hit.chunk_id || hit.parent_id;
          await open(id, true);
        };
        $("search-results").append(button);
      }
      if (!data.results.length) $("search-status").textContent = "No embedded matches. You can still browse the library and graph.";
    } catch(error) { if (ticket === epoch && request === searchSequence) $("search-status").textContent = error.message; }
  };
  window.workspace = {
    database: () => state.db,
    initialConversation: () => ({db: deepLink.db, binding: deepLink.conversation}),
    references: () => refs.map(({db, graph, kind, id, excerpt}) => ({db, graph, kind, id, excerpt})),
    empty(message) { $("reader-title").textContent="Workspace unavailable"; $("reader-body").textContent=message; },
    loading() { window.workspaceChat?.sync(); epoch++; searchSequence++; selected=null; $("add-context").disabled=true; $("library").replaceChildren(); $("search-results").replaceChildren(); $("search-status").textContent=""; $("reader-title").textContent="Loading workspace…"; $("reader-body").textContent=""; $("reader-meta").replaceChildren(); $("reader-links").replaceChildren(); },
    loaded() { renderLibrary(); const root = M.library(snapshot(),"outline")[0]?.node; if (root && !pendingLink.node) selectNode(root.id); },
    expanded() { renderLibrary(); },
    node(id) { selectionSequence++; if (id) show("node",{id,...graph.getNodeAttributes(id)}); else {selected=null;$("add-context").disabled=true;} },
    edge(id) { selectionSequence++; show("edge",{id,source:graph.source(id),target:graph.target(id),...graph.getEdgeAttributes(id)}); },
  };
  renderContext();
})();
