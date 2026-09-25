/* 论文问答前端：三栏（会话 | 原文 | 对话），原生 JS，无构建步骤。

   关键实现都标了"为什么"，几个容易踩的点先列在这儿：

   1. **流式用 fetch 读 body，不用 EventSource**：EventSource 只能发 GET，
      而提问要带 JSON body。所以自己切 SSE 帧（见 parseFrame）。
   2. **高亮用百分比定位**：后端 bbox 是归一化 0-1000（PLAN 踩坑 13），
      换成百分比铺在 canvas 上，不用管 PDF 点尺寸 / 渲染 scale / CSS 缩放。
   3. **证据卡"两处同源"**：实时回答用 SSE 里的 cards，历史消息用落库的快照
      `citations._cards` —— 两者结构一样，所以渲染函数共用一个。
   4. **发送前保证有会话**：没选中就自动建一个，否则对话不会被记住，
      而"历史会话"这个功能就等于没有。
*/

const API = "";
const LS_SIDEBAR = "rag.sidebar.collapsed";
const LS_PDF_WIDTH = "rag.pdf.width";

const state = {
  papers: [],
  selected: new Set(),
  sessions: [],
  sessionId: null,
  messages: [],       // [{role, content, citations}]
  streaming: null,    // 正在流式渲染的那条 assistant 消息（对象引用）
  asking: false,
  uploading: false,   // 入库期间不让提问：向量还没写完，问了也是白问
  viewer: { paperId: null, page: 0, total: 0, boxes: [], doc: null, loaded: null },
};

const el = (id) => document.getElementById(id);

// ---------------------------------------------------------------- 启动
async function boot() {
  applyStoredLayout();
  bindEvents();
  await Promise.all([loadHealth(), loadPapers(), loadSessions()]);
}

function bindEvents() {
  el("sidebar-toggle").addEventListener("click", toggleSidebar);
  el("new-session").addEventListener("click", newSession);
  el("ask-btn").addEventListener("click", send);
  bindScopePicker();
  el("pdf-close").addEventListener("click", collapsePdf);
  el("pdf-prev").addEventListener("click", () => showPage(state.viewer.page - 1));
  el("pdf-next").addEventListener("click", () => showPage(state.viewer.page + 1));
  el("upload-btn").addEventListener("click", () => el("file-input").click());
  el("file-input").addEventListener("change", uploadPaper);

  el("question").addEventListener("keydown", (event) => {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      send();
    }
  });
  el("question").addEventListener("input", autoGrow);
  document.addEventListener("keydown", (event) => {
    if (event.key === "Escape") collapsePdf();
  });
  bindResizer();
}

function autoGrow() {
  const node = el("question");
  node.style.height = "auto";
  node.style.height = `${Math.min(node.scrollHeight, 160)}px`;
}

// ---------------------------------------------------------------- 布局
function applyStoredLayout() {
  if (localStorage.getItem(LS_SIDEBAR) === "1") setSidebar(true);
  const width = Number(localStorage.getItem(LS_PDF_WIDTH) || 0);
  if (width > 0) el("pdf-pane").style.width = `${width}px`;
  syncResizer();
}

function setSidebar(collapsed) {
  el("sidebar").classList.toggle("collapsed", collapsed);
  const button = el("sidebar-toggle");
  button.title = collapsed ? "展开侧栏" : "收起侧栏";
  button.setAttribute("aria-label", button.title);
  localStorage.setItem(LS_SIDEBAR, collapsed ? "1" : "0");
}

function toggleSidebar() {
  setSidebar(!el("sidebar").classList.contains("collapsed"));
}

function expandPdf() {
  el("pdf-pane").classList.remove("collapsed");
  if (!el("pdf-pane").style.width) el("pdf-pane").style.width = "42%";
  syncResizer();
}

function collapsePdf() {
  el("pdf-pane").classList.add("collapsed");
  state.viewer.boxes = [];
  syncResizer();
}

function syncResizer() {
  el("resizer").classList.toggle("hidden", el("pdf-pane").classList.contains("collapsed"));
}

function bindResizer() {
  const resizer = el("resizer");
  let dragging = false;
  resizer.addEventListener("mousedown", () => {
    dragging = true;
    document.body.style.cursor = "col-resize";
  });
  window.addEventListener("mousemove", (event) => {
    if (!dragging) return;
    // 原文栏在左边，所以宽度 = 鼠标位置 - 侧栏宽度
    const sidebar = el("sidebar").getBoundingClientRect().width;
    const width = Math.max(280, Math.min(event.clientX - sidebar, window.innerWidth * 0.65));
    el("pdf-pane").style.width = `${width}px`;
  });
  window.addEventListener("mouseup", () => {
    if (!dragging) return;
    dragging = false;
    document.body.style.cursor = "";
    localStorage.setItem(LS_PDF_WIDTH, String(Math.round(el("pdf-pane").getBoundingClientRect().width)));
  });
}

// ---------------------------------------------------------------- 顶栏 / 论文
async function loadHealth() {
  try {
    const data = await (await fetch(`${API}/health`)).json();
    const cls = data.ok ? "dot-ok" : "dot-bad";
    const drift = data.ok ? "" : ` · 对账差异 ${data.drift}`;
    el("health").innerHTML =
      `<span class="dot ${cls}"></span>${data.papers} 篇 · ${data.chunks} chunk · ${data.vectors} 向量${drift}`;
  } catch (error) {
    el("health").innerHTML = `<span class="dot dot-bad"></span>读不到库（服务在跑吗？）`;
  }
}

async function loadPapers() {
  const data = await (await fetch(`${API}/papers`)).json();
  state.papers = data.items || [];
  // 论文被删掉时，把已选中的范围里失效的 id 清掉
  const alive = new Set(state.papers.map((paper) => paper.paper_id));
  state.selected = new Set([...state.selected].filter((id) => alive.has(id)));
  renderScope();
}

function renderScope() {
  renderScopeSummary();
  renderScopeList();
}

/* 范围选择器：平时只显示一行摘要（占位固定，论文再多也不会把输入区挤高），
   点开才是可搜索的列表 —— 50 篇论文铺成芯片墙是没法用的。 */
function renderScopeSummary() {
  const count = state.selected.size;
  const button = el("scope-summary");
  if (count === 0) button.textContent = "范围：全库";
  else if (count === state.papers.length && count > 1) button.textContent = `范围：全部 ${count} 篇`;
  else if (count === 1) button.textContent = `范围：${[...state.selected][0]}`;
  else button.textContent = `范围：已选 ${count} 篇`;
  button.classList.toggle("on", count > 0);
  button.title = count === 0
    ? "不选 = 全库；问句里出现论文名会自动识别"
    : [...state.selected].join("、");
}

function renderScopeList() {
  const box = el("scope-list");
  const keyword = (el("scope-search").value || "").trim().toLowerCase();
  const papers = state.papers.filter((paper) => {
    if (!keyword) return true;
    return `${paper.paper_id} ${paper.title || ""}`.toLowerCase().includes(keyword);
  });

  box.innerHTML = "";
  if (papers.length === 0) {
    box.innerHTML = `<div class="session-empty">没有匹配的论文</div>`;
  }
  for (const paper of papers) {
    const row = document.createElement("label");
    row.className = "scope-item";
    row.innerHTML = `
      <input type="checkbox" ${state.selected.has(paper.paper_id) ? "checked" : ""}>
      <span class="scope-item-main">
        <span class="scope-item-title" title="${escapeHtml(paper.title || paper.paper_id)}">${escapeHtml(paper.title || paper.paper_id)}</span>
        <span class="muted">${escapeHtml(paper.paper_id)} · ${paper.chunks ?? 0} chunk</span>
      </span>
      <button class="scope-del" type="button" title="删除这篇论文">✕</button>`;
    row.querySelector("input").onchange = () => {
      if (state.selected.has(paper.paper_id)) state.selected.delete(paper.paper_id);
      else state.selected.add(paper.paper_id);
      renderScopeSummary();
      renderScopeCount();
    };
    // 删除按钮在 <label> 里：不拦的话点它等于点了整行 → 顺手把这篇勾上/取消。
    // 两个都要拦：stopPropagation 挡冒泡，preventDefault 挡 label 的默认激活行为。
    row.querySelector(".scope-del").onclick = (event) => {
      event.preventDefault();
      event.stopPropagation();
      deletePaper(paper);
    };
    box.appendChild(row);
  }
  renderScopeCount();
}

function renderScopeCount() {
  const count = state.selected.size;
  const total = state.papers.length;
  if (count === 0) el("scope-count").textContent = `共 ${total} 篇 · 当前全库检索`;
  else if (count === total && total > 1) el("scope-count").textContent = `已选全部 ${total} 篇 · 等同全库检索`;
  else el("scope-count").textContent = `已选 ${count} 篇 · 只在这几篇里检索`;
}

/* 删除一篇论文。**一次只删被点名的那一篇** —— 不复用上面那套勾选：
   勾选的意思是"只在这几篇里检索"，用户勾 3 篇想看对比，一点删除就删 3 篇。

   确认框里写清三件事：删的是哪一篇、会删什么、**不会删什么**（MinerU 产物和历史会话）。
   后端那边 `indexer.delete_paper` 定死了顺序（先向量 → 再 FTS → 最后 SQLite）。 */
async function deletePaper(paper) {
  const name = paper.title || paper.paper_id;
  const ok = window.confirm(
    `删除《${name}》？\n\n` +
      `会删掉：这篇论文的 ${paper.chunks ?? 0} 个 chunk、对应的向量与关键词索引。\n` +
      `不会删：MinerU 解析产物（storage/mineru/）和历史会话 —— 历史回答与引用都留着，` +
      `只是点卡片看原文会提示"这篇论文已删除"。\n\n` +
      `删完要重新上传才能再问这篇。`
  );
  if (!ok) return;

  try {
    const response = await fetch(`${API}/papers/${paper.paper_id}`, { method: "DELETE" });
    if (!response.ok) {
      const detail = await response.json().catch(() => ({}));
      throw new Error(detail.detail || `HTTP ${response.status}`);
    }
    // 正开着这篇的原文就先收起来（否则预览停在已删的论文上），并把缓存的 PDF 丢掉 ——
    // 否则将来重传同名论文（paper_id 相同）时，`loaded === paperId` 会让它显示旧文档
    if (state.viewer.paperId === paper.paper_id) {
      state.viewer.doc = null;
      state.viewer.loaded = null;
      collapsePdf();
    }
    state.selected.delete(paper.paper_id);
    await loadPapers(); // 重新拉列表 + 渲染范围面板（失效的勾选会被清掉）
    setStatus(`已删除 ${paper.paper_id}`, false);
  } catch (error) {
    setStatus(`删除失败：${error.message}`, true);
  }
}

function bindScopePicker() {
  el("scope-summary").addEventListener("click", (event) => {
    event.stopPropagation();
    const panel = el("scope-panel");
    const willOpen = panel.classList.contains("hidden");
    panel.classList.toggle("hidden", !willOpen);
    if (willOpen) {
      el("scope-search").value = "";
      renderScopeList();
      el("scope-search").focus();
    }
  });
  el("scope-panel").addEventListener("click", (event) => event.stopPropagation());
  el("scope-search").addEventListener("input", renderScopeList);
  el("scope-all").addEventListener("click", () => {
    // 全选 = 把所有论文勾上。后端会把"选满全部"归一成全库（见 conversation.py），
    // 否则 5 篇会走多篇配额（每篇 3 条）一次返回 15 条证据，那不是用户想要的。
    state.selected = new Set(state.papers.map((paper) => paper.paper_id));
    renderScope();
  });
  el("scope-clear").addEventListener("click", () => {
    state.selected.clear();
    renderScope();
  });
  // 点面板外面就收起
  document.addEventListener("click", () => el("scope-panel").classList.add("hidden"));
}

// ---------------------------------------------------------------- 会话
async function loadSessions() {
  const data = await (await fetch(`${API}/sessions`)).json();
  state.sessions = data.items || [];
  renderSessions();
}

function renderSessions() {
  const box = el("session-list");
  box.innerHTML = "";
  if (state.sessions.length === 0) {
    box.innerHTML = `<div class="session-empty">还没有会话。提问会自动新建一个。</div>`;
    return;
  }
  for (const session of state.sessions) {
    const node = document.createElement("div");
    node.className = "session" + (session.session_id === state.sessionId ? " active" : "");
    node.title = session.title || "新会话";
    node.innerHTML = `<span class="session-title">${escapeHtml(session.title || "新会话")}</span>
      <button class="session-del" title="删除会话" aria-label="删除会话">×</button>`;
    node.onclick = () => openSession(session.session_id);
    node.querySelector(".session-del").onclick = async (event) => {
      event.stopPropagation();
      await deleteSession(session.session_id);
    };
    box.appendChild(node);
  }
}

function newSession() {
  /* **懒创建**：点"新建会话"只是清空视图，不落库。

     反过来的做法（点一下就在库里建一条）会留一堆 `message_count = 0` 的空会话，
     历史列表很快就脏了；而且用户点两下就多两条，看着像 bug。
     真正的会话在**第一句话发出去时**才创建 —— 见 ensureSession()。 */
  state.sessionId = null;
  state.selected.clear();
  state.messages = [];
  el("chat-title").textContent = "新会话";
  renderScope();
  renderSessions();
  renderMessages();
  el("question").focus();
}

async function ensureSession() {
  if (state.sessionId) return state.sessionId;
  const session = await (await fetch(`${API}/sessions`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ paper_ids: [...state.selected] }),
  })).json();
  state.sessionId = session.session_id;
  if (state.selected.size === 0) state.selected = new Set(session.scope || []);
  await loadSessions();
  renderScope();
  el("chat-title").textContent = session.title || "新会话";
  return state.sessionId;
}

async function openSession(sessionId) {
  const data = await (await fetch(`${API}/sessions/${sessionId}`)).json();
  state.sessionId = sessionId;
  state.selected = new Set(data.scope || []);
  state.messages = (data.messages || []).map((message) => ({
    role: message.role,
    content: message.content,
    citations: message.citations || {},
  }));
  el("chat-title").textContent = data.title || "新会话";
  renderScope();
  renderSessions();
  renderMessages(true);
}

async function deleteSession(sessionId) {
  await fetch(`${API}/sessions/${sessionId}`, { method: "DELETE" });
  if (state.sessionId === sessionId) {
    state.sessionId = null;
    state.messages = [];
    el("chat-title").textContent = "新会话";
    renderMessages();
  }
  await loadSessions();
}

// ---------------------------------------------------------------- 消息渲染
function renderMessages(scrollToEnd = true) {
  const box = el("messages");
  box.innerHTML = "";
  if (state.messages.length === 0) {
    const welcome = document.createElement("div");
    welcome.className = "welcome";
    welcome.innerHTML = `<div class="welcome-title">问点什么</div>
      <div class="muted">先点下面的论文 chip 缩小范围，或者直接提问（问句里出现论文名会自动识别）。<br>
      回答里的 <span class="cite">[E1]</span> 可以点，证据卡在回答下方，点卡片看原文。</div>`;
    box.appendChild(welcome);
    return;
  }
  for (const message of state.messages) box.appendChild(messageNode(message));
  if (scrollToEnd) box.scrollTop = box.scrollHeight;
}

function messageNode(message) {
  const node = document.createElement("div");
  node.className = `msg msg-${message.role}`;

  if (message.role === "user") {
    const bubble = document.createElement("div");
    bubble.className = "bubble";
    bubble.textContent = message.content;
    node.appendChild(bubble);
    return node;
  }

  const answer = document.createElement("div");
  answer.className = "answer";
  answer.innerHTML = renderAnswerHtml(message.content);
  // 档位徽标：**只有 unknown 才出现**（grounded 是常态，不加东西打扰阅读）。
  // 文案由 citations.unknown_reason 决定，"不在范围"和"库里没有"因此看得出区别。
  const badge = modeBadge(message.citations || {});
  if (badge) node.appendChild(badge);
  node.appendChild(answer);

  // "范围继承上一轮 / 追问改写为…" 这类提示：解析时算出来的，不是模型输出
  if (message.note) {
    const note = document.createElement("div");
    note.className = "msg-note";
    note.textContent = message.note;
    node.appendChild(note);
  }

  const cards = message.cards || (message.citations && message.citations._cards) || [];
  if (cards.length > 0) node.appendChild(evidenceNode(cards, message.citations || {}));
  return node;
}

/* 两档的呈现：grounded 什么都不加；unknown 挂一个徽标，措辞按 reason 分。
   老会话（Step 9 之前存的）没有 mode 字段 —— 那时按 insufficient 兜一下，
   否则刷新历史会看到"该有徽标却没有"。 */
const MODE_LABELS = {
  out_of_scope: "不在知识库范围内",
  not_in_corpus: "知识库未收录",
  downgraded: "依据不足",
  failed: "生成失败",
  chitchat: "闲聊",
};

function modeBadge(citations) {
  const reason = citations.unknown_reason || "";
  const isUnknown = citations.mode === "unknown" || (!citations.mode && citations.insufficient);
  if (!isUnknown) return null;
  const node = document.createElement("div");
  // 闲聊/失败不是"知识库里没有"，样式上也分开（见 app.css）
  node.className = `mode-badge mode-${reason || "unknown"}`;
  node.textContent = MODE_LABELS[reason] || "未收录";
  return node;
}

function renderAnswerHtml(text) {
  // 先转义再插引用标记 —— 顺序反了等于开了个 XSS 口子（模型输出里含论文原文）
  const escaped = String(text || "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;");
  return escaped.replace(
    /\[E(\d+)\]/g,
    (match) => `<span class="cite" data-label="${match.slice(1, -1)}">${match}</span>`
  );
}

function evidenceNode(cards, citations) {
  const wrap = document.createElement("div");
  // **默认折叠**：5 张卡铺开能占大半屏，而多数时候用户只关心回答；
  // 要核对来源时点一下展开（回答里的 [En] 也会自动展开并滚到那张卡）。
  wrap.className = "evidence collapsed";

  const used = (citations && citations.cited) || [];
  const unknown = (citations && citations.unknown) || [];
  const head = document.createElement("div");
  head.className = "evidence-head";
  const warn = unknown.length ? ` · 编造引用 ${unknown.join("、")}` : "";
  head.innerHTML = `<span>证据 ${cards.length} 张${used.length ? ` · 用到 ${used.join("、")}` : ""}${warn}</span>
    <span class="muted"></span>`;
  const hint = head.querySelector(".muted");
  const syncHint = () => {
    hint.textContent = wrap.classList.contains("collapsed") ? "点击展开" : "点击收起";
  };
  syncHint();
  head.onclick = () => {
    wrap.classList.toggle("collapsed");
    syncHint();
  };
  wrap.appendChild(head);

  const list = document.createElement("div");
  list.className = "evidence-list";
  for (const card of cards) list.appendChild(cardNode(card));
  wrap.appendChild(list);
  return wrap;
}

function cardNode(card) {
  const node = document.createElement("div");
  node.className = "card";
  node.dataset.label = card.label;
  node.onclick = () => openCard(card, node);

  const pages =
    card.page_start === null || card.page_start === undefined
      ? ""
      : card.page_end && card.page_end !== card.page_start
        ? `p.${card.page_start}-${card.page_end}`
        : `p.${card.page_start}`;
  // 每个 chip = 一个"本体就在这个 chunk 里"的资产。这个列表来自占位符展开
  // （retriever.expand_assets），所以正文只是**提到**、本体在别处的资产不会
  // 出现在这里 —— 那种交叉引用只记在 chunk_assets 里，不渲染。
  // caption 可能是空的（论文本来就没写图注/表注），这时只显示类型，别留个孤零零的冒号。
  const assets = (card.assets || [])
    .map((asset) => {
      const type = escapeHtml(asset.asset_type || "asset");
      const caption = truncate(asset.caption || "", 40);
      return `<span class="asset">${type}${caption ? `：${escapeHtml(caption)}` : ""}</span>`;
    })
    .join("");

  // 图资产额外给一张缩略图 —— 模型看得到图（挂进多模态消息了），用户也该看得到。
  // 点缩略图跟点卡片一样（冒泡到卡片的 onclick）→ 跳到原文那一页。后端是 `/assets/{id}/image`。
  const figures = (card.assets || [])
    .filter((asset) => asset.asset_type === "figure")
    .map(
      (asset) =>
        `<img class="card-figure" loading="lazy" alt="图" src="${API}/assets/${encodeURIComponent(asset.asset_id)}/image">`
    )
    .join("");

  // 显示**决定名次的那个分**：有重排分就显示重排分，重排没跑（或这一步失败）时
  // rank_score 是合并名次分（≤1）。两者都不是余弦 —— 别拿余弦冒充相关度。
  const score = card.rank_score ? `重排 ${card.rank_score}` : `余弦 ${card.similarity ?? ""}`;
  node.innerHTML = `
    <div class="card-head">
      <span class="label">${escapeHtml(card.label || "")}</span>
      <span class="card-meta">${escapeHtml(card.section_title || "（无章节）")} · ${pages} · ${escapeHtml(score)}</span>
    </div>
    <div class="card-paper">${escapeHtml(card.paper_title || card.paper_id || "")}</div>
    <div class="card-text">${escapeHtml(truncate(card.text || "", 320))}</div>
    ${figures ? `<div class="card-figures">${figures}</div>` : ""}
    ${assets ? `<div class="card-assets">${assets}</div>` : ""}
  `;
  return node;
}

function focusEvidence(label, root) {
  // **只在这条消息里找卡片**：同一个编号（E1/E2…）在多轮里会重复出现，
  // 全局找再取最后一张 = 点旧回答的 [E3] 永远跳到最新那条（真实踩到的 bug）。
  const scope = root || el("messages");
  const nodes = scope.querySelectorAll(`.card[data-label="${label}"]`);
  if (nodes.length === 0) return;
  const node = nodes[nodes.length - 1];  // 同一条消息里不该有重号，留个兜底
  const wrap = node.closest(".evidence");
  if (wrap) {
    wrap.classList.remove("collapsed");
    // 提示文案要跟着一起变，否则展开后还写着"点击展开"
    const hint = wrap.querySelector(".evidence-head .muted");
    if (hint) hint.textContent = "点击收起";
  }
  el("messages").querySelectorAll(".card").forEach((item) => item.classList.remove("active"));
  node.classList.add("active");
  node.scrollIntoView({ behavior: "smooth", block: "center" });
}

// 回答里的 [En] 是动态插入的，用事件委托统一处理。
// **作用域必须限定在"这条消息"**：编号跨轮会重复（每条回答都有自己的 E1/E2…），
// 不限定的话点旧回答的 [E3] 会跳到最新那条的卡片上。
el("messages").addEventListener("click", (event) => {
  const cite = event.target.closest(".cite");
  if (cite && cite.dataset.label) focusEvidence(cite.dataset.label, cite.closest(".msg"));
});

// ---------------------------------------------------------------- 提问
async function send() {
  const question = el("question").value.trim();
  if (!question || state.asking) return;
  if (state.uploading) {
    setStatus("正在入库，等这一步跑完再提问（向量还没写完，检索不到这篇）", true);
    return;
  }

  await ensureSession();

  state.asking = true;
  el("ask-btn").disabled = true;
  setStatus("检索中…");
  el("question").value = "";
  autoGrow();

  state.messages.push({ role: "user", content: question });
  const assistant = { role: "assistant", content: "", cards: [], citations: {} };
  state.messages.push(assistant);
  renderMessages();

  // 实时更新的是**最后一个** assistant 节点的 .answer ——
  // 用 :last-of-type 不可靠（它按标签名算，不看 class），所以自己取数组末位
  const liveNodes = el("messages").querySelectorAll(".msg-assistant");
  const answerNode = liveNodes[liveNodes.length - 1].querySelector(".answer");
  const payload = { question, session_id: state.sessionId };
  // **总是发**（包括空数组）：空数组 = "这一轮显式全库"。
  // 只在有勾选时才发的话，"取消勾选"会被后端当成"没给范围"，于是 sticky 把上一轮的
  // 范围又粘回来 —— 界面写着"全库"，实际还在旧范围里搜。
  payload.paper_ids = [...state.selected];

  let buffer = "";
  const started = performance.now();

  try {
    const response = await fetch(`${API}/chat`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    if (!response.ok) {
      const detail = await response.json().catch(() => ({}));
      throw new Error(detail.detail || `HTTP ${response.status}`);
    }

    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      let cut;
      while ((cut = buffer.indexOf("\n\n")) >= 0) {
        const frame = buffer.slice(0, cut);
        buffer = buffer.slice(cut + 2);
        const { event, data } = parseFrame(frame);
        if (event === "intent") {
          // 入口理解做完就先到了：面板/改写提示立刻更新，不用等检索（慢一个量级）。
          // 这时证据还没回来，先把"正在检索"告诉用户，别让他对着空白等。
          assistant.note = scopeHintText(data) ? `${scopeHintText(data)} · 正在检索…` : "正在检索…";
          syncScopeFromEvidence(data);
          renderMessages();
        } else if (event === "evidence") {
          assistant.cards = data.cards || [];
          assistant.note = scopeHintText(data);
          syncScopeFromEvidence(data);
        } else if (event === "token") {
          assistant.content += data.text || "";
          answerNode.innerHTML = renderAnswerHtml(assistant.content);
          el("messages").scrollTop = el("messages").scrollHeight;
        } else if (event === "done") {
          assistant.content = data.answer || assistant.content;
          // 模型一个字都没返回时，别让用户对着空白区发呆 —— 这种情况以前真的发生过
          // （流式展平写错，token 全被吞掉，还没报错），至少给个能看懂的现象
          if (!assistant.content.trim()) assistant.content = "（模型没有返回任何内容，请重试一次）";
          assistant.citations = data.citations || {};
          replaceLastMessage(assistant);
          const ms = Math.round(performance.now() - started);
          const cite = assistant.citations;
          const unknown = (cite.unknown || []).length ? ` · 编造引用 ${cite.unknown.join("、")}` : "";
          setStatus(`完成 ${ms} ms · 用到 ${(cite.cited || []).join("、") || "（无）"}${unknown}`, Boolean(unknown));
        } else if (event === "error") {
          setStatus(`出错（${data.stage}）：${data.message}`, true);
        }
      }
    }
  } catch (error) {
    setStatus(`请求失败：${error.message}`, true);
    assistant.content = assistant.content || `（失败：${error.message}）`;
    replaceLastMessage(assistant);
  } finally {
    state.asking = false;
    // 入库期间提问按钮要保持禁用（`answer` 刚回来就把按钮点亮会绕过那道闸）
    el("ask-btn").disabled = state.uploading;
    await loadSessions();   // 标题是首问自动生成的，跑完刷新列表
  }
}

function replaceLastMessage(message) {
  const nodes = el("messages").querySelectorAll(".msg-assistant");
  if (nodes.length > 0) {
    nodes[nodes.length - 1].replaceWith(messageNode(message));
    el("messages").scrollTop = el("messages").scrollHeight;
  }
}

function scopeHintText(data) {
  const parts = [];
  if (data.scope_source === "sticky") parts.push("范围继承上一轮");
  else if (data.scope_source === "question") parts.push("范围来自问句");
  else if (data.scope_source === "explicit") parts.push("范围来自选择");
  if (data.rewritten) parts.push(`追问改写为：「${data.standalone}」`);
  return parts.join(" · ");
}


/** 问句里点名了别的论文时，把面板的勾选同步过去。

    `scope_source === "question"` 说明这一轮的范围是**问句说了算**（压过了面板上的勾选），
    界面就得跟着换 —— 否则勾选还停在旧论文上，用户以为搜的是 A、实际搜的是 B。
    其它来源（explicit / sticky / all）本来就和面板一致，不用动。 */
function syncScopeFromEvidence(data) {
  if (data.scope_source !== "question") return;
  const next = new Set(data.scope || []);
  const same =
    next.size === state.selected.size && [...next].every((id) => state.selected.has(id));
  if (same) return;
  state.selected = next;
  renderScope();
}


function setStatus(text, isError = false) {
  el("status").textContent = text;
  el("status").className = isError ? "status error" : "status";
}

function parseFrame(frame) {
  let event = "";
  let data = "";
  for (const line of frame.split("\n")) {
    if (line.startsWith("event: ")) event = line.slice(7).trim();
    else if (line.startsWith("data: ")) data += line.slice(6);
  }
  if (!event) return { event: "", data: {} };
  try {
    return { event, data: JSON.parse(data || "{}") };
  } catch (error) {
    return { event: "error", data: { stage: "parse", message: "返回的 JSON 解析不了" } };
  }
}

// ---------------------------------------------------------------- 原文
async function openCard(card, node) {
  el("messages").querySelectorAll(".card").forEach((item) => item.classList.remove("active"));
  // 激活**被点的那一张**。不能按编号全页找最后一张 —— 编号跨轮重复（见 focusEvidence）。
  const target =
    node || [...el("messages").querySelectorAll(`.card[data-label="${card.label}"]`)].pop();
  if (target) target.classList.add("active");

  expandPdf();
  state.viewer.paperId = card.paper_id;
  state.viewer.boxes = card.trace || [];
  el("pdf-title").textContent = card.paper_title || card.paper_id || "原文";
  el("pdf-sub").textContent = `${card.label || ""} · ${card.section_title || ""}`;

  const first = state.viewer.boxes[0];
  const page = first && typeof first.page_idx === "number" ? first.page_idx : (card.page_start ?? 0);
  await showPage(page);
}

async function showPage(pageIndex) {
  const viewer = state.viewer;
  if (!viewer.paperId) return;

  if (!window.pdfjsLib) {
    el("pdf-empty").textContent =
      "原文预览需要 pdf.js（CDN）。离线环境下证据卡照常可用，坐标在接口的 trace 字段里。";
    el("pdf-empty").style.display = "block";
    el("pdf-wrap").style.display = "none";
    return;
  }

  // 这篇论文已经不在了（历史会话里的卡片会走到这儿）：不用去拿 PDF，直接说清楚。
  // 历史回答和引用照旧保留 —— 删论文不该改写历史记录，只是原文看不到了。
  if (!state.papers.some((paper) => paper.paper_id === viewer.paperId)) {
    el("pdf-empty").textContent = "这篇论文已删除，原文不可查看。";
    el("pdf-empty").style.display = "block";
    el("pdf-wrap").style.display = "none";
    return;
  }

  try {
    if (!viewer.doc || viewer.loaded !== viewer.paperId) {
      viewer.doc = await pdfjsLib.getDocument(`${API}/papers/${viewer.paperId}/file`).promise;
      viewer.loaded = viewer.paperId;
    }
  } catch (error) {
    // 本地列表是打开页面时拉的，可能已经过期（比如另一个标签页把它删了）—— 404 也认成"已删除"
    const gone = /404/.test(String(error && error.message));
    el("pdf-empty").textContent = gone
      ? "这篇论文已删除，原文不可查看。"
      : `拿不到原始 PDF（${error.message}）。入库时记的那个路径可能已经不在了。`;
    el("pdf-empty").style.display = "block";
    el("pdf-wrap").style.display = "none";
    return;
  }

  viewer.total = viewer.doc.numPages;
  viewer.page = Math.max(0, Math.min(pageIndex, viewer.total - 1));

  const page = await viewer.doc.getPage(viewer.page + 1);
  const viewport = page.getViewport({ scale: 1.6 });
  const canvas = el("pdf-canvas");
  canvas.width = viewport.width;
  canvas.height = viewport.height;
  await page.render({ canvasContext: canvas.getContext("2d"), viewport }).promise;

  el("pdf-wrap").style.display = "block";
  el("pdf-empty").style.display = "none";
  el("pdf-page").textContent = `${viewer.page + 1} / ${viewer.total}`;
  drawHighlights();
}

function drawHighlights() {
  const viewer = state.viewer;
  const overlay = el("pdf-overlay");
  overlay.innerHTML = "";
  const boxes = viewer.boxes.filter((item) => item.page_idx === viewer.page && item.bbox);
  for (const item of boxes) {
    const [x0, y0, x1, y1] = item.bbox;
    const node = document.createElement("div");
    node.className = "hl";
    node.style.left = `${(x0 / 1000) * 100}%`;
    node.style.top = `${(y0 / 1000) * 100}%`;
    node.style.width = `${((x1 - x0) / 1000) * 100}%`;
    node.style.height = `${((y1 - y0) / 1000) * 100}%`;
    overlay.appendChild(node);
  }
  if (boxes.length && overlay.firstChild) {
    overlay.firstChild.scrollIntoView({ block: "center" });
  }
}

// ---------------------------------------------------------------- 上传
const STEP_LABELS = { parse: "解析论文", index: "切分入库" };

/** 入库期间锁住提问：MinerU 要跑几分钟，向量也是最后才写 ——
    这中间提问只会得到"这篇论文不存在"的结果，用户会以为是系统坏了。 */
function setUploading(flag, note = "") {
  state.uploading = flag;
  el("ask-btn").disabled = flag;
  el("upload-btn").disabled = flag;
  el("file-input").disabled = flag;
  if (flag) {
    el("progress").classList.remove("hidden", "failed");
    el("progress-fill").style.width = "0%";
    el("progress-label").textContent = "排队中…";
    setStatus(note || "正在入库，完成后会自动选中这篇论文");
  }
}

/** 两段进度：MinerU 只在提交后轮询 pending/running/done，没有"解析了 37%"，
    所以进度只能按**步骤**算，当前这步算半格。 */
function renderProgress(job) {
  const steps = job.steps || [];
  const total = steps.length || 1;
  const done = steps.filter((step) => step.status === "done").length;
  const running = steps.some((step) => step.status === "running");
  const ratio = job.status === "failed" ? done / total : (done + (running ? 0.5 : 0)) / total;
  el("progress-fill").style.width = `${Math.round(ratio * 100)}%`;
  el("progress").classList.toggle("failed", job.status === "failed");

  const current = steps.find((step) => step.status === "running")
    || steps.find((step) => step.status === "queued");
  if (job.status === "done") el("progress-label").textContent = `入库完成（${total}/${total}）`;
  else if (job.status === "failed") el("progress-label").textContent = "入库失败";
  else el("progress-label").textContent =
    `${STEP_LABELS[current?.name] || current?.name || "排队中"}…（${done}/${total}）`;

  el("status").innerHTML = `<div class="steps">${steps.map((step) => {
    const cls = step.status === "done" ? "done" : step.status === "failed" ? "failed" : "";
    return `<span class="step ${cls}">${escapeHtml(STEP_LABELS[step.name] || step.name)} ${escapeHtml(step.status)}${step.detail ? ` · ${escapeHtml(step.detail)}` : ""}</span>`;
  }).join("")}</div>`;
}

async function uploadPaper() {
  const input = el("file-input");
  const file = input.files[0];
  if (!file) return;
  if (state.asking) {
    setStatus("等这一轮回答结束再上传（上传期间不让提问，两件事别叠在一起）", true);
    input.value = "";
    return;
  }

  setUploading(true, `已提交 ${file.name}，正在入库`);
  const form = new FormData();
  form.append("file", file);

  let task;
  try {
    const response = await fetch(`${API}/papers`, { method: "POST", body: form });
    task = await response.json();
    if (!response.ok) throw new Error(task.detail || `HTTP ${response.status}`);
  } catch (error) {
    setUploading(false);
    el("progress").classList.add("hidden");
    setStatus(`上传失败：${error.message}`, true);
    input.value = "";
    return;
  }

  // 解析要几分钟，所以轮询任务状态；进度显示在输入框下面
  const timer = setInterval(async () => {
    const job = await (await fetch(`${API}/tasks/${task.task_id}`)).json();
    renderProgress(job);

    if (job.status === "done") {
      clearInterval(timer);
      await Promise.all([loadHealth(), loadPapers()]);
      // 入库完**自动选中这篇**：否则范围还是"全库"，用户不问就不知道得手动勾。
      // 用 add 而不是整个替换 —— 用户可能先选了几篇做对比，上传新论文不该把选择清掉。
      state.selected.add(task.paper_id);
      renderScope();
      setUploading(false);
      const title = job.result.title || task.paper_id;
      setStatus(`《${title}》已入库（${job.result.chunks ?? 0} 个 chunk），已自动选中，可以直接提问`);
      setTimeout(() => el("progress").classList.add("hidden"), 4000);
      input.value = "";
    } else if (job.status === "failed") {
      clearInterval(timer);
      setUploading(false);
      setStatus(`入库失败：${job.error}`, true);
      setTimeout(() => el("progress").classList.add("hidden"), 8000);
      input.value = "";
    }
  }, 800);
}

// ---------------------------------------------------------------- 小工具
function truncate(text, limit) {
  const clean = String(text || "").replace(/\s+/g, " ").trim();
  return clean.length > limit ? `${clean.slice(0, limit)}…` : clean;
}

function escapeHtml(text) {
  return String(text ?? "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

boot();
