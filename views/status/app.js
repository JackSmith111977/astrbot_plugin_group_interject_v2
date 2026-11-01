/* Group Interject v2 control view.  Uses the AstrBot plugin view bridge.
 * The bridge SDK is injected AFTER this script by the host page, so never
 * capture window.AstrBotPluginView at module load; resolve it lazily. */
const BRIDGE_WAIT_MS = 5000;

function getBridge() {
  return window.AstrBotPluginView || null;
}

function bridgeUsable(candidate) {
  return Boolean(candidate && typeof candidate.ready === "function");
}

async function waitForBridge(timeoutMs = BRIDGE_WAIT_MS) {
  const started = Date.now();
  while (Date.now() - started < timeoutMs) {
    const candidate = getBridge();
    if (bridgeUsable(candidate)) return candidate;
    await new Promise((resolve) => window.setTimeout(resolve, 50));
  }
  return null;
}

const CONFIG_TABS = [
  {
    id: "runtime",
    label: "运行",
    sections: [
      { title: "开关与模式", keys: ["enable", "mode", "dry_run"] },
      { title: "模型", keys: ["provider_id"] },
      { title: "数据与隐私", keys: ["image_caption_enable", "memory_bridge_enable"] },
      { title: "自定义参数", keys: ["custom"] },
    ],
  },
  {
    id: "scope",
    label: "范围",
    sections: [
      { title: "群范围", keys: ["group_whitelist", "group_blacklist"] },
      { title: "发送者", keys: ["ignore_senders"] },
    ],
  },
  {
    id: "interaction",
    label: "交互",
    sections: [
      { title: "唤醒", keys: ["wake_names"] },
      { title: "静默", keys: ["quiet_hours"] },
      { title: "风格", keys: ["extra_rules"] },
    ],
  },
];

const state = {
  fields: [],
  values: {},
  providers: [],
  presets: {},
  groups: [],
  dirty: false,
  activeTab: "config",
  configTab: "runtime",
};

function esc(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
}

function seconds(value) {
  const n = Number(value);
  if (!Number.isFinite(n) || n <= 0) return "—";
  if (n < 60) return `${Math.round(n)}s`;
  if (n < 3600) return `${Math.round(n / 60)}m`;
  return `${(n / 3600).toFixed(1)}h`;
}

function setConn(ok, label) {
  const el = document.getElementById("conn");
  el.textContent = label;
  el.className = `chip ${ok ? "ok" : "bad"}`;
}

function toast(message, bad = false) {
  const el = document.getElementById("toast");
  el.textContent = message;
  el.className = `toast ${bad ? "bad" : "ok"}`;
  el.hidden = false;
  window.clearTimeout(toast.timer);
  toast.timer = window.setTimeout(() => {
    el.hidden = true;
  }, 3200);
}

function setDirty(value) {
  state.dirty = value;
  document.getElementById("dirty").hidden = !value;
}

async function ensureBridge() {
  const bridge = await waitForBridge();
  if (!bridge) {
    setConn(false, "PREVIEW / NO BRIDGE");
    return null;
  }
  try {
    await bridge.ready();
    return bridge;
  } catch (error) {
    setConn(false, "LINK FAILED");
    console.error(error);
    return null;
  }
}

// --------------------------------------------------------------- config

function getPath(source, path) {
  let current = source;
  for (const part of String(path || "").split(".")) {
    if (!current || typeof current !== "object" || !(part in current)) {
      return undefined;
    }
    current = current[part];
  }
  return current;
}

function fieldValue(field) {
  const value = getPath(state.values, field.key);
  return value ?? field.default;
}

function makeControl(field) {
  const value = fieldValue(field);
  if (field.type === "bool") {
    const wrap = document.createElement("label");
    wrap.className = "switch";
    const input = document.createElement("input");
    input.type = "checkbox";
    input.checked = Boolean(value);
    input.dataset.key = field.key;
    input.addEventListener("change", () => setDirty(true));
    const text = document.createElement("span");
    text.textContent = input.checked ? "开启" : "关闭";
    input.addEventListener("change", () => {
      text.textContent = input.checked ? "开启" : "关闭";
    });
    wrap.append(input, text);
    return wrap;
  }

  if (field.special === "select_provider") {
    const select = document.createElement("select");
    select.dataset.key = field.key;
    const empty = document.createElement("option");
    empty.value = "";
    empty.textContent = "跟随当前会话";
    select.append(empty);
    for (const provider of state.providers) {
      const option = document.createElement("option");
      option.value = provider.id;
      option.textContent = `${provider.name || provider.id}${provider.model ? ` / ${provider.model}` : ""}`;
      select.append(option);
    }
    select.value = String(value || "");
    select.addEventListener("change", () => setDirty(true));
    return select;
  }

  if (field.type === "list") {
    const textarea = document.createElement("textarea");
    textarea.dataset.key = field.key;
    textarea.rows = 3;
    textarea.placeholder = "每行一个，也可用逗号分隔";
    textarea.value = Array.isArray(value) ? value.join("\n") : String(value || "");
    textarea.addEventListener("input", () => setDirty(true));
    return textarea;
  }

  if (field.options && field.options.length) {
    const select = document.createElement("select");
    select.dataset.key = field.key;
    for (const option of field.options) {
      const item = document.createElement("option");
      item.value = String(option);
      item.textContent = String(option);
      select.append(item);
    }
    select.value = String(value ?? "");
    select.addEventListener("change", () => setDirty(true));
    return select;
  }

  if (field.type === "int" || field.type === "float") {
    const input = document.createElement("input");
    input.type = "number";
    input.dataset.key = field.key;
    if (field.min !== undefined) input.min = String(field.min);
    if (field.max !== undefined) input.max = String(field.max);
    input.step =
      field.step !== undefined
        ? String(field.step)
        : field.type === "float"
          ? "0.01"
          : "1";
    input.value = String(value ?? "");
    input.addEventListener("input", () => setDirty(true));
    return input;
  }

  if (field.key === "extra_rules" || field.type === "text") {
    const textarea = document.createElement("textarea");
    textarea.dataset.key = field.key;
    textarea.rows = 4;
    textarea.placeholder = "例如：多用短句、不要用感叹号";
    textarea.value = String(value || "");
    textarea.addEventListener("input", () => setDirty(true));
    return textarea;
  }

  const input = document.createElement("input");
  input.type = field.key === "quiet_hours" ? "text" : "text";
  input.dataset.key = field.key;
  input.placeholder = field.key === "quiet_hours" ? "01:00-08:00" : "";
  input.value = String(value ?? "");
  input.addEventListener("input", () => setDirty(true));
  return input;
}

function fillCustomFromPreset(details, field) {
  const modeField = document.querySelector('[data-key="mode"]');
  const mode = String(modeField?.value || state.values.mode || "balanced");
  const preset = state.presets[mode];
  if (!preset) {
    toast(`没有 ${mode} 档位参数可填充`, true);
    return;
  }
  for (const item of field.items || []) {
    if (item.key === "enabled") continue;
    const control = details.querySelector(
      `[data-key="${field.key}.${item.key}"]`,
    );
    if (control && preset[item.key] !== undefined) {
      control.value = String(preset[item.key]);
    }
  }
  setDirty(true);
}

function setCustomControlsEnabled(details, enabled) {
  details.classList.toggle("is-disabled", !enabled);
  for (const control of details.querySelectorAll("[data-key]")) {
    if (control.dataset.key === "custom.enabled") continue;
    control.disabled = !enabled;
  }
}

function renderObjectField(field) {
  const details = document.createElement("details");
  details.className = "custom-tuning";
  const enabled = Boolean(getPath(state.values, `${field.key}.enabled`));
  details.open = enabled;

  const summary = document.createElement("summary");
  summary.textContent = "自定义参数（可选覆盖当前档位）";
  details.append(summary);

  const body = document.createElement("div");
  body.className = "custom-body";

  const enabledField = (field.items || []).find((item) => item.key === "enabled");
  if (enabledField) {
    const row = renderField({
      ...enabledField,
      key: `${field.key}.enabled`,
    });
    body.append(row);
    const checkbox = row.querySelector("input[type='checkbox']");
    if (checkbox) {
      checkbox.addEventListener("change", () => {
        if (checkbox.checked) fillCustomFromPreset(details, field);
        setCustomControlsEnabled(details, checkbox.checked);
      });
    }
  }

  const direction = document.createElement("p");
  direction.className = "custom-direction";
  direction.textContent =
    "更活跃：降低最小消息数/冷却/置信度，提高话题轮次；更沉默则反向调整。";
  body.append(direction);

  const grid = document.createElement("div");
  grid.className = "fields custom-fields";
  for (const item of field.items || []) {
    if (item.key === "enabled") continue;
    grid.append(
      renderField({
        ...item,
        key: `${field.key}.${item.key}`,
      }),
    );
  }
  body.append(grid);

  const actions = document.createElement("div");
  actions.className = "custom-actions";
  const fill = document.createElement("button");
  fill.type = "button";
  fill.className = "btn btn-ghost";
  fill.textContent = "用当前档位填充";
  fill.addEventListener("click", () => fillCustomFromPreset(details, field));
  actions.append(fill);
  body.append(actions);
  if (field.hint) {
    const hint = document.createElement("div");
    hint.className = "hint";
    hint.textContent = field.hint;
    body.append(hint);
  }

  details.append(body);
  setCustomControlsEnabled(details, enabled);
  return details;
}

function renderField(field) {
  if (field.type === "object") {
    return renderObjectField(field);
  }
  const row = document.createElement("div");
  row.className = "field";
  const label = document.createElement("label");
  label.textContent = field.description || field.key;
  const control = makeControl(field);
  const hint = document.createElement("div");
  hint.className = "hint";
  hint.textContent = field.hint || "";
  row.append(label, control);
  if (field.hint) row.append(hint);
  return row;
}

function renderConfig() {
  const fields = new Map(state.fields.map((field) => [field.key, field]));
  const tabsHost = document.getElementById("config-tabs");
  const panesHost = document.getElementById("config-panes");
  tabsHost.innerHTML = "";
  panesHost.innerHTML = "";

  const knownTab = CONFIG_TABS.some((tab) => tab.id === state.configTab);
  if (!knownTab) state.configTab = CONFIG_TABS[0].id;

  for (const tab of CONFIG_TABS) {
    const active = tab.id === state.configTab;
    const button = document.createElement("button");
    button.type = "button";
    button.className = `config-tab${active ? " active" : ""}`;
    button.dataset.configTab = tab.id;
    button.setAttribute("role", "tab");
    button.setAttribute("aria-selected", String(active));
    button.textContent = tab.label;
    button.addEventListener("click", () => switchConfigTab(tab.id));
    tabsHost.append(button);

    const pane = document.createElement("div");
    pane.className = "config-pane";
    pane.dataset.configPane = tab.id;
    pane.hidden = !active;
    for (const section of tab.sections) {
      const sectionNode = document.createElement("section");
      sectionNode.className = "config-section";
      if (section.keys.includes("custom")) {
        sectionNode.classList.add("config-section--wide");
      }
      const heading = document.createElement("h3");
      heading.textContent = section.title;
      const grid = document.createElement("div");
      grid.className = "fields";
      for (const key of section.keys) {
        const field = fields.get(key);
        if (field) grid.append(renderField(field));
      }
      sectionNode.append(heading, grid);
      pane.append(sectionNode);
    }
    panesHost.append(pane);
  }
}

function switchConfigTab(tabId) {
  state.configTab = tabId;
  for (const button of document.querySelectorAll("[data-config-tab]")) {
    const active = button.dataset.configTab === tabId;
    button.classList.toggle("active", active);
    button.setAttribute("aria-selected", String(active));
  }
  for (const pane of document.querySelectorAll("[data-config-pane]")) {
    pane.hidden = pane.dataset.configPane !== tabId;
  }
}

function collectField(field) {
  if (field.type === "object") {
    const result = {};
    for (const item of field.items || []) {
      result[item.key] = collectField({
        ...item,
        key: `${field.key}.${item.key}`,
      });
    }
    return result;
  }
  const el = document.querySelector(`[data-key="${field.key}"]`);
  if (!el) return field.default ?? null;
  if (field.type === "bool") return Boolean(el.checked);
  if (field.type === "int" || field.type === "float") {
    const parsed = Number(el.value);
    return Number.isFinite(parsed) ? parsed : field.default;
  }
  if (field.type === "list") {
    return String(el.value || "")
      .split(/[\n,，;；]+/)
      .map((item) => item.trim())
      .filter(Boolean);
  }
  return String(el.value ?? "").trim();
}

function collectConfig() {
  const payload = {};
  for (const field of state.fields) {
    payload[field.key] = collectField(field);
  }
  return payload;
}

async function loadConfig() {
  const bridge = await ensureBridge();
  if (!bridge) return;
  try {
    const data = await bridge.apiGet("config");
    state.fields = data.fields || [];
    state.values = data.values || {};
    state.providers = data.providers || [];
    state.presets = data.presets || {};
    renderConfig();
    setDirty(false);
    setConn(true, "CONFIG LINK");
    updateHeader(data.runtime || {});
  } catch (error) {
    setConn(false, "CONFIG FAILED");
    toast(`读取配置失败：${error}`, true);
  }
}

async function saveConfig() {
  const bridge = await ensureBridge();
  if (!bridge) return;
  const button = document.getElementById("save");
  button.disabled = true;
  try {
    const result = await bridge.apiPost("config", collectConfig());
    if (result && result.status === "error") {
      throw new Error(result.message || "保存失败");
    }
    toast("配置已保存并即时生效");
    setDirty(false);
    await loadConfig();
    await loadStats();
  } catch (error) {
    toast(`保存失败：${error.message || error}`, true);
  } finally {
    button.disabled = false;
  }
}

function updateHeader(runtime) {
  const mode = runtime.mode || state.values.mode || "—";
  const customEnabled =
    runtime.custom_enabled ?? Boolean(getPath(state.values, "custom.enabled"));
  const dry = runtime.dry_run ?? state.values.dry_run;
  document.getElementById("mode-chip").textContent = customEnabled
    ? `MODE ${mode} + CUSTOM`
    : `MODE ${mode}`;
  document.getElementById("dry-chip").textContent = dry ? "DRY RUN" : "LIVE";
  document.getElementById("mode-chip").className = "chip";
  document.getElementById("dry-chip").className = `chip ${dry ? "warn" : "ok"}`;
}

// --------------------------------------------------------------- stats

function renderSummary(data) {
  const groups = state.groups;
  const cold = groups.reduce((sum, group) => sum + Number(group.cold_summaries || 0), 0);
  const pending = groups.reduce((sum, group) => sum + Number(group.pending_spill || 0), 0);
  const calls = groups.reduce((sum, group) => sum + Number(group.summary_calls || 0), 0);
  document.getElementById("summary").innerHTML = [
    [
      "模式",
      `${data.mode || "—"}${data.custom_enabled ? " + CUSTOM" : ""} / ${
        data.dry_run ? "DRY" : "LIVE"
      }`,
    ],
    ["群数", groups.length],
    ["COLD 摘要", cold],
    ["PENDING SPILL", pending],
    ["摘要调用", calls],
  ]
    .map(
      ([key, value]) =>
        `<div class="stat"><div class="k">${esc(key)}</div><div class="v">${esc(value)}</div></div>`,
    )
    .join("");
}

function decisionRows(group, limit = 4) {
  const rows = (group.recent_decisions || []).slice(-limit).reverse();
  if (!rows.length) return '<div class="recent-row muted"><span>—</span><span>—</span><span>暂无决策</span></div>';
  return rows
    .map(
      (item) =>
        `<div class="recent-row"><span>${esc(seconds(item.ago_seconds))}前</span><span>${esc(item.result || "—")}</span><span class="text">${esc(item.reason || item.text || "—")}</span></div>`,
    )
    .join("");
}

function renderGroups() {
  const host = document.getElementById("groups");
  const select = document.getElementById("umo");
  const selected = select.value;
  select.innerHTML = state.groups
    .map((group) => {
      const name = group.group_name || group.group_id || group.umo;
      return `<option value="${esc(group.umo)}">${esc(name)}</option>`;
    })
    .join("");
  if (selected) select.value = selected;
  if (!state.groups.length) {
    host.innerHTML = '<div class="empty">还没有群窗口。等群里出现消息后刷新。</div>';
    return;
  }
  host.innerHTML = state.groups
    .map((group) => {
      const cold = Number(group.cold_summaries || 0);
      const summaryAge = group.last_summary_ago_seconds;
      const summaryStatus = group.last_summary_status || "—";
      const turns = Number(group.episode_turns || 0);
      const burst = Number(group.episode_burst_limit || 0);
      const extra = Number(group.episode_extra_turns || 0);
      const episodeText =
        extra > 0
          ? `episode <b>${turns}</b><span class="muted"> (+${extra} slow)</span>`
          : `episode <b>${turns}</b>/${burst || turns + Number(group.episode_remaining_turns || 0)}`;
      const slowText =
        extra > 0 || group.episode_slow
          ? `<span>slow <b>${seconds(group.episode_next_extra_in_seconds)}</b> / <b>${Number(group.episode_reentry_messages_remaining || 0)}</b> 条</span>`
          : "";
      return `<article class="group">
        <div class="group-head">
          <div>
            <div class="group-name">${esc(group.group_name || group.group_id || group.umo)}</div>
            <div class="group-meta">${esc(group.umo)}</div>
          </div>
          <span class="state ${esc(group.state)}">${esc(group.state)}</span>
        </div>
        <div class="facts">
          <span>hot <b>${esc(group.window_messages)}</b>/${esc(group.window_capacity)}</span>
          <span>${episodeText}</span>
          ${slowText}
          <span>cold <b>${esc(cold)}</b> 条 / <b>${esc(group.summary_chars || 0)}</b> 字</span>
          <span>pending spill <b>${esc(group.pending_spill || 0)}</b></span>
          <span>摘要 <b>${esc(summaryStatus)}</b>${summaryAge ? ` · ${esc(seconds(summaryAge))}前` : ""}</span>
          <span>冷却 <b>${esc(seconds(group.cooldown_remaining_seconds))}</b></span>
          <span>抑制 <b>${esc(seconds(group.suppressed_remaining_seconds))}</b></span>
          <span>agent busy <b>${group.agent_busy ? "是" : "否"}</b></span>
        </div>
        <div class="recent">${decisionRows(group)}</div>
      </article>`;
    })
    .join("");
}

async function loadStats() {
  const bridge = await ensureBridge();
  if (!bridge) return;
  try {
    const data = await bridge.apiGet("stats");
    state.groups = data.groups || [];
    renderSummary(data);
    renderGroups();
    document.getElementById("server-time").textContent = data.server_time || "";
    updateHeader(data);
    setConn(true, "LINK ESTABLISHED");
  } catch (error) {
    setConn(false, "STATS FAILED");
    console.error(error);
  }
}

async function runProbe() {
  const bridge = await ensureBridge();
  if (!bridge) return;
  const umo = document.getElementById("umo").value;
  if (!umo) return;
  const output = document.getElementById("probe-output");
  output.textContent = "判定中…";
  try {
    const result = await bridge.apiPost("probe", { umo });
    output.textContent = JSON.stringify(result, null, 2);
  } catch (error) {
    output.textContent = `试判失败：${error}`;
  }
}

// --------------------------------------------------------------- tabs

function switchTab(tab) {
  state.activeTab = tab;
  for (const button of document.querySelectorAll("[data-tab]")) {
    button.classList.toggle("active", button.dataset.tab === tab);
  }
  document.getElementById("panel-config").hidden = tab !== "config";
  document.getElementById("panel-monitor").hidden = tab !== "monitor";
  if (tab === "monitor") loadStats();
}

document.getElementById("tabs").addEventListener("click", (event) => {
  const button = event.target.closest("[data-tab]");
  if (button) switchTab(button.dataset.tab);
});
document.getElementById("refresh").addEventListener("click", () => {
  loadStats();
  if (state.activeTab === "config") loadConfig();
});
document.getElementById("save").addEventListener("click", saveConfig);
document.getElementById("reset").addEventListener("click", loadConfig);
document.getElementById("probe").addEventListener("click", runProbe);

async function init() {
  await loadConfig();
  await loadStats();
  window.setInterval(loadStats, 15000);
}

if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", init);
} else {
  init();
}
