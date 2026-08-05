/* QuantDesk dashboard — vanilla, no dependencies, no build step.
 *
 * There is no HTMX here either, and that is the same decision as the charts:
 * the parts of it this page needs are a fetch and an innerHTML swap, roughly
 * forty lines below. Vendoring a library for that would mean third-party code
 * in the control plane of a trading system, and a Content-Security-Policy that
 * has to make an exception. Everything here is `script-src 'self'`.
 *
 * The page is fully rendered by the server. Nothing below is required to read
 * the dashboard — with JavaScript off you lose live updates, the palette and
 * the crosshair, and keep every number.
 */
(() => {
  "use strict";

  const $  = (sel, root = document) => root.querySelector(sel);
  const $$ = (sel, root = document) => Array.from(root.querySelectorAll(sel));
  const CSRF = document.body?.dataset.csrf || "";

  /* ── toasts ──────────────────────────────────────────────────────────── */

  let lastToast = "";

  function toast(title, detail, kind = "ok", ttl = 6000) {
    const host = $("#toasts");
    if (!host) return;
    // Five identical criticals in a row is one problem, not five notices.
    // Without this a repeating event buries the page under its own alarm.
    const key = `${title}|${detail}`;
    if (key === lastToast && host.children.length) return;
    lastToast = key;
    while (host.children.length >= 4) host.firstElementChild.remove();
    const el = document.createElement("div");
    el.className = `toast toast--${kind}`;
    const b = document.createElement("b");
    b.textContent = title;
    el.appendChild(b);
    if (detail) {
      const s = document.createElement("small");
      s.textContent = detail;
      el.appendChild(s);
    }
    host.appendChild(el);
    setTimeout(() => {
      el.classList.add("is-going");
      setTimeout(() => el.remove(), 300);
    }, ttl);
  }

  /* ── confirm modal ───────────────────────────────────────────────────── */

  let confirmResolve = null;

  function confirmDialog(message, typedWord) {
    const modal = $("#confirm");
    if (!modal) return Promise.resolve(true);
    $("#confirm-body").textContent = message;
    const typed = $("#confirm-typed");
    const input = $("#confirm-input");
    if (typedWord) {
      typed.hidden = false;
      $("#confirm-word").textContent = typedWord;
      input.value = "";
    } else {
      typed.hidden = true;
    }
    modal.hidden = false;
    (typedWord ? input : $("#confirm-go")).focus();
    return new Promise((resolve) => {
      confirmResolve = (ok) => {
        // A typed confirmation must match exactly. The same discipline as
        // `qd arm-live`: a destructive action should cost a sentence.
        if (ok && typedWord && input.value.trim() !== typedWord) {
          toast("Not confirmed", `Type ${typedWord} exactly.`, "warn");
          return;
        }
        modal.hidden = true;
        confirmResolve = null;
        resolve(ok);
      };
    });
  }

  $$("[data-close-modal]").forEach((el) =>
    el.addEventListener("click", () => confirmResolve && confirmResolve(false)));
  $("#confirm-go")?.addEventListener("click", () => confirmResolve && confirmResolve(true));
  $("#confirm-input")?.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && confirmResolve) confirmResolve(true);
  });

  /* ── control actions ─────────────────────────────────────────────────── */

  async function sendControl(command, args = {}) {
    const res = await fetch(`/control/${command}`, {
      method: "POST",
      headers: { "Content-Type": "application/json", "X-CSRF-Token": CSRF },
      body: JSON.stringify(args),
    });
    let body = {};
    try { body = await res.json(); } catch { /* non-JSON error page */ }
    if (!res.ok) {
      toast("Refused", body.error || `${res.status}`, "error", 9000);
      return;
    }
    // "Queued", never "done": the engine applies it on its next cycle, and
    // saying otherwise would be a lie told at the worst possible moment.
    toast(`Queued ${command}`,
          body.note || `Command #${body.id}. The engine applies it next cycle.`,
          body.note ? "warn" : "ok", body.note ? 12000 : 6000);
    setTimeout(refreshAll, 1200);
  }

  document.addEventListener("click", async (e) => {
    const btn = e.target.closest("[data-control]");
    if (!btn) return;
    e.preventDefault();
    const command = btn.dataset.control;
    if (command === "disarm-note") return;
    const ask = btn.dataset.confirm;
    if (ask && !(await confirmDialog(ask, btn.dataset.typed))) return;
    const args = {};
    if (btn.dataset.symbol) args.symbol = btn.dataset.symbol;
    if (btn.dataset.strategy) args.strategy = btn.dataset.strategy;
    if (btn.dataset.eventId) args.id = btn.dataset.eventId;
    sendControl(command, args);
  });

  /* ── partial refresh (the small part of HTMX this page actually uses) ── */

  async function refreshPanel(el) {
    const url = el.dataset.refresh;
    if (!url) return;
    try {
      const res = await fetch(url, { headers: { "X-Partial": "1" } });
      if (!res.ok) return;
      const html = await res.text();
      // Skip the swap when nothing changed: a table that re-renders every
      // 20 seconds loses your text selection and scroll position for nothing.
      if (html.trim() === el.dataset.lastHtml) return;
      el.dataset.lastHtml = html.trim();
      el.innerHTML = html;
      bindCharts(el);
      // The swapped-in markup carries server-rendered placeholders; refresh
      // them now rather than leaving the clock reading `--:--:--` for up to a
      // second after every status-bar update.
      tickClock();
      retime();
    } catch { /* offline; the next tick tries again */ }
  }

  function refreshAll() {
    $$("[data-refresh]").forEach(refreshPanel);
  }

  $$("[data-refresh]").forEach((el) => {
    const every = parseInt(el.dataset.every || "30", 10) * 1000;
    setInterval(() => {
      if (!document.hidden) refreshPanel(el);
    }, every);
  });

  /* ── SSE ─────────────────────────────────────────────────────────────── */

  const liveEl = $("[data-live]");
  let stream = null;
  let backoff = 1000;

  function setLive(on) {
    if (!liveEl) return;
    liveEl.classList.toggle("is-on", on);
    liveEl.classList.toggle("is-off", !on);
    $(".live-text", liveEl).textContent = on ? "live" : "offline";
  }

  function connect() {
    if (!window.EventSource) return;
    stream = new EventSource("/stream");

    stream.addEventListener("open", () => { setLive(true); backoff = 1000; });

    stream.addEventListener("qd", (e) => {
      let ev;
      try { ev = JSON.parse(e.data); } catch { return; }
      prependEvent(ev);
      if (ev.severity === "critical") {
        toast(ev.type.split(".").pop().replace(/_/g, " "),
              JSON.stringify(ev.payload).slice(0, 160), "error", 15000);
      }
      if (/order_filled|position_(opened|closed)/.test(ev.type)) {
        refreshAll();
      }
    });

    stream.addEventListener("state", () => {
      const bar = $("#statusbar");
      if (bar) refreshPanel(bar);
    });

    stream.addEventListener("error", () => {
      setLive(false);
      stream.close();
      // Backoff, capped. A dashboard hammering a database that is already
      // struggling is a way to make an outage worse.
      backoff = Math.min(backoff * 2, 30000);
      setTimeout(connect, backoff);
    });
  }

  function prependEvent(ev) {
    const rows = $("#event-rows");
    const follow = $("#follow");
    if (!rows || (follow && !follow.checked)) return;
    const tr = document.createElement("tr");
    tr.className = ev.severity === "critical" ? "sev-critical"
                 : ev.severity === "warning" ? "sev-warn" : "";
    const cells = [
      ["r num dim", String(ev.seq)],
      ["dim", "just now"],
      ["", ""],
      ["dim", ev.source],
      ["ev-payload num", JSON.stringify(ev.payload)],
      ["r", ""],
    ];
    cells.forEach(([cls, text], i) => {
      const td = document.createElement("td");
      td.className = cls;
      if (i === 2) {
        const b = document.createElement("b");
        b.textContent = ev.type.split(".").pop().replace(/_/g, " ");
        const g = document.createElement("span");
        g.className = "tl-group";
        g.textContent = ev.type.split(".")[0];
        td.append(b, g);
      } else if (i === 5 && ev.correlation_id) {
        const a = document.createElement("a");
        a.className = "cell-link";
        a.href = `/trades/${ev.correlation_id}`;
        a.textContent = "chain →";
        td.appendChild(a);
      } else {
        td.textContent = text;
      }
      tr.appendChild(td);
    });
    tr.classList.add("flash-up");
    rows.prepend(tr);
    while (rows.children.length > 300) rows.lastElementChild.remove();
    const counter = $("#event-count");
    if (counter) counter.textContent = rows.children.length;
  }

  /* ── chart crosshair ─────────────────────────────────────────────────── */

  function bindCharts(root = document) {
    $$(".chart-hit", root).forEach((hit) => {
      let points;
      try { points = JSON.parse(hit.dataset.points || "[]"); } catch { return; }
      if (!points.length) return;
      const svg = hit.closest("svg");
      const cross = $(".crosshair", svg);
      const tip = $("#tooltip");

      hit.addEventListener("mousemove", (e) => {
        const box = svg.getBoundingClientRect();
        const vb = svg.viewBox.baseVal;
        const x = ((e.clientX - box.left) / box.width) * vb.width;
        let best = points[0], dist = Infinity;
        for (const p of points) {
          const d = Math.abs(p.x - x);
          if (d < dist) { dist = d; best = p; }
        }
        if (cross) {
          cross.hidden = false;
          $(".crosshair-line", cross).setAttribute("x1", best.x);
          $(".crosshair-line", cross).setAttribute("x2", best.x);
          $(".crosshair-dot", cross).setAttribute("cx", best.x);
          $(".crosshair-dot", cross).setAttribute("cy", best.y);
        }
        if (tip) {
          tip.hidden = false;
          tip.innerHTML = "";
          const b = document.createElement("b");
          b.textContent = best.v.toLocaleString(undefined,
            { minimumFractionDigits: 2, maximumFractionDigits: 2 });
          const s = document.createElement("small");
          s.textContent = `${best.t.slice(0, 16)}  ·  dd ${(best.d * 100).toFixed(2)}%`;
          tip.append(b, s);
          const tw = tip.offsetWidth || 160;
          tip.style.left = `${Math.min(e.clientX + 14, window.innerWidth - tw - 12)}px`;
          tip.style.top = `${e.clientY - 46}px`;
        }
      });

      hit.addEventListener("mouseleave", () => {
        if (cross) cross.hidden = true;
        const tip = $("#tooltip");
        if (tip) tip.hidden = true;
      });
    });
  }

  /* ── command palette ─────────────────────────────────────────────────── */

  const COMMANDS = [
    { label: "Home",            hint: "g h", run: () => (location.href = "/") },
    { label: "Overview",        hint: "g o", run: () => (location.href = "/overview") },
    { label: "Positions & orders", hint: "g p", run: () => (location.href = "/positions") },
    { label: "Trades",          hint: "g t", run: () => (location.href = "/trades") },
    { label: "Strategies",      hint: "g s", run: () => (location.href = "/strategies") },
    { label: "Risk",            hint: "g r", run: () => (location.href = "/risk") },
    { label: "System",          hint: "g y", run: () => (location.href = "/system") },
    { label: "Activity",        hint: "g a", run: () => (location.href = "/activity") },
    { label: "HALT — stop opening positions", hint: "danger", run: async () => {
        if (await confirmDialog(
          "Halt the engine? New positions stop; existing stops and exits keep working."))
          sendControl("halt");
      } },
    { label: "Resume trading", hint: "", run: async () => {
        if (await confirmDialog("Resume opening positions?")) sendControl("resume");
      } },
    { label: "Request flatten all", hint: "danger", run: async () => {
        if (await confirmDialog(
          "Close every open position at market? The engine submits the closes on its next cycle and reports FLATTEN_COMPLETE when the book is flat.",
          "FLATTEN")) sendControl("flatten_all");
      } },
    { label: "Refresh every panel", hint: "", run: refreshAll },
    { label: "Sign out", hint: "", run: () => {
        const f = document.createElement("form");
        f.method = "post"; f.action = "/logout";
        document.body.appendChild(f); f.submit();
      } },
  ];

  const palette = $("#palette");
  const palInput = $("#palette-input");
  const palList = $("#palette-list");
  let palSel = 0, palShown = COMMANDS;

  function renderPalette(filter = "") {
    const needle = filter.toLowerCase().trim();
    palShown = COMMANDS.filter((c) => c.label.toLowerCase().includes(needle));
    palSel = 0;
    palList.innerHTML = "";
    palShown.forEach((c, i) => {
      const li = document.createElement("li");
      li.className = "palette-item" + (i === 0 ? " is-sel" : "");
      li.textContent = c.label;
      if (c.hint) {
        const s = document.createElement("small");
        s.textContent = c.hint;
        li.appendChild(s);
      }
      li.addEventListener("click", () => { closePalette(); c.run(); });
      palList.appendChild(li);
    });
  }

  function openPalette() {
    if (!palette) return;
    palette.hidden = false;
    palInput.value = "";
    renderPalette();
    palInput.focus();
  }

  function closePalette() { if (palette) palette.hidden = true; }

  function movePalette(delta) {
    const items = $$(".palette-item", palList);
    if (!items.length) return;
    items[palSel]?.classList.remove("is-sel");
    palSel = (palSel + delta + items.length) % items.length;
    items[palSel].classList.add("is-sel");
    items[palSel].scrollIntoView({ block: "nearest" });
  }

  palInput?.addEventListener("input", () => renderPalette(palInput.value));
  palInput?.addEventListener("keydown", (e) => {
    if (e.key === "ArrowDown") { e.preventDefault(); movePalette(1); }
    else if (e.key === "ArrowUp") { e.preventDefault(); movePalette(-1); }
    else if (e.key === "Enter") {
      e.preventDefault();
      const cmd = palShown[palSel];
      closePalette();
      cmd?.run();
    }
  });
  $$("[data-close-palette]").forEach((el) => el.addEventListener("click", closePalette));
  $$("[data-open-palette]").forEach((el) => el.addEventListener("click", openPalette));

  /* ── keyboard ────────────────────────────────────────────────────────── */

  let chord = null, chordTimer = null;
  const GOTO = { h: "/", o: "/overview", p: "/positions", t: "/trades",
                 s: "/strategies", r: "/risk", y: "/system", a: "/activity" };

  document.addEventListener("keydown", (e) => {
    const typing = /^(INPUT|TEXTAREA|SELECT)$/.test(e.target.tagName);

    if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === "k") {
      e.preventDefault();
      palette?.hidden ? openPalette() : closePalette();
      return;
    }
    if (e.key === "Escape") {
      closePalette();
      if (confirmResolve) confirmResolve(false);
      return;
    }
    if (typing) return;

    if (chord === "g" && GOTO[e.key]) {
      location.href = GOTO[e.key];
      chord = null;
      return;
    }
    if (e.key === "g") {
      chord = "g";
      clearTimeout(chordTimer);
      chordTimer = setTimeout(() => (chord = null), 1200);
      return;
    }
    if (e.key === "?") { openPalette(); return; }
    if (e.key === "r") { refreshAll(); toast("Refreshed", "", "ok", 1800); }
  });

  /* ── clock & relative times ──────────────────────────────────────────── */

  function tickClock() {
    const el = $("[data-clock]");
    if (el) el.textContent = new Date().toISOString().slice(11, 19) + "Z";
  }

  function retime() {
    const now = Date.now();
    $$("time.when").forEach((el) => {
      const t = Date.parse(el.getAttribute("datetime"));
      if (Number.isNaN(t)) return;
      const s = Math.max(0, (now - t) / 1000);
      el.textContent =
        s < 10 ? "just now" :
        s < 60 ? `${Math.floor(s)}s ago` :
        s < 3600 ? `${Math.floor(s / 60)}m ago` :
        s < 86400 ? `${Math.floor(s / 3600)}h ago` :
        `${Math.floor(s / 86400)}d ago`;
    });
  }

  /* ── boot ────────────────────────────────────────────────────────────── */

  // Remember whether the Details nav group was left open. Only ever opens:
  // the server renders it open when the current page lives inside it, and
  // this must not fight that.
  const adv = document.getElementById("nav-advanced");
  if (adv) {
    if (localStorage.getItem("nav-adv") === "open") adv.open = true;
    adv.addEventListener("toggle", () =>
      localStorage.setItem("nav-adv", adv.open ? "open" : "closed"));
  }

  bindCharts();
  tickClock();
  setInterval(tickClock, 1000);
  setInterval(retime, 15000);
  connect();
})();
