(function () {
  "use strict";

  var reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  /* ------------------------------------------------------------------
   * Tilt-series viewer: three frame-locked canvases driven by one clock.
   * Each stack is a sprite sheet (cols x rows of w x h frames).
   * ------------------------------------------------------------------ */
  function initViewer(root) {
    var n = +root.dataset.frames, cols = +root.dataset.cols;
    var fw = +root.dataset.w, fh = +root.dataset.h;
    var start = +root.dataset.start, step = +root.dataset.step;
    var canvases = [].slice.call(root.querySelectorAll("canvas"));
    var range = root.querySelector("input[type=range]");
    var out = root.querySelector(".angle");
    var needle = root.querySelector(".gauge-needle");
    var playBtn = root.querySelector(".play");
    var sheets = [], loaded = 0;
    var frame = Math.round((0 - start) / step); // start at 0°
    var dir = 1, playing = !reduceMotion, visible = false, timer = null;
    var FRAME_MS = 120, HOLD_MS = 600;

    function draw() {
      if (loaded < canvases.length) return;
      var sx = (frame % cols) * fw, sy = Math.floor(frame / cols) * fh;
      canvases.forEach(function (c, i) {
        c.getContext("2d").drawImage(sheets[i], sx, sy, fw, fh, 0, 0, fw, fh);
      });
    }

    function render() {
      var a = start + frame * step;
      range.value = frame;
      range.style.setProperty("--fill", (frame / (n - 1)) * 100 + "%");
      out.textContent = (a > 0 ? "+" : a < 0 ? "−" : "") + Math.abs(a) + "°";
      range.setAttribute("aria-valuetext", a + " degrees");
      if (needle) needle.setAttribute("transform", "rotate(" + a + ")");
      draw();
    }

    function tick() {
      var next = frame + dir, delay = FRAME_MS;
      if (next < 0 || next > n - 1) { dir = -dir; next = frame + dir; }
      frame = next;
      if (frame === 0 || frame === n - 1) delay = HOLD_MS;
      render();
      timer = setTimeout(tick, delay);
    }

    function sync() {
      clearTimeout(timer); timer = null;
      root.classList.toggle("paused", !playing);
      playBtn.setAttribute("aria-label", playing ? "Pause" : "Play");
      if (playing && visible && loaded === canvases.length) timer = setTimeout(tick, FRAME_MS);
    }

    canvases.forEach(function (c, i) {
      var img = new Image();
      img.decoding = "async";
      img.onload = function () {
        loaded++;
        if (loaded === canvases.length) { root.classList.add("ready"); render(); sync(); }
      };
      img.src = c.dataset.src;
      sheets[i] = img;
    });

    range.addEventListener("input", function () {
      frame = +range.value; playing = false; render(); sync();
    });
    playBtn.addEventListener("click", function () { playing = !playing; sync(); });

    root.tabIndex = 0;
    root.addEventListener("keydown", function (e) {
      if (e.target === range && (e.key === "ArrowLeft" || e.key === "ArrowRight")) return; // native
      if (e.key === "ArrowLeft" || e.key === "ArrowRight") {
        frame = Math.min(n - 1, Math.max(0, frame + (e.key === "ArrowRight" ? 1 : -1)));
        playing = false; render(); sync(); e.preventDefault();
      } else if (e.key === " " && e.target.tagName !== "BUTTON") {
        playing = !playing; sync(); e.preventDefault();
      }
    });

    if ("IntersectionObserver" in window) {
      new IntersectionObserver(function (es) { visible = es[0].isIntersecting; sync(); }, { threshold: 0.15 }).observe(root);
    } else { visible = true; }
    render(); sync();
  }

  /* ------------------------------------------------------------------
   * Results: F1-vs-runtime scatter + table, driven by dataset/prompt filters.
   * ------------------------------------------------------------------ */
  var METHODS = {
    FullTilt:  { label: "FullTilt (ours)", color: "var(--c-fulltilt)", group: "cryo" },
    TomoTwin:  { label: "TomoTwin",        color: "var(--c-tomotwin)", group: "cryo" },
    ProPicker: { label: "ProPicker",       color: "var(--c-propicker)", group: "cryo" },
    CryoSAM:   { label: "CryoSAM",         color: "var(--c-cryosam)", group: "cryo" },
    Zeng:      { label: "Zeng et al.†", color: "var(--c-zeng)", group: "cryo" },
    DETR3D:    { label: "DETR3D†", group: "mv" },
    PETR:      { label: "PETR†",   group: "mv" }
  };
  var CHART_ORDER = ["Zeng", "CryoSAM", "ProPicker", "TomoTwin", "FullTilt"];
  var TABLE_ORDER = ["DETR3D", "PETR", "Zeng", "TomoTwin", "CryoSAM", "ProPicker", "FullTilt"];
  var DS_NAME = { czii: "CZII", "10304": "EMPIAR-10304", "10499": "EMPIAR-10499" };
  var SVGNS = "http://www.w3.org/2000/svg";

  function el(tag, attrs, parent, text) {
    var e = document.createElementNS(SVGNS, tag);
    for (var k in attrs) e.setAttribute(k, attrs[k]);
    if (text != null) e.textContent = text;
    if (parent) parent.appendChild(e);
    return e;
  }
  function mean(s) { return parseFloat(s); }
  function fmtT(t) { return t < 1 ? t.toFixed(2) + " s" : t < 100 ? t.toFixed(1) + " s" : Math.round(t).toLocaleString("en-US") + " s"; }
  function fmtGB(mb) { return (mb / 1000).toFixed(1); }
  function speedup(r) { // floor to two significant figures, e.g. 7554 -> 7,500
    var p = Math.pow(10, Math.max(0, Math.floor(Math.log10(r)) - 1));
    return (Math.floor(r / p) * p).toLocaleString("en-US");
  }

  function initResults(root) {
    var state = { ds: "10304", p: "1" };
    var chartBox = root.querySelector(".chart");
    var tip = root.querySelector(".chart-tip");
    var card = root.querySelector(".chart-card");
    var tbody = root.querySelector(".rtable tbody");

    root.querySelectorAll(".seg").forEach(function (seg) {
      var key = seg.dataset.key;
      seg.addEventListener("click", function (e) {
        var b = e.target.closest("button"); if (!b) return;
        seg.querySelectorAll("button").forEach(function (x) { x.setAttribute("aria-checked", x === b ? "true" : "false"); });
        state[key] = b.dataset.v; update();
      });
    });

    function rows() {
      var map = {};
      window.FT_RESULTS[state.ds][state.p].forEach(function (r) {
        map[r[0]] = { m05: r[1], m1: r[2], f1: r[3], t: +r[4], mem: +r[5], bold: r[6] };
      });
      return map;
    }

    function drawChart(data) {
      var W = Math.round(Math.max(340, Math.min(760, chartBox.clientWidth || 760)));
      var H = W < 560 ? 340 : 380, m = { l: W < 560 ? 44 : 58, r: 20, t: 26, b: 50 };
      var iw = W - m.l - m.r, ih = H - m.t - m.b;
      var x0 = -1, x1 = 3.8;
      var ymax = 0;
      CHART_ORDER.forEach(function (k) { ymax = Math.max(ymax, mean(data[k].f1)); });
      var ystep = ymax > 0.4 ? 0.1 : 0.05;
      ymax = Math.ceil((ymax * 1.12) / ystep) * ystep;
      function X(t) { return m.l + ((Math.log10(t) - x0) / (x1 - x0)) * iw; }
      function Y(v) { return m.t + ih - (v / ymax) * ih; }

      chartBox.innerHTML = "";
      var svg = el("svg", { viewBox: "0 0 " + W + " " + H }, chartBox);
      var defs = el("defs", {}, svg);
      var mk = el("marker", { id: "ah", viewBox: "0 0 10 10", refX: "8", refY: "5", markerWidth: "7", markerHeight: "7", orient: "auto-start-reverse" }, defs);
      el("path", { d: "M0 0L10 5L0 10z", class: "arrow-head" }, mk);

      var g = el("g", { class: "grid" }, svg);
      for (var e = -1; e <= 3; e++) {
        el("line", { x1: X(Math.pow(10, e)), x2: X(Math.pow(10, e)), y1: m.t, y2: m.t + ih }, g);
        var tx = el("text", { x: X(Math.pow(10, e)), y: m.t + ih + 20, "text-anchor": "middle", class: "tick" }, svg, "10");
        el("tspan", { dy: "-6", class: "tick-sup" }, tx, String(e).replace("-", "−"));
      }
      for (var v = 0; v <= ymax + 1e-9; v += ystep) {
        el("line", { x1: m.l, x2: m.l + iw, y1: Y(v), y2: Y(v) }, g);
        el("text", { x: m.l - 10, y: Y(v) + 4, "text-anchor": "end", class: "tick" }, svg, v.toFixed(ystep < 0.1 ? 2 : 1));
      }
      el("text", { x: m.l + iw / 2, y: H - 8, "text-anchor": "middle", class: "axis-label" }, svg, W < 560 ? "Runtime (s, log scale)" : "Runtime per tilt-series / tomogram (s, log scale)");
      el("text", { x: -(m.t + ih / 2), y: 12, transform: "rotate(-90)", "text-anchor": "middle", class: "axis-label" }, svg, "F1 score");

      // speed-up arrows from FullTilt to the two strongest baselines
      var ft = data.FullTilt, fx = X(ft.t), fy = Y(mean(ft.f1));
      [["TomoTwin", -1], ["ProPicker", 1]].forEach(function (a) {
        var o = data[a[0]], ox = X(o.t), oy = Y(mean(o.f1));
        var dx = ox - fx, dy = oy - fy, len = Math.hypot(dx, dy);
        var ux = dx / len, uy = dy / len;
        var sx = fx + ux * 14, sy = fy + uy * 14, ex = ox - ux * 14, ey = oy - uy * 14;
        var bend = a[1] * Math.min(70, len * 0.22);
        var cx = (sx + ex) / 2 - uy * bend, cy = (sy + ey) / 2 + ux * bend;
        if (a[0] === "TomoTwin") { cx = (sx + ex) / 2; cy = Math.min(sy, ey) - Math.max(10, len * 0.05); }
        el("path", { d: "M" + sx + " " + sy + "Q" + cx + " " + cy + " " + ex + " " + ey, class: "arrow", "marker-end": "url(#ah)" }, svg);
        var mx = 0.25 * sx + 0.5 * cx + 0.25 * ex, my = 0.25 * sy + 0.5 * cy + 0.25 * ey;
        var label = "> " + speedup(o.t / ft.t) + "× faster";
        if (a[0] === "TomoTwin") el("text", { x: mx, y: my - 10, "text-anchor": "middle", class: "speed" }, svg, label);
        else el("text", { x: mx + 12, y: my - 6, "text-anchor": "start", class: "speed" }, svg, label);
      });

      // points + direct labels
      var LBL = {
        FullTilt:  function (x, y) { return [x, y + 26, "middle"]; },
        TomoTwin:  function (x, y) { return [x - 14, y + 22, "end"]; },
        ProPicker: function (x, y) { return [x + 13, y + 4, "start"]; },
        CryoSAM:   function (x, y) { return [x - 4, y - 13, "start"]; },
        Zeng:      function (x, y) { return [x + 4, y - 13, "end"]; }
      };
      CHART_ORDER.forEach(function (k) {
        var d = data[k], x = X(d.t), y = Y(mean(d.f1));
        var pt = el("g", { class: "pt", tabindex: "0", "aria-label": METHODS[k].label + ": F1 " + d.f1 + ", runtime " + fmtT(d.t) }, svg);
        el("circle", { cx: x, cy: y, r: 18, class: "hit" }, pt);
        el("circle", { cx: x, cy: y, r: k === "FullTilt" ? 8 : 6.5, class: "dot", style: "fill:" + METHODS[k].color }, pt);
        var p = LBL[k](x, y);
        el("text", { x: p[0], y: p[1], "text-anchor": p[2], class: "lbl" + (k === "FullTilt" ? " lbl-ft" : "") }, svg, METHODS[k].label);
        function show() {
          var r = svg.getBoundingClientRect(), c = card.getBoundingClientRect(), s = r.width / W;
          tip.innerHTML = "<b><i style=\"background:" + METHODS[k].color + "\"></i>" + METHODS[k].label + "</b>" +
            "<span>F1<em>" + d.f1 + "</em></span><span>Runtime<em>" + fmtT(d.t) + "</em></span>" +
            "<span>Peak VRAM<em>" + fmtGB(d.mem) + " GB</em></span>";
          tip.hidden = false;
          var left = r.left - c.left + x * s, top = r.top - c.top + y * s;
          left = Math.max(95, Math.min(c.width - 95, left));
          tip.style.left = left + "px"; tip.style.top = top + "px";
        }
        function hide() { tip.hidden = true; }
        pt.addEventListener("mouseenter", show); pt.addEventListener("focus", show);
        pt.addEventListener("mouseleave", hide); pt.addEventListener("blur", hide);
      });
      chartBox.setAttribute("aria-label", "F1 against runtime on " + DS_NAME[state.ds] + " with " + state.p + " prompt(s)");
    }

    function cell(v, bold) {
      if (v === "n.a.") return "<td class=\"na\">n.a.</td>";
      var parts = v.split("±");
      var main = parts[0].trim(), sd = parts[1] ? "<span class=\"sd\">±" + parts[1].trim() + "</span>" : "";
      return "<td>" + (bold ? "<b>" + main + "</b>" : main) + sd + "</td>";
    }

    function drawTable(data) {
      var html = "", lastGroup = null;
      TABLE_ORDER.forEach(function (k) {
        var d = data[k], meta = METHODS[k];
        if (meta.group !== lastGroup) {
          html += "<tr class=\"group\"><td colspan=\"6\">" + (meta.group === "mv" ? "General multi-view 3D detectors" : "Cryo-ET methods") + "</td></tr>";
          lastGroup = meta.group;
        }
        var b = d.bold;
        var swatch = meta.color ? "<span class=\"mk\" style=\"background:" + meta.color + "\"></span>" : "<span class=\"mk mk-none\"></span>";
        html += "<tr" + (k === "FullTilt" ? " class=\"ours\"" : "") + ">" +
          "<td class=\"l\">" + swatch + meta.label + "</td>" +
          cell(d.m05, b[0] === "1") + cell(d.m1, b[1] === "1") + cell(d.f1, b[2] === "1") +
          cell(d.t.toLocaleString("en-US"), b[3] === "1") + cell(fmtGB(d.mem), b[4] === "1") + "</tr>";
      });
      tbody.innerHTML = html;
    }

    var last;
    function update() { last = rows(); drawChart(last); drawTable(last); }
    var rt, lastW = chartBox.clientWidth;
    window.addEventListener("resize", function () {
      clearTimeout(rt);
      rt = setTimeout(function () {
        if (Math.abs(chartBox.clientWidth - lastW) > 20) { lastW = chartBox.clientWidth; drawChart(last); }
      }, 150);
    });
    update();
  }

  /* ------------------------------------------------------------------
   * Tabs (qualitative tomogram slices)
   * ------------------------------------------------------------------ */
  var CAPTIONS = {
    "q-ft": "<strong>FullTilt</strong> &middot; 0.23&nbsp;s, from the tilt-series alone.",
    "q-gt": "<strong>Ground truth</strong> &middot; size-based picking followed by manual curation.",
    "q-tt": "<strong>TomoTwin</strong> &middot; 1,730&nbsp;s of sliding-window inference on the tomogram.",
    "q-pp": "<strong>ProPicker</strong> &middot; 49.2&nbsp;s of sliding-window inference on the tomogram."
  };
  function initTabs(root) {
    var tabs = [].slice.call(root.querySelectorAll("[role=tab]"));
    var cap = root.querySelector("[data-caption]");
    function select(t, focus) {
      tabs.forEach(function (x) {
        var on = x === t;
        x.setAttribute("aria-selected", on ? "true" : "false");
        x.tabIndex = on ? 0 : -1;
        document.getElementById(x.getAttribute("aria-controls")).hidden = !on;
      });
      cap.innerHTML = CAPTIONS[t.getAttribute("aria-controls")];
      if (focus) t.focus();
    }
    tabs.forEach(function (t, i) {
      t.tabIndex = i === 0 ? 0 : -1;
      t.addEventListener("click", function () { select(t); });
      t.addEventListener("keydown", function (e) {
        if (e.key !== "ArrowRight" && e.key !== "ArrowLeft") return;
        var j = (i + (e.key === "ArrowRight" ? 1 : tabs.length - 1)) % tabs.length;
        select(tabs[j], true); e.preventDefault();
      });
    });
  }

  /* ------------------------------------------------------------------
   * Misc: copy BibTeX, sticky-bar border, active section link
   * ------------------------------------------------------------------ */
  document.querySelectorAll("[data-copy]").forEach(function (b) {
    b.addEventListener("click", function () {
      var txt = document.querySelector(b.dataset.copy).innerText;
      var done = function () { b.textContent = "Copied"; b.classList.add("done"); setTimeout(function () { b.textContent = "Copy"; b.classList.remove("done"); }, 1600); };
      if (navigator.clipboard) navigator.clipboard.writeText(txt).then(done, function () {});
    });
  });
  document.querySelectorAll(".btn[data-soon]").forEach(function (a) {
    a.addEventListener("click", function (e) { e.preventDefault(); });
  });

  var themeBtn = document.querySelector(".theme-toggle");
  function setThemeLabel() {
    var dark = document.documentElement.dataset.theme === "dark";
    themeBtn.setAttribute("aria-label", dark ? "Switch to light theme" : "Switch to dark theme");
  }
  themeBtn.addEventListener("click", function () {
    var root = document.documentElement, dark = root.dataset.theme !== "dark";
    if (dark) root.dataset.theme = "dark"; else delete root.dataset.theme;
    try { localStorage.setItem("ft-theme", dark ? "dark" : "light"); } catch (e) {}
    setThemeLabel();
  });
  setThemeLabel();

  var bar = document.querySelector(".topbar");
  window.addEventListener("scroll", function () { bar.classList.toggle("scrolled", window.scrollY > 8); }, { passive: true });

  if ("IntersectionObserver" in window) {
    var links = {};
    document.querySelectorAll(".toc a").forEach(function (a) { links[a.getAttribute("href").slice(1)] = a; });
    var io = new IntersectionObserver(function (es) {
      es.forEach(function (e) {
        if (!e.isIntersecting) return;
        for (var k in links) links[k].classList.toggle("active", k === e.target.id);
      });
    }, { rootMargin: "-45% 0px -50% 0px" });
    Object.keys(links).forEach(function (id) { var s = document.getElementById(id); if (s) io.observe(s); });
  }

  document.querySelectorAll(".viewer").forEach(initViewer);
  document.querySelectorAll("[data-results]").forEach(initResults);
  document.querySelectorAll("[data-tabs]").forEach(initTabs);
})();
