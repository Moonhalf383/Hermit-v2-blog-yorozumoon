import { createCaret } from "./caret.js";

(() => {
  "use strict";

  const $ = (selector, root = document) => root.querySelector(selector);
  const $$ = (selector, root = document) => [...root.querySelectorAll(selector)];
  const documentPane = $("#document-pane");
  const prefersReducedMotion = matchMedia("(prefers-reduced-motion: reduce)");
  const scrollBehavior = () => prefersReducedMotion.matches ? "auto" : "smooth";
  let caret;
  const cursor = $("#vim-cursor");
  const modeIndicator = $("#mode-indicator");
  const keyIndicator = $("#key-indicator");
  const toast = $("#editor-toast");
  const help = $("#help-dialog");
  const fileIndex = JSON.parse($("#site-file-index")?.textContent || "[]");
  const config = JSON.parse($("#editor-config")?.textContent || "{}");
  const paneOrder = ["left", "editor", "right"];
  const paneElements = {
    left: $("#explorer-pane"),
    editor: documentPane,
    right: $("#outline-pane")
  };
  const scrollElements = {
    left: $("#file-tree"),
    editor: documentPane,
    right: $("#outline-tree")
  };
  const state = {
    pane: "editor",
    mode: "normal",
    current: {left: null, editor: null, right: null},
    visualLinewise: false,
    pending: "",
    pendingTimer: 0,
    helpIndex: 0,
    toastTimer: 0
  };

  function normalizePath(value) {
    return String(value || "").replaceAll("\\", "/").replace(/^\.\//, "").replace(/^\//, "");
  }

  function safeDecode(value) {
    try { return decodeURIComponent(value); } catch (_) { return value; }
  }

  function normalizeKey(value) {
    return safeDecode(String(value || ""))
      .trim().toLowerCase().replaceAll("\\", "/")
      .replace(/^\.\//, "").replace(/^\//, "").replace(/\/$/, "")
      .replace(/\.md$/i, "").replace(/\/index$/i, "");
  }

  function setIcon(container, value, kind = "file") {
    const raw = String(value || "").trim();
    const key = raw.replace(/^Li(?=[A-Z])/, "").replace(/^lucide[:/-]/i, "").toLowerCase();
    const paths = {
      file: "M5 2h9l5 5v15H5z M14 2v6h5",
      folder: "M2 5h8l2 3h10v13H2z",
      house: "M2 11 12 2l10 9 M5 9v13h5v-7h4v7h5V9",
      terminal: "M3 4h18v16H3z M6 8l4 4-4 4 M13 16h5",
      book: "M3 3h7l2 2 2-2h7v18h-7l-2 1-2-1H3z M12 5v17",
      image: "M3 3h18v18H3z M3 17l6-6 4 4 3-3 5 5 M16 7h.01",
      link: "M10 13a4 4 0 0 0 6 0l4-4a4 4 0 0 0-6-6l-2 2 M14 11a4 4 0 0 0-6 0l-4 4a4 4 0 0 0 6 6l2-2",
      tag: "M3 3h9l10 10-9 9L3 12z M7 7h.01",
      cpu: "M5 5h14v14H5z M9 9h6v6H9z M9 1v4 M15 1v4 M9 19v4 M15 19v4 M1 9h4 M1 15h4 M19 9h4 M19 15h4",
      brain: "M12 5c-2-5-8-3-7 2-5 1-4 7-1 8-2 5 5 9 8 4 3 5 10 1 8-4 3-1 4-7-1-8 1-5-5-7-7-2z M12 5v14 M5 7l3 2 M4 15l4-2 M19 7l-3 2 M20 15l-4-2",
      chart: "M3 3v18h18 M7 17v-5 M12 17V7 M17 17V3",
      notebook: "M5 2h15v20H5z M2 6h6 M2 12h6 M2 18h6 M11 7h5 M11 11h5",
      gamepad: "M7 6h10c4 0 7 13 4 14-2 1-4-4-5-4H8c-1 0-3 5-5 4C0 19 3 6 7 6z M5 11h6 M8 8v6 M16 10h.01 M19 13h.01",
      music: "M9 18V5l12-3v13 M9 5v4l12-3 M9 18a3 3 0 1 1-3-3c2 0 3 1 3 3 M21 15a3 3 0 1 1-3-3c2 0 3 1 3 3",
      pen: "m15 3 6 6L8 22H2v-6z M12 6l6 6 M2 16l6 6",
      box: "m12 2 10 5v10l-10 5-10-5V7z M2 7l10 5 10-5 M12 12v10 M7 4.5l10 5V15",
      flask: "M8 2h8 M9 2v7L3 19q-1 3 2 3h14q3 0 2-3L15 9V2 M6 15h12",
      cloud: "M6 19a5 5 0 0 1-1-10 7 7 0 0 1 13-2 6 6 0 0 1 0 12z",
      github: "M9 19c-5 2-5-3-7-3 M15 22v-4c0-1 .5-2 1-2 4-.5 6-2 6-6 0-2-1-3-2-4 0-1 0-3-1-4-2 0-3 1-4 1a14 14 0 0 0-6 0C8 3 6 2 4 2c-1 1-1 3-1 4-1 1-2 2-2 4 0 4 2 5.5 6 6 .5 0 1 1 1 2v4"
    };
    let name = Object.hasOwn(paths, key) ? key : key.includes("home") || key.includes("house") ? "house" :
      key.includes("code") || key.includes("terminal") ? "terminal" :
      Object.keys(paths).find(name => key.includes(name));
    if (!name && raw && !/^(Li[A-Z]|lucide[:/-])/i.test(raw)) {
      container.textContent = raw;
    } else {
      name ||= kind;
      const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
      svg.setAttribute("viewBox", "0 0 24 24");
      svg.setAttribute("fill", "none");
      svg.setAttribute("stroke", "currentColor");
      svg.setAttribute("stroke-width", "1.5");
      const path = document.createElementNS(svg.namespaceURI, "path");
      path.setAttribute("d", paths[name] || paths.file);
      svg.append(path);
      container.replaceChildren(svg);
    }
    container.setAttribute("aria-hidden", "true");
    if (raw) container.title = raw;
  }

  function readStored(key, fallback) {
    try {
      const value = localStorage.getItem(key);
      return value === null ? fallback : JSON.parse(value);
    } catch (_) { return fallback; }
  }

  function writeStored(key, value) {
    try { localStorage.setItem(key, JSON.stringify(value)); } catch (_) { /* private mode */ }
  }

  function buildFileTree() {
    const entries = fileIndex
      .filter(entry => !normalizePath(entry.path).split("/").some(part => part.startsWith(".")))
      .sort((a, b) => a.path.localeCompare(b.path, undefined, {numeric: true}));
    const newRoot = () => ({name: "", path: "", dirs: new Map(), files: [], icon: ""});
    const filesRoot = newRoot();
    const tagsRoot = newRoot();
    const currentSource = normalizePath(document.body.dataset.currentSource);
    const container = $("#file-tree");

    for (const entry of entries) {
      const path = normalizePath(entry.path);
      const parts = path.split("/").filter(Boolean);
      if (!parts.length) continue;
      let node = filesRoot;
      parts.slice(0, -1).forEach((part, index) => {
        if (!node.dirs.has(part)) {
          const dirPath = parts.slice(0, index + 1).join("/");
          node.dirs.set(part, {name: part, path: dirPath, dirs: new Map(), files: [], icon: ""});
        }
        node = node.dirs.get(part);
      });
      node.files.push({...entry, path, name: parts.at(-1)});
      if (entry.folderIcon && parts.length > 1) node.icon = entry.folderIcon;

      const entryTags = Array.isArray(entry.tags) ? entry.tags : [entry.tags].filter(Boolean);
      for (const tag of entryTags) {
        const tagName = String(tag).trim().replace(/^#/, "");
        if (!tagName) continue;
        const tagPath = `@tag/${tagName}`;
        if (!tagsRoot.dirs.has(tagName)) {
          tagsRoot.dirs.set(tagName, {name: tagName, path: tagPath, dirs: new Map(), files: [], icon: "tag"});
        }
        tagsRoot.dirs.get(tagName).files.push({...entry, path, name: parts.at(-1)});
      }
    }

    function containsCurrent(node) {
      return node.files.some(file => normalizePath(file.source) === currentSource) ||
        [...node.dirs.values()].some(containsCurrent);
    }

    function renderNode(node, isRoot = false) {
      const list = document.createElement("ul");
      list.className = "tree-list";
      if (isRoot) list.setAttribute("role", "group");

      [...node.dirs.values()]
        .sort((a, b) => a.name.localeCompare(b.name, undefined, {numeric: true}))
        .forEach(dir => {
          const item = document.createElement("li");
          item.className = "tree-dir";
          const row = document.createElement("button");
          const activeBranch = containsCurrent(dir);
          const savedDirs = new Set(readStored("kode-editor:open-dirs", []));
          const closedDirs = new Set(readStored("kode-editor:closed-dirs", []));
          const topLevel = dir.path.split("/").length === 1;
          const open = !closedDirs.has(dir.path) && (activeBranch || savedDirs.has(dir.path) || topLevel);
          row.type = "button";
          row.className = "tree-row";
          row.dataset.kind = "dir";
          row.dataset.path = dir.path;
          row.setAttribute("role", "treeitem");
          row.setAttribute("aria-expanded", String(open));
          row.innerHTML = '<span class="tree-caret" aria-hidden="true">›</span>';
          const icon = document.createElement("span");
          icon.className = "file-icon";
          setIcon(icon, dir.icon, "folder");
          const label = document.createElement("span");
          label.className = "tree-label";
          label.textContent = dir.path.startsWith("@tag/") ? `#${dir.name}` : dir.name;
          row.append(icon, label);
          item.append(row, renderNode(dir));
          list.append(item);
        });

      node.files
        .sort((a, b) => (b.date ?? -Infinity) - (a.date ?? -Infinity) ||
          a.name.localeCompare(b.name, undefined, {numeric: true}) || a.path.localeCompare(b.path))
        .forEach(file => {
          const item = document.createElement("li");
          const row = document.createElement("a");
          row.className = "tree-row";
          row.dataset.kind = "file";
          row.dataset.path = file.path;
          row.dataset.source = file.source;
          row.href = file.url;
          row.setAttribute("role", "treeitem");
          if (normalizePath(file.source) === currentSource) row.classList.add("active");
          const spacer = document.createElement("span");
          spacer.className = "tree-caret";
          spacer.setAttribute("aria-hidden", "true");
          const icon = document.createElement("span");
          icon.className = "file-icon";
          setIcon(icon, file.icon || (file.path === "README.md" ? "house" : ""));
          if (file.iconColor) icon.style.color = file.iconColor;
          const label = document.createElement("span");
          label.className = "tree-label";
          label.textContent = file.name;
          row.append(spacer, icon, label);
          item.append(row);
          list.append(item);
        });
      if (isRoot) {
        const readme = $(".tree-row[data-path='README.md']", list)?.parentElement;
        if (readme) list.prepend(readme);
      }
      return list;
    }

    function renderView(view) {
      container.replaceChildren(renderNode(view === "tags" ? tagsRoot : filesRoot, true));
      container.dataset.view = view;
      container.setAttribute("aria-label", view === "tags" ? "标签" : "文件");
      $$('[data-explorer-view]').forEach(button => button.setAttribute("aria-selected", String(button.dataset.explorerView === view)));
      state.current.left = $(".tree-row.active", container) || $(".tree-row", container);
      writeStored("kode-editor:explorer-view", view);
      if (state.pane === "left") setCurrent("left", state.current.left);
    }

    container.addEventListener("click", event => {
      const row = event.target.closest(".tree-row");
      if (!row) return;
      setCurrent("left", row);
      focusPane("left");
      if (row.dataset.kind === "dir") {
        event.preventDefault();
        toggleDirectory(row);
      }
    });
    $$('[data-explorer-view]').forEach(button => button.addEventListener("click", () => renderView(button.dataset.explorerView)));
    const initialView = readStored("kode-editor:explorer-view", "files");
    renderView(initialView === "tags" ? "tags" : "files");
  }

  function toggleDirectory(row, force) {
    if (!row || row.dataset.kind !== "dir") return;
    const expanded = force ?? row.getAttribute("aria-expanded") !== "true";
    row.setAttribute("aria-expanded", String(expanded));
    const open = new Set(readStored("kode-editor:open-dirs", []));
    const closed = new Set(readStored("kode-editor:closed-dirs", []));
    if (expanded) {
      open.add(row.dataset.path);
      closed.delete(row.dataset.path);
    } else {
      open.delete(row.dataset.path);
      closed.add(row.dataset.path);
    }
    writeStored("kode-editor:open-dirs", [...open]);
    writeStored("kode-editor:closed-dirs", [...closed]);
    requestCursorUpdate();
  }

  const wikiExact = new Map();
  const wikiShort = new Map();
  function addShortWikilink(key, url) {
    if (!key) return;
    if (!wikiShort.has(key)) wikiShort.set(key, url);
    else if (wikiShort.get(key) !== url) wikiShort.set(key, null);
  }

  function buildWikiLookup() {
    for (const entry of fileIndex) {
      const path = normalizePath(entry.path);
      const pathKey = normalizeKey(path);
      const filename = path.split("/").at(-1) || "";
      const stem = filename.replace(/\.md$/i, "");
      wikiExact.set(pathKey, entry.url);
      [entry.title, stem, ...(entry.aliases || [])]
        .map(normalizeKey).filter(Boolean).forEach(key => addShortWikilink(key, entry.url));
    }
  }

  function resolveWikilink(target) {
    const [pagePart, heading = ""] = target.split("#", 2);
    const key = normalizeKey(pagePart || document.body.dataset.currentPath);
    let url = wikiExact.get(key) || wikiShort.get(key);
    if (!url && key.includes("/")) {
      const matches = [...wikiExact.entries()].filter(([candidate]) => candidate.endsWith(`/${key}`));
      const urls = new Set(matches.map(([, candidateURL]) => candidateURL));
      if (urls.size === 1) url = [...urls][0];
    }
    if (!url) return "";
    return heading ? `${url}#${slugify(heading)}` : url;
  }

  function slugify(value) {
    return String(value).trim().toLowerCase().replace(/[^\p{L}\p{N}\s_-]/gu, "").replace(/\s/g, "-");
  }

  function assetURL(target) {
    if (/^(https?:|data:|\/)/i.test(target)) return target;
    const encoded = target.split("/").map(part => encodeURIComponent(safeDecode(part))).join("/");
    return new URL(encoded, location.href).href;
  }

  function transformWikilinks() {
    const root = $("[data-wikilinks]");
    if (!root) return;
    const walker = document.createTreeWalker(root, NodeFilter.SHOW_TEXT, {
      acceptNode(node) {
        if (!node.nodeValue.includes("[[")) return NodeFilter.FILTER_REJECT;
        return node.parentElement.closest("pre,code,a,script,style,mjx-container,.mermaid")
          ? NodeFilter.FILTER_REJECT : NodeFilter.FILTER_ACCEPT;
      }
    });
    const nodes = [];
    while (walker.nextNode()) nodes.push(walker.currentNode);
    const pattern = /(!?)\[\[([^\]\n]+)\]\]/g;

    nodes.forEach(node => {
      const text = node.nodeValue;
      let match;
      let last = 0;
      const fragment = document.createDocumentFragment();
      pattern.lastIndex = 0;
      while ((match = pattern.exec(text))) {
        fragment.append(text.slice(last, match.index));
        const embedded = match[1] === "!";
        const [rawTarget, rawLabel = ""] = match[2].split("|", 2);
        const target = rawTarget.trim();
        const label = rawLabel.trim();
        const isImage = /\.(avif|gif|jpe?g|png|svg|webp)$/i.test(target);
        if (embedded && isImage) {
          const frame = document.createElement("span");
          frame.className = "media-frame wikilink-asset";
          const image = document.createElement("img");
          image.src = assetURL(target);
          image.alt = label && !/^\d+(x\d+)?$/.test(label) ? label : target.split("/").at(-1);
          image.loading = "lazy";
          image.decoding = "async";
          const size = label.match(/^(\d+)(?:x(\d+))?$/);
          if (size) {
            image.width = Number(size[1]);
            frame.style.width = `${Number(size[1])}px`;
            if (size[2]) {
              image.height = Number(size[2]);
              image.style.aspectRatio = `${Number(size[1])} / ${Number(size[2])}`;
            }
          }
          frame.append(image);
          fragment.append(frame);
        } else {
          const link = document.createElement("a");
          link.className = "wikilink";
          link.dataset.wikilink = target;
          link.textContent = label || target.replace(/\.md$/i, "");
          const href = resolveWikilink(target);
          link.href = href || "#";
          if (!href) {
            link.classList.add("unresolved");
            link.title = `未找到：${target}`;
          }
          fragment.append(link);
        }
        last = pattern.lastIndex;
      }
      if (last) {
        fragment.append(text.slice(last));
        node.replaceWith(fragment);
      }
    });
  }

  function buildOutline() {
    const container = $("#outline-tree");
    const headings = $$(".content h1,.content h2,.content h3,.content h4,.content h5,.content h6");
    if (!headings.length) {
      container.innerHTML = '<p class="empty-tree">no symbols</p>';
      return;
    }
    headings.forEach((heading, index) => {
      if (!heading.id) heading.id = `${slugify(heading.textContent) || "section"}-${index + 1}`;
      const link = document.createElement("a");
      const level = Number(heading.tagName.slice(1));
      link.className = "outline-row";
      link.href = `#${encodeURIComponent(heading.id)}`;
      link.dataset.level = String(level);
      link.style.setProperty("--depth", String(level));
      link.innerHTML = `<span class="outline-index">${String(index + 1).padStart(2, "0")}</span><span class="tree-label"></span>`;
      $(".tree-label", link).textContent = heading.textContent;
      link.addEventListener("click", event => {
        event.preventDefault();
        heading.scrollIntoView({behavior: scrollBehavior(), block: "start"});
        history.replaceState(null, "", link.hash);
        setCurrent("right", link);
      });
      container.append(link);
    });
    state.current.right = $(".outline-row", container);
  }

  function navigationUnits() { return caret?.units() || []; }

  function navigationUnitFor(target) {
    return navigationUnits().find(unit => unit.contains(target)) || null;
  }

  function clearCharacterCursor() { caret?.enable(false); }

  function moveCharacter(direction) { caret?.horizontal(direction); }

  function visibleRows(pane) {
    const selector = pane === "left" ? ".tree-row" : pane === "right" ? ".outline-row" : "";
    return selector ? $$(selector, paneElements[pane]).filter(row => row.offsetParent !== null) : navigationUnits();
  }

  function setCurrent(pane, element) {
    if (!element) return;
    if (pane === "editor") caret?.activate(element);
    visibleRows(pane).forEach(item => {
      item.classList.remove("selected");
      if (pane !== "editor" || !item.matches("a[href]")) item.tabIndex = -1;
    });
    element.classList.add("selected");
    element.tabIndex = 0;
    state.current[pane] = element;
    if (pane === state.pane && !help.open) {
      element.focus({preventScroll: true});
      requestCursorUpdate();
    }
  }

  function focusPane(pane) {
    if (!paneElements[pane] || isPaneCollapsed(pane)) return;
    if (pane !== "editor") clearCharacterCursor();
    state.pane = pane;
    Object.entries(paneElements).forEach(([name, element]) => element.classList.toggle("focused", name === pane));
    const rows = visibleRows(pane);
    if (!state.current[pane] || state.current[pane].offsetParent === null) {
      state.current[pane] = rows[0] || null;
    }
    setCurrent(pane, state.current[pane]);
    if (pane !== "editor") scrollWithMargin(state.current[pane], scrollElements[pane]);
    showKey(pane.toUpperCase());
  }

  function isPaneCollapsed(pane) {
    return document.documentElement.classList.contains(`${pane}-collapsed`);
  }

  function cyclePane(direction) {
    let index = paneOrder.indexOf(state.pane);
    for (let count = 0; count < paneOrder.length; count += 1) {
      index = (index + direction + paneOrder.length) % paneOrder.length;
      if (!isPaneCollapsed(paneOrder[index])) return focusPane(paneOrder[index]);
    }
  }

  function togglePane(side) {
    const className = `${side}-collapsed`;
    const collapsed = document.documentElement.classList.toggle(className);
    writeStored(`kode-editor:${className}`, collapsed);
    $$(`[data-toggle-pane="${side}"]`).forEach(button => button.setAttribute("aria-expanded", String(!collapsed)));
    if (collapsed && (state.pane === side || paneElements[side].contains(document.activeElement))) focusPane("editor");
    paneElements[side].inert = collapsed;
    caret?.invalidate();
    requestCursorUpdate();
  }

  function move(direction) {
    if (state.pane === "editor") return caret.vertical(direction);
    const rows = visibleRows(state.pane);
    if (!rows.length) return;
    const index = Math.max(0, rows.indexOf(state.current[state.pane]));
    const next = rows[Math.max(0, Math.min(rows.length - 1, index + direction))];
    setCurrent(state.pane, next);
    scrollWithMargin(next, scrollElements[state.pane]);
  }

  function scrollWithMargin(element, scroller) {
    if (!element || !scroller) return;
    const rect = element.getBoundingClientRect();
    const bounds = scroller.getBoundingClientRect();
    const line = parseFloat(getComputedStyle(element).lineHeight) || 20;
    const margin = Math.min(bounds.height * .28, line * 4);
    if (rect.top < bounds.top + margin) {
      scroller.scrollBy({top: rect.top - bounds.top - margin, behavior: scrollBehavior()});
    } else if (rect.bottom > bounds.bottom - margin) {
      scroller.scrollBy({top: rect.bottom - bounds.bottom + margin, behavior: scrollBehavior()});
    }
  }

  function horizontal(direction) {
    const current = state.current[state.pane];
    if (!current) return;
    if (state.pane === "left") {
      if (current.dataset.kind === "dir") {
        const expanded = current.getAttribute("aria-expanded") === "true";
        if (direction > 0 && !expanded) toggleDirectory(current, true);
        else if (direction < 0 && expanded) toggleDirectory(current, false);
        else if (direction < 0) {
          const parent = current.closest("ul")?.parentElement?.querySelector(":scope > .tree-row");
          if (parent) setCurrent("left", parent);
        }
      } else if (direction > 0) {
        location.href = current.href;
      } else {
        const parent = current.closest("ul")?.parentElement?.querySelector(":scope > .tree-row");
        if (parent) setCurrent("left", parent);
      }
    } else if (state.pane === "right" && direction > 0) {
      current.click();
    } else if (state.pane === "editor") {
      moveCharacter(direction);
    }
  }

  function enterVisual(linewise = false) {
    if (state.pane !== "editor") focusPane("editor");
    state.mode = "visual";
    document.body.classList.add("visual-mode");
    modeIndicator.textContent = linewise ? "V-LINE" : "VISUAL";
    caret.visual(linewise);
  }

  function exitVisual() {
    state.mode = "normal";
    document.body.classList.remove("visual-mode");
    modeIndicator.textContent = "NORMAL";
    caret.normal();
    requestCursorUpdate();
  }

  async function copyText(text, message = "copied") {
    if (!text) return;
    try {
      await navigator.clipboard.writeText(text);
    } catch (_) {
      const field = document.createElement("textarea");
      field.value = text;
      field.style.position = "fixed";
      field.style.opacity = "0";
      document.body.append(field);
      field.select();
      document.execCommand("copy");
      field.remove();
    }
    showToast(message);
  }

  function yankCurrent() {
    copyText(caret.yank(), "copied");
  }

  function goDefinition() {
    if (state.pane !== "editor") return;
    const card = caret.current();
    if (card?.matches("a.friend-card")) return card.click();
    const link = caret.position()?.node.parentElement.closest("a[data-wikilink]");
    if (!link) return showToast("no link under cursor");
    if (link.classList.contains("unresolved")) return showToast(link.title || "unresolved wikilink");
    location.href = link.href;
  }

  function showToast(message) {
    toast.textContent = message;
    toast.classList.add("show");
    clearTimeout(state.toastTimer);
    state.toastTimer = setTimeout(() => toast.classList.remove("show"), 1300);
  }

  function showKey(message) {
    keyIndicator.textContent = message;
    clearTimeout(state.pendingTimer);
    state.pendingTimer = setTimeout(() => {
      state.pending = "";
      keyIndicator.textContent = "? : help";
    }, 900);
  }

  function toggleHelp(force) {
    const open = force ?? !help.open;
    if (open && !help.open) {
      help.showModal();
      state.helpIndex = 0;
      selectHelpRow();
    } else if (!open && help.open) {
      help.close();
      requestCursorUpdate();
    }
  }

  function selectHelpRow() {
    const rows = $$(".help-list p", help);
    rows.forEach((row, index) => row.classList.toggle("selected", index === state.helpIndex));
    rows[state.helpIndex]?.scrollIntoView({block: "nearest"});
    cursor.classList.remove("visible");
  }

  function handleKey(event) {
    const target = event.target instanceof Element ? event.target : null;
    if (event.defaultPrevented || event.metaKey || event.altKey || target?.isContentEditable || /^(INPUT|TEXTAREA|SELECT)$/.test(target?.tagName || "")) return;
    if (help.open) {
      if (event.ctrlKey) return;
      if (event.key === "j" || event.key === "k") {
        event.preventDefault();
        const length = $$(".help-list p", help).length;
        state.helpIndex = Math.max(0, Math.min(length - 1, state.helpIndex + (event.key === "j" ? 1 : -1)));
        selectHelpRow();
      } else if (event.key === "Escape" || event.key === "?") {
        event.preventDefault();
        toggleHelp(false);
      }
      return;
    }

    if (event.ctrlKey && (event.key === "h" || event.key === "l")) {
      event.preventDefault();
      cyclePane(event.key === "h" ? -1 : 1);
      return;
    }
    if (event.ctrlKey) return;
    if (event.key === "H" || event.key === "L") {
      event.preventDefault();
      togglePane(event.key === "H" ? "left" : "right");
      return;
    }
    if (event.key === "?") {
      event.preventDefault();
      toggleHelp(true);
      return;
    }
    if (event.key === "Escape") {
      if (state.mode === "visual") exitVisual();
      state.pending = "";
      showKey("NORMAL");
      return;
    }
    if (event.key === "j" || event.key === "k") {
      event.preventDefault();
      move(event.key === "j" ? 1 : -1);
      return;
    }
    if (event.key === "h" || event.key === "l") {
      event.preventDefault();
      horizontal(event.key === "l" ? 1 : -1);
      return;
    }
    if (state.mode === "normal" && (event.key === "v" || event.key === "V")) {
      event.preventDefault();
      enterVisual(event.key === "V");
      return;
    }
    if (state.mode === "visual" && event.key === "y") {
      event.preventDefault();
      copyText(getSelection()?.toString() || "", "selection copied");
      exitVisual();
      return;
    }
    if (state.mode === "normal" && event.key === "y") {
      event.preventDefault();
      if (state.pending === "y") {
        state.pending = "";
        yankCurrent();
      } else {
        state.pending = "y";
        showKey("y");
      }
      return;
    }
    if (state.pane === "editor" && ["w", "e", "b", "0", "$", "G"].includes(event.key)) {
      event.preventDefault();
      if (["w", "e", "b"].includes(event.key)) caret.word(event.key);
      else caret.boundary(event.key);
      return;
    }
    if (event.key === "g") {
      event.preventDefault();
      if (state.pending === "g" && state.pane === "editor") {
        caret.boundary("gg");
        state.pending = "";
      } else { state.pending = "g"; showKey("g"); }
      return;
    }
    if (state.mode === "normal" && event.key === "d" && state.pending === "g") {
      event.preventDefault();
      state.pending = "";
      goDefinition();
    }
  }

  function requestCursorUpdate() {
    caret?.enable(state.pane === "editor" && !help.open);
  }

  async function renderMermaid() {
    if (!$(".mermaid")) return;
    try {
      const module = await import(config.mermaidLib);
      const mermaid = module.default;
      mermaid.initialize({
        startOnLoad: false,
        securityLevel: "strict",
        theme: "base",
        fontFamily: '"Kode Mono", monospace',
        themeVariables: {
          background: "#ffffff", primaryColor: "#ffffff", primaryTextColor: "#111111",
          primaryBorderColor: "#111111", lineColor: "#111111", secondaryColor: "#f7f7f5",
          tertiaryColor: "#ffffff", clusterBkg: "#ffffff", clusterBorder: "#111111",
          edgeLabelBackground: "#ffffff", fontFamily: '"Kode Mono", monospace'
        },
        flowchart: {curve: "linear", htmlLabels: false}
      });
      await mermaid.run({nodes: $$(".mermaid")});
    } catch (error) {
      console.warn("Mermaid failed to load", error);
      showToast("diagram render failed");
    }
  }

  function restoreLayout() {
    const mobile = matchMedia("(max-width: 900px)").matches;
    const left = readStored("kode-editor:left-collapsed", mobile);
    const right = readStored("kode-editor:right-collapsed", mobile);
    document.documentElement.classList.toggle("left-collapsed", left);
    document.documentElement.classList.toggle("right-collapsed", right);
    paneElements.left.inert = left;
    paneElements.right.inert = right;
    $$('[data-toggle-pane="left"]').forEach(button => button.setAttribute("aria-expanded", String(!left)));
    $$('[data-toggle-pane="right"]').forEach(button => button.setAttribute("aria-expanded", String(!right)));
    requestAnimationFrame(() => requestAnimationFrame(() => document.documentElement.classList.remove("layout-boot")));
  }

  function bindMouseNavigation() {
    document.addEventListener("focusin", event => {
      if (!paneElements.editor.contains(event.target)) clearCharacterCursor();
    });
    Object.entries(paneElements).forEach(([pane, element]) => {
      element.addEventListener("focusin", event => {
        if (pane !== "editor") clearCharacterCursor();
        state.pane = pane;
        Object.entries(paneElements).forEach(([name, candidate]) => candidate.classList.toggle("focused", name === pane));
        const unit = pane === "editor" ? navigationUnitFor(event.target) : event.target.closest(pane === "left" ? ".tree-row" : ".outline-row");
        if (unit) {
          if (pane === "editor" && caret.current() !== unit) caret.activate(unit);
          $$(".selected", element).forEach(item => item.classList.remove("selected"));
          unit.classList.add("selected");
          state.current[pane] = unit;
        }
        requestCursorUpdate();
      });
    });
    paneElements.editor.addEventListener("click", event => {
      if (event.target.closest("a,button,input,textarea,.giscus") || !getSelection().isCollapsed) return;
      state.pane = "editor";
      if (caret.click(event.target, event.clientX, event.clientY)) focusPane("editor");
    });
    $("#outline-tree").addEventListener("click", event => {
      const row = event.target.closest(".outline-row");
      if (row) {
        setCurrent("right", row);
        focusPane("right");
      }
    });
    $$('[data-toggle-pane]').forEach(button => button.addEventListener("click", () => togglePane(button.dataset.togglePane)));
    $("[data-close-help]").addEventListener("click", () => toggleHelp(false));
    help.addEventListener("click", event => {
      if (event.target === help) toggleHelp(false);
    });
  }

  function loadComments() {
    const container = $("[data-giscus]");
    if (!container) return;
    const load = () => {
      const script = document.createElement("script");
      script.src = "https://giscus.app/client.js";
      script.async = true;
      script.crossOrigin = "anonymous";
      for (const attr of container.attributes) {
        if (attr.name.startsWith("data-") && attr.name !== "data-giscus") script.setAttribute(attr.name, attr.value);
      }
      container.append(script);
    };
    const observer = new IntersectionObserver(entries => {
      if (entries.some(entry => entry.isIntersecting)) { observer.disconnect(); load(); }
    }, {root: documentPane, rootMargin: "250px"});
    observer.observe(container);
  }

  restoreLayout();
  buildWikiLookup();
  buildFileTree();
  transformWikilinks();
  buildOutline();
  caret = createCaret({root: $("#document"), scroller: documentPane, cursor,
    onUnit(unit) {
      state.current.editor = unit;
      if (!unit.matches("a[href]")) unit.tabIndex = -1;
      if (document.activeElement !== unit && state.pane === "editor") unit.focus({preventScroll: true});
    }
  });
  bindMouseNavigation();
  document.addEventListener("keydown", handleKey);
  documentPane.addEventListener("scroll", requestCursorUpdate, {passive: true, capture: true});
  $("#file-tree").addEventListener("scroll", requestCursorUpdate, {passive: true});
  $("#outline-tree").addEventListener("scroll", requestCursorUpdate, {passive: true});
  addEventListener("resize", requestCursorUpdate, {passive: true});
  $("#editor-app").addEventListener("transitionend", () => caret.invalidate());

  const units = navigationUnits();
  state.current.editor = units[0] || $(".document-header h1");
  focusPane("editor");
  renderMermaid();
  loadComments();
})();
