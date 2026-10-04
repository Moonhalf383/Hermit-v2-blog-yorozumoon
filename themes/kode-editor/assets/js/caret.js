// Read-only, rendered-text navigation. No character spans or copied glyphs are
// inserted into the article: the cursor only inverts the pixels beneath it.
export function createCaret({root, scroller, cursor, onUnit}) {
  const segmenter = new Intl.Segmenter(undefined, {granularity: "grapheme"});
  const reduced = matchMedia("(prefers-reduced-motion: reduce)");
  const behavior = () => reduced.matches ? "instant" : "smooth";
  const skip = "script,style,svg,.mermaid,mjx-container,.giscus,.ln,.lnt,[aria-hidden=true]";
  let unitList = [];
  let textCache = new WeakMap();
  let geometryCache = new WeakMap();
  let unit = null, index = 0, goalX = null, anchor = null, linewise = false;
  let enabled = false, frame = 0;

  function units() {
    if (!unitList.length) {
      const candidates = [...root.querySelectorAll(".document-header h1,.content h1,.content h2,.content h3,.content h4,.content h5,.content h6,.content p,.content li,.content pre,.content table,.content a.friend-card")];
      const parents = new Set(candidates);
      unitList = candidates.filter(element => {
        if (element.closest(skip)) return false;
        for (let parent = element.parentElement; parent !== root; parent = parent.parentElement) {
          if (!parent || parents.has(parent)) return false;
        }
        return true;
      });
    }
    return unitList;
  }

  function positions(element) {
    if (textCache.has(element)) return textCache.get(element);
    const result = [];
    const card = element.matches("a.friend-card");
    const textRoot = card ? element.querySelector(".friend-name") || element : element;
    const walker = document.createTreeWalker(textRoot, NodeFilter.SHOW_TEXT, {
      acceptNode: node => node.parentElement.closest(skip) ? NodeFilter.FILTER_REJECT : NodeFilter.FILTER_ACCEPT
    });
    while (walker.nextNode()) {
      const node = walker.currentNode;
      // Cards are atomic navigation stops: select their label, not every glyph.
      if (card) {
        if (!node.data.trim()) continue;
        result.push({node, start: 0, end: node.length, text: node.data});
        break;
      }
      for (const part of segmenter.segment(node.data)) {
        result.push({node, start: part.index, end: part.index + part.segment.length, text: part.segment});
      }
    }
    textCache.set(element, result);
    return result;
  }

  function rangeFor(position) {
    const range = document.createRange();
    range.setStart(position.node, position.start);
    range.setEnd(position.node, position.end);
    return range;
  }

  // Measure one block on demand, not the entire document on every keystroke.
  // Coordinates are relative to the block, so ordinary scrolling needs no
  // per-character layout reads. Font/width/content changes invalidate the cache.
  function geometry(element) {
    if (geometryCache.has(element)) return geometryCache.get(element);
    if (element.matches("a.friend-card")) {
      const cell = {index: 0, row: 0, top: 0, left: 0, width: 10, height: 10};
      const value = {cells: [cell], rows: [{middle: 5, height: 10, cells: [cell]}]};
      geometryCache.set(element, value);
      return value;
    }
    const origin = element.getBoundingClientRect();
    const style = getComputedStyle(element);
    const lineHeight = parseFloat(style.lineHeight) || parseFloat(style.fontSize) * 1.6;
    const code = element.matches("pre");
    const points = positions(element);
    const cells = [], rows = [];
    let previous = null;
    points.forEach((point, i) => {
      const rect = rangeFor(point).getBoundingClientRect();
      const newline = /[\r\n]/.test(point.text);
      if ((!rect.width || !rect.height) && !(code && newline)) return;
      let top = rect.top - origin.top + element.scrollTop;
      let left = rect.left - origin.left + element.scrollLeft;
      let height = rect.height;
      let width = rect.width;
      if (newline && code) {
        width = Math.max(6, parseFloat(style.fontSize) * .6);
        if (!height) {
          height = previous?.height || lineHeight;
          top = previous ? previous.top + (previous.newline ? lineHeight : 0) : parseFloat(style.paddingTop);
          left = previous && !previous.newline ? previous.left + previous.width : parseFloat(style.paddingLeft);
        }
      }
      const cell = {index: i, top, left, width: Math.max(2, width), height, newline};
      cells[i] = cell;
      previous = cell;
    });
    // DOM order is not vertical order in tables and other multi-column blocks.
    for (const cell of cells.filter(Boolean).sort((a, b) => (a.top + a.height / 2) - (b.top + b.height / 2) || a.left - b.left)) {
      const middle = cell.top + cell.height / 2;
      let row = rows.at(-1);
      if (!row || Math.abs(middle - row.middle) > Math.min(cell.height, row.height) * .45) {
        row = {middle, height: cell.height, cells: []};
        rows.push(row);
      }
      cell.row = rows.length - 1;
      row.cells.push(cell);
    }
    rows.forEach(row => row.cells.sort((a, b) => a.left - b.left));
    const value = {cells, rows};
    geometryCache.set(element, value);
    return value;
  }

  function cellAt(element, at) {
    const data = geometry(element);
    return data.cells[at] || data.rows[0]?.cells[0];
  }

  function rectFor(element, cell) {
    const origin = element.getBoundingClientRect();
    return {left: origin.left + cell.left - element.scrollLeft,
      top: origin.top + cell.top - element.scrollTop,
      width: cell.width, height: cell.height};
  }

  function paint() {
    frame = 0;
    const cell = unit && cellAt(unit, index);
    if (!enabled || anchor || !cell || !unit.isConnected) {
      cursor.classList.remove("visible");
      return;
    }
    const rect = rectFor(unit, cell);
    const clip = scroller.getBoundingClientRect();
    let left = Math.max(rect.left, clip.left), top = Math.max(rect.top, clip.top);
    let right = Math.min(rect.left + rect.width, clip.right), bottom = Math.min(rect.top + rect.height, clip.bottom);
    for (let parent = unit; parent && parent !== scroller; parent = parent.parentElement) {
      if (/(auto|scroll|hidden|clip)/.test(getComputedStyle(parent).overflow)) {
        const bounds = parent.getBoundingClientRect();
        left = Math.max(left, bounds.left + parent.clientLeft);
        right = Math.min(right, bounds.left + parent.clientLeft + parent.clientWidth);
        top = Math.max(top, bounds.top + parent.clientTop);
        bottom = Math.min(bottom, bounds.top + parent.clientTop + parent.clientHeight);
      }
    }
    if (right <= left || bottom <= top) {
      cursor.classList.remove("visible");
      return;
    }
    cursor.style.width = `${right - left}px`;
    cursor.style.height = `${bottom - top}px`;
    cursor.style.transform = `translate3d(${left}px,${top}px,0)`;
    cursor.classList.add("visible");
  }

  function schedule() { if (!frame) frame = requestAnimationFrame(paint); }
  function invalidate(text = false) {
    goalX = null;
    geometryCache = new WeakMap();
    if (text) { textCache = new WeakMap(); unitList = []; }
    schedule();
  }

  function reveal() {
    const cell = unit && cellAt(unit, index);
    if (!cell) return;
    const rect = rectFor(unit, cell), bounds = scroller.getBoundingClientRect();
    const margin = Math.min(bounds.height / 4, Math.max(24, cell.height) * 3);
    let delta = 0;
    if (rect.top < bounds.top + margin) delta = rect.top - bounds.top - margin;
    else if (rect.top + rect.height > bounds.bottom - margin) delta = rect.top + rect.height - bounds.bottom + margin;
    if (delta) scroller.scrollBy({top: delta, behavior: behavior()});
    if (unit.matches("pre")) {
      const codeBounds = unit.getBoundingClientRect();
      const x = rect.left < codeBounds.left + 20 ? rect.left - codeBounds.left - 20 :
        rect.left + rect.width > codeBounds.right - 20 ? rect.left + rect.width - codeBounds.right + 20 : 0;
      if (x) unit.scrollBy({left: x, behavior: behavior()});
    }
  }

  function activate(element, at = null, {scroll = false, preserveX = false} = {}) {
    if (!element) return;
    const changed = element !== unit;
    unit = element;
    const data = geometry(unit);
    if (at !== null) index = at;
    else if (changed) index = data.rows[0]?.cells[0]?.index || 0;
    index = Math.max(0, Math.min(positions(unit).length - 1, index));
    const cell = cellAt(unit, index);
    if (cell) index = cell.index;
    if (!preserveX) goalX = cell ? rectFor(unit, cell).left + cell.width / 2 : null;
    onUnit(unit);
    if (anchor) updateSelection();
    if (scroll) reveal();
    schedule();
  }

  function nearest(row, x, element) {
    return row.cells.reduce((best, cell) => {
      const rect = rectFor(element, cell);
      const distance = Math.abs(rect.left + rect.width / 2 - x);
      return distance < best.distance ? {distance, index: cell.index} : best;
    }, {distance: Infinity, index: row.cells[0].index}).index;
  }

  function vertical(direction) {
    if (!unit) return;
    const data = geometry(unit), cell = cellAt(unit, index);
    const desiredX = goalX ?? (cell ? rectFor(unit, cell).left + cell.width / 2 : 0);
    let destination = unit, row = cell ? data.rows[cell.row + direction] : null;
    if (!row) {
      const list = units();
      let n = list.indexOf(unit) + direction;
      while (n >= 0 && n < list.length) {
        const rows = geometry(list[n]).rows;
        if (rows.length) { destination = list[n]; row = direction > 0 ? rows[0] : rows.at(-1); break; }
        n += direction;
      }
    }
    if (!row) {
      if (direction < 0) scroller.scrollTo({top: 0, behavior: behavior()});
      else scroller.scrollTo({top: scroller.scrollHeight, behavior: behavior()});
      return;
    }
    activate(destination, nearest(row, desiredX, destination), {scroll: true, preserveX: true});
    goalX = desiredX;
  }

  function horizontal(direction) {
    if (!unit) return;
    const data = geometry(unit), cell = cellAt(unit, index);
    if (!cell) return;
    const row = data.rows[cell.row];
    const n = row.cells.findIndex(candidate => candidate.index === index);
    const next = row.cells[Math.max(0, Math.min(row.cells.length - 1, n + direction))];
    activate(unit, next.index, {scroll: true});
  }

  // Vim's small-word classes: whitespace, letters/numbers/underscore, punctuation.
  const wordClass = text => /^\s+$/.test(text) ? 0 : /^[\p{L}\p{N}_\p{M}]+$/u.test(text) ? 1 : 2;
  function word(key) {
    if (!unit) return;
    const list = units();
    let u = list.indexOf(unit), i = index;
    const point = () => positions(list[u])[i];
    const step = direction => {
      i += direction;
      while (u >= 0 && u < list.length && (i < 0 || i >= positions(list[u]).length)) {
        u += direction;
        if (u < 0 || u >= list.length) return false;
        i = direction > 0 ? 0 : positions(list[u]).length - 1;
      }
      return u >= 0 && u < list.length;
    };
    const direction = key === "b" ? -1 : 1;
    const initialClass = wordClass(point()?.text || " ");
    let previousClass = initialClass;
    let started = false, last = {u, i};
    while (step(direction)) {
      if (key !== "w" && started && u !== last.u) { u = last.u; i = last.i; break; }
      const kind = wordClass(point()?.text || " ");
      if (key === "w" && kind && (kind !== previousClass || u !== last.u)) break;
      if (key === "b") {
        if (started && kind !== previousClass) { u = last.u; i = last.i; break; }
        if (kind) started = true;
      }
      if (key === "e") {
        if (started && kind !== previousClass) { u = last.u; i = last.i; break; }
        if (kind) started = true;
      }
      previousClass = kind;
      last = {u, i};
    }
    if (u < 0 || u >= list.length) { u = last.u; i = last.i; }
    activate(list[u], i, {scroll: true});
  }

  function boundary(key) {
    if (!unit) return;
    if (key === "gg" || key === "G") {
      const list = key === "gg" ? units() : [...units()].reverse();
      const dest = list.find(element => geometry(element).rows.length);
      if (!dest) return;
      const rows = geometry(dest).rows;
      activate(dest, key === "gg" ? rows[0].cells[0].index : rows.at(-1).cells.at(-1).index, {scroll: true});
      if (key === "gg") scroller.scrollTo({top: 0, behavior: behavior()});
    } else {
      const row = geometry(unit).rows[cellAt(unit, index).row];
      activate(unit, key === "0" ? row.cells[0].index : row.cells.at(-1).index, {scroll: true});
    }
  }

  function click(target, x, y) {
    const dest = units().find(element => element.contains(target));
    if (!dest) return false;
    const rows = geometry(dest).rows;
    if (!rows.length) return false;
    const origin = dest.getBoundingClientRect();
    const row = rows.reduce((best, candidate) => Math.abs(origin.top + candidate.middle - dest.scrollTop - y) < Math.abs(origin.top + best.middle - dest.scrollTop - y) ? candidate : best);
    activate(dest, nearest(row, x, dest));
    return true;
  }

  function updateSelection() {
    if (!anchor || !unit) return;
    const list = units();
    let a = anchor, b = {unit, index};
    if (list.indexOf(a.unit) > list.indexOf(b.unit) || a.unit === b.unit && a.index > b.index) [a, b] = [b, a];
    let ai = a.index, bi = b.index;
    if (linewise) {
      const ag = geometry(a.unit), bg = geometry(b.unit);
      ai = ag.rows[cellAt(a.unit, ai).row].cells[0].index;
      bi = bg.rows[cellAt(b.unit, bi).row].cells.at(-1).index;
    }
    const start = positions(a.unit)[ai], end = positions(b.unit)[bi];
    if (!start || !end) return;
    const range = rangeFor(start);
    range.setEnd(end.node, end.end);
    getSelection().removeAllRanges();
    getSelection().addRange(range);
  }

  function visual(lines) { anchor = {unit, index}; linewise = lines; updateSelection(); schedule(); }
  function normal() { anchor = null; getSelection().removeAllRanges(); schedule(); }
  function yank() {
    if (!unit) return "";
    const code = unit.matches("pre") ? unit : positions(unit)[index]?.node.parentElement.closest("pre");
    if (code) return code.textContent;
    const cell = cellAt(unit, index);
    if (!cell) return "";
    const row = geometry(unit).rows[cell.row], points = positions(unit);
    const range = rangeFor(points[row.cells[0].index]);
    const end = points[row.cells.at(-1).index];
    range.setEnd(end.node, end.end);
    return range.toString();
  }

  const observer = new ResizeObserver(() => invalidate());
  observer.observe(root);
  new MutationObserver(() => invalidate(true)).observe(root, {childList: true, subtree: true, characterData: true});
  document.fonts.ready.then(() => invalidate());
  document.fonts.addEventListener("loadingdone", () => invalidate());
  root.addEventListener("load", () => invalidate(), true);
  scroller.addEventListener("scroll", schedule, {passive: true, capture: true});
  window.addEventListener("resize", () => invalidate(), {passive: true});
  window.addEventListener("pageshow", () => { cursor.classList.add("snap"); invalidate(); requestAnimationFrame(() => requestAnimationFrame(() => cursor.classList.remove("snap"))); });

  return {units, activate, vertical, horizontal, word, boundary, click, visual, normal, yank, schedule, invalidate,
    position: () => positions(unit)[index],
    enable(value) { enabled = value; schedule(); },
    hide() { cursor.classList.remove("visible"); },
    current: () => unit};
}
