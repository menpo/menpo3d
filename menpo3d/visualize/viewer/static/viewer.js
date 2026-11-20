// Streaming 3D viewer client. Any number of Viewer objects on a page share ONE websocket per server (a Connection).
// Frames arrive as H.264 (WebCodecs) or JPEG, tagged with the viewer id and the view-projection matrix they were rendered with,
// so landmark labels and picked points are drawn at exactly the position of the frame on screen.
const HEADER = 87;  // kind u8, viewer u16, seq u32, echo f64, render f32, encode f32, 16 x f32 matrix
const connections = (window.__meshStreams = window.__meshStreams || new Map());

class Connection {
  constructor(url, token) {
    this.url = url.replace(/^http/, 'ws') + '/ws?token=' + encodeURIComponent(token || '');
    this.viewers = new Map(); this.nextId = 1; this.ws = null; this.closing = false; this.rtt = null;
    addEventListener('beforeunload', () => { this.closing = true; });
    addEventListener('pageshow', () => {   // restored from the back/forward cache: the server may have closed us while frozen
      this.closing = false;
      if (this.viewers.size && (!this.ws || this.ws.readyState > 1)) this.open();
    });
    this.ping = setInterval(() => this.send({t: 'ping', ts: performance.now()}), 5000);  // keeps the link measured; a visible idle page proves it is alive
    this.open();
  }
  open() {
    const ws = this.ws = new WebSocket(this.url); ws.binaryType = 'arraybuffer';
    ws.onopen = () => { for (const v of this.viewers.values()) v.attach(); };
    ws.onmessage = ev => this.onmessage(ev);
    ws.onclose = () => {
      for (const v of this.viewers.values()) v.detach();
      if (!this.closing && this.viewers.size) setTimeout(() => this.open(), 1000);
    };
  }
  send(o) { if (this.ws && this.ws.readyState === 1) this.ws.send(JSON.stringify(o)); }
  add(viewer) { viewer.id = this.nextId++; this.viewers.set(viewer.id, viewer); if (this.ws.readyState === 1) viewer.attach(); }
  remove(viewer) {
    this.send({t: 'close', v: viewer.id}); this.viewers.delete(viewer.id);
    if (!this.viewers.size) {  // last viewer gone: close the socket so the server can free everything (and exit if asked to)
      this.closing = true; clearInterval(this.ping); this.ws.close(); connections.delete(this.key);
    }
  }
  onmessage(ev) {
    if (typeof ev.data === 'string') {
      const m = JSON.parse(ev.data);
      if (m.t === 'pong') { this.rtt = performance.now() - m.ts; return; }
      const v = this.viewers.get(m.v); if (v) v.onjson(m);
      return;
    }
    const dv = new DataView(ev.data), v = this.viewers.get(dv.getUint16(1, true));
    if (v) v.onframe(ev.data, dv);
  }
}

export class Viewer {
  constructor(el, opts = {}) {
    this.opts = Object.assign({url: location.origin, token: '', scene: null, fit: false, codec: typeof VideoDecoder === 'undefined' ? 'jpeg' : 'h264', quality: 0, bw: 0, rtt: 0, win: 0, adapt: true, w: 640, h: 480, hud: false, labels: true}, opts);
    const {w, h, fit} = this.opts;
    this.root = document.createElement('div');
    this.root.style.cssText = fit ? 'position:relative;width:100%;height:100%;background:#fff;line-height:0;overflow:hidden' : `position:relative;width:${w}px;height:${h}px;background:#fff;display:inline-block;line-height:0`;
    this.canvas = document.createElement('canvas'); this.canvas.width = w; this.canvas.height = h; this.canvas.style.cursor = 'grab'; this.canvas.tabIndex = 0; this.canvas.style.outline = 'none';
    this.overlay = document.createElement('canvas'); this.overlay.width = w; this.overlay.height = h; this.overlay.style.cssText = 'position:absolute;left:0;top:0;pointer-events:none';
    for (const c of [this.canvas, this.overlay]) { c.style.width = fit ? '100%' : w + 'px'; c.style.height = fit ? '100%' : h + 'px'; }
    this.root.append(this.canvas, this.overlay); el.appendChild(this.root);
    if (this.opts.hud) {
      this.hudEl = document.createElement('div'); this.hudEl.style.cssText = 'position:absolute;top:4px;left:8px;font:12px monospace;color:#ddd;white-space:pre;pointer-events:none;text-shadow:0 0 3px #000;line-height:1.2';
      this.root.appendChild(this.hudEl); this.hudTimer = setInterval(() => this.updateHud(), 1000);
    }
    this.g = this.canvas.getContext('2d'); this.og = this.overlay.getContext('2d');
    this.decoder = null; this.needKey = true; this.pending = new Map(); this.meta = {labels: [], fov: 35}; this.mvp = null; this.picked = null;
    this.reqs = new Map(); this.nextReq = 1; this.resetStats();
    this.inView = true; this.tabShown = document.visibilityState === 'visible'; this.vis = true;
    this.bindInput(); this.bindVisibility();
    const key = this.opts.url + '|' + this.opts.token; this.conn = connections.get(key) || new Connection(this.opts.url, this.opts.token); this.conn.key = key; connections.set(key, this.conn);
    if (fit) { this.ro = new ResizeObserver(() => this.refit()); this.ro.observe(this.root); this.refit(true); }
    this.conn.add(this);
  }
  refit(initial) {   // fill the parent at device resolution; the server clamps to 3840x2160
    const r = this.root.getBoundingClientRect(), d = window.devicePixelRatio || 1;
    const w = Math.max(16, Math.round(r.width * d)), h = Math.max(16, Math.round(r.height * d));
    if (w === this.canvas.width && h === this.canvas.height) return;
    this.opts.w = this.canvas.width = this.overlay.width = w; this.opts.h = this.canvas.height = this.overlay.height = h;
    if (!initial) this.send({t: 'resize', w, h});
  }
  send(o) { o.v = this.id; this.conn.send(o); }
  attach() {  // (re)open the session on the server and send the configuration
    const o = this.opts;
    this.send({t: 'open'});
    if (o.scene) this.send({t: 'scene', id: o.scene});
    this.send({t: 'cfg', codec: o.codec, quality: o.quality, bw: o.bw, rtt: o.rtt, win: o.win, adapt: o.adapt});
    this.send({t: 'visible', on: this.vis});
    this.send({t: 'resize', w: o.w, h: o.h});
    if (this.style) this.send(Object.assign({t: 'style'}, this.style));
    this.send({t: 'drag', dx: 0, dy: 0, ts: performance.now()});
  }
  detach() { if (this.decoder) { try { this.decoder.close(); } catch (e) {} } this.decoder = null; this.needKey = true; this.pending.clear(); }
  destroy() { clearInterval(this.hudTimer); this.ro && this.ro.disconnect(); this.io && this.io.disconnect(); this.removeVis && this.removeVis(); this.detach(); this.conn.remove(this); this.root.remove(); }

  // ---- public API
  setStyle(style) { this.style = Object.assign(this.style || {}, style); this.send(Object.assign({t: 'style'}, this.style)); }
  reset() { this.send({t: 'reset', ts: performance.now()}); }
  setPose(pose) { this.send({t: 'pose', pose, ts: performance.now()}); }
  getPose() { return new Promise(res => { this._poseRes = res; this.send({t: 'getpose'}); }); }
  pick(x, y) { return new Promise(res => { const req = this.nextReq++; this.reqs.set(req, res); this.send({t: 'pick', x, y, req}); }); }
  still() { return this.canvas.toDataURL('image/png'); }

  // ---- messages
  onjson(m) {
    if (m.t === 'meta') { this.meta = m; this.drawOverlay(); }
    else if (m.t === 'error') { this.error = m.message; console.error('menpo3d viewer:', m.message); this.drawOverlay(); }
    else if (m.t === 'pose') { this.pose = m.pose; if (this._poseRes) { this._poseRes(m.pose); this._poseRes = null; } }
    else if (m.t === 'picked') { const r = this.reqs.get(m.req); this.reqs.delete(m.req); this.picked = m.hit ? {hit: m.hit, x: this._px, y: this._py} : null; this.drawOverlay(); if (r) r(m.hit); if (this.onpick) this.onpick(m.hit); }
  }
  makeDecoder() {
    this.decoder = new VideoDecoder({
      output: frame => { const info = this.pending.get(frame.timestamp); this.pending.delete(frame.timestamp); this.g.drawImage(frame, 0, 0, this.canvas.width, this.canvas.height); frame.close(); if (info) this.done(info); },
      error: e => { this.stats.errors++; console.error('decoder', e); this.needKey = true; this.makeDecoder(); this.send({t: 'key'}); this.send({t: 'ack'}); },
    });
    this.decoder.configure({codec: 'avc1.42E02A', avc: {format: 'annexb'}, optimizeForLatency: true});
  }
  async onframe(buf, dv) {
    const kind = dv.getUint8(0), seq = dv.getUint32(3, true), echo = dv.getFloat64(7, true), st = this.stats;
    st.render.push(dv.getFloat32(15, true)); st.enc.push(dv.getFloat32(19, true)); st.bytes += buf.byteLength;
    const mvp = new Float32Array(16); for (let i = 0; i < 16; i++) mvp[i] = dv.getFloat32(23 + 4 * i, true);
    const payload = new Uint8Array(buf, HEADER), info = {kind, echo, mvp, tRecv: performance.now()};
    if (kind === 2 || kind === 3) {
      const bmp = await createImageBitmap(new Blob([payload], {type: 'image/jpeg'}));
      this.g.drawImage(bmp, 0, 0, this.canvas.width, this.canvas.height); bmp.close(); this.done(info);
      if (kind === 3) this.onstill();
    } else {
      if (!this.decoder) this.makeDecoder();
      if (this.needKey && kind !== 1) { this.send({t: 'key'}); this.send({t: 'ack'}); return; }
      this.needKey = false; this.pending.set(seq, info);
      this.decoder.decode(new EncodedVideoChunk({type: kind === 1 ? 'key' : 'delta', timestamp: seq, data: payload}));
    }
  }
  onstill() { if (this.onstillframe) this.onstillframe(this.still()); }
  done(info) {
    const st = this.stats, now = performance.now();
    st.frames++; st.kinds[info.kind] = (st.kinds[info.kind] || 0) + 1;
    if (info.echo > 0) st.lat.push(now - info.echo);
    st.dec.push(now - info.tRecv);
    this.mvp = info.mvp; this.drawOverlay();
    this.send({t: 'ack'});
  }

  // ---- labels and the picked point, projected with the matrix of the frame that is on screen
  project(p) {
    const m = this.mvp; if (!m) return null;
    const x = m[0] * p[0] + m[1] * p[1] + m[2] * p[2] + m[3], y = m[4] * p[0] + m[5] * p[1] + m[6] * p[2] + m[7], w = m[12] * p[0] + m[13] * p[1] + m[14] * p[2] + m[15];
    if (w <= 0) return null;
    return [(x / w + 1) * this.canvas.width / 2, (1 - y / w) * this.canvas.height / 2];
  }
  drawOverlay() {
    const g = this.og; g.clearRect(0, 0, this.overlay.width, this.overlay.height);
    if (this.error) { g.fillStyle = '#c00'; g.font = '14px sans-serif'; g.fillText(this.error, 10, 20); }
    if (!this.mvp) return;
    g.font = '12px sans-serif'; g.textBaseline = 'middle';
    if (this.opts.labels && this.meta.labels) {
      const focal = 1 / Math.tan(this.meta.fov * Math.PI / 360), h = this.canvas.height;
      for (const l of this.meta.labels) {
        const s = this.project(l.p); if (!s) continue;
        const m = this.mvp, w = m[12] * l.p[0] + m[13] * l.p[1] + m[14] * l.p[2] + m[15];
        const px = Math.max(7, Math.min(48, l.size * focal * (h / 2) / w * 1.5));
        g.font = `${px}px sans-serif`; g.textBaseline = 'alphabetic'; g.fillStyle = `rgb(${l.colour.map(c => Math.round(c * 255)).join(',')})`; g.fillText(l.text, s[0], s[1]);
      }
    }
    if (this.picked) {
      const {x, y, hit} = this.picked;
      g.strokeStyle = '#f33'; g.lineWidth = 2; g.beginPath(); g.moveTo(x - 8, y); g.lineTo(x + 8, y); g.moveTo(x, y - 8); g.lineTo(x, y + 8); g.stroke();
      const txt = `tri ${hit.tri}  vertex ${hit.vertex}`; g.fillStyle = '#fff'; g.strokeStyle = '#000'; g.lineWidth = 3; g.strokeText(txt, x + 10, y - 10); g.fillText(txt, x + 10, y - 10);
    }
  }

  // ---- input and visibility
  bindInput() {
    const c = this.canvas; let down = false, moved = 0, pan = false;
    c.oncontextmenu = e => e.preventDefault();
    c.onpointerdown = e => { down = true; moved = 0; pan = e.button === 1 || e.button === 2 || e.shiftKey; c.focus(); c.setPointerCapture(e.pointerId); c.style.cursor = pan ? 'move' : 'grabbing'; };
    c.onpointerup = e => {
      down = false; c.style.cursor = 'grab';
      if (moved < 4 && e.button === 0 && !pan) { const r = c.getBoundingClientRect(); this._px = (e.clientX - r.left) * c.width / r.width; this._py = (e.clientY - r.top) * c.height / r.height; this.pick(this._px, this._py); }
    };
    c.onpointermove = e => {
      if (!down) return;
      const k = c.width / c.getBoundingClientRect().width;   // CSS pixels to canvas pixels
      moved += Math.abs(e.movementX) + Math.abs(e.movementY); this.picked = null;
      this.send({t: pan ? 'pan' : 'drag', dx: e.movementX * (pan ? k : 1), dy: e.movementY * (pan ? k : 1), ts: performance.now()});
    };
    c.onwheel = e => { e.preventDefault(); this.send({t: 'wheel', dy: e.deltaY, ts: performance.now()}); };
    c.onkeydown = e => { if (e.key === 'r') this.reset(); };
  }
  bindVisibility() {  // scrolled out of view / another tab / page navigated away: the server stops rendering and frees the encoder
    const set = () => { const v = this.inView && this.tabShown; if (v !== this.vis) { this.vis = v; this.send({t: 'visible', on: v}); } };
    this.io = new IntersectionObserver(es => { this.inView = es[es.length - 1].isIntersecting; set(); }); this.io.observe(this.root);
    const vc = () => { this.tabShown = document.visibilityState === 'visible'; set(); }, hide = () => { this.tabShown = false; set(); }, show = () => { this.tabShown = true; set(); };
    document.addEventListener('visibilitychange', vc); addEventListener('pagehide', hide); addEventListener('pageshow', show);
    this.removeVis = () => { document.removeEventListener('visibilitychange', vc); removeEventListener('pagehide', hide); removeEventListener('pageshow', show); };
  }

  // ---- statistics
  resetStats() { this.stats = {t0: performance.now(), frames: 0, bytes: 0, lat: [], dec: [], render: [], enc: [], kinds: {}, errors: 0}; }
  summary() {
    const s = this.stats, dt = (performance.now() - s.t0) / 1000, avg = a => a.length ? a.reduce((x, y) => x + y, 0) / a.length : 0;
    const pct = (a, p) => { if (!a.length) return 0; const b = [...a].sort((x, y) => x - y); return b[Math.min(b.length - 1, Math.floor(p * b.length))]; };
    return {codec: this.opts.codec, seconds: +dt.toFixed(1), frames: s.frames, fps: +(s.frames / dt).toFixed(1), mbps: +(s.bytes * 8 / dt / 1e6).toFixed(2),
      kb_per_frame: s.frames ? +(s.bytes / s.frames / 1024).toFixed(1) : 0, lat_p50: pct(s.lat, .5), lat_p95: pct(s.lat, .95),
      decode_ms: avg(s.dec), render_ms: avg(s.render), encode_ms: avg(s.enc), kinds: s.kinds, errors: s.errors};
  }
  updateHud() { const s = this.summary(), r = this.conn.rtt; this.hudEl.textContent = `${s.codec} ${s.fps} fps ${s.mbps} Mbit/s ${s.kb_per_frame} KB/frame latency p50 ${s.lat_p50 && s.lat_p50.toFixed(0)} ms` + (r === null ? '' : ` ping ${r.toFixed(0)} ms`); }
}
