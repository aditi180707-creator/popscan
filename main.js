import {
  FilesetResolver,
  HandLandmarker,
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/vision_bundle.mjs";

const video = document.getElementById("video");
const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");
const statusEl = document.getElementById("status");

// offscreen canvas used to grab + posterize a snapshot of the video before warping it into the quad
const sample = document.createElement("canvas");
const sctx = sample.getContext("2d", { willReadFrequently: true });

// ---- tunables ----
const RECT_COLOR      = "90, 220, 255";
const CORNER_SIZE     = 14;
const RECT_HOLD_MS    = 250;   // a dropped hand vanishes the frame quickly instead of freezing it in place

const SAMPLE_W      = 260;     // down from 340 — still smooth, notably cheaper per frame
const BLUR_PASSES   = 2;
const EDGE_DARKEN    = 0.18;

const PALETTES = [
  [ // original — cyan/magenta/orange
    [10, 5, 20],
    [180, 20, 90],
    [255, 110, 20],
    [0, 210, 200],
    [255, 230, 40],
  ],
  [ // thermal — black/red/orange/yellow/white
    [5, 0, 0],
    [120, 10, 10],
    [220, 60, 10],
    [255, 160, 0],
    [255, 245, 200],
  ],
  [ // acid green/purple
    [8, 0, 15],
    [110, 10, 140],
    [40, 200, 90],
    [180, 255, 60],
    [240, 240, 255],
  ],
  [ // original particle-scan cyan, monochrome-ish
    [5, 10, 15],
    [20, 60, 90],
    [40, 120, 160],
    [80, 190, 230],
    [190, 240, 255],
  ],
];
let paletteIndex = 0;
function getPalette() { return PALETTES[paletteIndex]; }

const BURST_DURATION   = 900;   // ms the burst animation plays
const BURST_COOLDOWN   = 400;   // ms after a burst before the frame can reform
const BURST_GRID_X     = 26;    // particle grid resolution sampled from the posterized frame
const BURST_GRID_Y     = 16;
const BURST_SPEED_MIN  = 220;   // px/sec outward
const BURST_SPEED_MAX  = 520;
const BURST_GRAVITY    = 650;   // px/sec^2 downward drift
const BURST_SIZE       = 5;

const FLASH_DURATION   = 350;   // ms the activation flash plays when the frame first locks in

let handLandmarker;
let running = false;

// the 4 tracked corner points: thumb tip + index tip of each hand, named (not index-order),
// so the frame doesn't flip or jump when the model's hand order changes between frames
let rawPoints = { leftThumb: null, leftIndex: null, rightThumb: null, rightIndex: null };
let smoothPoints = { leftThumb: null, leftIndex: null, rightThumb: null, rightIndex: null };
let lastSeen = { leftThumb: 0, leftIndex: 0, rightThumb: 0, rightIndex: 0 };

// burst / flash / palm-gesture state
let burstActive = false;
let burstStart = 0;
let burstParticles = [];
let cooldownUntil = 0;
let flashStart = -Infinity;
let prevQuadReady = false;
let palmOpenPrev = false;

function dist2(a, b) {
  const dx = a.x - b.x, dy = a.y - b.y;
  return Math.sqrt(dx * dx + dy * dy);
}

// a finger counts as extended if its tip sits farther from the wrist than its middle
// joint does — open palm needs 3+ of the 4 main fingers extended at once
function isPalmOpen(lm) {
  const wrist = lm[0];
  const fingers = [[6, 8], [10, 12], [14, 16], [18, 20]]; // [pip, tip] per finger
  let extended = 0;
  for (const [pip, tip] of fingers) {
    if (dist2(lm[tip], wrist) > dist2(lm[pip], wrist)) extended++;
  }
  return extended >= 3;
}

// small on-screen hint for the keyboard controls
const hintEl = document.createElement("div");
hintEl.style.cssText =
  "position:fixed;bottom:12px;right:16px;color:rgba(122,215,255,0.65);" +
  "font:12px -apple-system,sans-serif;letter-spacing:0.04em;text-transform:uppercase;" +
  "pointer-events:none;z-index:10;text-align:right;line-height:1.6;";
hintEl.textContent = "C: color  ·  S: screenshot  ·  open palm: burst";
document.body.appendChild(hintEl);

window.addEventListener("keydown", (e) => {
  if (e.key === "c" || e.key === "C") {
    paletteIndex = (paletteIndex + 1) % PALETTES.length;
  } else if (e.key === "s" || e.key === "S") {
    takeScreenshot();
  }
});

function takeScreenshot() {
  const link = document.createElement("a");
  link.download = `particle-scan-${Date.now()}.png`;
  link.href = canvas.toDataURL("image/png");
  link.click();
}

function resize() {
  canvas.width = window.innerWidth * devicePixelRatio;
  canvas.height = window.innerHeight * devicePixelRatio;
  canvas.style.width = window.innerWidth + "px";
  canvas.style.height = window.innerHeight + "px";
  ctx.setTransform(devicePixelRatio, 0, 0, devicePixelRatio, 0, 0);
}
window.addEventListener("resize", resize);
resize();

async function setup() {
  statusEl.textContent = "loading model…";
  const filesetResolver = await FilesetResolver.forVisionTasks(
    "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm"
  );

  handLandmarker = await HandLandmarker.createFromOptions(filesetResolver, {
    baseOptions: {
      modelAssetPath:
        "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
      delegate: "GPU", // safe now — hand tracking is the only model running, no more GPU contention
    },
    runningMode: "VIDEO",
    numHands: 2,
    minHandDetectionConfidence: 0.35,
    minHandPresenceConfidence: 0.35,
    minTrackingConfidence: 0.35,
  });

  statusEl.textContent = "requesting camera…";
  const stream = await navigator.mediaDevices.getUserMedia({
    video: { width: 960, height: 720, facingMode: "user" },
    audio: false,
  });
  video.srcObject = stream;
  await new Promise((res) => (video.onloadedmetadata = res));
  video.play();

  statusEl.textContent = "tracking";
  running = true;
  requestAnimationFrame(loop);
}

let lastVideoTime = -1;
let quadReady = false;
let TL, TR, BR, BL;

function loop() {
  if (!running) return;

  if (video.currentTime !== lastVideoTime) {
    lastVideoTime = video.currentTime;
    const now = performance.now();
    const handResult = handLandmarker.detectForVideo(video, now);
    updatePoints(handResult, now);

    // the expensive posterize pipeline only needs to run once per actual camera frame,
    // not once per animation frame (which can fire faster than the webcam delivers frames)
    const have = Object.values(smoothPoints).filter(Boolean).length;
    const inCooldown = now < cooldownUntil;
    quadReady = have === 4 && !burstActive && !inCooldown;

    if (quadReady && !prevQuadReady) flashStart = now; // just locked in — trigger the activation flash
    prevQuadReady = quadReady;

    if (quadReady) {
      const w = window.innerWidth, h = window.innerHeight;
      const { tl, tr, br, bl } = orderQuad(smoothPoints);
      TL = { x: tl.x * w, y: tl.y * h };
      TR = { x: tr.x * w, y: tr.y * h };
      BR = { x: br.x * w, y: br.y * h };
      BL = { x: bl.x * w, y: bl.y * h };
      preparePosterizedSample(tl, tr, br, bl);
    }
  }

  draw();
  requestAnimationFrame(loop);
}

function toPoint(lm) {
  return { x: 1 - lm.x, y: lm.y }; // mirrored
}

const SMOOTH = 0.45;
function lerpPoint(prev, target) {
  if (!prev) return target;
  return {
    x: prev.x + (target.x - prev.x) * SMOOTH,
    y: prev.y + (target.y - prev.y) * SMOOTH,
  };
}

function updatePoints(result, now) {
  const hands = result.landmarks || [];
  const handedness = result.handednesses || result.handedness || [];

  rawPoints = { leftThumb: null, leftIndex: null, rightThumb: null, rightIndex: null };

  let palmOpenNow = false;
  for (const lm of hands) {
    if (isPalmOpen(lm)) palmOpenNow = true;
  }
  // rising edge only — fire once per open-palm gesture, not every frame it's held
  if (palmOpenNow && !palmOpenPrev && quadReady && !burstActive) {
    triggerBurst(now);
  }
  palmOpenPrev = palmOpenNow;

  for (let i = 0; i < hands.length; i++) {
    const label = handedness[i] && handedness[i][0] && handedness[i][0].categoryName;
    const thumb = toPoint(hands[i][4]);
    const index = toPoint(hands[i][8]);
    if (label === "Left") {
      rawPoints.leftThumb = thumb;
      rawPoints.leftIndex = index;
    } else if (label === "Right") {
      rawPoints.rightThumb = thumb;
      rawPoints.rightIndex = index;
    } else if (!rawPoints.leftThumb) {
      rawPoints.leftThumb = thumb;
      rawPoints.leftIndex = index;
    } else {
      rawPoints.rightThumb = thumb;
      rawPoints.rightIndex = index;
    }
  }

  for (const key of Object.keys(rawPoints)) {
    if (rawPoints[key]) {
      smoothPoints[key] = lerpPoint(smoothPoints[key], rawPoints[key]);
      lastSeen[key] = now;
    } else if (now - lastSeen[key] > RECT_HOLD_MS) {
      smoothPoints[key] = null;
    }
  }

  statusEl.textContent = `tracking — hands: ${hands.length}` +
    ` | pts: ${Object.values(smoothPoints).filter(Boolean).length}/4`;
}

// order the 4 tracked points into TL/TR/BR/BL by position — robust to the hands tilting
// or getting close together (what makes the reference frame warp into a triangle shape)
function orderQuad(points) {
  const pts = Object.values(points);
  const sorted = [...pts].sort((a, b) => a.x - b.x);
  const leftTwo = sorted.slice(0, 2).sort((a, b) => a.y - b.y);
  const rightTwo = sorted.slice(2, 4).sort((a, b) => a.y - b.y);
  return {
    tl: leftTwo[0], bl: leftTwo[1],
    tr: rightTwo[0], br: rightTwo[1],
  };
}

function quantizeToPalette(d) {
  const palette = getPalette();
  for (let i = 0; i < d.length; i += 4) {
    const luma = 0.299 * d[i] + 0.587 * d[i + 1] + 0.114 * d[i + 2];
    const bucket = Math.min(palette.length - 1, Math.floor((luma / 256) * palette.length));
    const [pr, pg, pb] = palette[bucket];
    d[i] = pr; d[i + 1] = pg; d[i + 2] = pb;
  }
}

// separable box blur (horizontal pass then vertical pass) — same visual effect as a full
// 3x3 kernel at roughly a third of the reads, and reuses one scratch buffer instead of
// allocating a new array every call (that allocation churn was adding up every frame)
let blurScratch = null;
function boxBlur(d, w, h) {
  if (!blurScratch || blurScratch.length !== d.length) {
    blurScratch = new Uint8ClampedArray(d.length);
  }
  const tmp = blurScratch;

  // horizontal
  for (let y = 0; y < h; y++) {
    const row = y * w;
    for (let x = 0; x < w; x++) {
      let r = 0, g = 0, b = 0, n = 0;
      for (let dx = -1; dx <= 1; dx++) {
        const nx = x + dx;
        if (nx < 0 || nx >= w) continue;
        const i = (row + nx) * 4;
        r += d[i]; g += d[i + 1]; b += d[i + 2]; n++;
      }
      const o = (row + x) * 4;
      tmp[o] = r / n; tmp[o + 1] = g / n; tmp[o + 2] = b / n; tmp[o + 3] = d[o + 3];
    }
  }

  // vertical
  for (let x = 0; x < w; x++) {
    for (let y = 0; y < h; y++) {
      let r = 0, g = 0, b = 0, n = 0;
      for (let dy = -1; dy <= 1; dy++) {
        const ny = y + dy;
        if (ny < 0 || ny >= h) continue;
        const i = (ny * w + x) * 4;
        r += tmp[i]; g += tmp[i + 1]; b += tmp[i + 2]; n++;
      }
      const o = (y * w + x) * 4;
      d[o] = r / n; d[o + 1] = g / n; d[o + 2] = b / n;
    }
  }
}

function drawEdges(d, w, h) {
  for (let y = 1; y < h - 1; y++) {
    for (let x = 1; x < w - 1; x++) {
      const i = (y * w + x) * 4;
      const iR = i + 4;
      const iD = i + w * 4;
      const diff = Math.abs(d[i] - d[iR]) + Math.abs(d[i + 1] - d[iR + 1]) + Math.abs(d[i + 2] - d[iR + 2]) +
                   Math.abs(d[i] - d[iD]) + Math.abs(d[i + 1] - d[iD + 1]) + Math.abs(d[i + 2] - d[iD + 2]);
      if (diff > 40) {
        d[i]     *= 1 - EDGE_DARKEN;
        d[i + 1] *= 1 - EDGE_DARKEN;
        d[i + 2] *= 1 - EDGE_DARKEN;
      }
    }
  }
}

// standard 2D-canvas triangle texture mapping: solves the affine transform that sends
// the source triangle to the destination triangle, then clips + draws the image through it
function drawTriangle(img, sx0, sy0, sx1, sy1, sx2, sy2, dx0, dy0, dx1, dy1, dx2, dy2) {
  ctx.save();
  ctx.beginPath();
  ctx.moveTo(dx0, dy0);
  ctx.lineTo(dx1, dy1);
  ctx.lineTo(dx2, dy2);
  ctx.closePath();
  ctx.clip();

  const denom = sx0 * (sy1 - sy2) + sx1 * (sy2 - sy0) + sx2 * (sy0 - sy1);
  if (Math.abs(denom) < 1e-6) { ctx.restore(); return; }

  const a = (dx0 * (sy1 - sy2) + dx1 * (sy2 - sy0) + dx2 * (sy0 - sy1)) / denom;
  const b = (dy0 * (sy1 - sy2) + dy1 * (sy2 - sy0) + dy2 * (sy0 - sy1)) / denom;
  const c = (dx0 * (sx2 - sx1) + dx1 * (sx0 - sx2) + dx2 * (sx1 - sx0)) / denom;
  const d = (dy0 * (sx2 - sx1) + dy1 * (sx0 - sx2) + dy2 * (sx1 - sx0)) / denom;
  const e = (dx0 * (sx1 * sy2 - sx2 * sy1) + dx1 * (sx2 * sy0 - sx0 * sy2) + dx2 * (sx0 * sy1 - sx1 * sy0)) / denom;
  const f = (dy0 * (sx1 * sy2 - sx2 * sy1) + dy1 * (sx2 * sy0 - sx0 * sy2) + dy2 * (sx0 * sy1 - sx1 * sy0)) / denom;

  ctx.transform(a, b, c, d, e, f);
  ctx.imageSmoothingEnabled = true;
  ctx.imageSmoothingQuality = "high";
  ctx.drawImage(img, 0, 0);
  ctx.restore();
}

function triggerBurst(now) {
  if (!TL || !TR || !BR || !BL) return;
  const sw = sample.width, sh = sample.height;
  if (!sw || !sh) return;

  const imgData = sctx.getImageData(0, 0, sw, sh).data;
  const cx = (TL.x + TR.x + BR.x + BL.x) / 4;
  const cy = (TL.y + TR.y + BR.y + BL.y) / 4;
  const particles = [];

  for (let gy = 0; gy < BURST_GRID_Y; gy++) {
    for (let gx = 0; gx < BURST_GRID_X; gx++) {
      const u = (gx + 0.5) / BURST_GRID_X;
      const v = (gy + 0.5) / BURST_GRID_Y;
      const px = Math.min(sw - 1, Math.floor(u * sw));
      const py = Math.min(sh - 1, Math.floor(v * sh));
      const idx = (py * sw + px) * 4;

      // bilinear position of this grid cell within the current quad
      const topX = TL.x * (1 - u) + TR.x * u, topY = TL.y * (1 - u) + TR.y * u;
      const botX = BL.x * (1 - u) + BR.x * u, botY = BL.y * (1 - u) + BR.y * u;
      const x = topX * (1 - v) + botX * v;
      const y = topY * (1 - v) + botY * v;

      const dx = x - cx, dy = y - cy;
      const d = Math.sqrt(dx * dx + dy * dy) || 1;
      const speed = BURST_SPEED_MIN + Math.random() * (BURST_SPEED_MAX - BURST_SPEED_MIN);
      const jitter = (Math.random() - 0.5) * 0.6; // radians, so the burst isn't perfectly radial
      const cos = Math.cos(jitter), sin = Math.sin(jitter);
      const ndx = dx / d, ndy = dy / d;

      particles.push({
        x, y,
        vx: (ndx * cos - ndy * sin) * speed,
        vy: (ndx * sin + ndy * cos) * speed,
        r: imgData[idx], g: imgData[idx + 1], b: imgData[idx + 2],
        size: BURST_SIZE * (0.6 + Math.random() * 0.8),
      });
    }
  }

  burstParticles = particles;
  burstActive = true;
  burstStart = now;
  cooldownUntil = now + BURST_DURATION + BURST_COOLDOWN;
}

function drawBurst(now) {
  const elapsed = now - burstStart;
  if (elapsed > BURST_DURATION) { burstActive = false; return; }

  const t = elapsed / 1000;
  const alpha = Math.max(0, 1 - elapsed / BURST_DURATION);

  for (const p of burstParticles) {
    const x = p.x + p.vx * t;
    const y = p.y + p.vy * t + 0.5 * BURST_GRAVITY * t * t;
    ctx.fillStyle = `rgba(${p.r}, ${p.g}, ${p.b}, ${alpha})`;
    ctx.fillRect(x - p.size / 2, y - p.size / 2, p.size, p.size);
  }
}

function drawFlash(now) {
  const elapsed = now - flashStart;
  if (elapsed > FLASH_DURATION || !TL) return;
  const alpha = (1 - elapsed / FLASH_DURATION) * 0.7;

  ctx.save();
  ctx.beginPath();
  ctx.moveTo(TL.x, TL.y);
  ctx.lineTo(TR.x, TR.y);
  ctx.lineTo(BR.x, BR.y);
  ctx.lineTo(BL.x, BL.y);
  ctx.closePath();
  ctx.clip();
  ctx.fillStyle = `rgba(255, 255, 255, ${alpha})`;
  ctx.fillRect(0, 0, window.innerWidth, window.innerHeight);
  ctx.restore();
}

function draw() {
  const w = window.innerWidth;
  const h = window.innerHeight;
  ctx.clearRect(0, 0, w, h);

  const now = performance.now();

  if (burstActive) {
    drawBurst(now);
    return;
  }

  if (!quadReady) return;

  // two triangles cover the quad: TL-TR-BR and TL-BR-BL
  drawTriangle(sample,
    0, 0, SAMPLE_W, 0, SAMPLE_W, sample.height,
    TL.x, TL.y, TR.x, TR.y, BR.x, BR.y);
  drawTriangle(sample,
    0, 0, SAMPLE_W, sample.height, 0, sample.height,
    TL.x, TL.y, BR.x, BR.y, BL.x, BL.y);

  drawFlash(now);

  // frame border + corner handles
  ctx.strokeStyle = `rgba(${RECT_COLOR}, 0.9)`;
  ctx.lineWidth = 2;
  ctx.shadowColor = `rgba(${RECT_COLOR}, 0.8)`;
  ctx.shadowBlur = 6;
  ctx.beginPath();
  ctx.moveTo(TL.x, TL.y);
  ctx.lineTo(TR.x, TR.y);
  ctx.lineTo(BR.x, BR.y);
  ctx.lineTo(BL.x, BL.y);
  ctx.closePath();
  ctx.stroke();
  ctx.shadowBlur = 0;

  for (const p of [TL, TR, BR, BL]) drawCorner(p.x, p.y);
}

function drawCorner(cx, cy) {
  ctx.fillStyle = `rgba(${RECT_COLOR}, 1)`;
  ctx.fillRect(cx - CORNER_SIZE / 2, cy - CORNER_SIZE / 2, CORNER_SIZE, CORNER_SIZE);
}

function preparePosterizedSample(tl, tr, br, bl) {
  const vw = video.videoWidth, vh = video.videoHeight;
  if (!vw || !vh) return;

  const screenW = window.innerWidth, screenH = window.innerHeight;
  const videoAspect = vw / vh, screenAspect = screenW / screenH;
  let drawW, drawH, offX, offY;
  if (screenAspect > videoAspect) {
    drawW = screenW; drawH = screenW / videoAspect;
    offX = 0; offY = (screenH - drawH) / 2;
  } else {
    drawH = screenH; drawW = screenH * videoAspect;
    offX = (screenW - drawW) / 2; offY = 0;
  }
  const scaleX = vw / drawW, scaleY = vh / drawH;

  // bounding box of the 4 (normalized, mirrored) points -> video pixel space
  const minX = Math.min(tl.x, tr.x, br.x, bl.x);
  const maxX = Math.max(tl.x, tr.x, br.x, bl.x);
  const minY = Math.min(tl.y, tr.y, br.y, bl.y);
  const maxY = Math.max(tl.y, tr.y, br.y, bl.y);

  const screenX1 = minX * screenW, screenX2 = maxX * screenW;
  const screenY1 = minY * screenH, screenY2 = maxY * screenH;

  const srcX1 = vw - (screenX2 - offX) * scaleX; // mirrored
  const srcY1 = (screenY1 - offY) * scaleY;
  const srcW = Math.max(1, (screenX2 - screenX1) * scaleX);
  const srcH = Math.max(1, (screenY2 - screenY1) * scaleY);

  const sampleH = Math.max(1, Math.round(SAMPLE_W * (srcH / srcW)));
  sample.width = SAMPLE_W;
  sample.height = sampleH;
  sctx.save();
  sctx.scale(-1, 1);
  sctx.drawImage(video, srcX1, srcY1, srcW, srcH, -SAMPLE_W, 0, SAMPLE_W, sampleH);
  sctx.restore();

  const imgData = sctx.getImageData(0, 0, SAMPLE_W, sampleH);
  const d = imgData.data;

  for (let p = 0; p < BLUR_PASSES; p++) boxBlur(d, SAMPLE_W, sampleH);
  drawEdges(d, SAMPLE_W, sampleH);
  quantizeToPalette(d);

  sctx.putImageData(imgData, 0, 0);
}

setup().catch((err) => {
  console.error(err);
  statusEl.textContent = "error: " + err.message;
});
