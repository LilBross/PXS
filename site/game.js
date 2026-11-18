/* global module -- optional CommonJS export for controller regression tests */
(() => {
  "use strict";

  const EPS = 1e-4;
  const GROUND_PROBE = 0.035;
  const STEP_DOWN_SNAP = 0.30;
  const MAX_LIGHTS = 8;

  // ---------- Base64 binary helpers ----------
  function b64ToUint8(b64) {
    const bin = atob(b64);
    const len = bin.length;
    const out = new Uint8Array(len);
    for (let i = 0; i < len; i++) out[i] = bin.charCodeAt(i);
    return out;
  }

  function b64ToFloat32(b64) {
    const bytes = b64ToUint8(b64);
    return new Float32Array(bytes.buffer, bytes.byteOffset, bytes.byteLength / 4);
  }

  // ---------- 3D Math (matches engine/math3d.py 1:1, row-major -> col-major for GL) ----------
  function vec3Sub(a, b) { return [a[0] - b[0], a[1] - b[1], a[2] - b[2]]; }
  function vec3Add(a, b) { return [a[0] + b[0], a[1] + b[1], a[2] + b[2]]; }
  function vec3Dot(a, b) { return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]; }
  function vec3Cross(a, b) {
    return [
      a[1] * b[2] - a[2] * b[1],
      a[2] * b[0] - a[0] * b[2],
      a[0] * b[1] - a[1] * b[0],
    ];
  }
  function normalize(v) {
    const len = Math.hypot(v[0], v[1], v[2]);
    if (len < 1e-8) return [0, 0, 0];
    return [v[0] / len, v[1] / len, v[2] / len];
  }
  function mat4Identity() {
    return [1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1];
  }
  function mat4Mul(a, b) {
    const out = new Array(16);
    for (let r = 0; r < 4; r++) {
      const r4 = r * 4;
      const a0 = a[r4], a1 = a[r4+1], a2 = a[r4+2], a3 = a[r4+3];
      for (let c = 0; c < 4; c++) {
        out[r4 + c] = a0*b[c] + a1*b[4+c] + a2*b[8+c] + a3*b[12+c];
      }
    }
    return out;
  }
  function mat4Translate(tx, ty, tz) {
    return [1,0,0,tx, 0,1,0,ty, 0,0,1,tz, 0,0,0,1];
  }
  function mat4Scale(sx, sy, sz) {
    return [sx,0,0,0, 0,sy,0,0, 0,0,sz,0, 0,0,0,1];
  }
  function mat4TransformPt(m, p) {
    return [
      m[0]*p[0] + m[1]*p[1] + m[2]*p[2] + m[3],
      m[4]*p[0] + m[5]*p[1] + m[6]*p[2] + m[7],
      m[8]*p[0] + m[9]*p[1] + m[10]*p[2] + m[11],
    ];
  }
  function clamp01(v) { return v < 0.0 ? 0.0 : (v > 1.0 ? 1.0 : v); }
  function sstep(edge0, edge1, x) {
    if (edge1 <= edge0) return x >= edge1 ? 1.0 : 0.0;
    const t = clamp01((x - edge0) / (edge1 - edge0));
    return t * t * (3.0 - 2.0 * t);
  }
  function s5(edge0, edge1, x) {
    if (edge1 <= edge0) return x >= edge1 ? 1.0 : 0.0;
    const t = clamp01((x - edge0) / (edge1 - edge0));
    return t * t * t * (t * (t * 6.0 - 15.0) + 10.0);
  }
  function lerp3(a, b, t) {
    return [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, a[2] + (b[2] - a[2]) * t];
  }
  function aimForearmMatrix(wristPos, elbowPos, upHint = [0, 1, 0]) {
    const zAxis = normalize(vec3Sub(elbowPos, wristPos));
    let xAxis = vec3Cross(upHint, zAxis);
    if (Math.hypot(xAxis[0], xAxis[1], xAxis[2]) < 1e-5) xAxis = [1, 0, 0];
    else xAxis = normalize(xAxis);
    const yAxis = vec3Cross(zAxis, xAxis);
    return [
      xAxis[0], yAxis[0], zAxis[0], wristPos[0],
      xAxis[1], yAxis[1], zAxis[1], wristPos[1],
      xAxis[2], yAxis[2], zAxis[2], wristPos[2],
      0, 0, 0, 1,
    ];
  }
  function mat4RotX(rad) {
    const c = Math.cos(rad), s = Math.sin(rad);
    return [1,0,0,0, 0,c,-s,0, 0,s,c,0, 0,0,0,1];
  }
  function mat4RotY(rad) {
    const c = Math.cos(rad), s = Math.sin(rad);
    return [c,0,s,0, 0,1,0,0, -s,0,c,0, 0,0,0,1];
  }
  function mat4RotZ(rad) {
    const c = Math.cos(rad), s = Math.sin(rad);
    return [c,-s,0,0, s,c,0,0, 0,0,1,0, 0,0,0,1];
  }
  function mat4Perspective(fovDeg, aspect, near, far) {
    const f = 1.0 / Math.tan((fovDeg * Math.PI / 180) * 0.5);
    const a = Math.max(aspect, 1e-5);
    const nf = near - far;
    return [
      f / a, 0, 0, 0,
      0, f, 0, 0,
      0, 0, (far + near) / nf, (2 * far * near) / nf,
      0, 0, -1, 0,
    ];
  }
  function mat4LookAt(eye, target, up) {
    const forward = normalize(vec3Sub(target, eye));
    const right = normalize(vec3Cross(forward, up));
    const camUp = vec3Cross(right, forward);
    return [
      right[0], right[1], right[2], -vec3Dot(right, eye),
      camUp[0], camUp[1], camUp[2], -vec3Dot(camUp, eye),
      -forward[0], -forward[1], -forward[2], vec3Dot(forward, eye),
      0, 0, 0, 1,
    ];
  }
  function mat4CameraToWorld(eye, forward, up) {
    const fwd = normalize(forward);
    const right = normalize(vec3Cross(fwd, up));
    const camUp = vec3Cross(right, fwd);
    return [
      right[0], camUp[0], -fwd[0], eye[0],
      right[1], camUp[1], -fwd[1], eye[1],
      right[2], camUp[2], -fwd[2], eye[2],
      0, 0, 0, 1,
    ];
  }
  function toGlMat4(m) {
    return new Float32Array([
      m[0], m[4], m[8],  m[12],
      m[1], m[5], m[9],  m[13],
      m[2], m[6], m[10], m[14],
      m[3], m[7], m[11], m[15],
    ]);
  }
  function aabbIntersects(a, b) {
    return (
      a[0] < b[3] && a[3] > b[0] &&
      a[1] < b[4] && a[4] > b[1] &&
      a[2] < b[5] && a[5] > b[2]
    );
  }
  function raycastAABB(origin, dir, box, maxDist) {
    let tMin = 0.0, tMax = maxDist;
    for (let i = 0; i < 3; i++) {
      const o = origin[i], d = dir[i], bmin = box[i], bmax = box[i + 3];
      if (Math.abs(d) < 1e-8) {
        if (o < bmin || o > bmax) return null;
      } else {
        const inv = 1.0 / d;
        let t1 = (bmin - o) * inv;
        let t2 = (bmax - o) * inv;
        if (t1 > t2) { const tmp = t1; t1 = t2; t2 = tmp; }
        if (t1 > tMin) tMin = t1;
        if (t2 < tMax) tMax = t2;
        if (tMin > tMax) return null;
      }
    }
    return (tMin >= 0 && tMin <= maxDist) ? tMin : null;
  }

  // ---------- Dynamic Light Evaluation (matches engine/world.py) ----------
  function evalLightIntensity(lt, t) {
    let factor = 1.0;
    if (lt.pulseAmount > 0) {
      factor *= 1.0 - lt.pulseAmount * (0.5 + 0.5 * Math.sin(t * lt.pulseSpeed + lt.phase));
    }
    if (lt.flickerAmount > 0) {
      const s1 = Math.sin(t * lt.flickerSpeed * 13.7 + lt.phase);
      const s2 = Math.sin(t * lt.flickerSpeed * 29.3 + lt.phase * 1.7);
      const s3 = Math.sin(t * lt.flickerSpeed * 61.1 + lt.phase * 0.3);
      const dip = (s1 * s2 + 0.5 * s3) > -0.82 ? 1.0 : 0.52;
      const subtle = 1.0 + 0.08 * s2 * lt.flickerAmount;
      factor *= (1.0 - lt.flickerAmount) + lt.flickerAmount * dip * subtle;
    }
    return Math.max(0.0, lt.intensity * factor);
  }

  // ---------- Kick Animation (driven by Python _KEYFRAMES) ----------
  function easeMode(u, mode) {
    u = Math.max(0, Math.min(1, u));
    if (mode === "snap") return 1.0 - Math.pow(1.0 - u, 3.6);
    if (mode === "ease_out") return 1.0 - Math.pow(1.0 - u, 2.4);
    if (mode === "smooth" || mode === "ease_in_out") return u * u * (3.0 - 2.0 * u);
    return u;
  }
  function lerp(a, b, t) { return a + (b - a) * t; }

  class KickController {
    constructor(legMeta) {
      this.thighLen = legMeta.thighLength;
      this.shinLen = legMeta.shinLength;
      this.duration = legMeta.duration;
      this.cooldownMax = legMeta.cooldown;
      this.keyframes = legMeta.keyframes;
      this.active = false;
      this.timer = 0;
      this.cooldown = 0;
      this.impactChecked = false;
      this.shakeTimer = 0;
      this.shakeDur = 0.12;
      this.shakeStrength = 0;
      this.onStart = null;
      this.onHit = null;
    }

    trigger() {
      if (this.active || this.cooldown > 0) return false;
      this.active = true;
      this.timer = 0;
      this.cooldown = this.cooldownMax;
      this.impactChecked = false;
      if (this.onStart) this.onStart();
      return true;
    }

    samplePose(t) {
      const kfs = this.keyframes;
      if (t <= kfs[0].t) return kfs[0].pose;
      if (t >= kfs[kfs.length - 1].t) return kfs[kfs.length - 1].pose;
      for (let i = 0; i < kfs.length - 1; i++) {
        const k0 = kfs[i], k1 = kfs[i + 1];
        if (t >= k0.t && t <= k1.t) {
          const u = (t - k0.t) / Math.max(k1.t - k0.t, 1e-6);
          const w = easeMode(u, k1.mode);
          const p0 = k0.pose, p1 = k1.pose;
          return {
            hip_pos: [
              lerp(p0.hip_pos[0], p1.hip_pos[0], w),
              lerp(p0.hip_pos[1], p1.hip_pos[1], w),
              lerp(p0.hip_pos[2], p1.hip_pos[2], w),
            ],
            hip_pitch_deg: lerp(p0.hip_pitch_deg, p1.hip_pitch_deg, w),
            hip_yaw_deg: lerp(p0.hip_yaw_deg, p1.hip_yaw_deg, w),
            hip_roll_deg: lerp(p0.hip_roll_deg, p1.hip_roll_deg, w),
            knee_pitch_deg: lerp(p0.knee_pitch_deg, p1.knee_pitch_deg, w),
            ankle_pitch_deg: lerp(p0.ankle_pitch_deg, p1.ankle_pitch_deg, w),
            ankle_yaw_deg: lerp(p0.ankle_yaw_deg, p1.ankle_yaw_deg, w),
            cam_roll_deg: lerp(p0.cam_roll_deg, p1.cam_roll_deg, w),
            cam_pitch_deg: lerp(p0.cam_pitch_deg, p1.cam_pitch_deg, w),
            cam_fov_deg: lerp(p0.cam_fov_deg, p1.cam_fov_deg, w),
          };
        }
      }
      return kfs[kfs.length - 1].pose;
    }

    update(dt, eyePos, lookDir, colliders) {
      if (this.cooldown > 0) this.cooldown = Math.max(0, this.cooldown - dt);
      let camPitch = 0, camRoll = 0, camFov = 0;
      let camShake = [0, 0, 0];
      let recoilVel = [0, 0, 0];

      if (this.active) {
        this.timer += dt;
        if (!this.impactChecked && this.timer >= 0.185) {
          this.impactChecked = true;
          const rays = [
            eyePos,
            [eyePos[0], eyePos[1] - 0.45, eyePos[2]],
            [eyePos[0], eyePos[1] - 0.80, eyePos[2]],
          ];
          let bestHit = null;
          for (const origin of rays) {
            for (const box of colliders) {
              const d = raycastAABB(origin, lookDir, box, 1.95);
              if (d !== null && (bestHit === null || d < bestHit)) bestHit = d;
            }
          }
          if (bestHit !== null) {
            this.shakeTimer = this.shakeDur;
            this.shakeStrength = 0.048 * (1.0 - 0.35 * (bestHit / 1.95));
            const push = 2.6 * (1.0 - 0.35 * (bestHit / 1.95));
            recoilVel = [-lookDir[0] * push, Math.max(0, -lookDir[1] * push * 0.4), -lookDir[2] * push];
            if (this.onHit) this.onHit(bestHit);
          }
        }
        if (this.timer >= this.duration) {
          this.active = false;
          this.timer = 0;
        } else {
          const pose = this.samplePose(this.timer);
          camPitch = pose.cam_pitch_deg * Math.PI / 180;
          camRoll = pose.cam_roll_deg * Math.PI / 180;
          camFov = pose.cam_fov_deg;
        }
      }

      if (this.shakeTimer > 0) {
        this.shakeTimer = Math.max(0, this.shakeTimer - dt);
        const decay = this.shakeTimer / Math.max(this.shakeDur, 1e-5);
        const phase = (this.shakeDur - this.shakeTimer) * 105.0;
        const amp = this.shakeStrength * decay * decay;
        camShake = [
          Math.sin(phase * 1.3) * amp,
          Math.cos(phase * 1.9) * amp * 0.85,
          Math.sin(phase * 0.9) * amp * 0.5,
        ];
      }
      return { camPitch, camRoll, camFov, camShake, recoilVel };
    }

    buildDrawCalls(legBatches, vmToWorld) {
      if (!this.active) return [];
      const pose = this.samplePose(this.timer);
      const [hx, hy, hz] = pose.hip_pos;
      const deg = Math.PI / 180;

      const mHip = mat4Mul(
        mat4Translate(hx, hy, hz),
        mat4Mul(
          mat4RotY(pose.hip_yaw_deg * deg),
          mat4Mul(mat4RotZ(pose.hip_roll_deg * deg), mat4RotX(pose.hip_pitch_deg * deg))
        )
      );
      const mKnee = mat4Mul(
        mHip,
        mat4Mul(mat4Translate(0, -this.thighLen, 0), mat4RotX(pose.knee_pitch_deg * deg))
      );
      const mAnkle = mat4Mul(
        mKnee,
        mat4Mul(
          mat4Translate(0, -this.shinLen, 0),
          mat4Mul(mat4RotY(pose.ankle_yaw_deg * deg), mat4RotX(pose.ankle_pitch_deg * deg))
        )
      );

      return [
        { batches: legBatches.thigh, model: mat4Mul(vmToWorld, mHip) },
        { batches: legBatches.shin, model: mat4Mul(vmToWorld, mKnee) },
        { batches: legBatches.foot, model: mat4Mul(vmToWorld, mAnkle) },
      ];
    }
  }

  // ---------- Player + Camera (matches engine/player.py & camera.py 1:1) ----------
  class Player {
    constructor(spawnPos, spawnYaw, cfg, legMeta) {
      this.cfg = cfg;
      this.spawnPos = [...spawnPos];
      this.spawnYaw = spawnYaw;
      this.legMeta = legMeta;
      this.reset();
    }

    reset() {
      this.pos = [...this.spawnPos];
      this.vel = [0, 0, 0];
      this.yaw = this.spawnYaw;
      this.pitch = 0.0;
      this.targetYaw = this.spawnYaw;
      this.targetPitch = 0.0;
      this.onGround = false;
      this.crouching = false;
      this.sliding = false;
      this.noclip = false;

      // CS2-style jump state (no auto-bhop, landing slowdown & stamina)
      this.prevJump = false;
      this.coyoteTimer = 0;
      this.jumpLockTimer = 0;
      this.landingSlowTimer = 0;
      this.jumpStamina = 0;
      this.airSpeedCap = this.cfg.walk_speed;
      this.fallSpeed = 0;
      this.landingDip = 0;

      // Sprint-slide & silky camera feel state
      this.slideTimer = 0;
      this.slideCooldown = 0;
      this.slideDir = [0, -1];
      this.slideRollSign = 1;
      this.slideRoll = 0;
      this.slideFovOffset = 0;
      this.strafeRoll = 0;
      this.sprintFovOffset = 0;

      this.eyeHeight = this.cfg.stand_eye_height;
      this.smoothFootY = this.pos[1];
      this.recoilTimer = 0;
      this.flashlightOn = false;

      this.bobPhase = 0;
      this.bobAmount = 0;
      this.walkPhase = 0;
      this.lastStepIdx = 0;
      this.stepCooldown = 0;
      this.idlePhase = 0;
      this.bobOffset = [0, 0, 0];

      this.kickPitch = 0;
      this.kickRoll = 0;
      this.kickFovOffset = 0;
      this.kickOffset = [0, 0, 0];
      const prevStart = this.kick ? this.kick.onStart : null;
      const prevHit = this.kick ? this.kick.onHit : null;
      this.kick = new KickController(this.legMeta);
      this.kick.onStart = prevStart;
      this.kick.onHit = prevHit;
    }

    get height() { return (this.crouching || this.sliding) ? this.cfg.crouch_height : this.cfg.stand_height; }

    getAABB(x = this.pos[0], y = this.pos[1], z = this.pos[2], h = this.height) {
      const r = this.cfg.radius;
      return [x - r, y, z - r, x + r, y + h, z + r];
    }

    collidesAny(box, colliders) {
      for (let i = 0; i < colliders.length; i++) {
        if (aabbIntersects(box, colliders[i])) return true;
      }
      return false;
    }

    findGroundBelow(x, y, z, probeDist, colliders) {
      const r = this.cfg.radius - 1e-4;
      const probe = [x - r, y - probeDist, z - r, x + r, y + 0.01, z + r];
      let best = null;
      for (let i = 0; i < colliders.length; i++) {
        const c = colliders[i];
        if (aabbIntersects(probe, c) && c[4] <= y + 0.015) {
          if (best === null || c[4] > best) best = c[4];
        }
      }
      return best;
    }

    canStand(colliders) {
      return !this.collidesAny(this.getAABB(this.pos[0], this.pos[1], this.pos[2], this.cfg.stand_height), colliders);
    }

    rotate(dx, dy) {
      this.targetYaw += dx * this.cfg.mouse_sensitivity;
      this.targetPitch -= dy * this.cfg.mouse_sensitivity;
      const maxP = 85 * Math.PI / 180;
      this.targetPitch = Math.max(-maxP, Math.min(maxP, this.targetPitch));
    }

    updateLookSmoothing(dt) {
      const alpha = 1.0 - Math.exp(-dt * 32.0);
      this.yaw += (this.targetYaw - this.yaw) * alpha;
      this.pitch += (this.targetPitch - this.pitch) * alpha;
    }

    get effectivePitch() {
      const maxP = 87 * Math.PI / 180;
      return Math.max(-maxP, Math.min(maxP, this.pitch + this.kickPitch));
    }
    get forward() {
      const ep = this.effectivePitch, cp = Math.cos(ep);
      return [Math.sin(this.yaw) * cp, Math.sin(ep), -Math.cos(this.yaw) * cp];
    }
    get flatForward() { return [Math.sin(this.yaw), 0, -Math.cos(this.yaw)]; }
    get right() { return [Math.cos(this.yaw), 0, Math.sin(this.yaw)]; }
    get rolledUp() {
      const totalRoll = this.kickRoll + this.slideRoll + this.strafeRoll;
      if (Math.abs(totalRoll) < 1e-6) return [0, 1, 0];
      const fwd = this.forward;
      const br = normalize(vec3Cross(fwd, [0, 1, 0]));
      const bu = vec3Cross(br, fwd);
      const cr = Math.cos(totalRoll), sr = Math.sin(totalRoll);
      return [bu[0]*cr + br[0]*sr, bu[1]*cr + br[1]*sr, bu[2]*cr + br[2]*sr];
    }
    get eyePos() {
      return [
        this.pos[0] + this.bobOffset[0] + this.kickOffset[0],
        this.smoothFootY + this.eyeHeight + this.bobOffset[1] + this.kickOffset[1] - this.landingDip,
        this.pos[2] + this.bobOffset[2] + this.kickOffset[2],
      ];
    }

    slideHorizontal(sx, y, sz, dx, dz, colliders) {
      let x = sx, z = sz, hitAny = false;
      const r = this.cfg.radius;

      if (Math.abs(dx) > 1e-8) {
        const nx = x + dx;
        const box = this.getAABB(nx, y, z);
        const hits = [];
        for (const c of colliders) if (aabbIntersects(box, c)) hits.push(c);
        if (hits.length === 0) {
          x = nx;
        } else {
          hitAny = true;
          if (dx > 0) x = Math.min(...hits.map(c => c[0])) - r - EPS;
          else x = Math.max(...hits.map(c => c[3])) + r + EPS;
        }
      }
      if (Math.abs(dz) > 1e-8) {
        const nz = z + dz;
        const box = this.getAABB(x, y, nz);
        const hits = [];
        for (const c of colliders) if (aabbIntersects(box, c)) hits.push(c);
        if (hits.length === 0) {
          z = nz;
        } else {
          hitAny = true;
          if (dz > 0) z = Math.min(...hits.map(c => c[2])) - r - EPS;
          else z = Math.max(...hits.map(c => c[5])) + r + EPS;
        }
      }
      return { x, z, hitAny };
    }

    maxClearLift(x, y, z, maxStep, colliders) {
      const r = this.cfg.radius;
      const testBox = [x - r, y, z - r, x + r, y + this.height + maxStep, z + r];
      let allowed = maxStep;
      for (const c of colliders) {
        if (aabbIntersects(testBox, c) && c[1] >= y + this.height - 0.01) {
          const clearance = c[1] - (y + this.height) - EPS;
          if (clearance < allowed) allowed = clearance;
        }
      }
      return Math.max(0, allowed);
    }

    stepSlideMove(dx, dz, allowStep, colliders) {
      const [sx, sy, sz] = this.pos;
      const norm = this.slideHorizontal(sx, sy, sz, dx, dz, colliders);
      if (!norm.hitAny || !allowStep) {
        this.pos[0] = norm.x;
        this.pos[2] = norm.z;
        return;
      }

      const lift = this.maxClearLift(sx, sy, sz, this.cfg.step_height, colliders);
      if (lift <= 0.02) {
        this.pos[0] = norm.x;
        this.pos[2] = norm.z;
        return;
      }

      const stepped = this.slideHorizontal(sx, sy + lift, sz, dx, dz, colliders);
      const normD2 = (norm.x - sx) ** 2 + (norm.z - sz) ** 2;
      const stepD2 = (stepped.x - sx) ** 2 + (stepped.z - sz) ** 2;
      if (stepD2 <= normD2 + 1e-10) {
        this.pos[0] = norm.x;
        this.pos[2] = norm.z;
        return;
      }

      const landedY = this.findGroundBelow(stepped.x, sy + lift, stepped.z, lift + 0.05, colliders);
      if (landedY !== null && landedY > sy + 1e-4 && (landedY - sy) <= this.cfg.step_height + 0.01) {
        if (!this.collidesAny(this.getAABB(stepped.x, landedY, stepped.z), colliders)) {
          this.pos[0] = stepped.x;
          this.pos[1] = landedY;
          this.pos[2] = stepped.z;
          return;
        }
      }
      this.pos[0] = norm.x;
      this.pos[2] = norm.z;
    }

    applyLandingImpact(impactVy) {
      if (impactVy < -2.2) {
        this.landingSlowTimer = 0.30;
        this.jumpLockTimer = 0.22;
        this.landingDip = Math.max(this.landingDip, Math.min(0.14, Math.abs(impactVy) * 0.015));
        this.vel[0] *= 0.55;
        this.vel[2] *= 0.55;
        if (this.onLand) this.onLand(Math.abs(impactVy));
      }
    }

    endSlide(keepCrouch, colliders) {
      this.sliding = false;
      this.slideTimer = 0;
      this.slideCooldown = this.cfg.slide_cooldown;
      if (keepCrouch || !this.canStand(colliders)) this.crouching = true;
      else this.crouching = false;
    }

    moveVertical(delta, colliders) {
      this.onGround = false;
      const newY = this.pos[1] + delta;
      const box = this.getAABB(this.pos[0], newY, this.pos[2]);
      const hits = [];
      for (const c of colliders) if (aabbIntersects(box, c)) hits.push(c);
      if (hits.length === 0) {
        this.pos[1] = newY;
        return;
      }
      if (delta <= 0) {
        const impactVy = this.fallSpeed;
        this.pos[1] = Math.max(...hits.map(c => c[4]));
        this.vel[1] = 0;
        this.fallSpeed = 0;
        this.onGround = true;
        this.applyLandingImpact(impactVy);
      } else {
        this.pos[1] = Math.min(...hits.map(c => c[1])) - this.height - EPS;
        this.vel[1] = 0;
      }
    }

    update(dt, inp, colliders) {
      dt = Math.min(dt, 0.05);

      if (this.noclip) {
        this.updateLookSmoothing(dt);
        this.onGround = false;
        this.crouching = false;
        this.sliding = false;
        this.fallSpeed = 0;
        this.landingDip = 0;
        this.slideTimer = 0;

        const fwd = this.forward;
        const rr = this.right;
        let vx = fwd[0] * inp.forward + rr[0] * inp.strafe;
        let vy = fwd[1] * inp.forward + (inp.jump ? 1.0 : 0.0) - (inp.crouch ? 1.0 : 0.0);
        let vz = fwd[2] * inp.forward + rr[2] * inp.strafe;
        const vlen = Math.hypot(vx, vy, vz);
        if (vlen > 1e-5) {
          vx /= vlen;
          vy /= vlen;
          vz /= vlen;
        }
        const flySpeed = inp.sprint ? 18.0 : 9.5;
        const blend = 1.0 - Math.exp(-dt * 12.0);
        this.vel[0] += (vx * flySpeed - this.vel[0]) * blend;
        this.vel[1] += (vy * flySpeed - this.vel[1]) * blend;
        this.vel[2] += (vz * flySpeed - this.vel[2]) * blend;

        this.pos[0] += this.vel[0] * dt;
        this.pos[1] += this.vel[1] * dt;
        this.pos[2] += this.vel[2] * dt;
        this.smoothFootY = this.pos[1];

        this.eyeHeight += (this.cfg.stand_eye_height - this.eyeHeight) * (1.0 - Math.exp(-dt * 13.0));
        this.slideRoll += (0 - this.slideRoll) * (1.0 - Math.exp(-dt * 11.5));
        this.slideFovOffset += (0 - this.slideFovOffset) * (1.0 - Math.exp(-dt * 10.0));
        const targetStrafeRoll = -inp.strafe * (1.6 * Math.PI / 180);
        this.strafeRoll += (targetStrafeRoll - this.strafeRoll) * (1.0 - Math.exp(-dt * 9.5));
        const targetFlyFov = (inp.sprint && vlen > 1e-5) ? 4.5 : 0.0;
        this.sprintFovOffset += (targetFlyFov - this.sprintFovOffset) * (1.0 - Math.exp(-dt * 7.5));
        this.landingDip += (0 - this.landingDip) * (1.0 - Math.exp(-dt * 10.5));
        this.idlePhase += dt * 1.9;
        this.bobAmount += (0 - this.bobAmount) * (1.0 - Math.exp(-dt * 12.0));
        this.bobOffset = [0, 0, 0];

        if (inp.kick) this.kick.trigger();
        const kb = this.kick.update(dt, this.eyePos, this.forward, []);
        this.kickPitch = Number.isFinite(kb.camPitch) ? kb.camPitch : 0;
        this.kickRoll = Number.isFinite(kb.camRoll) ? kb.camRoll : 0;
        this.kickFovOffset = Number.isFinite(kb.camFov) ? kb.camFov : 0;
        this.kickOffset = Array.isArray(kb.camShake) ? kb.camShake : [0, 0, 0];
        this.prevJump = Boolean(inp.jump);
        return;
      }

      const prevOnGround = this.onGround;
      if (this.vel[1] <= 0.01) {
        const gy = this.findGroundBelow(this.pos[0], this.pos[1], this.pos[2], GROUND_PROBE, colliders);
        if (gy !== null) {
          if (!prevOnGround) this.applyLandingImpact(this.fallSpeed);
          this.pos[1] = gy;
          this.vel[1] = 0;
          this.fallSpeed = 0;
          this.onGround = true;
        } else {
          this.onGround = false;
        }
      } else {
        this.onGround = false;
      }
      const wasGrounded = this.onGround;

      if (this.jumpLockTimer > 0) this.jumpLockTimer = Math.max(0, this.jumpLockTimer - dt);
      if (this.landingSlowTimer > 0) this.landingSlowTimer = Math.max(0, this.landingSlowTimer - dt);
      if (this.slideCooldown > 0) this.slideCooldown = Math.max(0, this.slideCooldown - dt);
      if (this.onGround) this.jumpStamina = Math.max(0, this.jumpStamina - 0.58 * dt);
      if (this.recoilTimer > 0) this.recoilTimer = Math.max(0, this.recoilTimer - dt);

      // Silky frame-rate-independent exponential look smoothing before reading movement basis
      this.updateLookSmoothing(dt);

      const ff = this.flatForward, rr = this.right;
      let mx = ff[0] * inp.forward + rr[0] * inp.strafe;
      let mz = ff[2] * inp.forward + rr[2] * inp.strafe;
      const moveLen = Math.hypot(mx, mz);
      if (moveLen > 1e-5) { mx /= moveLen; mz /= moveLen; }

      // Sprint-slide trigger on fresh C press while sprinting
      if (
        inp.slidePress &&
        this.onGround &&
        !this.sliding &&
        !this.crouching &&
        this.slideCooldown <= 0 &&
        inp.sprint &&
        inp.forward > 0 &&
        moveLen > 1e-5
      ) {
        this.sliding = true;
        this.crouching = true;
        this.slideTimer = 0;
        this.slideDir = [mx, mz];
        this.slideRollSign = inp.strafe < -0.1 ? -1 : 1;
        this.landingSlowTimer = 0;
        if (this.onSlideStart) this.onSlideStart();
      }

      if (!this.sliding) {
        if (inp.crouch) this.crouching = true;
        else if (this.crouching && this.canStand(colliders)) this.crouching = false;
      }

      if (inp.kick) this.kick.trigger();

      const jumpPressed = Boolean(inp.jump) && !this.prevJump;
      if (this.onGround) this.coyoteTimer = 0.075;
      else this.coyoteTimer = Math.max(0, this.coyoteTimer - dt);

      if (this.sliding) {
        this.slideTimer += dt;
        if (this.slideTimer >= this.cfg.slide_duration) {
          this.endSlide(Boolean(inp.crouch), colliders);
        } else {
          const u = this.slideTimer / Math.max(this.cfg.slide_duration, 1e-5);
          if (Math.abs(inp.strafe) > 0.1) {
            const sx = this.slideDir[0] + rr[0] * inp.strafe * 0.45 * dt;
            const sz = this.slideDir[1] + rr[2] * inp.strafe * 0.45 * dt;
            const sl = Math.hypot(sx, sz);
            if (sl > 1e-5) this.slideDir = [sx / sl, sz / sl];
          }
          const decel = 1.0 - Math.pow(1.0 - u, 2.1);
          const curSlideSpeed = this.cfg.slide_speed + (this.cfg.slide_end_speed - this.cfg.slide_speed) * decel;
          this.vel[0] = this.slideDir[0] * curSlideSpeed;
          this.vel[2] = this.slideDir[1] * curSlideSpeed;
        }
      }

      if (!this.sliding) {
        let maxSpeed = this.crouching
          ? this.cfg.crouch_speed
          : (inp.sprint && inp.forward > 0 ? this.cfg.sprint_speed : this.cfg.walk_speed);

        if (this.onGround && this.landingSlowTimer > 0) {
          const slowU = this.landingSlowTimer / 0.30;
          maxSpeed *= 1.0 - 0.45 * slowU;
        }

        const targetVx = mx * maxSpeed, targetVz = mz * maxSpeed;

        if (this.onGround && this.recoilTimer <= 0) {
          if (moveLen > 1e-5) {
            this.vel[0] = targetVx;
            this.vel[2] = targetVz;
          } else {
            this.vel[0] = 0;
            this.vel[2] = 0;
          }
        } else if (this.onGround) {
          const decay = Math.max(0, 1.0 - 18.0 * dt);
          this.vel[0] = targetVx + (this.vel[0] - targetVx) * decay;
          this.vel[2] = targetVz + (this.vel[2] - targetVz) * decay;
        } else {
          if (moveLen > 1e-5) {
            const airBlend = Math.min(1.0, this.cfg.air_accel * dt / Math.max(this.airSpeedCap, 1.0));
            this.vel[0] += (mx * this.airSpeedCap - this.vel[0]) * airBlend;
            this.vel[2] += (mz * this.airSpeedCap - this.vel[2]) * airBlend;
          }
          const curH = Math.hypot(this.vel[0], this.vel[2]);
          if (curH > this.airSpeedCap && curH > 1e-5) {
            const s = this.airSpeedCap / curH;
            this.vel[0] *= s;
            this.vel[2] *= s;
          }
        }
      }

      let jumped = false;
      if (jumpPressed && this.coyoteTimer > 0 && this.jumpLockTimer <= 0) {
        if (this.sliding) {
          this.endSlide(false, colliders);
          const curH = Math.hypot(this.vel[0], this.vel[2]);
          if (curH > this.cfg.sprint_speed) {
            const s = this.cfg.sprint_speed / curH;
            this.vel[0] *= s;
            this.vel[2] *= s;
          }
        }
        const staminaScale = 1.0 - 0.26 * this.jumpStamina;
        this.vel[1] = this.cfg.jump_velocity * staminaScale;
        this.jumpStamina = Math.min(1.0, this.jumpStamina + 0.42);
        const takeoffH = Math.hypot(this.vel[0], this.vel[2]);
        this.airSpeedCap = Math.min(Math.max(takeoffH, this.cfg.walk_speed), this.cfg.sprint_speed);
        this.onGround = false;
        this.coyoteTimer = 0;
        jumped = true;
        if (this.onJump) this.onJump();
      }

      const prevX = this.pos[0], prevY = this.pos[1], prevZ = this.pos[2];
      const dxReq = this.vel[0] * dt, dzReq = this.vel[2] * dt;
      this.stepSlideMove(dxReq, dzReq, wasGrounded && !jumped, colliders);

      // Distance-synced R16 walkPhase + synchronized concrete footstep impulse!
      if (this.stepCooldown > 0) this.stepCooldown = Math.max(0, this.stepCooldown - dt);
      if (this.onGround && !this.sliding) {
        const stepDist = Math.hypot(this.pos[0] - prevX, this.pos[2] - prevZ);
        const hNow = Math.hypot(this.vel[0], this.vel[2]);
        if (stepDist > 1e-4 && hNow > 0.25) {
          const sprintBlend = Math.min(1.0, Math.max(0.0, (hNow - 4.6) / 2.8));
          const strideHalf = this.crouching ? 1.55 : (2.38 + 0.46 * sprintBlend);
          this.walkPhase += (stepDist / strideHalf) * Math.PI;
          const stepIdx = Math.floor((this.walkPhase + Math.PI * 0.5) / Math.PI);
          if (stepIdx > this.lastStepIdx) {
            this.lastStepIdx = stepIdx;
            if (this.stepCooldown <= 0 && this.onFootstep) {
              this.stepCooldown = 0.26;
              const stepMode = this.crouching ? "crouch" : (hNow > 5.4 ? "sprint" : "walk");
              this.onFootstep(stepMode);
            }
          }
        } else if (hNow <= 0.25) {
          const nearestPi = Math.round(this.walkPhase / Math.PI) * Math.PI;
          this.walkPhase += (nearestPi - this.walkPhase) * (1.0 - Math.exp(-dt * 9.0));
          this.lastStepIdx = Math.floor((this.walkPhase + Math.PI * 0.5) / Math.PI);
        }
      }

      if (this.sliding) {
        const reqDist = Math.hypot(dxReq, dzReq);
        const actDist = Math.hypot(this.pos[0] - prevX, this.pos[2] - prevZ);
        if (reqDist > 1e-4 && actDist < reqDist * 0.25) {
          this.endSlide(Boolean(inp.crouch), colliders);
          this.vel[0] = 0;
          this.vel[2] = 0;
        }
      }

      const steppedUp = this.pos[1] > prevY + 1e-4;
      if (wasGrounded && !jumped && !steppedUp) {
        const snapY = this.findGroundBelow(this.pos[0], this.pos[1], this.pos[2], STEP_DOWN_SNAP, colliders);
        if (snapY !== null) {
          this.pos[1] = snapY;
          this.vel[1] = 0;
          this.onGround = true;
        } else {
          this.onGround = false;
          if (this.sliding) {
            this.endSlide(Boolean(inp.crouch), colliders);
            const curH = Math.hypot(this.vel[0], this.vel[2]);
            this.airSpeedCap = Math.min(Math.max(curH, this.cfg.walk_speed), this.cfg.sprint_speed);
          }
        }
      }

      if (!this.onGround) {
        this.vel[1] -= this.cfg.gravity * dt;
        if (this.vel[1] < this.fallSpeed) this.fallSpeed = this.vel[1];
        this.moveVertical(this.vel[1] * dt, colliders);
      }

      if (Math.abs(this.pos[1] - this.smoothFootY) > 0.6) {
        this.smoothFootY = this.pos[1];
      } else {
        this.smoothFootY += (this.pos[1] - this.smoothFootY) * (1.0 - Math.exp(-dt * 15.0));
      }

      const targetEye = this.sliding
        ? this.cfg.slide_eye_height
        : (this.crouching ? this.cfg.crouch_eye_height : this.cfg.stand_eye_height);
      this.eyeHeight += (targetEye - this.eyeHeight) * (1.0 - Math.exp(-dt * 13.0));

      const hSpeed = Math.hypot(this.vel[0], this.vel[2]);
      let targetRoll = 0, targetFov = 0, targetStrafeRoll = 0, targetSprintFov = 0;
      if (this.sliding) {
        const u = this.slideTimer / Math.max(this.cfg.slide_duration, 1e-5);
        targetRoll = (5.2 * Math.PI / 180) * this.slideRollSign;
        targetFov = 8.0 * (1.0 - 0.65 * u);
      } else {
        targetStrafeRoll = (this.onGround && !this.crouching) ? (-inp.strafe * (1.35 * Math.PI / 180)) : 0;
        const isSprinting = Boolean(inp.sprint) && this.onGround && !this.crouching && hSpeed > this.cfg.walk_speed * 1.08;
        targetSprintFov = isSprinting ? 4.2 : 0;
      }
      this.slideRoll += (targetRoll - this.slideRoll) * (1.0 - Math.exp(-dt * 11.5));
      this.slideFovOffset += (targetFov - this.slideFovOffset) * (1.0 - Math.exp(-dt * 10.0));
      this.strafeRoll += (targetStrafeRoll - this.strafeRoll) * (1.0 - Math.exp(-dt * 9.5));
      this.sprintFovOffset += (targetSprintFov - this.sprintFovOffset) * (1.0 - Math.exp(-dt * 7.5));
      this.landingDip += (0 - this.landingDip) * (1.0 - Math.exp(-dt * 10.5));

      // Gentle, organic head-bob synchronized with stride cadence (suppressed cleanly during slide)
      this.idlePhase += dt * 1.9;
      let targetBob = 0;
      if (this.onGround && !this.sliding && hSpeed > 0.25) {
        const sf = Math.min(hSpeed / 4.6, 1.60);
        const sb = Math.min(1.0, Math.max(0.0, (hSpeed - 4.6) / 2.8));
        const strideHalf = this.crouching ? 1.55 : (2.38 + 0.46 * sb);
        const freq = (hSpeed / strideHalf) * Math.PI;
        this.bobPhase += dt * freq;
        targetBob = (this.crouching ? 0.014 : 0.025) * Math.min(sf, 1.25);
      }
      this.bobAmount += (targetBob - this.bobAmount) * (1.0 - Math.exp(-dt * 11.0));
      const idleBreath = Math.sin(this.idlePhase) * 0.004 * (1.0 - Math.min(1.0, this.bobAmount / 0.02));
      const bobV = Math.cos(this.bobPhase * 2.0) * this.bobAmount + idleBreath;
      const bobL = -Math.cos(this.bobPhase) * this.bobAmount * 0.48;
      this.bobOffset = [rr[0] * bobL, bobV, rr[2] * bobL];

      // Kick update
      const kb = this.kick.update(dt, this.eyePos, this.forward, colliders);
      this.kickPitch = kb.camPitch;
      this.kickRoll = kb.camRoll;
      this.kickFovOffset = kb.camFov;
      this.kickOffset = kb.camShake;

      if (Math.abs(kb.recoilVel[0]) > 1e-4 || Math.abs(kb.recoilVel[2]) > 1e-4) {
        this.vel[0] += kb.recoilVel[0];
        this.vel[2] += kb.recoilVel[2];
        if (!this.onGround) this.vel[1] += kb.recoilVel[1];
        this.recoilTimer = 0.085;
      }

      this.prevJump = Boolean(inp.jump);
    }
  }

  // ---------- WebGL2 Helpers ----------
  function compileShader(gl, type, src) {
    const sh = gl.createShader(type);
    gl.shaderSource(sh, src);
    gl.compileShader(sh);
    if (!gl.getShaderParameter(sh, gl.COMPILE_STATUS)) {
      const log = gl.getShaderInfoLog(sh);
      throw new Error("Shader compile error: " + log);
    }
    return sh;
  }

  function createProgram(gl, vsSrc, fsSrc) {
    const prog = gl.createProgram();
    gl.attachShader(prog, compileShader(gl, gl.VERTEX_SHADER, vsSrc));
    gl.attachShader(prog, compileShader(gl, gl.FRAGMENT_SHADER, fsSrc));
    gl.linkProgram(prog);
    if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) {
      throw new Error("Program link error: " + gl.getProgramInfoLog(prog));
    }
    return prog;
  }

  function uLoc(gl, prog, name) {
    return gl.getUniformLocation(prog, name) || gl.getUniformLocation(prog, name + "[0]");
  }

  function uploadBatches(gl, batchList) {
    return batchList.map(item => {
      const floats = b64ToFloat32(item.dataB64);
      const vao = gl.createVertexArray();
      gl.bindVertexArray(vao);
      const vbo = gl.createBuffer();
      gl.bindBuffer(gl.ARRAY_BUFFER, vbo);
      gl.bufferData(gl.ARRAY_BUFFER, floats, gl.STATIC_DRAW);
      gl.enableVertexAttribArray(0);
      gl.vertexAttribPointer(0, 3, gl.FLOAT, false, 32, 0);
      gl.enableVertexAttribArray(1);
      gl.vertexAttribPointer(1, 3, gl.FLOAT, false, 32, 12);
      gl.enableVertexAttribArray(2);
      gl.vertexAttribPointer(2, 2, gl.FLOAT, false, 32, 24);
      gl.bindVertexArray(null);
      return {
        texture: item.texture,
        emissive: item.emissive || 0.0,
        tintSlot: item.tintSlot || null,
        vertexCount: item.vertexCount,
        vao,
      };
    });
  }

  function uploadBoneBatchMap(gl, boneMap) {
    const out = {};
    for (const [bone, list] of Object.entries(boneMap)) {
      out[bone] = uploadBatches(gl, list);
    }
    return out;
  }

  // ---------- R16 Biomechanical 16-Bone Character Skeleton Animator (matches engine/character.py 1:1) ----------
  const WALK_THIGH   = [-0.68, -0.52, -0.06,  0.34,  0.52,  0.16, -0.44, -0.70];
  const WALK_SHIN    = [ 0.08,  0.28,  0.10,  0.09,  0.44,  1.12,  0.86,  0.20];
  const WALK_WFOOT   = [-0.24,  0.00,  0.00,  0.12,  0.44,  0.26, -0.06, -0.22];

  const SPRINT_THIGH = [-0.84, -0.46,  0.28,  0.78,  0.44, -0.42, -1.12, -1.06];
  const SPRINT_SHIN  = [ 0.24,  0.64,  0.22,  0.16,  1.64,  1.82,  1.42,  0.48];
  const SPRINT_WFOOT = [-0.12,  0.02,  0.22,  0.64,  0.56,  0.24, -0.04, -0.16];

  const CROUCH_THIGH = [-1.12, -0.96, -0.66, -0.44, -0.48, -0.78, -1.08, -1.18];
  const CROUCH_SHIN  = [ 1.10,  1.38,  1.24,  0.96,  1.18,  1.66,  1.52,  1.16];
  const CROUCH_WFOOT = [-0.12,  0.00,  0.00,  0.12,  0.34,  0.22, -0.04, -0.12];

  const WALK_ARM_U   = [-0.54, -0.32,  0.02,  0.32,  0.46,  0.28, -0.06, -0.42];
  const WALK_ARM_L   = [-0.54, -0.46, -0.28, -0.16, -0.14, -0.20, -0.36, -0.50];
  const WALK_ARM_H   = [ 0.08,  0.10,  0.04, -0.04, -0.08, -0.06,  0.02,  0.06];

  const SPRINT_ARM_U = [-1.02, -0.58,  0.14,  0.76,  0.96,  0.52, -0.22, -0.84];
  const SPRINT_ARM_L = [-1.78, -1.62, -1.28, -1.04, -1.10, -1.26, -1.52, -1.72];
  const SPRINT_ARM_H = [-0.08, -0.05, -0.02,  0.04,  0.06,  0.02, -0.04, -0.08];

  function sampleCycle(keys, u) {
    const n = keys.length;
    const um = ((u % 1.0) + 1.0) % 1.0;
    const x = um * n;
    const i1 = Math.floor(x) % n;
    const t = x - Math.floor(x);
    const p0 = keys[(i1 - 1 + n) % n];
    const p1 = keys[i1];
    const p2 = keys[(i1 + 1) % n];
    const p3 = keys[(i1 + 2) % n];
    const t2 = t * t;
    const t3 = t2 * t;
    return 0.5 * (
      (2.0 * p1) +
      (-p0 + p2) * t +
      (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * t2 +
      (-p0 + 3.0 * p1 - 3.0 * p2 + p3) * t3
    );
  }

  function gaitLegAngles(phi, amp, sprintT, pelvisPitch, crouchT = 0.0) {
    const u = ((phi - Math.PI * 0.5) / (2.0 * Math.PI));
    let th = sampleCycle(WALK_THIGH, u);
    let sh = sampleCycle(WALK_SHIN, u);
    let wf = sampleCycle(WALK_WFOOT, u);

    if (sprintT > 1e-4) {
      th += (sampleCycle(SPRINT_THIGH, u) - th) * sprintT;
      sh += (sampleCycle(SPRINT_SHIN, u) - sh) * sprintT;
      wf += (sampleCycle(SPRINT_WFOOT, u) - wf) * sprintT;
    }
    if (crouchT > 1e-4) {
      th += (sampleCycle(CROUCH_THIGH, u) - th) * crouchT;
      sh += (sampleCycle(CROUCH_SHIN, u) - sh) * crouchT;
      wf += (sampleCycle(CROUCH_WFOOT, u) - wf) * crouchT;
    }

    const idleTh = -0.82 * crouchT;
    const idleSh = 1.34 * crouchT;
    const thighX = idleTh + (th - idleTh) * amp;
    const shinX = Math.max(0.03, idleSh + (sh - idleSh) * amp);
    const targetWf = wf * amp;
    const footX = Math.max(-0.78, Math.min(0.95, targetWf - (pelvisPitch + thighX + shinX)));
    return [thighX, shinX, footX];
  }

  function gaitArmAngles(phi, amp, sprintT, sideSign, crouchT = 0.0) {
    const u = ((phi - Math.PI * 0.5) / (2.0 * Math.PI));
    let ux = sampleCycle(WALK_ARM_U, u);
    let lx = sampleCycle(WALK_ARM_L, u);
    let hx = sampleCycle(WALK_ARM_H, u);

    if (sprintT > 1e-4) {
      ux += (sampleCycle(SPRINT_ARM_U, u) - ux) * sprintT;
      lx += (sampleCycle(SPRINT_ARM_L, u) - lx) * sprintT;
      hx += (sampleCycle(SPRINT_ARM_H, u) - hx) * sprintT;
    }
    if (crouchT > 1e-4) {
      ux = ux * (1.0 - 0.50 * crouchT) - 0.16 * crouchT;
      lx = lx * (1.0 - 0.35 * crouchT) - 0.52 * crouchT;
    }

    const idleLx = -0.16 - 0.48 * crouchT;
    const uX = ux * amp;
    const lX = idleLx + (lx - idleLx) * amp;
    const hX = hx * amp;
    const fwdDrive = Math.max(0.0, -ux);
    const uY = -sideSign * (0.08 + 0.18 * sprintT) * fwdDrive * amp;
    return [uX, lX, hX, uY];
  }

  function footLowestYFromPelvis(j, pelvisPitch, thighX, shinX, footX) {
    const yHip = j.thigh_l_offset[1];
    const lThigh = Math.abs(j.shin_offset[1]);
    const lShin = Math.abs(j.foot_offset[1]);
    const hSole = Math.abs(j.foot_sole_y || -0.09);
    const a1 = pelvisPitch + thighX;
    const a2 = a1 + shinX;
    const a3 = a2 + footX;
    const yAnkle = yHip - lThigh * Math.cos(a1) - lShin * Math.cos(a2);
    const c3 = Math.cos(a3);
    const s3 = Math.sin(a3);
    const yHeel = -hSole * c3 + 0.048 * s3;
    const yToe = -hSole * c3 - 0.132 * s3;
    return yAnkle + Math.min(yHeel, yToe);
  }

  function rotYXZ(yaw, pitch, roll) {
    let m = Math.abs(yaw) > 1e-6 ? mat4RotY(yaw) : mat4Identity();
    if (Math.abs(pitch) > 1e-6) m = mat4Mul(m, mat4RotX(pitch));
    if (Math.abs(roll) > 1e-6) m = mat4Mul(m, mat4RotZ(roll));
    return m;
  }

  function rotZYX(roll, yaw, pitch) {
    let m = Math.abs(roll) > 1e-6 ? mat4RotZ(roll) : mat4Identity();
    if (Math.abs(yaw) > 1e-6) m = mat4Mul(m, mat4RotY(yaw));
    if (Math.abs(pitch) > 1e-6) m = mat4Mul(m, mat4RotX(pitch));
    return m;
  }

  function smooth01(t) {
    const x = Math.max(0.0, Math.min(1.0, t));
    return x * x * (3.0 - 2.0 * x);
  }

  function evalSmokingCycle(elapsed) {
    const period = 10.5;
    const t = ((elapsed % period) + period) % period;
    const cycleIdx = Math.floor(elapsed / period);

    let puffArm = 0.0;
    if (t >= 2.2 && t < 3.4) {
      puffArm = smooth01((t - 2.2) / 1.2);
    } else if (t >= 3.4 && t < 4.5) {
      puffArm = 1.0;
    } else if (t >= 4.5 && t < 5.7) {
      puffArm = 1.0 - smooth01((t - 4.5) / 1.2);
    }

    let emberGlow = 0.0;
    if (t >= 3.25 && t < 4.75) {
      emberGlow = Math.sin(Math.PI * (t - 3.25) / 1.5);
    }

    let exhaleAmt = 0.0;
    if (t >= 5.15 && t < 7.25) {
      exhaleAmt = Math.sin(Math.PI * (t - 5.15) / 2.10);
    }

    const dirSign = (cycleIdx % 2 === 0) ? 1.0 : -1.0;
    const baseLook =
      Math.sin(elapsed * 0.41) * 0.28 +
      Math.sin(elapsed * 0.19 + 1.3) * 0.24 +
      dirSign * 0.16 * smooth01(Math.max(0.0, Math.min(1.0, (t - 6.8) / 2.0)));
    const lookYaw = baseLook * (1.0 - 0.72 * puffArm);
    const lookPitch =
      -0.02 + Math.sin(elapsed * 0.33 + 0.5) * 0.06
      - 0.21 * exhaleAmt
      + 0.05 * puffArm;

    return { puffArm, emberGlow, exhaleAmt, lookYaw, lookPitch };
  }

  const GARMENT_BUST_TUNING = {
    tshirt:    [102.0,  7.4, 1.00],
    sweater:   [112.0,  8.6, 0.90],
    tracksuit: [118.0,  9.2, 0.86],
    hoodie:    [128.0, 10.4, 0.78],
    jacket:    [142.0, 11.8, 0.68],
    coat:      [148.0, 12.4, 0.64],
  };

  class BustPhysics {
    constructor() {
      this.yl = 0.0; this.vyl = 0.0;
      this.yr = 0.0; this.vyr = 0.0;
      this.xl = 0.0; this.vxl = 0.0;
      this.xr = 0.0; this.vxr = 0.0;
      this._prevVelY = 0.0;
      this._prevYaw = 0.0;
      this._prevSpeed = 0.0;
      this._prevOnGround = true;
      this._prevKick = 0.0;
      this._prevRecoil = 0.0;
      this._initialized = false;
    }
    reset() {
      this.yl = this.vyl = this.yr = this.vyr = 0.0;
      this.xl = this.vxl = this.xr = this.vxr = 0.0;
      this._initialized = false;
    }
    impulse(vy = 0.0, vx = 0.0) {
      this.vyl += vy * 1.03;
      this.vyr += vy * 0.97;
      this.vxl += vx;
      this.vxr += vx * 0.96;
    }
    update(dt, elapsed, facingYaw, opts = {}) {
      const horizSpeed = opts.horizSpeed || 0.0;
      const velY = opts.velY || 0.0;
      const onGround = (opts.onGround !== undefined) ? opts.onGround : true;
      const crouching = !!opts.crouching;
      const sliding = !!opts.sliding;
      const kickPhase = opts.kickPhase || 0.0;
      const recoil = opts.recoil || 0.0;
      const walkPhase = opts.walkPhase || 0.0;
      const topId = opts.topId || "tshirt";
      const bustSize = Math.max(0.0, Math.min(1.0, (opts.bustSize !== undefined) ? Number(opts.bustSize) : 0.5));

      dt = Math.max(1e-4, Math.min(0.05, dt));
      const [kRaw, cRaw, gRaw] = GARMENT_BUST_TUNING[topId] || [115.0, 8.8, 0.90];
      // Larger bust has more soft-body inertia & lively jiggle; smaller bust stays taut & subtle
      const sizeScale = 0.58 + 0.92 * bustSize;
      const kBase = kRaw * (1.16 - 0.34 * bustSize);
      const cBase = cRaw * (0.94 - 0.18 * bustSize);
      const gAmp = gRaw * 1.48 * sizeScale;

      if (!this._initialized) {
        this._prevVelY = velY;
        this._prevYaw = facingYaw;
        this._prevSpeed = horizSpeed;
        this._prevOnGround = onGround;
        this._prevKick = kickPhase;
        this._prevRecoil = recoil;
        this._initialized = true;
      }

      let dyaw = facingYaw - this._prevYaw;
      while (dyaw > Math.PI) dyaw -= Math.PI * 2.0;
      while (dyaw < -Math.PI) dyaw += Math.PI * 2.0;
      const yawVel = Math.max(-14.0, Math.min(14.0, dyaw / dt));

      const ayBody = Math.max(-42.0, Math.min(42.0, (velY - this._prevVelY) / dt));
      const speedAcc = Math.max(-18.0, Math.min(18.0, (horizSpeed - this._prevSpeed) / dt));

      if (onGround && !this._prevOnGround) {
        const impact = Math.min(1.0, Math.max(0.25, Math.abs(this._prevVelY) / 6.5));
        this.vyl -= 0.62 * impact * gAmp;
        this.vyr -= 0.58 * impact * gAmp;
        this.vxl += 0.32 * impact * gAmp;
        this.vxr -= 0.32 * impact * gAmp;
      }
      if (!onGround && this._prevOnGround && velY > 1.5) {
        this.vyl -= 0.34 * gAmp;
        this.vyr -= 0.34 * gAmp;
      }
      if (kickPhase > 0.28 && this._prevKick <= 0.28) {
        this.vyl += 0.30 * gAmp;
        this.vyr += 0.34 * gAmp;
        this.vxl -= 0.68 * gAmp;
        this.vxr -= 0.60 * gAmp;
      }
      const dRec = recoil - this._prevRecoil;
      if (dRec > 0.015) {
        const kr = Math.min(1.0, dRec * 4.5) * gAmp;
        this.vyl += 0.26 * kr;
        this.vyr += 0.28 * kr;
        this.vxl += 0.28 * kr;
        this.vxr -= 0.28 * kr;
      }

      const strideF = Math.min(1.0, horizSpeed / 5.4);
      let gaitYL = 0.0, gaitYR = 0.0, gaitXL = 0.0, gaitXR = 0.0;
      if (onGround && !sliding && strideF > 0.04) {
        let runBoost = 1.0 + 0.62 * Math.min(1.0, Math.max(0.0, (horizSpeed - 4.2) / 3.2));
        if (crouching) runBoost *= 0.52;
        gaitYL = -Math.sin(walkPhase * 2.0 + 0.14) * 1.95 * strideF * runBoost;
        gaitYR = -Math.sin(walkPhase * 2.0 - 0.14) * 1.95 * strideF * runBoost;
        gaitXL = Math.cos(walkPhase + 0.18) * 1.95 * strideF * runBoost;
        gaitXR = Math.cos(walkPhase - 0.18) * 1.95 * strideF * runBoost;
      }

      const breathF = Math.sin(elapsed * 1.65) * 0.24;
      const airLift = !onGround ? 0.44 : (sliding ? 0.18 : 0.0);

      const fyL = (-ayBody * 0.046 + gaitYL + breathF + airLift - speedAcc * 0.016) * gAmp;
      const fyR = (-ayBody * 0.044 + gaitYR + breathF + airLift - speedAcc * 0.016) * gAmp;
      const fxL = (-yawVel * 0.85 + gaitXL) * gAmp;
      const fxR = (-yawVel * 0.80 + gaitXR) * gAmp;

      const subDt = dt * 0.5;
      const kl = kBase, kr = kBase * 1.045;
      const cl = cBase, cr = cBase * 1.02;
      const kx = kBase * 0.80, cx = cBase * 0.86;
      const yLim = 0.024 + 0.014 * bustSize;
      const xLim = 0.22 + 0.14 * bustSize;
      for (let s = 0; s < 2; s++) {
        this.vyl += (fyL - kl * this.yl - cl * this.vyl) * subDt;
        this.yl += this.vyl * subDt;
        this.vyr += (fyR - kr * this.yr - cr * this.vyr) * subDt;
        this.yr += this.vyr * subDt;

        this.vxl += (fxL - kx * this.xl - cx * this.vxl) * subDt;
        this.xl += this.vxl * subDt;
        this.vxr += (fxR - kx * 1.04 * this.xr - cx * this.vxr) * subDt;
        this.xr += this.vxr * subDt;

        if (this.yl > yLim) { this.yl = yLim; this.vyl = Math.min(0.0, this.vyl * -0.28); }
        else if (this.yl < -yLim * 1.1) { this.yl = -yLim * 1.1; this.vyl = Math.max(0.0, this.vyl * -0.28); }
        if (this.yr > yLim) { this.yr = yLim; this.vyr = Math.min(0.0, this.vyr * -0.28); }
        else if (this.yr < -yLim * 1.1) { this.yr = -yLim * 1.1; this.vyr = Math.max(0.0, this.vyr * -0.28); }
        this.xl = Math.max(-xLim, Math.min(xLim, this.xl));
        this.xr = Math.max(-xLim, Math.min(xLim, this.xr));
      }

      this._prevVelY = velY;
      this._prevYaw = facingYaw;
      this._prevSpeed = horizSpeed;
      this._prevOnGround = onGround;
      this._prevKick = kickPhase;
      this._prevRecoil = recoil;

      const pitchL = Math.max(-0.36, Math.min(0.38, -this.yl * 14.5));
      const pitchR = Math.max(-0.36, Math.min(0.38, -this.yr * 14.5));
      return [pitchL, this.xl * 1.08, this.yl * 0.78, pitchR, this.xr * 1.08, this.yr * 0.78];
    }
  }

  function evaluateCharacterBones(jointsByGender, gender, rootPos, facingYaw, opts) {
    const j = jointsByGender[gender] || jointsByGender.male;
    const elapsed = opts.elapsed || 0;
    const horizSpeed = opts.horizSpeed || 0;
    const onGround = opts.onGround !== false;
    const crouching = Boolean(opts.crouching);
    const sliding = Boolean(opts.sliding);
    const kickProgress = opts.kickProgress || 0;
    const showcase = Boolean(opts.showcase);
    const smokingMenu = Boolean(opts.smokingMenu);
    const hasWalkPhase = typeof opts.walkPhase === "number";
    const velY = opts.velY || 0;

    let pelvisX = 0.0;
    let pelvisY = j.pelvis_y;
    let pelvisPitch = 0.0, pelvisYaw = 0.0, pelvisRoll = 0.0;
    let torsoPitch = 0.0, torsoYaw = 0.0, torsoRoll = 0.0;
    let neckPitch = 0.0, neckYaw = 0.0, neckRoll = 0.0;
    let headPitch = 0.0, headYaw = 0.0, headRoll = 0.0;

    let uaLX = 0.02, uaLY = -0.04, uaLZ = 0.075, laLX = -0.18, hLX = 0.04;
    let uaRX = 0.02, uaRY =  0.04, uaRZ = -0.075, laRX = -0.18, hRX = 0.04;
    let lThighX = 0.0, lShinX = 0.0, lFootX = 0.0;
    let rThighX = 0.0, rShinX = 0.0, rFootX = 0.0;

    if (smokingMenu) {
      const { puffArm, emberGlow, exhaleAmt, lookYaw, lookPitch } = evalSmokingCycle(elapsed);
      const coughPhase = ((elapsed % 47.0) + 47.0) % 47.0;
      const cough = smooth01((coughPhase - 23.0) / 0.28) * (1.0 - smooth01((coughPhase - 24.25) / 0.34));
      const ashPhase = ((elapsed % 31.0) + 31.0) % 31.0;
      const ashFlick = smooth01((ashPhase - 15.0) / 0.12) * (1.0 - smooth01((ashPhase - 15.55) / 0.16));
      const breath = Math.sin(elapsed * 1.45);
      const sway = Math.sin(elapsed * 0.55);

      pelvisX = 0.016 + sway * 0.008;
      pelvisRoll = 0.025 + sway * 0.010;
      torsoPitch = 0.028 + breath * 0.014 - 0.025 * exhaleAmt + 0.018 * emberGlow;
      torsoRoll = -pelvisRoll * 0.88;
      torsoYaw = lookYaw * 0.18 + sway * 0.025;

      neckYaw = lookYaw * 0.36;
      headYaw = lookYaw * 0.64;
      neckPitch = lookPitch * 0.42 - torsoPitch * 0.45;
      headPitch = lookPitch * 0.58 - torsoPitch * 0.35;
      headRoll = -lookYaw * 0.12 + Math.sin(elapsed * 0.48) * 0.022;

      uaLX = 0.06 + Math.sin(elapsed * 0.72) * 0.025;
      uaLY = -0.08;
      uaLZ = 0.065 + breath * 0.008;
      laLX = -0.34 - breath * 0.020;
      hLX = 0.08 + Math.sin(elapsed * 2.8 + 0.6) * 0.028;
      uaLX += -0.76 * cough;
      uaLY += 0.28 * cough;
      laLX += -0.78 * cough;
      hLX += 0.24 * cough;
      headPitch += 0.16 * cough;
      torsoPitch += 0.035 * cough;

      const restUaRX = -0.16 + Math.sin(elapsed * 0.95) * 0.022;
      const restUaRY = 0.02;
      const restUaRZ = -0.28 - breath * 0.012;
      const restLaRX = -0.78 + Math.sin(elapsed * 0.72) * 0.035;
      const restHRX = 0.08;

      let puffUaRX, puffUaRY, puffUaRZ, puffLaRX;
      if (gender === "female") {
        puffUaRX = -1.28; puffUaRY = 0.80; puffUaRZ = 0.08; puffLaRX = -1.84;
      } else {
        puffUaRX = -1.12; puffUaRY = 0.96; puffUaRZ = 0.24; puffLaRX = -1.84;
      }

      uaRX = restUaRX + (puffUaRX - restUaRX) * puffArm;
      uaRY = restUaRY + (puffUaRY - restUaRY) * puffArm;
      uaRZ = restUaRZ + (puffUaRZ - restUaRZ) * puffArm;
      laRX = restLaRX + (puffLaRX - restLaRX) * puffArm;
      hRX = restHRX + (0.18 - restHRX) * puffArm + 0.018 * Math.sin(elapsed * 2.3) * (1.0 - puffArm) - 0.40 * ashFlick;
      laRX += 0.18 * ashFlick;

      lThighX = -0.02; lShinX = 0.05; lFootX = -(lThighX + lShinX);
      rThighX = 0.05; rShinX = 0.10; rFootX = -(rThighX + rShinX);
      const ly = footLowestYFromPelvis(j, pelvisPitch, lThighX, lShinX, lFootX);
      const ry = footLowestYFromPelvis(j, pelvisPitch, rThighX, rShinX, rFootX);
      pelvisY = -Math.min(ly, ry) + breath * 0.003;
    } else if (showcase) {
      const breath = Math.sin(elapsed * 1.80);
      const sway = Math.sin(elapsed * 0.80);
      pelvisX = 0.014 + sway * 0.008;
      pelvisRoll = 0.028 + sway * 0.012;
      torsoPitch = 0.020 + breath * 0.014;
      torsoRoll = -pelvisRoll * 0.90;
      torsoYaw = sway * 0.028;

      const lookYaw = Math.sin(elapsed * 0.65) * 0.17;
      const lookPitch = -0.02 + Math.sin(elapsed * 0.46) * 0.042;
      neckYaw = lookYaw * 0.42;
      headYaw = lookYaw * 0.58;
      neckPitch = lookPitch * 0.40 - torsoPitch * 0.45;
      headPitch = lookPitch * 0.60 - torsoPitch * 0.35;
      headRoll = Math.sin(elapsed * 0.52) * 0.022;

      uaLX = 0.03 + Math.sin(elapsed * 1.15) * 0.025;
      uaRX = -0.02 - Math.sin(elapsed * 1.15) * 0.025;
      uaLY = -0.05; uaRY = 0.05;
      uaLZ = 0.082 + breath * 0.010;
      uaRZ = -0.082 - breath * 0.010;
      laLX = -0.19 - breath * 0.022;
      laRX = -0.23 + breath * 0.018;
      hLX = 0.05; hRX = 0.04;

      lThighX = -0.03; lShinX = 0.06; lFootX = -(lThighX + lShinX);
      rThighX = 0.05; rShinX = 0.10; rFootX = -(rThighX + rShinX);
      const ly = footLowestYFromPelvis(j, pelvisPitch, lThighX, lShinX, lFootX);
      const ry = footLowestYFromPelvis(j, pelvisPitch, rThighX, rShinX, rFootX);
      pelvisY = -Math.min(ly, ry) + breath * 0.004;
    } else if (sliding) {
      pelvisY = 0.34;
      pelvisPitch = -0.58;
      pelvisYaw = -0.22;
      torsoPitch = -0.24;
      torsoYaw = 0.26;
      neckPitch = 0.36;
      headPitch = 0.32;
      headYaw = -0.05;

      rThighX = -1.06; rShinX = 0.14; rFootX = -0.26;
      lThighX = -0.42; lShinX = 1.52; lFootX = 0.35;
      uaLX = 0.58; uaLZ = 0.48; laLX = -0.35; hLX = 0.25;
      uaRX = -0.45; uaRZ = -0.42; laRX = -0.85; hRX = -0.10;
    } else if (!onGround) {
      const upT = Math.min(1.0, Math.max(0.0, velY / 7.5));
      const fallT = Math.min(1.0, Math.max(0.0, -velY / 9.0));
      const airSway = Math.sin(elapsed * 4.2) * 0.04;

      pelvisY += 0.05 * upT - 0.03 * fallT;
      pelvisPitch = 0.04 + 0.08 * fallT;
      torsoPitch = 0.08 + 0.14 * fallT - 0.06 * upT;
      neckPitch = -(pelvisPitch + torsoPitch) * 0.45;
      headPitch = -(pelvisPitch + torsoPitch) * 0.50;

      lThighX = -0.78 * upT - 0.36 * (1.0 - upT) + 0.08 * fallT + airSway;
      lShinX = 1.24 * upT + 0.64 * (1.0 - upT) - 0.16 * fallT;
      lFootX = 0.34 * upT + 0.18 * (1.0 - upT) + 0.10 * fallT;

      rThighX = 0.22 * upT - 0.26 * (1.0 - upT) + 0.06 * fallT - airSway;
      rShinX = 0.42 * upT + 0.54 * (1.0 - upT) - 0.10 * fallT;
      rFootX = 0.44 * upT + 0.22 * (1.0 - upT) + 0.10 * fallT;

      uaLX = -0.92 * upT - 0.36 * (1.0 - upT) - 0.20 * fallT;
      uaLZ = 0.26 + 0.34 * fallT;
      laLX = -1.12 * upT - 0.62 * (1.0 - upT) - 0.12 * fallT;
      hLX = 0.10;

      uaRX = 0.24 * upT - 0.26 * (1.0 - upT) - 0.18 * fallT;
      uaRZ = -0.26 - 0.34 * fallT;
      laRX = -0.88 * upT - 0.56 * (1.0 - upT) - 0.12 * fallT;
      hRX = 0.10;
    } else {
      const moveAmp = Math.min(1.0, Math.max(0.0, (horizSpeed - 0.12) / 1.85));
      const sprintT = crouching ? 0.0 : Math.min(1.0, Math.max(0.0, (horizSpeed - 4.7) / 2.6));
      const crouchT = crouching ? 1.0 : 0.0;

      if (moveAmp > 0.01) {
        const phi = hasWalkPhase
          ? opts.walkPhase
          : elapsed * (crouching ? 4.9 : (6.1 + 2.2 * sprintT));
        const s1 = Math.sin(phi);
        const c1 = Math.cos(phi);

        pelvisPitch = (0.045 + 0.12 * sprintT + 0.10 * crouchT) * moveAmp;
        pelvisYaw = -s1 * (0.105 + 0.085 * sprintT) * (1.0 - 0.30 * crouchT) * moveAmp;
        pelvisRoll = -c1 * (0.038 + 0.018 * sprintT) * (1.0 - 0.25 * crouchT) * moveAmp;
        pelvisX = -c1 * (0.020 * (1.0 - 0.35 * sprintT)) * moveAmp;

        torsoPitch = (0.055 + 0.26 * sprintT + 0.26 * crouchT) * moveAmp + (0.28 * crouchT * (1.0 - moveAmp));
        torsoYaw = s1 * (0.20 + 0.17 * sprintT) * (1.0 - 0.30 * crouchT) * moveAmp;
        torsoRoll = -pelvisRoll * 1.18;

        const compPitch = -(pelvisPitch + torsoPitch) * 0.88;
        const compYaw = -(pelvisYaw + torsoYaw) * 0.85;
        neckPitch = compPitch * 0.45 + Math.sin(2.0 * phi - 0.6) * 0.018 * moveAmp;
        headPitch = compPitch * 0.55 + Math.sin(2.0 * phi - 0.9) * 0.015 * moveAmp;
        neckYaw = compYaw * 0.42;
        headYaw = compYaw * 0.58;
        headRoll = -torsoRoll * 0.55;

        [lThighX, lShinX, lFootX] = gaitLegAngles(phi, moveAmp, sprintT, pelvisPitch, crouchT);
        [rThighX, rShinX, rFootX] = gaitLegAngles(phi + Math.PI, moveAmp, sprintT, pelvisPitch, crouchT);

        const ly = footLowestYFromPelvis(j, pelvisPitch, lThighX, lShinX, lFootX);
        const ry = footLowestYFromPelvis(j, pelvisPitch, rThighX, rShinX, rFootX);
        const groundedPy = -Math.min(ly, ry);
        const flightWave = Math.pow(Math.max(0.0, Math.sin(2.0 * phi - 0.65)), 2.2);
        pelvisY = groundedPy + sprintT * 0.052 * flightWave * moveAmp;

        [uaLX, laLX, hLX, uaLY] = gaitArmAngles(phi + Math.PI, moveAmp, sprintT, 1.0, crouchT);
        [uaRX, laRX, hRX, uaRY] = gaitArmAngles(phi, moveAmp, sprintT, -1.0, crouchT);
        const armOut = 0.082 + 0.052 * sprintT + 0.055 * crouchT;
        uaLZ = armOut;
        uaRZ = -armOut;
      } else {
        const breath = Math.sin(elapsed * 1.90);
        if (crouching) {
          pelvisPitch = 0.08;
          torsoPitch = 0.30 + breath * 0.012;
          neckPitch = -0.18;
          headPitch = -0.18;
          lThighX = -0.82; lShinX = 1.34; lFootX = -(pelvisPitch + lThighX + lShinX);
          rThighX = -0.82; rShinX = 1.34; rFootX = -(pelvisPitch + rThighX + rShinX);
          uaLX = -0.18; uaRX = -0.18;
          uaLZ = 0.13; uaRZ = -0.13;
          laLX = -0.64; laRX = -0.64;
          const ly = footLowestYFromPelvis(j, pelvisPitch, lThighX, lShinX, lFootX);
          pelvisY = -ly + breath * 0.004;
        } else {
          pelvisX = Math.sin(elapsed * 0.72) * 0.008;
          pelvisRoll = Math.sin(elapsed * 0.72) * 0.012;
          torsoRoll = -pelvisRoll * 0.85;
          torsoPitch = 0.015 + breath * 0.014;
          neckPitch = -torsoPitch * 0.55;
          headPitch = -torsoPitch * 0.45;
          lThighX = -0.02; lShinX = 0.05; lFootX = -(lThighX + lShinX);
          rThighX = 0.03; rShinX = 0.06; rFootX = -(rThighX + rShinX);
          uaLZ = 0.082 + breath * 0.010;
          uaRZ = -0.082 - breath * 0.010;
          laLX = -0.17 - breath * 0.020;
          laRX = -0.19 + breath * 0.015;
          const ly = footLowestYFromPelvis(j, pelvisPitch, lThighX, lShinX, lFootX);
          const ry = footLowestYFromPelvis(j, pelvisPitch, rThighX, rShinX, rFootX);
          pelvisY = -Math.min(ly, ry) + breath * 0.005;
        }
      }
    }

    if (kickProgress > 0.001) {
      const sinK = Math.sin(kickProgress * Math.PI);
      if (kickProgress < 0.32) {
        const u = kickProgress / 0.32;
        rThighX = -1.32 * u;
        rShinX = 1.62 * u;
        rFootX = 0.30 * u;
        pelvisYaw = 0.24 * u;
        torsoPitch -= 0.14 * u;
        torsoYaw -= 0.32 * u;
      } else if (kickProgress < 0.62) {
        const u = (kickProgress - 0.32) / 0.30;
        rThighX = -1.32 - 0.28 * u;
        rShinX = 1.62 * (1.0 - u) + 0.06 * u;
        rFootX = 0.30 + 0.28 * u;
        pelvisYaw = 0.24 + 0.12 * u;
        torsoPitch -= 0.14 + 0.14 * u;
        torsoYaw -= 0.32 + 0.14 * u;
      } else {
        const u = (kickProgress - 0.62) / 0.38;
        const rem = 1.0 - u;
        rThighX = -1.60 * rem;
        rShinX = 0.06 * rem;
        rFootX = 0.58 * rem;
        pelvisYaw = 0.36 * rem;
        torsoPitch -= 0.28 * rem;
        torsoYaw -= 0.46 * rem;
      }
      neckPitch += 0.16 * sinK;
      headPitch += 0.14 * sinK;
      uaLX = -0.68 * sinK;
      uaLZ = 0.14 + 0.26 * sinK;
      laLX = -0.85 * sinK - 0.15;
      uaRX = 0.56 * sinK;
      uaRZ = -0.14 - 0.24 * sinK;
      laRX = -0.95 * sinK - 0.15;
    }

    // Two-handed tactical M1911 Pistol Pose & Reload Animation Override (3rd-person F5!)
    if (opts.holdingPistol && !showcase && !sliding && kickProgress <= 0.001) {
      const aimPitch = Math.max(-0.65, Math.min(0.65, opts.cameraPitch || 0.0));
      const rec = Math.max(0.0, Math.min(0.5, opts.pistolRecoil || 0.0));
      if (opts.pistolReloading) {
        const u = Math.max(0.0, Math.min(1.0, opts.pistolReloadU || 0.0));
        uaRX = -0.96 - aimPitch * 0.45;
        uaRY = 0.34;
        uaRZ = -0.12;
        laRX = -0.78;
        hRX = 0.10;
        if (u < 0.38) {
          const t = u / 0.38;
          uaLX = -1.16 * (1.0 - t) - 0.32 * t;
          uaLY = -0.34 * (1.0 - t) - 0.08 * t;
          uaLZ = 0.12 + 0.08 * t;
          laLX = -0.46 * (1.0 - t) - 0.96 * t;
          hLX = 0.08;
        } else if (u < 0.68) {
          const t = (u - 0.38) / 0.30;
          uaLX = -0.32 * (1.0 - t) - 1.04 * t;
          uaLY = -0.08 * (1.0 - t) - 0.44 * t;
          uaLZ = 0.20 * (1.0 - t) + 0.10 * t;
          laLX = -0.96 * (1.0 - t) - 0.78 * t;
          hLX = 0.12;
        } else if ((opts.pistolReloadType || "empty") === "empty" && u < 0.88) {
          const t = (u - 0.68) / 0.20;
          const rack = Math.sin(t * Math.PI);
          uaLX = -1.18 + 0.22 * rack;
          uaLY = -0.44;
          uaLZ = 0.16;
          laLX = -0.74 - 0.34 * rack;
          hLX = 0.06;
        } else {
          uaLX = -1.16 - aimPitch * 0.55;
          uaLY = -0.36;
          uaLZ = 0.12;
          laLX = -0.46;
          hLX = 0.08;
        }
      } else {
        uaRX = -1.26 - aimPitch * 0.65 - rec * 0.75;
        uaRY = 0.26;
        uaRZ = -0.08;
        laRX = -0.30 - rec * 0.28;
        hRX = 0.04;

        uaLX = -1.18 - aimPitch * 0.62 - rec * 0.55;
        uaLY = -0.38;
        uaLZ = 0.10;
        laLX = -0.44 - rec * 0.22;
        hLX = 0.06;
      }
    }

    // NPC handheld prop arm-pose overrides (winter street passersby carrying bag, book, coffee, briefcase, box!)
    const npcProp = (opts && opts.npcProp) ? opts.npcProp : "none";
    if(opts.listen){neckYaw+=.26*opts.listen;headYaw+=.38*opts.listen;headPitch-=.075*opts.listen;}
    if (opts.throwAmount) {uaRX-=opts.throwAmount*.76;laRX+=opts.throwAmount*.34;hRX-=opts.throwAmount*.46;}
    if (npcProp !== "none" && !smokingMenu && !showcase && !opts.holdingPistol) {
      const bob = Math.sin(elapsed * 2.4) * 0.025;
      if (npcProp === "coffee") {
        uaRX = -0.52 + bob; uaRY = 0.14; uaRZ = -0.10; laRX = -1.18; hRX = 0.06;
      } else if (npcProp === "book") {
        uaRX = -0.44 + bob; uaRY = 0.22; uaRZ = -0.08; laRX = -1.05; hRX = 0.12;
      } else if (npcProp === "box") {
        uaLX = -0.78 + bob; uaLY = -0.16; uaLZ = 0.08; laLX = -0.92; hLX = 0.08;
        uaRX = -0.78 - bob; uaRY = 0.16; uaRZ = -0.08; laRX = -0.92; hRX = 0.08;
      } else if (npcProp === "bag" || npcProp === "briefcase") {
        uaRX = 0.04 + bob * 0.5; uaRY = 0.04; uaRZ = -0.14; laRX = -0.14; hRX = 0.04;
      }
    }

    if(opts.seatedDinner){
      const breath=.008*Math.sin(elapsed*1.24);pelvisY=.53;pelvisX=pelvisRoll=pelvisYaw=pelvisPitch=0;
      lThighX=-1.47;rThighX=-1.52;lShinX=1.54;rShinX=1.68;lFootX=-.07;rFootX=-.16;
      torsoPitch=.32+breath;torsoRoll=.13;torsoYaw=0;headPitch=.17;headRoll=.18;neckPitch=-.17;neckRoll=.07;headYaw=neckYaw=0;
      uaRX=-1.22;uaRY=.10;uaRZ=-.12;laRX=-2.48;hRX=.32;
      uaLX=-1.13;uaLY=-.13;uaLZ=.14;laLX=-.58;hLX=.11;
    }
    if(opts.heroSit!==undefined){const q=opts.heroSit;
      pelvisY=j.pelvis_y*(1-q)+.53*q;pelvisX=pelvisPitch=pelvisYaw=pelvisRoll=0;torsoPitch=.09+.16*q;torsoRoll=torsoYaw=0;
      lThighX=-1.47*q;rThighX=-1.51*q;lShinX=1.53*q;rShinX=1.6*q;lFootX=-.06*q;rFootX=-.09*q;
      headPitch=.08;neckPitch=-.14;headRoll=neckRoll=0;headYaw=.23*Math.sin(elapsed*5.5)*opts.heroShake+.58*opts.doorLook;neckYaw=.30*opts.doorLook;
      uaRX=-.20-.80*q;laRX=-.23-.74*q;uaLX=-.15-.84*q;laLX=-.25-.62*q;uaLZ=.12;uaRZ=-.11;
    }
    if(opts.worried){neckPitch+=.17*opts.worried;headPitch+=.30*opts.worried;torsoPitch+=.065*opts.worried;}
    if(opts.acting){const a=opts.acting;
      torsoPitch+=.022*a.breath+.13*a.ask+.085*a.doubt+.155*a.settle-.095*a.flinch;
      torsoRoll+=.031*a.hope-.034*a.comfort;neckPitch+=.12*a.drawLook-.095*a.flinch;headPitch+=.047*a.drawLook+.09*a.hope;
      headRoll+=.080*a.ask-.055*a.hope;headYaw+=.095*a.drawLook;neckYaw+=.045*a.drawLook;
      // A small palm-up question, a slower offering gesture and a hitch on the bell.
      uaLY-=.26*a.ask+.29*a.palm;uaLZ+=.19*a.ask+.21*a.palm;uaLX-=.20*a.ask+.16*a.palm+.11*a.flinch;
      laLX-=.62*a.ask+.65*a.palm+.38*a.doubt+.22*a.flinch;hLX-=.23*a.ask+.22*a.palm;
      uaRX-=.18*a.comfort;uaRY+=.17*a.comfort;laRX-=.26*a.comfort;hRX+=.14*a.comfort;
      // Overlap the wrist with the elbow; it settles a fraction after the gesture.
      const wristBeat=.5+.5*Math.sin(elapsed*4.15);hLX+=.095*wristBeat*(a.ask+a.doubt)+.068*a.breath*a.palm;
      uaLY+=.080*a.doubt*wristBeat;torsoYaw+=.036*a.ask-.046*a.drawLook;headRoll+=.023*a.breath*(.4+a.doubt);
      lThighX+=.042*a.doubt+.062*a.settle;lShinX-=.053*a.doubt+.08*a.settle;
    }
    if(opts.beats){const {ask,stop,chest,tight,nod}=opts.beats;
      uaLX-=.57*ask+.98*stop+.42*chest;laLX-=.87*ask+.34*stop+1.18*chest;uaLZ+=.43*ask+.21*stop;uaLY+=.65*chest;hLX+=.38*stop-.21*ask;
      uaRX-=.30*ask+.16*tight;laRX-=.46*ask+.39*tight;uaRY-=.21*ask;torsoYaw+=.10*chest-.09*stop;torsoPitch+=.065*ask-.057*stop+.045*tight;
      neckPitch+=.12*nod+.07*tight;headRoll+=.085*ask-.055*tight;headPitch+=.055*nod;}
    // A thought reaches the eyes before the chin and shoulders. Speech accents are
    // one-shot envelopes from the current line, not a looping talking animation.
    if(opts.delivery){const d=opts.delivery,br=d.breath,side=d.side;
      torsoPitch+=.020*br+.085*d.offer-.06*d.catch;torsoRoll+=side*(.015*br+.041*d.offer);neckPitch+=.095*d.ack-.085*d.catch;headPitch+=.065*d.ack+.052*d.release;
      headRoll+=side*(.041*d.ack-.037*d.catch);neckYaw+=side*.060*d.release;headYaw+=side*.049*d.release;
      uaLX-=.24*d.offer+.12*d.catch;laLX-=.34*d.offer+.14*d.release;uaLZ+=.09*d.offer;hLX+=.21*d.tail;
      uaRX+=.036*br;headPitch+=.014*br;if(!d.seated){pelvisX+=side*.032*d.release;pelvisRoll+=side*.026*d.release;lThighX+=.038*d.release;}}
    if(opts.entryPush){const k=opts.entryPush;torsoPitch+=.15*k;uaRX-=1.40*k;uaLX-=1.21*k;laRX-=.18*k;laLX-=.24*k;uaRY+=.24*k;uaLY-=.19*k;}
    if(opts.recoilBody){torsoPitch-=.29*opts.recoilBody;headPitch+=.13*opts.recoilBody;uaLZ+=.30*opts.recoilBody;}
    if(opts.doorReach){uaRX-=.93*opts.doorReach;laRX-=.55*opts.doorReach;}
    if(opts.onlineInspect){uaRX-=.40*opts.onlineInspect;hRX-=.37*opts.onlineInspect;headPitch+=.09*opts.onlineInspect;}if(opts.onlineADS){uaRX-=.15*opts.onlineADS;}
    if(opts.samBeat){const b=opts.samBeat;headYaw+=.30*b.glance;neckYaw+=.10*b.follow;torsoYaw+=.085*b.follow;pelvisX+=.033*b.shift;pelvisRoll+=.030*b.shift;torsoRoll-=.047*b.shift;uaLX+=.13*b.shift;laLX-=.18*b.follow;hRX+=.09*b.settle;headPitch+=.057*b.settle;}
    if(opts.pocketReach){uaRX+=.19*opts.pocketReach;uaRY+=.28*opts.pocketReach;laRX-=.38*opts.pocketReach;torsoRoll-=.07*opts.pocketReach;}
    if(opts.showProp){uaRX-=.48*opts.showProp;laRX-=.59*opts.showProp;uaRZ-=.17*opts.showProp;hRX+=.11*opts.showProp;}
    if(opts.balconyIdle!==undefined){
      const breath=.5+.5*Math.cos(elapsed*.85);
      if(opts.balconyIdle===0){const lift=Math.pow(breath,3);uaRX-=.57*lift;laRX-=.63*lift;headPitch-=.07*lift;}
      else if(opts.balconyIdle===1){headPitch+=.20;laLX-=.48+.10*breath;uaLX-=.24;}
      else{torsoPitch+=.13;uaRX-=.37;laRX-=.40;uaLX-=.34;laLX-=.42;}
    }
    // Expressive look & talking gestures for interactive map Test NPC
    const npcHeadLookYaw = (opts && typeof opts.npcHeadLookYaw === "number") ? opts.npcHeadLookYaw : 0.0;
    const npcTalking = (opts && typeof opts.npcTalking === "number") ? opts.npcTalking : 0.0;
    const isHoldingPistol = Boolean(opts && opts.holdingPistol);
    if (npcHeadLookYaw !== 0.0 && !isHoldingPistol && !smokingMenu && !showcase) {
      neckYaw += npcHeadLookYaw * 0.42;
      headYaw += npcHeadLookYaw * 0.58;
    }
    if (npcTalking > 0.01 && !isHoldingPistol && !smokingMenu && !showcase) {
      const tb = Math.max(0.0, Math.min(1.0, npcTalking));
      const g1 = Math.sin(elapsed * 5.2);
      const g2 = Math.sin(elapsed * 3.1 + 1.1);
      const g3 = Math.cos(elapsed * 2.4);
      torsoYaw += g3 * 0.045 * tb;
      torsoPitch += (0.02 + 0.025 * g2) * tb;
      headPitch += (0.045 * Math.sin(elapsed * 4.4) - 0.03) * tb;
      headRoll += 0.055 * g3 * tb;
      uaRX = uaRX * (1.0 - tb) + (-0.42 + 0.16 * g1) * tb;
      uaRY = uaRY * (1.0 - tb) + (0.22 + 0.14 * g2) * tb;
      uaRZ = uaRZ * (1.0 - tb) + (-0.18 - 0.07 * g3) * tb;
      laRX = laRX * (1.0 - tb) + (-1.08 + 0.22 * g2) * tb;
      hRX  = hRX  * (1.0 - tb) + (0.14 * g1) * tb;
      uaLX = uaLX * (1.0 - tb) + (-0.18 + 0.09 * g2) * tb;
      uaLY = uaLY * (1.0 - tb) + (-0.14 + 0.08 * g3) * tb;
      laLX = laLX * (1.0 - tb) + (-0.58 + 0.12 * g1) * tb;
    }

    if(opts.waking!==undefined){const u=opts.waking,q=sstep(.035,.74,u),calm=sstep(2.8,10.8,u),pivot=sstep(8.35,12.74,u),plant=sstep(10.52,13.24,u),up=sstep(13.95,17.62,u),step=sstep(18.1,20.32,u),lean=sstep(12.62,14.42,u)*(1-sstep(15,17.2,u));
      const breaths=[.03,1.18,2.44,3.87,5.49,7.38,9.63,12.02,15.02,18.38,21.75,24.8];let air=0;for(let i=0;i<breaths.length-1;i++){if(u>=breaths[i]&&u<breaths[i+1]){const k=(u-breaths[i])/(breaths[i+1]-breaths[i]);air=k<.27?sstep(0,.27,k):1-sstep(.27,.94,k);break;}}
      const gasp=(air-.32)*(1-.72*calm),scan=.27*Math.sin(u*1.34)*sstep(.89,2,u)*(1-calm)-.36*sstep(7,8.6,u)+.64*sstep(10.6,13.1,u)-.96*sstep(16.8,18.1,u)+1.12*sstep(18.9,20.8,u)-.42*sstep(21,22,u);
      pelvisY=lerp(.642+.009*q,j.pelvis_y,up)-.030*lean;pelvisPitch=(-1.37+1.27*q)*(1-plant);torsoPitch=(-.07+.22*q)*(1-up)+.091*gasp*q+.28*lean;neckPitch=.10-.18*q+.14*lean;headPitch=-.09+.18*q+.06*gasp+.16*pivot*(1-up);headYaw=scan;headRoll=.11*scan;torsoYaw=.18*scan;
      uaRX=-.12-.69*q+.77*calm-.47*lean;laRX=-.35-1.13*q+1.21*calm-.15*lean;uaRY=.29*q*(1-calm);uaLX=-.19+.43*q-.19*up-.68*lean;laLX=-.24-.55*q+.52*up;uaLZ=.23+.18*q-.30*up;uaRZ=-.16-.13*gasp;
      // Load the supporting thigh before lifting, counter-rotate the shoulders on the two steps.
      const weight=sstep(10.10,11.42,u)*(1-sstep(12.89,14.10,u)),push=sstep(13.22,14.51,u)*(1-sstep(15.36,16.90,u)),sway=Math.sin(step*2*Math.PI);pelvisRoll+=.055*weight-.046*push+.032*sway;torsoRoll-=.085*weight-.055*push+.04*sway;pelvisX+=.043*weight+.026*sway;torsoYaw+=.058*sway;hLX-=.27*push;neckPitch-=.08*push;
      // Swing knees over the mattress, then solve both legs against planted ankles while rising.
      const L=-j.shin_offset[1],S=-j.foot_offset[1];function leg(isLeft){const stride=(isLeft?1:-1)*Math.sin(step*2*Math.PI)*.18,forward=.49-.46*up+stride,dy=pelvisY+j.thigh_l_offset[1]+j.foot_sole_y-.075*Math.max(0,(isLeft?1:-1)*Math.sin(step*2*Math.PI)),d=Math.min(L+S-.0001,Math.hypot(forward,dy)),knee=Math.acos(Math.max(-1,Math.min(1,(d*d-L*L-S*S)/(2*L*S)))),hip=-Math.atan2(forward,dy)-Math.atan2(S*Math.sin(knee),L+S*Math.cos(knee));return [lerp(-.1-1.34*q,hip,plant),lerp(.12+.15*q,knee,plant),-(hip+knee)*plant];}
      [lThighX,lShinX,lFootX]=leg(true);[rThighX,rShinX,rFootX]=leg(false);}
    // Forward kinematics for all 16 R16 bones (with lateral pelvic weight transfer & inward arm yaw!)
    const pxW = rootPos[0] + Math.cos(facingYaw) * pelvisX;
    const pzW = rootPos[2] - Math.sin(facingYaw) * pelvisX;
    const mPelvis = mat4Mul(
      mat4Translate(pxW, rootPos[1] + pelvisY, pzW),
      rotYXZ(facingYaw + pelvisYaw, pelvisPitch, pelvisRoll)
    );
    const mTorso = mat4Mul(
      mat4Mul(mPelvis, mat4Translate(j.torso_offset[0], j.torso_offset[1], j.torso_offset[2])),
      rotYXZ(torsoYaw, torsoPitch, torsoRoll)
    );

    // Female soft-body bust child bone with customizable bustSize & lively jiggle physics!
    const bustSize = Math.max(0.0, Math.min(1.0, (opts && opts.bustSize !== undefined) ? Number(opts.bustSize) : 0.5));
    let bpL = 0.0, byL = 0.0, bdL = 0.0, bpR = 0.0, byR = 0.0, bdR = 0.0;
    if (opts && opts.bustState) {
      [bpL, byL, bdL, bpR, byR, bdR] = opts.bustState;
    } else if (gender === "female") {
      const ph = (opts && opts.walkPhase !== undefined) ? opts.walkPhase : (elapsed * Math.max(horizSpeed, 0.0) * 1.8);
      const sf = (onGround && !sliding) ? Math.min(1.0, horizSpeed / 5.0) : 0.0;
      const szAmp = 0.60 + 0.85 * bustSize;
      const br = Math.sin(elapsed * 1.65) * 0.024 * szAmp;
      bpL = br + Math.sin(ph * 2.0 + 0.14) * 0.18 * sf * szAmp;
      bpR = br + Math.sin(ph * 2.0 - 0.14) * 0.18 * sf * szAmp;
      byL = Math.cos(ph + 0.18) * 0.11 * sf * szAmp;
      byR = Math.cos(ph - 0.18) * 0.11 * sf * szAmp;
      bdL = -bpL * 0.055;
      bdR = -bpR * 0.055;
    }
    const blo = j.bust_l_offset || [0.0, 0.296, 0.012];
    const bSx = 0.94 + bustSize * 0.14;
    const bSy = 0.86 + bustSize * 0.28;
    const bSz = 0.66 + bustSize * 0.74;
    const gravDrop = -0.004 * Math.max(0.0, bustSize - 0.40);
    const cPitch = 0.5 * (bpL + bpR) * 0.90;
    const cYaw   = 0.5 * (byL + byR) * 0.78;
    const cRoll  = (bpL - bpR) * 0.72;
    const cDrop  = 0.5 * (bdL + bdR) * 1.15 + gravDrop;
    const mBustL = mat4Mul(
      mat4Mul(mTorso, mat4Translate(blo[0], blo[1] + cDrop, blo[2])),
      mat4Mul(rotYXZ(cYaw, cPitch, cRoll), mat4Scale(bSx, bSy, bSz))
    );
    const mBustR = mBustL;

    const mNeck = mat4Mul(
      mat4Mul(mTorso, mat4Translate(j.neck_offset[0], j.neck_offset[1], j.neck_offset[2])),
      rotYXZ(neckYaw, neckPitch, neckRoll)
    );
    const mHead = mat4Mul(
      mat4Mul(mNeck, mat4Translate(j.head_offset[0], j.head_offset[1], j.head_offset[2])),
      rotYXZ(headYaw, headPitch, headRoll)
    );

    const mUpperArmL = mat4Mul(
      mat4Mul(mTorso, mat4Translate(j.upper_arm_l_offset[0], j.upper_arm_l_offset[1], j.upper_arm_l_offset[2])),
      rotZYX(uaLZ, uaLY, uaLX)
    );
    const mLowerArmL = mat4Mul(
      mat4Mul(mUpperArmL, mat4Translate(j.lower_arm_offset[0], j.lower_arm_offset[1], j.lower_arm_offset[2])),
      mat4RotX(laLX)
    );
    const mHandL = mat4Mul(
      mat4Mul(mLowerArmL, mat4Translate(j.hand_offset[0], j.hand_offset[1], j.hand_offset[2])),
      mat4RotX(hLX)
    );

    const mUpperArmR = mat4Mul(
      mat4Mul(mTorso, mat4Translate(j.upper_arm_r_offset[0], j.upper_arm_r_offset[1], j.upper_arm_r_offset[2])),
      rotZYX(uaRZ, uaRY, uaRX)
    );
    const mLowerArmR = mat4Mul(
      mat4Mul(mUpperArmR, mat4Translate(j.lower_arm_offset[0], j.lower_arm_offset[1], j.lower_arm_offset[2])),
      mat4RotX(laRX)
    );
    const mHandR = mat4Mul(
      mat4Mul(mLowerArmR, mat4Translate(j.hand_offset[0], j.hand_offset[1], j.hand_offset[2])),
      mat4RotX(hRX)
    );

    const mThighL = mat4Mul(
      mat4Mul(mPelvis, mat4Translate(j.thigh_l_offset[0], j.thigh_l_offset[1], j.thigh_l_offset[2])),
      mat4RotX(lThighX)
    );
    const mShinL = mat4Mul(
      mat4Mul(mThighL, mat4Translate(j.shin_offset[0], j.shin_offset[1], j.shin_offset[2])),
      mat4RotX(lShinX)
    );
    const mFootL = mat4Mul(
      mat4Mul(mShinL, mat4Translate(j.foot_offset[0], j.foot_offset[1], j.foot_offset[2])),
      mat4RotX(lFootX)
    );

    const mThighR = mat4Mul(
      mat4Mul(mPelvis, mat4Translate(j.thigh_r_offset[0], j.thigh_r_offset[1], j.thigh_r_offset[2])),
      mat4RotX(rThighX)
    );
    const mShinR = mat4Mul(
      mat4Mul(mThighR, mat4Translate(j.shin_offset[0], j.shin_offset[1], j.shin_offset[2])),
      mat4RotX(rShinX)
    );
    const mFootR = mat4Mul(
      mat4Mul(mShinR, mat4Translate(j.foot_offset[0], j.foot_offset[1], j.foot_offset[2])),
      mat4RotX(rFootX)
    );

    return {
      pelvis: mPelvis,
      torso: mTorso,
      bust_l: mBustL,
      bust_r: mBustR,
      neck: mNeck,
      head: mHead,
      upper_arm_l: mUpperArmL,
      lower_arm_l: mLowerArmL,
      hand_l: mHandL,
      upper_arm_r: mUpperArmR,
      lower_arm_r: mLowerArmR,
      hand_r: mHandR,
      thigh_l: mThighL,
      shin_l: mShinL,
      foot_l: mFootL,
      thigh_r: mThighR,
      shin_r: mShinR,
      foot_r: mFootR,
    };
  }

  function getCharacterSlotColors(catalog, ap) {
    const skin = catalog.skinPalette[ap.skin % catalog.skinPalette.length].rgb;
    const hair = catalog.hairPalette[ap.hair_color % catalog.hairPalette.length].rgb;
    const top = catalog.topPalette[ap.top_color % catalog.topPalette.length].rgb;
    const bottom = catalog.bottomPalette[ap.bottom_color % catalog.bottomPalette.length].rgb;
    const shoes = catalog.shoePalette[ap.shoe_color % catalog.shoePalette.length].rgb;
    const acc = catalog.accessoryPalette[ap.accessory_color % catalog.accessoryPalette.length].rgb;
    const topLum = 0.299 * top[0] + 0.587 * top[1] + 0.114 * top[2];
    const topAccent = topLum < 0.42 ? [0.90, 0.89, 0.85] : [0.16, 0.17, 0.19];
    const botLum = 0.299 * bottom[0] + 0.587 * bottom[1] + 0.114 * bottom[2];
    const bottomAccent = botLum < 0.42 ? [0.88, 0.88, 0.85] : [0.15, 0.16, 0.18];
    return {
      skin,
      hair,
      top,
      top_accent: topAccent,
      bottom,
      bottom_accent: bottomAccent,
      shoes,
      sole: (ap.shoe_style === 1 || ap.shoe_style === 4) ? [0.86, 0.85, 0.80] : [0.12, 0.12, 0.13],
      accessory: acc,
      glass_dark: [0.09, 0.10, 0.12],
      metal: [0.76, 0.72, 0.58],
      white: [1.0, 1.0, 1.0],
    };
  }

  // ---------- Main Boot ----------
  function mat4InvRigid(m) {
    const r00 = m[0], r01 = m[1], r02 = m[2], tx = m[3];
    const r10 = m[4], r11 = m[5], r12 = m[6], ty = m[7];
    const r20 = m[8], r21 = m[9], r22 = m[10], tz = m[11];
    return [
      r00, r10, r20, -(r00 * tx + r10 * ty + r20 * tz),
      r01, r11, r21, -(r01 * tx + r11 * ty + r21 * tz),
      r02, r12, r22, -(r02 * tx + r12 * ty + r22 * tz),
      0.0, 0.0, 0.0, 1.0,
    ];
  }

  function pivotRot(pivot, yaw, pitch, roll) {
    return mat4Mul(
      mat4Translate(pivot[0], pivot[1], pivot[2]),
      mat4Mul(rotYXZ(yaw, pitch, roll), mat4Translate(-pivot[0], -pivot[1], -pivot[2]))
    );
  }

  function wrapAngle(rad) {
    return Math.atan2(Math.sin(rad), Math.cos(rad));
  }

  function propHalfHeight(rot, hx, hy, hz) {
    const pitch = rot[0], roll = rot[2];
    const sp = Math.sin(pitch), cp = Math.cos(pitch);
    const sr = Math.sin(roll), cr = Math.cos(roll);
    return Math.abs(cp * sr) * hx + Math.abs(cp * cr) * hy + Math.abs(sp) * hz;
  }

  function safePropSpawn(origin, target, colliders, radius, minY) {
    const dx = target[0] - origin[0];
    const dy = target[1] - origin[1];
    const dz = target[2] - origin[2];
    const dist = Math.hypot(dx, dy, dz);
    let tx = target[0], ty = target[1], tz = target[2];
    if (dist > 1e-5 && colliders && colliders.length) {
      const dir = [dx / dist, dy / dist, dz / dist];
      let bestT = dist;
      const margin = radius + 0.04;
      for (let i = 0; i < colliders.length; i++) {
        const hit = raycastAABB(origin, dir, colliders[i], dist + margin);
        if (hit !== null && hit - margin < bestT) {
          bestT = Math.max(0.05, hit - margin);
        }
      }
      tx = origin[0] + dir[0] * bestT;
      ty = origin[1] + dir[1] * bestT;
      tz = origin[2] + dir[2] * bestT;
    }
    tx = Math.max(-15.75, Math.min(15.75, tx));
    tz = Math.max(-15.75, Math.min(15.75, tz));
    ty = Math.max(minY, ty);
    return [tx, ty, tz];
  }

  function stepWorldProp(
    prop, dt, colliders, halfH, radius,
    halfExtents = [0.028, 0.095, 0.145], bounce = 0.28, friction = 0.56
  ) {
    if (prop.grounded) return false;
    const clampedDt = Math.min(Math.max(dt, 0.0), 0.05);
    if (clampedDt <= 1e-6) return false;

    const pos = prop.pos;
    const vel = prop.vel;
    const rot = prop.rot;
    const avel = prop.avel;
    const hx = halfExtents[0], hy = halfExtents[1], hz = halfExtents[2];

    const steps = Math.max(1, Math.min(8, Math.ceil(clampedDt / 0.008)));
    const subDt = clampedDt / steps;
    let impact = false;
    const cols = colliders || [];

    for (let s = 0; s < steps; s++) {
      if (prop.grounded) break;

      const curH = Math.max(halfH, propHalfHeight(rot, hx, hy, hz));
      const px = pos[0];
      const py = Math.max(pos[1], curH);
      const pz = pos[2];

      // 1. Horizontal Axis X (only walls/crates overlapping our vertical body, NEVER floor!)
      let nx = px + vel[0] * subDt;
      const yLo = py - curH + 0.025;
      const yHi = py + curH - 0.015;
      for (let i = 0; i < cols.length; i++) {
        const c = cols[i];
        if (c[4] <= yLo || c[1] >= yHi) continue;
        if (pz + radius > c[2] && pz - radius < c[5]) {
          if (nx + radius > c[0] && nx - radius < c[3]) {
            if (vel[0] > 0.0) nx = c[0] - radius - 0.002;
            else if (vel[0] < 0.0) nx = c[3] + radius + 0.002;
            else {
              const d0 = Math.abs((nx + radius) - c[0]);
              const d1 = Math.abs(c[3] - (nx - radius));
              nx = (d0 < d1) ? (c[0] - radius - 0.002) : (c[3] + radius + 0.002);
            }
            if (Math.abs(vel[0]) > 0.65) impact = true;
            vel[0] = -vel[0] * 0.38;
            vel[2] *= 0.85;
          }
        }
      }

      // 2. Horizontal Axis Z
      let nz = pz + vel[2] * subDt;
      for (let i = 0; i < cols.length; i++) {
        const c = cols[i];
        if (c[4] <= yLo || c[1] >= yHi) continue;
        if (nx + radius > c[0] && nx - radius < c[3]) {
          if (nz + radius > c[2] && nz - radius < c[5]) {
            if (vel[2] > 0.0) nz = c[2] - radius - 0.002;
            else if (vel[2] < 0.0) nz = c[5] + radius + 0.002;
            else {
              const d0 = Math.abs((nz + radius) - c[2]);
              const d1 = Math.abs(c[5] - (nz - radius));
              nz = (d0 < d1) ? (c[2] - radius - 0.002) : (c[5] + radius + 0.002);
            }
            if (Math.abs(vel[2]) > 0.65) impact = true;
            vel[2] = -vel[2] * 0.38;
            vel[0] *= 0.85;
          }
        }
      }

      // Outer arena perimeter safety clamp
      if (nx < -15.80) { nx = -15.80; vel[0] = Math.abs(vel[0]) * 0.38; }
      else if (nx > 15.80) { nx = 15.80; vel[0] = -Math.abs(vel[0]) * 0.38; }
      if (nz < -15.80) { nz = -15.80; vel[2] = Math.abs(vel[2]) * 0.38; }
      else if (nz > 15.80) { nz = 15.80; vel[2] = -Math.abs(vel[2]) * 0.38; }

      // 3. Vertical Axis Y (gravity, ceiling bounce, solid floor/table/podium landing & flat settling)
      vel[1] -= 12.5 * subDt;
      let ny = py + vel[1] * subDt;

      if (vel[1] > 0.0) {
        for (let i = 0; i < cols.length; i++) {
          const c = cols[i];
          if (nx + radius * 0.65 > c[0] && nx - radius * 0.65 < c[3] &&
              nz + radius * 0.65 > c[2] && nz - radius * 0.65 < c[5]) {
            if (py + curH <= c[1] + 0.06 && ny + curH >= c[1]) {
              ny = c[1] - curH - 0.002;
              vel[1] = -vel[1] * 0.25;
            }
          }
        }
      }

      let supportY = 0.0;
      for (let i = 0; i < cols.length; i++) {
        const c = cols[i];
        if (nx + radius * 0.65 > c[0] && nx - radius * 0.65 < c[3] &&
            nz + radius * 0.65 > c[2] && nz - radius * 0.65 < c[5]) {
          if (c[4] > supportY && c[4] <= (py - curH + 0.08)) {
            supportY = c[4];
          }
        }
      }

      const targetY = supportY + curH;
      if (ny <= targetY) {
        ny = targetY;
        if (vel[1] < -0.75) {
          impact = true;
          vel[1] = -vel[1] * bounce;
          vel[0] *= friction;
          vel[2] *= friction;
          avel[0] *= 0.45;
          avel[1] *= 0.55;
          avel[2] *= 0.45;
        } else {
          vel[1] = 0.0;
          const damp = Math.exp(-subDt * 11.0);
          vel[0] *= damp;
          vel[2] *= damp;
          const kFlat = 1.0 - Math.exp(-subDt * 16.0);
          const dp = wrapAngle(0.0 - rot[0]);
          const dr = wrapAngle(Math.PI * 0.5 - rot[2]);
          rot[0] = wrapAngle(rot[0] + dp * kFlat);
          rot[2] = wrapAngle(rot[2] + dr * kFlat);
          avel[0] *= 0.3;
          avel[1] *= damp;
          avel[2] *= 0.3;
          rot[1] += avel[1] * subDt;
          const newH = Math.max(halfH, propHalfHeight(rot, hx, hy, hz));
          ny = supportY + newH;
          if (Math.hypot(vel[0], vel[2]) < 0.045 && Math.abs(dp) < 0.035 && Math.abs(dr) < 0.035) {
            vel[0] = vel[1] = vel[2] = 0.0;
            avel[0] = avel[1] = avel[2] = 0.0;
            rot[0] = 0.0;
            rot[2] = Math.PI * 0.5;
            ny = supportY + halfH;
            prop.grounded = true;
            if (!prop.hasBounced) {
              impact = true;
              prop.hasBounced = true;
            }
          }
        }
        if (impact) prop.hasBounced = true;
      } else {
        rot[0] = wrapAngle(rot[0] + avel[0] * subDt);
        rot[1] += avel[1] * subDt;
        rot[2] = wrapAngle(rot[2] + avel[2] * subDt);
        const rotH = Math.max(halfH, propHalfHeight(rot, hx, hy, hz));
        if (ny < supportY + rotH) ny = supportY + rotH;
      }

      pos[0] = nx;
      pos[1] = Math.max(ny, halfH);
      pos[2] = nz;
    }

    return impact;
  }

  // ---------------------------------------------------------------------------
  // Weapon controller shared by all available firearms.
  // 1:1 parity with engine/weapon.py!
  // ---------------------------------------------------------------------------
  class WeaponController {
    constructor(consts, startOwned = false) {
      this.MAG_AXIS = consts.magAxis || [0.0, -0.9516, 0.3074];
      this.MAG_MESH_CENTER = consts.magMeshCenter || [0.0, -0.009, 0.008];
      this.MUZZLE_LOCAL = consts.muzzleLocal || [0.0, 0.0642, -0.2005];
      this.EJECT_LOCAL = consts.ejectLocal || [0.015, 0.070, -0.070];
      this.SLIDE_TRAVEL = consts.slideTravel || 0.046;
      this.MAG_CAPACITY = consts.magCapacity || 7;
      this.START_RESERVE = consts.startReserve || 14;
      this.FIRE_COOLDOWN = consts.fireCooldown || 0.165;
      this.RELOAD_EMPTY_DUR = consts.reloadEmptyDur || 2.35;
      this.RELOAD_TACTICAL_DUR = consts.reloadTacticalDur || 1.85;
      this.DRAW_DUR = consts.drawDur || 0.34;
      this.WEAPON_ORDER = consts.weaponOrder || ["m1911", "glock17", "remington11", "double_barrel", "revolver", "ak47", "m4a1", "mp7", "m110", "mp5s"];
      this.DB_HINGE_PIVOT = consts.dbHingePivot || [0.0, 0.044, -0.174];
      this.DB_LEVER_PIVOT = consts.dbLeverPivot || [0.0, 0.088, -0.082];
      this.REV_CYL_PIVOT = consts.revCylPivot || [0.0, 0.0565, -0.072];
      this.REV_CRANE_PIVOT = consts.revCranePivot || [-0.0175, 0.0365, -0.072];
      this.REV_HAMMER_PIVOT = consts.revHammerPivot || [0.0, 0.049, 0.004];
      this.REV_CHAMBER_RING_R = (typeof consts.revChamberRingR === "number") ? consts.revChamberRingR : 0.0138;
      this.REV_CHAMBER_REAR_Z = (typeof consts.revChamberRearZ === "number") ? consts.revChamberRearZ : -0.0402;
      this.AK_MAG_AXIS = consts.akMagAxis || [0.0, -0.9350, 0.3546];
      this.AK_MAG_PIVOT = consts.akMagPivot || [0.0, 0.024, -0.206];
      this.M4_MAG_AXIS = consts.m4MagAxis || [0.0, -0.9850, 0.1725];
      this.M4_MAG_CENTER = consts.m4MagCenter || [0.0, -0.036, -0.155];
      this.MP7_MAG_AXIS = consts.mp7MagAxis || [0.0, -0.9659, 0.2588];
      this.MP7_MAG_CENTER = consts.mp7MagCenter || [0.0, -0.075, 0.024];
      this.M110_MAG_AXIS = consts.m110MagAxis || [0.0, -0.996, 0.087];
      this.M110_MAG_CENTER = consts.m110MagCenter || [0.0, 0.036, -0.150];
      this.MP5S_MAG_AXIS = consts.mp5sMagAxis || [0.0, -0.974, -0.225];
      this.MP5S_MAG_CENTER = consts.mp5sMagCenter || [0.0, 0.032, -0.136];
      this.FP_FINGER_NAMES = consts.fpFingerNames || ["thumb", "index", "middle", "ring", "pinky"];
      this.FP_HAND_MCP_R = consts.fpHandMcpR || {
        thumb:  [-0.0255, -0.0038, -0.0205],
        index:  [-0.0218,  0.0022, -0.0705],
        middle: [-0.0068,  0.0036, -0.0748],
        ring:   [ 0.0082,  0.0020, -0.0715],
        pinky:  [ 0.0222, -0.0012, -0.0645],
      };
      this.FP_HAND_MCP_L = consts.fpHandMcpL || {
        thumb:  [ 0.0255, -0.0038, -0.0205],
        index:  [ 0.0218,  0.0022, -0.0705],
        middle: [ 0.0068,  0.0036, -0.0748],
        ring:   [-0.0082,  0.0020, -0.0715],
        pinky:  [-0.0222, -0.0012, -0.0645],
      };
      this.FP_HAND_PIP_LEN = consts.fpHandPipLen || {
        thumb: 0.0275, index: 0.0345, middle: 0.0385, ring: 0.0355, pinky: 0.0285,
      };
      this.SPECS = consts.specs || {
        m1911: {
          id: "m1911", name: "COLT M1911", caliber: ".45 ACP", weapon_type: "pistol",
          mag_capacity: 7, start_reserve: 14, fire_cooldown: 0.165, pump_duration: 0.0, full_auto: false,
          pellets: 1, spread_hip: 0.0, spread_ads: 0.0, slide_travel: 0.046,
          muzzle_local: [0.0, 0.0642, -0.2005], eject_local: [0.015, 0.070, -0.070], flash_scale: 1.0,
          recoil_pitch: 0.22, recoil_z: 0.038, recoil_y: 0.014,
          cam_pitch: 0.034, cam_roll: 0.012, sfx_fire: "fire", shell_reload: false, slot: 1,
        },
        glock17: {
          id: "glock17", name: "GLOCK-17", caliber: "9MM AUTO", weapon_type: "pistol",
          mag_capacity: 17, start_reserve: 34, fire_cooldown: 0.088, pump_duration: 0.0, full_auto: true,
          pellets: 1, spread_hip: 0.0, spread_ads: 0.0, slide_travel: 0.042,
          muzzle_local: [0.0, 0.0642, -0.2005], eject_local: [0.015, 0.070, -0.070], flash_scale: 0.92,
          recoil_pitch: 0.145, recoil_z: 0.026, recoil_y: 0.010,
          cam_pitch: 0.022, cam_roll: 0.014, sfx_fire: "fire_glock", shell_reload: false, slot: 2,
        },
        remington11: {
          id: "remington11", name: "REMINGTON 11", caliber: "12 GAUGE", weapon_type: "shotgun",
          mag_capacity: 5, start_reserve: 20, fire_cooldown: 0.84, pump_duration: 0.76, full_auto: false,
          pellets: 8, spread_hip: 0.052, spread_ads: 0.030, slide_travel: 0.072,
          muzzle_local: [0.0, 0.0805, -0.710], eject_local: [0.024, 0.072, -0.118], flash_scale: 1.65,
          recoil_pitch: 0.34, recoil_z: 0.066, recoil_y: 0.026,
          cam_pitch: 0.068, cam_roll: 0.022, sfx_fire: "fire_shotgun", shell_reload: true, slot: 3,
        },
        double_barrel: {
          id: "double_barrel", name: "ДВУСТВОЛКА", caliber: "12 GAUGE", weapon_type: "double_barrel",
          mag_capacity: 2, start_reserve: 4, fire_cooldown: 0.28, pump_duration: 0.0, full_auto: false,
          pellets: 10, spread_hip: 0.064, spread_ads: 0.036, slide_travel: 0.0,
          muzzle_local: [0.0, 0.0725, -0.652], eject_local: [0.0, 0.0725, -0.130], flash_scale: 1.85,
          recoil_pitch: 0.42, recoil_z: 0.076, recoil_y: 0.030,
          cam_pitch: 0.082, cam_roll: 0.026, sfx_fire: "fire_double_barrel",
          shell_reload: false, break_reload: true, slot: 4,
        },
        revolver: {
          id: "revolver", name: "COLT PYTHON .357", caliber: ".357 MAG", weapon_type: "revolver",
          mag_capacity: 8, start_reserve: 16, fire_cooldown: 0.14, pump_duration: 0.0, full_auto: false,
          cock_duration: 0.24, pellets: 1, spread_hip: 0.0, spread_ads: 0.0, slide_travel: 0.0,
          muzzle_local: [0.0, 0.0705, -0.322], eject_local: [-0.030, 0.056, -0.072], flash_scale: 1.54,
          recoil_pitch: 0.34, recoil_z: 0.052, recoil_y: 0.022,
          cam_pitch: 0.056, cam_roll: 0.018, sfx_fire: "fire_revolver",
          shell_reload: false, revolver_reload: true, slot: 5,
        },
        ak47: {
          id: "ak47", name: "АК-47", caliber: "7.62×39", weapon_type: "rifle",
          mag_capacity: 30, start_reserve: 60, fire_cooldown: 0.100, pump_duration: 0.0, full_auto: true,
          pellets: 1, spread_hip: 0.010, spread_ads: 0.002, slide_travel: 0.084,
          muzzle_local: [0.0, 0.0765, -0.668], eject_local: [0.022, 0.078, -0.140], flash_scale: 1.38,
          recoil_pitch: 0.175, recoil_z: 0.036, recoil_y: 0.014,
          cam_pitch: 0.028, cam_roll: 0.015, sfx_fire: "fire_ak47",
          shell_reload: false, slot: 6,
        },
        m4a1: {
          id: "m4a1", name: "M4A1", caliber: "5.56×45", weapon_type: "rifle",
          mag_capacity: 30, start_reserve: 60, fire_cooldown: 0.080, pump_duration: 0.0, full_auto: true,
          pellets: 1, spread_hip: 0.007, spread_ads: 0.001, slide_travel: 0.058,
          muzzle_local: [0.0, 0.0825, -0.630], eject_local: [0.021, 0.084, -0.130], flash_scale: 1.18,
          recoil_pitch: 0.135, recoil_z: 0.028, recoil_y: 0.010,
          cam_pitch: 0.020, cam_roll: 0.011, sfx_fire: "fire_m4a1",
          shell_reload: false, slot: 7,
        },
        mp7: {
          id: "mp7", name: "HK MP7A1", caliber: "4.6×30MM", weapon_type: "smg",
          mag_capacity: 30, start_reserve: 60, fire_cooldown: 0.068, pump_duration: 0.0, full_auto: true,
          pellets: 1, spread_hip: 0.0075, spread_ads: 0.0012, slide_travel: 0.044,
          muzzle_local: [0.0, 0.068, -0.308], eject_local: [0.020, 0.074, -0.068], flash_scale: 1.05,
          recoil_pitch: 0.092, recoil_z: 0.020, recoil_y: 0.009,
          cam_pitch: 0.017, cam_roll: 0.010, sfx_fire: "fire_mp7",
          shell_reload: false, slot: 8,
        },
      };
      this.FINGER_PIVOTS_R = consts.fingerPivotsR || {
        thumb:  [0.018,  0.014,  0.044],
        index:  [0.022,  0.020, -0.006],
        middle: [0.022, -0.002, -0.006],
        ring:   [0.022, -0.022,  0.002],
        pinky:  [0.022, -0.041,  0.008],
      };
      this.FINGER_PIVOTS_L = consts.fingerPivotsL || {
        thumb:  [-0.012,  0.018,  0.026],
        index:  [-0.014,  0.016, -0.012],
        middle: [-0.014, -0.003, -0.010],
        ring:   [-0.014, -0.022, -0.006],
        pinky:  [-0.014, -0.040, -0.001],
      };
      this.startOwned = startOwned;
      this.SPRAY_PATTERNS = consts.sprayPatterns || {};
      this.BALLISTICS = consts.ballistics || {};
      this.lastSpawnIndices = [];
      this.worldPickups = [];
      this.worldMags = [];
      this.weapons = {};
      this.activeWeapon = "m1911";
      this.reset(startOwned);
    }

    get activeSpec() {
      return this.SPECS[this.activeWeapon] || this.SPECS.m1911;
    }

    get isFullAuto() {
      return Boolean(this.activeSpec && this.activeSpec.full_auto);
    }

    get isShotgun() {
      return Boolean(this.activeSpec && this.activeSpec.weapon_type === "shotgun");
    }

    get isDoubleBarrel() {
      return Boolean(this.activeSpec && this.activeSpec.weapon_type === "double_barrel");
    }

    get isRevolver() {
      return Boolean(this.activeSpec && this.activeSpec.weapon_type === "revolver");
    }

    get isRifle() {
      return Boolean(this.activeSpec && this.activeSpec.weapon_type === "rifle");
    }

    get isAK47() {
      return this.activeWeapon === "ak47";
    }

    get isM4A1() {
      return this.activeWeapon === "m4a1";
    }

    get isMP7() {
      return this.activeWeapon === "mp7";
    }

    get isM110() {
      return this.activeWeapon === "m110";
    }

    get isMP5S() {
      return this.activeWeapon === "mp5s";
    }

    get isSniper() {
      return Boolean(this.activeSpec && this.activeSpec.weapon_type === "sniper");
    }

    get isLongGun() {
      return this.isShotgun || this.isDoubleBarrel || this.isRifle || this.isSniper;
    }

    startMagCheck() {
      if (!this.hasWeapon || !this.equipped || this.reloading || this.pumpActive || this.revCocking || this.suppAnimActive || this.equipT < 0.72) {
        return [];
      }
      if (this.magChecking) return [];
      this.cancelInspect();
      this.magChecking = true;
      this.magCheckEmpty = (this.magAmmo <= 0);
      this.magCheckTimer = 0.0;
      this.mcRevSpin = 0.0;
      const cap = Math.max(1, this.magCapacity || this.MAG_CAPACITY || 1);
      const frac = this.magAmmo / cap;
      if (this.isDoubleBarrel) {
        this.magCheckMode = "double_barrel";
        this.magCheckDuration = 2.25;
        if (this.magAmmo <= 0) this.magCheckLabel = `КАЗЁННИК ПУСТ (0/${cap})`;
        else if (this.magAmmo >= 2) this.magCheckLabel = `КАЗЁННИК: ОБА СТВОЛА ЗАРЯЖЕНЫ (${this.magAmmo}/${cap})`;
        else this.magCheckLabel = `КАЗЁННИК: ОДИН ПАТРОН (${this.magAmmo}/${cap})`;
      } else if (this.isRevolver) {
        this.magCheckMode = "revolver";
        this.magCheckDuration = 2.55;
        if (this.magAmmo <= 0) this.magCheckLabel = `БАРАБАН ПУСТ (0/${cap})`;
        else if (frac >= 0.99) this.magCheckLabel = `БАРАБАН ПОЛНЫЙ (${this.magAmmo}/${cap})`;
        else this.magCheckLabel = `БАРАБАН: ${this.magAmmo}/${cap} ПАТР.`;
      } else if (this.isShotgun) {
        this.magCheckMode = "shotgun";
        this.magCheckDuration = this.magCheckEmpty ? 2.35 : 2.15;
        if (this.magAmmo <= 0) this.magCheckLabel = `ПОДСТВОЛЬНИК ПУСТ (0/${cap}) — ЗАТВОР`;
        else this.magCheckLabel = `ПАТРОННИК / ТРУБКА (${this.magAmmo}/${cap})`;
      } else if (this.magCheckEmpty) {
        this.magCheckMode = "mag_empty";
        this.magCheckDuration = 3.15;
        this.magCheckLabel = `МАГАЗИН ПУСТ (0/${cap}) — ЗАТВОР`;
      } else {
        this.magCheckMode = "mag_loaded";
        this.magCheckDuration = 2.55;
        if (frac >= 0.99) this.magCheckLabel = `МАГАЗИН ПОЛНЫЙ (${this.magAmmo}/${cap})`;
        else if (frac >= 0.65) this.magCheckLabel = `МАГАЗИН: ПОЧТИ ПОЛНЫЙ (${this.magAmmo}/${cap})`;
        else if (frac >= 0.35) this.magCheckLabel = `МАГАЗИН: ОКОЛО ПОЛОВИНЫ (${this.magAmmo}/${cap})`;
        else this.magCheckLabel = `МАГАЗИН: ПОЧТИ ПУСТ (${this.magAmmo}/${cap})`;
      }
      this.mcEv1 = this.mcEv2 = this.mcEv3 = this.mcEv4 = this.mcEv5 = this.mcEv6 = this.mcEv7 = false;
      this.magCheckLabelTimer = this.magCheckDuration / 0.82 + 0.55;
      return ["inspect"];
    }

    cancelMagCheck() {
      this.magChecking = false;
    }

    toggleSuppressor() {
      if (!this.hasWeapon || !this.equipped || !this.isMP5S || this.reloading || this.pumpActive || this.suppAnimActive || this.equipT < 0.72) {
        return [];
      }
      this.cancelInspect();
      this.cancelMagCheck();
      this.suppAnimActive = true;
      this.suppAnimTimer = 0.0;
      this.suppAnimDuration = 1.05;
      this.suppTargetState = !this.mp5Suppressed;
      this.suppEvSfx = false;
      this.magCheckLabel = this.suppTargetState ? "ГЛУШИТЕЛЬ MP5-S: НАКРУЧИВАНИЕ..." : "ГЛУШИТЕЛЬ MP5-S: СНЯТИЕ...";
      this.magCheckLabelTimer = 1.65;
      return [];
    }

    startInspect() {
      if (!this.hasWeapon || !this.equipped || this.reloading || this.pumpActive || this.revCocking || this.equipT < 0.72) {
        return [];
      }
      if (this.inspecting && this.inspectTimer > 0.35) {
        this.inspectTimer = 0.0;
        this.evInsTurn = false;
        this.evInsEnd = false;
        return ["inspect"];
      }
      this.inspecting = true;
      this.inspectTimer = 0.0;
      this.inspectDuration = this.isLongGun ? 4.35 : 3.95;
      this.evInsTurn = false;
      this.evInsEnd = false;
      return ["inspect"];
    }

    cancelInspect() {
      this.inspecting = false;
      this.inspectHold = false;
    }

    consumeCamKick() {
      const p = this.pendingCamPitch;
      const r = this.pendingCamRoll;
      this.pendingCamPitch = 0.0;
      this.pendingCamRoll = 0.0;
      return [p, r];
    }

    _clearActionStates() {
      this.reloading = false;
      this.reloadTimer = 0.0;
      this.shellReloadPhase = "";
      this.reloadInterrupt = false;
      this.chamberThenLoad = false;
      this.pumpActive = false;
      this.pumpTimer = 0.0;
      this.dbReloadPhase = "";
      this.dbReloadTarget = 0;
      this.dbInsertedCount = 0;
      this.dbEjectCount = 0;
      this.dbEvOpen = false;
      this.dbEvEject = false;
      this.dbEvClose = false;
      this.revCocking = false;
      this.revCockTimer = 0.0;
      this.revReloadPhase = "";
      this.revLoadChamber = -1;
      this.sprayIndex = 0.0;
      this.sprayPitch = 0.0;
      this.sprayYaw = 0.0;
      this.fireBloom = 0.0;
      this.camRecoilPitch = 0.0;
      this.camRecoilYaw = 0.0;
      this.revEvOpen = false;
      this.revEvDump = false;
      this.revEvLoad = false;
      this.revEvRoundIn = false;
      this.revEvClose = false;
      this.inspecting = false;
      this.inspectTimer = 0.0;
      this.inspectDuration = 4.10;
      this.inspectHold = false;
      this.inspectBlend = 0.0;
      this.adsT = 0.0;
    }

    get worldPickup() {
      return this.worldPickups.length ? this.worldPickups[0] : null;
    }

    ownedWeapons() {
      const owned = this.WEAPON_ORDER.filter((wid) => this.weapons[wid] && this.weapons[wid].owned);
      for (const wid of owned.slice(2)) this.weapons[wid].owned = false; // clamp old saves to primary + sidearm
      const slots = owned.slice(0, 2);
      if (slots.length && !slots.includes(this.activeWeapon)) {
        this.activeWeapon = slots[0];
        this._syncFromActive();
      }
      return slots;
    }

    _saveToActive() {
      const st = this.weapons[this.activeWeapon];
      if (!st) return;
      st.magAmmo = this.magAmmo;
      st.reserveAmmo = this.reserveAmmo;
      st.slideLocked = this.slideLocked;
      st.spentShells = this.spentShells;
      if (this.activeWeapon === "revolver") {
        st.revChambers = this.revChambers.slice();
      }
    }

    _applyMagReload(take) {
      take = Math.max(0, Math.min(Math.floor(take), this.reserveAmmo));
      this.magAmmo += take;
      this.reserveAmmo -= take;
      this._saveToActive();
    }

    _syncFromActive() {
      const spec = this.activeSpec;
      const st = this.weapons[this.activeWeapon];
      this.MAG_CAPACITY = spec.mag_capacity;
      this.START_RESERVE = spec.start_reserve;
      this.SLIDE_TRAVEL = spec.slide_travel;
      this.FIRE_COOLDOWN = spec.fire_cooldown;
      if (st) {
        this.hasWeapon = Boolean(st.owned);
        this.magAmmo = st.magAmmo;
        this.reserveAmmo = st.reserveAmmo;
        this.slideLocked = Boolean(st.slideLocked);
        this.spentShells = st.spentShells || 0;
        if (this.activeWeapon === "revolver") {
          if (Array.isArray(st.revChambers) && st.revChambers.length === 8) {
            this.revChambers = st.revChambers.slice();
          } else {
            const liveN = Math.max(0, Math.min(8, this.magAmmo));
            const spN = Math.max(0, Math.min(8 - liveN, this.spentShells));
            this.revChambers = [];
            for (let i = 0; i < 8; i++) {
              this.revChambers.push(i < liveN ? "live" : (i < liveN + spN ? "spent" : "empty"));
            }
          }
        }
      } else {
        this.hasWeapon = false;
        this.magAmmo = spec.mag_capacity;
        this.reserveAmmo = spec.start_reserve;
        this.slideLocked = false;
        this.spentShells = 0;
      }
    }

    reset(owned = null, clearWorldMags = true) {
      const hasW = (owned === null) ? this.startOwned : Boolean(owned);
      this.weapons = {};
      for (const wid of this.WEAPON_ORDER) {
        const sp = this.SPECS[wid];
        this.weapons[wid] = {
          owned: hasW,
          magAmmo: sp.mag_capacity,
          reserveAmmo: sp.start_reserve,
          slideLocked: false,
          spentShells: 0,
          revChambers: wid === "revolver" ? new Array(8).fill("live") : null,
        };
      }
      this.revChambers = new Array(8).fill("live");
      this.activeWeapon = "m1911";
      this._syncFromActive();
      this.equipped = hasW;
      this.equipT = 0.0;
      this.adsT = 0.0;
      this.fireTimer = 0.0;
      this.pickupCd = 0.0;
      this.triggerPull = 0.0;
      this.hammerDrop = 0.0;
      this.slideKick = 0.0;
      this.muzzleTimer = 0.0;
      this.muzzleRoll = 0.0;
      this.flashLightTimer = 0.0;
      this.flashWorldPos = [0, 0, 0];

      this.recoilPitch = 0.0;
      this.recoilRoll = 0.0;
      this.recoilZ = 0.0;
      this.recoilY = 0.0;

      this.reloading = false;
      this.reloadType = "empty";
      this.reloadTimer = 0.0;
      this.reloadDuration = this.RELOAD_EMPTY_DUR;
      this.relEvOut = false;
      this.relEvIn = false;
      this.magChecking = false;
      this.magCheckEmpty = false;
      this.magCheckMode = "mag_loaded";
      this.magCheckTimer = 0.0;
      this.magCheckDuration = 2.55;
      this.magCheckBlend = 0.0;
      this.magCheckLabel = "";
      this.magCheckLabelTimer = 0.0;
      this.mcEv1 = this.mcEv2 = this.mcEv3 = this.mcEv4 = this.mcEv5 = this.mcEv6 = this.mcEv7 = false;
      this.mcRevSpin = 0.0;
      this.bodycamCamRoll = 0.0;
      this.bodycamCamPitch = 0.0;
      this.bodycamCamYaw = 0.0;
      // v14: Non-accumulating spring camera jolts for snappy reloads, bolt racks, breech snaps & mag slaps
      this.camJoltPitch = 0.0;
      this.camJoltYaw = 0.0;
      this.camJoltRoll = 0.0;
      this.camJoltPitchV = 0.0;
      this.camJoltYawV = 0.0;
      this.camJoltRollV = 0.0;
      // v14: Rich 3D tracers, particles, screen-space impact & speed-lines FX
      this.tracers = [];
      this.particles = [];
      this.fxImpactPulse = 0.0;
      this.fxSpeedLines = 0.0;
      this._slideFxAccum = 0.0;
      this._boltFxAccum = 0.0;
      this.scopeAmount = 0.0;
      this.scopeBlur = 0.0;
      this.mp5Suppressed = true;
      this.suppAnimActive = false;
      this.suppAnimTimer = 0.0;
      this.suppAnimDuration = 1.05;
      this.suppTargetState = true;
      this.suppEvSfx = false;
      this.relEvBack = false;
      this.relEvFwd = false;
      this.relCommitted = false;

      // Shotgun bolt-racking / pump cooldown & shell-by-shell reload state
      this.pumpActive = false;
      this.pumpTimer = 0.0;
      this.pumpDuration = 0.76;
      this.pumpEvBack = false;
      this.pumpEvEject = false;
      this.pumpEvFwd = false;
      this.shellReloadPhase = "";
      this.reloadInterrupt = false;
      this.reloadWasEmpty = false;
      this.evShellIn = false;
      this.chamberThenLoad = false;

      // Double-barrel break-action reload & alternating muzzle state
      this.dbLastBarrel = 1;
      this.dbReloadPhase = "";
      this.dbReloadTarget = 0;
      this.dbInsertedCount = 0;
      this.dbEjectCount = 0;
      this.dbEvOpen = false;
      this.dbEvEject = false;
      this.dbEvClose = false;

      // Revolver pre-shot hammer cocking, cylinder indexing & 3-phase swing-out reload state
      this.revCocking = false;
      this.revCockTimer = 0.0;
      this.revCockDuration = 0.24;
      this.revCylAngle = 0.0;
      this.revCylTarget = 0.0;
      this.revReloadPhase = "";
      this.revLoadChamber = -1;
      this.revEvOpen = false;
      this.revEvDump = false;
      this.revEvLoad = false;
      this.revEvRoundIn = false;
      this.revEvClose = false;
      this.pendingCamPitch = 0.0;
      this.pendingCamRoll = 0.0;

      this.swayX = 0.0;
      this.swayY = 0.0;
      this.bobPhase = 0.0;
      this.bobAmp = 0.0;
      this.sprintBlend = 0.0;
      this.elapsed = 0.0;

      this.shells = [];
      this.sparks = [];
      this.bulletHoles = [];
      this.sfxQueue = [];
      if (clearWorldMags) this.worldMags = [];

      // CS2-like dynamic spray pattern, inaccuracy bloom & camera recoil recovery state
      this.sprayIndex = 0.0;
      this.sprayPitch = 0.0;
      this.sprayYaw = 0.0;
      this.fireBloom = 0.0;
      this.timeSinceShot = 10.0;
      this.camRecoilPitch = 0.0;
      this.camRecoilYaw = 0.0;
      this.landingPenalty = 0.0;
      this.wasOnGround = true;
      this.lastHorizSpeed = 0.0;
      this.lastVelY = 0.0;
      this.lastOnGround = true;
      this.lastCrouching = false;
      this.lastSliding = false;
      this.currentSpread = 0.0028;
      this.crosshairSpreadPx = 2.0;
      this.lastShotDir = [0, 0, -1];
    }

    _getBallistics() {
      return this.BALLISTICS[this.activeWeapon] || this.BALLISTICS.m1911 || {
        inacc_stand: 0.0024, inacc_crouch: 0.0012, inacc_move: 0.032, inacc_jump: 0.098,
        inacc_fire: 0.012, max_bloom: 0.058, recovery_tau: 0.26, crouch_recoil: 0.78,
        cam_scale: 0.80, bullet_scale: 1.05,
      };
    }

    computeCurrentSpread() {
      const b = this._getBallistics();
      const isSg = (this.activeSpec.weapon_type === "shotgun" || this.activeSpec.weapon_type === "double_barrel");
      const isSn = (this.activeSpec.weapon_type === "sniper");
      let base = this.lastCrouching ? b.inacc_crouch : b.inacc_stand;
      if (isSn) {
        // M110 SASS: surgical pinpoint accuracy in scope when still (0.16x), wild hipfire when unscoped!
        base = (base * (1.0 - 0.84 * this.adsT)) + (1.0 - this.adsT) * 0.048;
      } else {
        base *= (1.0 - (isSg ? 0.38 : 0.72) * this.adsT);
      }

      const spd = Math.max(0.0, this.lastHorizSpeed);
      let movePen = 0.0;
      if (this.lastOnGround && spd > 0.30) {
        if (this.lastSliding) {
          movePen = b.inacc_move * 1.65;
        } else if (this.lastCrouching) {
          const mf = Math.min(1.0, (spd - 0.25) / 2.2);
          movePen = b.inacc_move * (isSn ? 0.68 : 0.28) * (mf * mf);
        } else {
          const mf = Math.max(0.0, (spd - 0.30) / (isSn ? 3.2 : 4.2));
          movePen = b.inacc_move * (Math.pow(Math.min(1.0, mf), isSn ? 1.15 : 1.65) + Math.max(0.0, mf - 1.0) * 0.95);
        }
        // Sniper gets ZERO moving ADS relief — walking makes it nearly impossible to hit!
        if (!isSn) movePen *= (1.0 - 0.45 * this.adsT);
      }

      let airPen = 0.0;
      if (!this.lastOnGround) {
        airPen = b.inacc_jump * (0.85 + 0.28 * Math.min(1.5, Math.abs(this.lastVelY) / 6.5));
      } else if (this.landingPenalty > 0.0) {
        airPen = b.inacc_jump * 0.42 * clamp01(this.landingPenalty / 0.18);
      }

      const bloomScale = (this.lastCrouching ? 0.78 : 1.0) * (1.0 - 0.48 * this.adsT);
      return base + movePen + airPen + this.fireBloom * bloomScale;
    }

    spawnWorldPickup(spawnPoints, forceIndex = null) {
      this.reset(false, true);
      this.worldPickups = [];
      if (!spawnPoints || !spawnPoints.length) return [];

      // Place M110 SASS (idx 9) & MP5-S (idx 10) directly on the Sector B armory workbench behind the door!
      const n = spawnPoints.length;
      const armorySet = (n >= 11);
      const idxs = [];
      const baseWids = armorySet
        ? this.WEAPON_ORDER.filter((w) => w !== "m110" && w !== "mp5s")
        : this.WEAPON_ORDER.slice();
      const poolN = armorySet ? (n - 2) : n;
      const all = [];
      for (let i = 0; i < poolN; i++) all.push(i);
      const pool = all.slice();
      const baseMap = {};
      for (let k = 0; k < baseWids.length; k++) {
        if (forceIndex !== null && forceIndex !== undefined) {
          baseMap[baseWids[k]] = (forceIndex + k) % poolN;
        } else if (pool.length > 0) {
          const pickI = Math.floor(Math.random() * pool.length);
          baseMap[baseWids[k]] = pool.splice(pickI, 1)[0];
        } else {
          baseMap[baseWids[k]] = k % poolN;
        }
      }
      if (armorySet) {
        baseMap.m110 = n - 2;
        baseMap.mp5s = n - 1;
      }
      for (const wid of this.WEAPON_ORDER) idxs.push(baseMap[wid] ?? 0);
      this.lastSpawnIndices = idxs.slice();

      for (let wi = 0; wi < this.WEAPON_ORDER.length; wi++) {
        const wid = this.WEAPON_ORDER[wi];
        const spec = this.SPECS[wid];
        const sp = spawnPoints[idxs[wi]];
        const isLg = (wid === "remington11" || wid === "double_barrel" || wid === "ak47" || wid === "m4a1" || wid === "mp7" || wid === "m110" || wid === "mp5s");
        this.worldPickups.push({
          weaponId: wid,
          pos: [sp.pos[0], sp.pos[1] + (isLg ? 0.038 : 0.028), sp.pos[2]],
          vel: [0.0, 0.0, 0.0],
          rot: [0.0, sp.yaw || 0.0, Math.PI * 0.5],
          avel: [0.0, 0.0, 0.0],
          grounded: true,
          hasBounced: true,
          magAmmo: spec.mag_capacity,
          reserveAmmo: spec.start_reserve,
          slideLocked: false,
          spentShells: 0,
          label: sp.label || "Арена",
        });
      }
      return this.worldPickups;
    }

    tryDropWeapon(eyePos, aimDir, colliders = null) {
      if (!this.hasWeapon) return false;
      this._saveToActive();
      const wid = this.activeWeapon;
      const fx = aimDir[0], fy = aimDir[1], fz = aimDir[2];
      const fLen = Math.hypot(fx, fy, fz) || 1.0;
      const dx = fx / fLen, dy = fy / fLen, dz = fz / fLen;
      const rx = -dz, rz = dx;
      const rLen = Math.hypot(rx, rz) || 1.0;
      const rnx = rx / rLen, rnz = rz / rLen;

      const desired = [
        eyePos[0] + dx * 0.44 + rnx * 0.11,
        eyePos[1] + dy * 0.44 - 0.12,
        eyePos[2] + dz * 0.44 + rnz * 0.11,
      ];
      const startPos = safePropSpawn(eyePos, desired, colliders, 0.085, 0.16);
      const speed = 4.2;
      const vel = [
        dx * speed + rnx * 0.28,
        Math.max(-0.6, dy * speed + 1.35),
        dz * speed + rnz * 0.28,
      ];
      const yaw = Math.atan2(dx, -dz);

      this.worldPickups.push({
        weaponId: wid,
        pos: startPos,
        vel,
        rot: [0.18, yaw, 0.35],
        avel: [-4.4, 2.8, 3.8],
        grounded: false,
        hasBounced: false,
        magAmmo: this.magAmmo,
        reserveAmmo: this.reserveAmmo,
        slideLocked: this.slideLocked,
        spentShells: this.spentShells,
        label: "Брошен на пол",
      });

      this.weapons[wid].owned = false;
      this._clearActionStates();
      this.pickupCd = 0.42;

      const remaining = this.ownedWeapons();
      if (remaining.length > 0) {
        this.activeWeapon = remaining[0];
        this._syncFromActive();
        this.equipped = true;
        this.equipT = 0.0;
      } else {
        this.hasWeapon = false;
        this.equipped = false;
        this.equipT = 0.0;
      }
      this.sfxQueue.push("draw");
      return true;
    }

    nearestPickup(playerPos, maxDist = 2.55) {
      if (this.pickupCd > 0.0 || !this.worldPickups.length) return null;
      let best = null;
      let bestD2 = maxDist * maxDist + 1.0;
      for (const wp of this.worldPickups) {
        const wid = wp.weaponId || "m1911";
        if (this.weapons[wid] && this.weapons[wid].owned) continue;
        const p = wp.pos;
        const dx = p[0] - playerPos[0];
        const dz = p[2] - playerPos[2];
        const dy = p[1] - (playerPos[1] + 0.95);
        const d2 = dx * dx + dz * dz;
        if (d2 <= maxDist * maxDist && Math.abs(dy) <= 1.75 && d2 < bestD2) {
          bestD2 = d2;
          best = wp;
        }
      }
      return best;
    }

    isNearPickup(playerPos, maxDist = 2.55) {
      return this.nearestPickup(playerPos, maxDist) !== null;
    }

    tryPickup(playerPos, maxDist = 2.55, granted = null) {
      const wp = granted || this.nearestPickup(playerPos, maxDist);
      if (!wp || this.ownedWeapons().length >= 2) return false;
      const idx = this.worldPickups.indexOf(wp);
      if (idx >= 0) this.worldPickups.splice(idx, 1);
      if (this.hasWeapon) this._saveToActive();
      const wid = wp.weaponId || "m1911";
      const spec = this.SPECS[wid] || this.SPECS.m1911;
      const st = this.weapons[wid];
      st.owned = true;
      st.magAmmo = (wp.magAmmo !== undefined) ? wp.magAmmo : spec.mag_capacity;
      st.reserveAmmo = (wp.reserveAmmo !== undefined) ? wp.reserveAmmo : spec.start_reserve;
      st.slideLocked = Boolean(wp.slideLocked);
      st.spentShells = (wp.spentShells !== undefined) ? wp.spentShells : Math.max(0, spec.mag_capacity - st.magAmmo);
      this.activeWeapon = wid;
      this._syncFromActive();
      this._clearActionStates();
      this.equipped = true;
      this.equipT = 0.0;
      this.sfxQueue.push("draw");
      if (wid === "double_barrel") this.sfxQueue.push("db_close");
      else if (wid === "revolver") this.sfxQueue.push("rev_close");
      else if (!this.slideLocked) this.sfxQueue.push((wid === "remington11") ? "pump_fwd" : "slide_fwd");
      this.spawnPickupFx(wp.pos || playerPos);
      this._addCamJolt(-0.08, 0.05, 0.14);
      return true;
    }

    selectWeapon(wid) {
      if (!this.weapons[wid] || !this.weapons[wid].owned || !this.ownedWeapons().includes(wid)) return false;
      if (this.activeWeapon === wid && this.equipped) {
        this.toggleEquip();
        return true;
      }
      if (this.hasWeapon) this._saveToActive();
      this.activeWeapon = wid;
      this._syncFromActive();
      this._clearActionStates();
      this.equipped = true;
      this.equipT = 0.0;
      this.sfxQueue.push("draw");
      return true;
    }

    cycleWeapon(dir = 1) {
      const owned = this.ownedWeapons();
      if (!owned.length) return false;
      if (owned.length === 1) {
        return this.equipped ? (this.toggleEquip(), true) : this.selectWeapon(owned[0]);
      }
      const curIdx = Math.max(0, owned.indexOf(this.activeWeapon));
      const nextWid = owned[(curIdx + (dir >= 0 ? 1 : -1) + owned.length) % owned.length];
      if (this.hasWeapon) this._saveToActive();
      this.activeWeapon = nextWid;
      this._syncFromActive();
      this._clearActionStates();
      this.equipped = true;
      this.equipT = 0.0;
      this.sfxQueue.push("draw");
      return true;
    }

    popSfxEvents() {
      if (!this.sfxQueue.length) return [];
      const evs = this.sfxQueue.slice();
      this.sfxQueue.length = 0;
      return evs;
    }

    toggleEquip() {
      if (!this.hasWeapon) {
        const owned = this.ownedWeapons();
        if (!owned.length) return;
        this.activeWeapon = owned[0];
        this._syncFromActive();
      }
      this.equipped = !this.equipped;
      if (this.equipped) {
        this.equipT = 0.0;
        this.sfxQueue.push("draw");
      } else {
        this._clearActionStates();
      }
    }

    // Keep the four articulated-part curves in sync with engine/weapon.py.
    _reloadProgress() { return clamp01(this.reloadTimer / Math.max(this.reloadDuration, 1e-4)); }
    _checkProgress() { return clamp01(this.magCheckTimer / Math.max(this.magCheckDuration, 1e-4)); }

    _dbBreakAngle() {
      if (this.magCheckBlend > 0.01 && this.magCheckMode === "double_barrel") {
        const u = this._checkProgress();
        return 0.88 * s5(0.14, 0.34, u) * (1 - s5(0.68, 0.86, u)) * this.magCheckBlend;
      }
      if (!this.reloading || this.reloadType !== "break_action") return 0;
      const u = this._reloadProgress();
      switch (this.dbReloadPhase) {
        case "open": return 0.96 * s5(0.22, 0.56, u);
        case "insert": return 0.96;
        case "close": return 0.96 * (1 - s5(0.18, 0.54, u));
        default: return 0;
      }
    }

    _dbLeverAngle() {
      if (this.magCheckBlend > 0.01 && this.magCheckMode === "double_barrel") {
        const u = this._checkProgress();
        return -0.58 * s5(0.05, 0.20, u) * (1 - s5(0.55, 0.78, u)) * this.magCheckBlend;
      }
      if (!this.reloading || this.reloadType !== "break_action") return 0;
      const u = this._reloadProgress();
      switch (this.dbReloadPhase) {
        case "open": return -0.62 * s5(0.04, 0.24, u) * (1 - 0.35 * s5(0.45, 0.78, u));
        case "insert": return -0.40;
        case "close": return -0.40 * (1 - s5(0.40, 0.62, u));
        default: return 0;
      }
    }

    _revCraneAngle() {
      if (this.magCheckBlend > 0.01 && this.magCheckMode === "revolver") {
        const u = this._checkProgress();
        return 1.42 * s5(0.08, 0.25, u) * (1 - s5(0.76, 0.90, u)) * this.magCheckBlend;
      }
      if (!this.reloading || this.reloadType !== "revolver") return 0;
      const u = this._reloadProgress();
      switch (this.revReloadPhase) {
        case "open_dump": return 1.42 * s5(0.08, 0.28, u);
        case "load": return 1.42;
        case "close": return 1.42 * (1 - s5(0.18, 0.56, u));
        default: return 0;
      }
    }

    _revHammerAngle() {
      if (!this.revCocking) return -0.14 * this.hammerDrop;
      return 1.08 * s5(0, 0.82, clamp01(this.revCockTimer / Math.max(this.revCockDuration, 1e-4)));
    }

    canReload() {
      return (
        this.hasWeapon && this.equipped && !this.reloading && !this.pumpActive && !this.revCocking
        && this.magAmmo < this.MAG_CAPACITY && this.reserveAmmo > 0
      );
    }

    startReload() {
      if (!this.canReload()) return false;
      this.cancelInspect();
      this.cancelMagCheck();
      if (this.activeSpec.break_reload) {
        this.reloading = true;
        this.reloadType = "break_action";
        this.reloadTimer = 0.0;
        this.dbReloadPhase = "open";
        this.reloadDuration = 0.50;
        this.dbReloadTarget = Math.min(this.MAG_CAPACITY - this.magAmmo, this.reserveAmmo);
        this.dbInsertedCount = 0;
        this.dbEjectCount = Math.max(1, Math.min(2, this.spentShells || (this.MAG_CAPACITY - this.magAmmo)));
        this.dbEvOpen = false;
        this.dbEvEject = false;
        this.dbEvClose = false;
        this.evShellIn = false;
        this._addCamJolt(-0.10, 0.06, -0.14);
        return true;
      }
      if (this.activeSpec.revolver_reload) {
        this.reloading = true;
        this.reloadType = "revolver";
        this.reloadTimer = 0.0;
        this.reloadInterrupt = false;
        this.revReloadPhase = "open_dump";
        this.reloadDuration = 0.80;
        this.revLoadChamber = -1;
        this.revEvOpen = false;
        this.revEvDump = false;
        this.revEvLoad = false;
        this.revEvRoundIn = false;
        this.revEvClose = false;
        this.relCommitted = false;
        this._addCamJolt(0.08, -0.06, -0.18);
        return true;
      }
      if (this.activeSpec.shell_reload) {
        this.reloading = true;
        this.reloadType = "shell";
        this.reloadTimer = 0.0;
        this.reloadInterrupt = false;
        this.reloadWasEmpty = (this.magAmmo === 0);
        this.evShellIn = false;
        this.relEvBack = false;
        this.relEvFwd = false;
        if (this.reloadWasEmpty) {
          this.slideLocked = true;
          this.chamberThenLoad = (this.reserveAmmo > 1 && this.MAG_CAPACITY > 1);
          this.shellReloadPhase = "chamber";
          this.reloadDuration = 1.44;
        } else {
          this.chamberThenLoad = false;
          this.shellReloadPhase = "enter";
          this.reloadDuration = 0.40;
        }
        this._addCamJolt(-0.08, 0.05, 0.14);
        return true;
      }
      this.reloading = true;
      this.reloadTimer = 0.0;
      this.reloadType = (this.magAmmo === 0) ? "empty" : "tactical";
      if (this.isAK47) {
        this.reloadDuration = (this.reloadType === "empty") ? 2.15 : 1.78;
      } else if (this.isM4A1) {
        this.reloadDuration = (this.reloadType === "empty") ? 1.98 : 1.62;
      } else if (this.isM110) {
        this.reloadDuration = (this.reloadType === "empty") ? 2.24 : 1.84;
      } else if (this.isMP5S) {
        this.reloadDuration = (this.reloadType === "empty") ? 1.88 : 1.54;
      } else if (this.isMP7) {
        this.reloadDuration = (this.reloadType === "empty") ? 1.82 : 1.48;
      } else {
        this.reloadDuration = (this.reloadType === "empty") ? 1.92 : 1.56;
      }
      this.relEvOut = false;
      this.relEvIn = false;
      this.relEvBack = false;
      this.relEvFwd = false;
      this.relCommitted = false;
      this._addCamJolt(-0.09, 0.07, -0.16);
      return true;
    }

    // ---- v14: Camera Jolts & Rich 3D / Screen-Space VFX Emitters ----
    _addCamJolt(dp = 0.0, dy = 0.0, dr = 0.0) {
      const scale = (this.reloading || this.magChecking || this.inspecting) ? 0.52 : 1.0;
      this.camJoltPitchV += dp * scale;
      this.camJoltYawV += dy * scale;
      this.camJoltRollV += dr * scale;
    }

    _spawnParticle(pos, vel, color, size0, size1, life, gravity = 0.0, drag = 2.0, emissive = false) {
      if (this.particles.length >= 160) this.particles.shift();
      this.particles.push({
        pos: [pos[0], pos[1], pos[2]],
        vel: [vel[0], vel[1], vel[2]],
        color: [color[0], color[1], color[2]],
        size0,
        size1,
        life,
        maxLife: Math.max(life, 0.01),
        gravity,
        drag,
        emissive: Boolean(emissive),
        rot: Math.random() * Math.PI * 2.0,
        avel: (Math.random() - 0.5) * 9.0,
      });
    }

    _spawnTracer(startPos, endPos, color, thickness = 0.010, life = 0.085) {
      if (this.tracers.length >= 32) this.tracers.shift();
      this.tracers.push({
        start: [startPos[0], startPos[1], startPos[2]],
        end: [endPos[0], endPos[1], endPos[2]],
        color: [color[0], color[1], color[2]],
        thickness,
        life,
        maxLife: Math.max(life, 0.01),
      });
    }

    _spawnMuzzleFx(muzzlePos, fwd, rightV, upV) {
      const wid = this.activeWeapon;
      const isMp5Sil = this.isMP5S && this.mp5Suppressed;
      const isHeavy = this.isShotgun || this.isDoubleBarrel;
      const isSn = this.isM110;
      const rnd = (a, b) => a + Math.random() * (b - a);
      if (isMp5Sil) {
        for (let i = 0; i < 4; i++) {
          const sp = rnd(0.55, 1.35);
          const rx = rnd(-0.26, 0.26), ry = rnd(0.08, 0.36);
          this._spawnParticle(
            [muzzlePos[0] + fwd[0] * 0.04, muzzlePos[1] + fwd[1] * 0.04, muzzlePos[2] + fwd[2] * 0.04],
            [fwd[0] * sp + rightV[0] * rx + upV[0] * ry, fwd[1] * sp + rightV[1] * rx + upV[1] * ry + 0.25, fwd[2] * sp + rightV[2] * rx + upV[2] * ry],
            [0.72, 0.78, 0.84], 0.014, 0.068, rnd(0.26, 0.44), -0.35, 3.6, false
          );
        }
        return;
      }
      if (isSn) {
        for (const side of [-1.0, 1.0]) {
          for (let i = 0; i < 4; i++) {
            const sp = rnd(1.8, 3.6);
            const fSp = rnd(0.2, 0.9);
            const uSp = rnd(-0.35, 0.45);
            this._spawnParticle(
              muzzlePos,
              [rightV[0] * side * sp + fwd[0] * fSp + upV[0] * uSp, rightV[1] * side * sp + fwd[1] * fSp + upV[1] * uSp + 0.25, rightV[2] * side * sp + fwd[2] * fSp + upV[2] * uSp],
              [0.78, 0.76, 0.72], 0.022, 0.135, rnd(0.32, 0.56), -0.42, 4.2, false
            );
          }
          for (let i = 0; i < 2; i++) {
            const sp = rnd(2.8, 5.2);
            this._spawnParticle(
              muzzlePos,
              [rightV[0] * side * sp + fwd[0] * rnd(0.5, 1.6), rightV[1] * side * sp + rnd(-0.6, 0.8), rightV[2] * side * sp + fwd[2] * rnd(0.5, 1.6)],
              [1.0, 0.80, 0.36], 0.014, 0.004, rnd(0.08, 0.16), 6.2, 1.4, true
            );
          }
        }
        this.fxImpactPulse = Math.max(this.fxImpactPulse, 0.62);
        return;
      }
      const smokeN = isHeavy ? 9 : (this.isRevolver ? 7 : 4);
      const emberN = isHeavy ? 8 : (this.isRevolver ? 6 : 3);
      for (let i = 0; i < smokeN; i++) {
        const sp = rnd(0.9, isHeavy ? 2.8 : 1.9);
        const rx = rnd(-0.55, 0.55), ry = rnd(-0.20, 0.55);
        this._spawnParticle(
          [muzzlePos[0] + fwd[0] * 0.03, muzzlePos[1] + fwd[1] * 0.03, muzzlePos[2] + fwd[2] * 0.03],
          [fwd[0] * sp + rightV[0] * rx + upV[0] * ry, fwd[1] * sp + rightV[1] * rx + upV[1] * ry + 0.28, fwd[2] * sp + rightV[2] * rx + upV[2] * ry],
          [0.64, 0.63, 0.60], 0.018, isHeavy ? 0.155 : 0.092, rnd(0.28, 0.52), -0.38, 3.8, false
        );
      }
      for (let i = 0; i < emberN; i++) {
        const sp = rnd(2.5, 6.2);
        const rx = rnd(-1.1, 1.1), ry = rnd(-0.8, 1.1);
        this._spawnParticle(
          muzzlePos,
          [fwd[0] * sp + rightV[0] * rx + upV[0] * ry, fwd[1] * sp + rightV[1] * rx + upV[1] * ry, fwd[2] * sp + rightV[2] * rx + upV[2] * ry],
          [1.0, rnd(0.55, 0.88), 0.24], 0.011, 0.003, rnd(0.09, 0.22), 7.5, 1.6, true
        );
      }
      if (isHeavy) this.fxImpactPulse = Math.max(this.fxImpactPulse, 0.58);
      else if (this.isRevolver || this.isAK47 || this.isM4A1) this.fxImpactPulse = Math.max(this.fxImpactPulse, 0.24);
      else if (wid === "mp5s" && !this.mp5Suppressed) this.fxImpactPulse = Math.max(this.fxImpactPulse, 0.18);
    }

    _spawnImpactFx(hp, nx, ny, nz, isHeavy = false) {
      const rnd = (a, b) => a + Math.random() * (b - a);
      const isSn = this.isM110;
      const dustN = isSn ? 6 : (isHeavy ? 2 : 3);
      const chipN = isSn ? 5 : (isHeavy ? 2 : 3);
      const sparkN = isSn ? 6 : (isHeavy ? 2 : 3);
      for (let i = 0; i < dustN; i++) {
        const sp = rnd(0.7, isSn ? 2.6 : 1.7);
        this._spawnParticle(
          [hp[0] + nx * 0.03, hp[1] + ny * 0.03, hp[2] + nz * 0.03],
          [nx * sp + rnd(-0.85, 0.85), ny * sp + rnd(0.15, 1.05), nz * sp + rnd(-0.85, 0.85)],
          [0.62, 0.60, 0.56], 0.022, isSn ? 0.15 : 0.095, rnd(0.32, 0.58), -0.18, 3.6, false
        );
      }
      for (let i = 0; i < chipN; i++) {
        const sp = rnd(1.5, isSn ? 4.5 : 3.2);
        this._spawnParticle(
          [hp[0] + nx * 0.025, hp[1] + ny * 0.025, hp[2] + nz * 0.025],
          [nx * sp + rnd(-1.4, 1.4), ny * sp + rnd(0.4, 2.2), nz * sp + rnd(-1.4, 1.4)],
          [0.42, 0.40, 0.37], rnd(0.012, 0.024), 0.008, rnd(0.34, 0.64), 10.5, 0.7, false
        );
      }
      for (let i = 0; i < sparkN; i++) {
        const sp = rnd(2.2, isSn ? 6.2 : 4.2);
        this._spawnParticle(
          [hp[0] + nx * 0.02, hp[1] + ny * 0.02, hp[2] + nz * 0.02],
          [nx * sp + rnd(-1.8, 1.8), ny * sp + rnd(0.3, 2.0), nz * sp + rnd(-1.8, 1.8)],
          [1.0, rnd(0.62, 0.94), 0.28], 0.012, 0.003, rnd(0.10, 0.22), 8.2, 1.2, true
        );
      }
    }

    spawnDoorBreachFx(pos, fwd) {
      const rnd = (a, b) => a + Math.random() * (b - a);
      for (let i = 0; i < 16; i++) {
        const sp = rnd(1.6, 4.6);
        this._spawnParticle(
          [pos[0] + rnd(-0.22, 0.22), pos[1] + rnd(-0.48, 0.48), pos[2] + rnd(-0.22, 0.22)],
          [fwd[0] * sp + rnd(-1.3, 1.3), rnd(0.25, 2.2), fwd[2] * sp + rnd(-1.3, 1.3)],
          [0.64, 0.60, 0.54], 0.045, 0.22, rnd(0.42, 0.85), -0.15, 2.8, false
        );
      }
      for (let i = 0; i < 12; i++) {
        const sp = rnd(2.4, 6.2);
        this._spawnParticle(
          [pos[0] + rnd(-0.18, 0.18), pos[1] + rnd(-0.32, 0.32), pos[2] + rnd(-0.18, 0.18)],
          [fwd[0] * sp + rnd(-1.9, 1.9), rnd(0.4, 2.8), fwd[2] * sp + rnd(-1.9, 1.9)],
          [0.36, 0.42, 0.38], rnd(0.018, 0.036), 0.010, rnd(0.45, 0.82), 11.2, 0.8, false
        );
      }
      for (let i = 0; i < 10; i++) {
        const sp = rnd(2.2, 5.6);
        this._spawnParticle(
          [pos[0], pos[1] + rnd(-0.25, 0.25), pos[2]],
          [fwd[0] * sp + rnd(-2.0, 2.0), rnd(0.3, 2.2), fwd[2] * sp + rnd(-2.0, 2.0)],
          [1.0, rnd(0.68, 0.92), 0.30], 0.016, 0.004, rnd(0.14, 0.28), 8.5, 1.2, true
        );
      }
      this.fxImpactPulse = Math.max(this.fxImpactPulse, 0.85);
      this._addCamJolt(0.22, 0.08, -0.26);
    }

    spawnDoorDustFx(pos) {
      const rnd = (a, b) => a + Math.random() * (b - a);
      for (let i = 0; i < 6; i++) {
        this._spawnParticle(
          [pos[0] + rnd(-0.25, 0.25), pos[1] + rnd(-0.45, 0.45), pos[2] + rnd(-0.25, 0.25)],
          [rnd(-0.6, 0.6), rnd(0.15, 0.75), rnd(-0.6, 0.6)],
          [0.62, 0.60, 0.56], 0.032, 0.14, rnd(0.35, 0.62), -0.10, 3.0, false
        );
      }
    }

    spawnLandingFx(feetPos, fallSpeed) {
      const rnd = (a, b) => a + Math.random() * (b - a);
      const amt = Math.min(1.0, Math.max(0.2, (fallSpeed - 1.5) / 6.5));
      const cnt = Math.floor(6 + amt * 8);
      for (let i = 0; i < cnt; i++) {
        const ang = (i / cnt) * Math.PI * 2.0 + rnd(-0.25, 0.25);
        const sp = rnd(0.8, 2.2) * (0.6 + 0.6 * amt);
        this._spawnParticle(
          [feetPos[0] + Math.cos(ang) * 0.18, feetPos[1] + 0.06, feetPos[2] + Math.sin(ang) * 0.18],
          [Math.cos(ang) * sp, rnd(0.25, 0.95) * (0.6 + 0.5 * amt), Math.sin(ang) * sp],
          [0.64, 0.62, 0.58], 0.032, 0.14 * (0.7 + 0.5 * amt), rnd(0.32, 0.58), 1.4, 3.4, false
        );
      }
      if (fallSpeed > 4.2) {
        this.fxImpactPulse = Math.max(this.fxImpactPulse, 0.35 * amt);
        this._addCamJolt(-0.18 * amt, 0.0, (Math.random() - 0.5) * 0.14 * amt);
      }
    }

    spawnSlideFx(feetPos, vel) {
      const rnd = (a, b) => a + Math.random() * (b - a);
      for (let i = 0; i < 3; i++) {
        this._spawnParticle(
          [feetPos[0] + rnd(-0.22, 0.22), feetPos[1] + 0.08, feetPos[2] + rnd(-0.22, 0.22)],
          [-vel[0] * 0.22 + rnd(-0.6, 0.6), rnd(0.25, 0.95), -vel[2] * 0.22 + rnd(-0.6, 0.6)],
          [0.66, 0.64, 0.60], 0.028, 0.13, rnd(0.28, 0.50), -0.15, 3.2, false
        );
      }
      if (Math.random() < 0.65) {
        this._spawnParticle(
          [feetPos[0] + rnd(-0.18, 0.18), feetPos[1] + 0.06, feetPos[2] + rnd(-0.18, 0.18)],
          [-vel[0] * 0.35 + rnd(-0.9, 0.9), rnd(0.4, 1.5), -vel[2] * 0.35 + rnd(-0.9, 0.9)],
          [1.0, 0.78, 0.32], 0.011, 0.003, rnd(0.10, 0.20), 7.8, 1.4, true
        );
      }
    }

    spawnPickupFx(pos) {
      const rnd = (a, b) => a + Math.random() * (b - a);
      for (let i = 0; i < 10; i++) {
        const ang = (i / 10) * Math.PI * 2.0 + rnd(-0.2, 0.2);
        const sp = rnd(0.8, 2.2);
        this._spawnParticle(
          [pos[0], pos[1] + 0.12, pos[2]],
          [Math.cos(ang) * sp, rnd(0.5, 1.7), Math.sin(ang) * sp],
          [0.85, 1.0, 0.72], 0.014, 0.004, rnd(0.22, 0.42), 2.5, 2.4, true
        );
      }
    }

    _spawnBreechSmoke(vmToWorld, cnt = 5) {
      if (!vmToWorld) return;
      const rnd = (a, b) => a + Math.random() * (b - a);
      const gunM = this.computeGunViewMatrix();
      const bWorld = mat4TransformPt(vmToWorld, mat4TransformPt(gunM, [0.0, 0.075, -0.15]));
      for (let i = 0; i < cnt; i++) {
        this._spawnParticle(
          [bWorld[0] + rnd(-0.02, 0.02), bWorld[1] + rnd(-0.01, 0.02), bWorld[2] + rnd(-0.02, 0.02)],
          [rnd(-0.28, 0.28), rnd(0.32, 0.85), rnd(-0.28, 0.28)],
          [0.72, 0.72, 0.70], 0.014, 0.078, rnd(0.32, 0.58), -0.35, 3.2, false
        );
      }
    }

    _spawnDbEjectedShell(vmToWorld, barrelIdx = 0) {
      if (!vmToWorld) return;
      const gunM = this.computeGunViewMatrix();
      const ang = this._dbBreakAngle();
      const hp = this.DB_HINGE_PIVOT;
      const mBbl = mat4Mul(gunM, pivotRot(hp, 0.0, -ang, 0.0));
      const bx = (barrelIdx === 0) ? -0.0138 : 0.0138;
      const pView = mat4TransformPt(mBbl, [bx, 0.0725, -0.125]);
      const pWorld = mat4TransformPt(vmToWorld, pView);
      const rightW = [vmToWorld[0], vmToWorld[4], vmToWorld[8]];
      const upW = [vmToWorld[1], vmToWorld[5], vmToWorld[9]];
      const fwdW = [-vmToWorld[2], -vmToWorld[6], -vmToWorld[10]];
      const bwdSpd = 1.85 + Math.random() * 0.60;
      const upSpd = 1.55 + Math.random() * 0.50;
      const sideSpd = (barrelIdx === 0 ? -0.52 : 0.58) + (Math.random() * 0.44 - 0.22);
      this.shells.push({
        pos: [pWorld[0], pWorld[1], pWorld[2]],
        vel: [
          rightW[0]*sideSpd + upW[0]*upSpd - fwdW[0]*bwdSpd,
          rightW[1]*sideSpd + upW[1]*upSpd - fwdW[1]*bwdSpd,
          rightW[2]*sideSpd + upW[2]*upSpd - fwdW[2]*bwdSpd,
        ],
        rot: [Math.random()*6.28, Math.random()*6.28, Math.random()*6.28],
        avel: [(Math.random()*28)-14, (Math.random()*24)-12, (Math.random()*32)-16],
        life: 1.95,
        shellType: "shotgun",
      });
      if (this.shells.length > 20) this.shells.shift();
    }

    _spawnRevDumpedShells(vmToWorld, chamberIndices = null) {
      if (!vmToWorld) return;
      const gunM = this.computeGunViewMatrix();
      const crAng = this._revCraneAngle();
      const mCyl = mat4Mul(
        mat4Mul(gunM, pivotRot(this.REV_CRANE_PIVOT, 0.0, 0.0, -crAng)),
        pivotRot(this.REV_CYL_PIVOT, 0.0, 0.0, this.revCylAngle)
      );
      const rightW = [vmToWorld[0], vmToWorld[4], vmToWorld[8]];
      const upW = [vmToWorld[1], vmToWorld[5], vmToWorld[9]];
      const fwdW = [-vmToWorld[2], -vmToWorld[6], -vmToWorld[10]];
      const indices = (chamberIndices && chamberIndices.length) ? chamberIndices : [0];
      const [cxR, cyR] = this.REV_CYL_PIVOT;
      for (let k = 0; k < indices.length; k++) {
        const ci = indices[k];
        const angC = Math.PI * 0.5 + ci * (Math.PI * 2.0 / 8.0);
        const pCyl = [
          cxR + Math.cos(angC) * this.REV_CHAMBER_RING_R,
          cyR + Math.sin(angC) * this.REV_CHAMBER_RING_R,
          this.REV_CHAMBER_REAR_Z + 0.014,
        ];
        const pWorld = mat4TransformPt(vmToWorld, mat4TransformPt(mCyl, pCyl));
        const s = -0.35 - Math.random() * 0.55 + Math.cos(angC) * 0.16;
        const u = -0.18 + Math.random() * 0.42 + Math.sin(angC) * 0.12;
        const f = -0.42 - Math.random() * 0.52;
        this.shells.push({
          pos: [pWorld[0], pWorld[1], pWorld[2]],
          vel: [
            rightW[0]*s + upW[0]*u + fwdW[0]*f,
            rightW[1]*s + upW[1]*u + fwdW[1]*f,
            rightW[2]*s + upW[2]*u + fwdW[2]*f,
          ],
          rot: [Math.random()*6.28, Math.random()*6.28, Math.random()*6.28],
          avel: [(Math.random()*22)-11, (Math.random()*22)-11, (Math.random()*26)-13],
          life: 4.6,
          shellType: "rev_spent",
          grounded: false,
          bounces: 0,
        });
      }
      while (this.shells.length > 28) this.shells.shift();
    }

    _spawnEjectedShell(vmToWorld, shellType = "pistol") {
      if (!vmToWorld) return;
      const spec = this.activeSpec;
      const gunM = this.computeGunViewMatrix();
      let ejectLocal = spec.eject_local || this.EJECT_LOCAL;
      const ejectWorld = mat4TransformPt(vmToWorld, mat4TransformPt(gunM, ejectLocal));
      const rightW = [vmToWorld[0], vmToWorld[4], vmToWorld[8]];
      const upW = [vmToWorld[1], vmToWorld[5], vmToWorld[9]];
      const fwdW = [-vmToWorld[2], -vmToWorld[6], -vmToWorld[10]];
      const isSg = (shellType === "shotgun");
      let rSpd = (isSg ? 2.05 : 1.85) + Math.random() * 0.85;
      const uSpd = (isSg ? 1.38 : 1.25) + Math.random() * 0.65;
      const fSpd = -0.35 + Math.random() * 0.45;
      this.shells.push({
        pos: [ejectWorld[0], ejectWorld[1], ejectWorld[2]],
        vel: [
          rightW[0]*rSpd + upW[0]*uSpd + fwdW[0]*fSpd,
          rightW[1]*rSpd + upW[1]*uSpd + fwdW[1]*fSpd,
          rightW[2]*rSpd + upW[2]*uSpd + fwdW[2]*fSpd,
        ],
        rot: [Math.random()*6.28, Math.random()*6.28, Math.random()*6.28],
        avel: [(Math.random()*28)-14, (Math.random()*28)-14, (Math.random()*36)-18],
        life: isSg ? 1.75 : 1.55,
        shellType,
      });
      if (this.shells.length > 20) this.shells.shift();
    }

    _executeFireShot(eyePos, aimDir, vmToWorld, colliders) {
      const spec = this.activeSpec;
      const isSg = (spec.weapon_type === "shotgun");
      const isDb = (spec.weapon_type === "double_barrel");
      const isRev = (spec.weapon_type === "revolver");
      const isHeavy = isSg || isDb;
      this.magAmmo = Math.max(0, this.magAmmo - 1);
      if (isDb) this.spentShells = Math.min(this.MAG_CAPACITY, this.spentShells + 1);
      if (isRev) {
        this.spentShells = Math.min(this.MAG_CAPACITY, this.spentShells + 1);
        const liveIdx = this.revChambers.indexOf("live");
        if (liveIdx >= 0) this.revChambers[liveIdx] = "spent";
      }
      if (isDb) this.dbLastBarrel = 1 - this.dbLastBarrel;
      this.fireTimer = spec.fire_cooldown;
      this.triggerPull = 1.0;
      this.hammerDrop = (this.activeWeapon === "m1911" || isRev) ? 1.0 : 0.0;
      this.slideKick = (isHeavy || isRev) ? 0.0 : 1.0;
      // Pistols and M4A1 lock bolt/slide open on empty; authentic AK-47 bolt closes on empty!
      this.slideLocked = (this.magAmmo === 0) && (!isHeavy && !isRev && !this.isAK47);
      this._saveToActive();
      this.muzzleTimer = isHeavy ? 0.078 : (isRev ? 0.062 : (spec.full_auto ? 0.042 : 0.048));
      this.muzzleRoll = Math.random() * Math.PI * 2.0;
      this.flashLightTimer = isHeavy ? 0.125 : (isRev ? 0.105 : (spec.full_auto ? 0.068 : 0.085));

      const adsDamp = 1.0 - (isHeavy ? 0.28 : 0.38) * this.adsT;
      this.recoilPitch = Math.min(isHeavy ? 0.62 : 0.48, this.recoilPitch + spec.recoil_pitch * adsDamp);
      this.recoilRoll += ((Math.random() * 0.12) - 0.055) * adsDamp;
      this.recoilZ = Math.min(isHeavy ? 0.098 : 0.072, this.recoilZ + spec.recoil_z * adsDamp);
      this.recoilY = Math.min(isHeavy ? 0.048 : 0.036, this.recoilY + spec.recoil_y * adsDamp);

      // CS2-style deterministic spray pattern + movement/jumping/crouch/rapid-fire inaccuracy!
      const bal = this._getBallistics();
      const pat = this.SPRAY_PATTERNS[this.activeWeapon] || this.SPRAY_PATTERNS.m1911 || [[0.0, 0.016]];
      const shotIdx = Math.min(pat.length - 1, Math.floor(this.sprayIndex));
      const [patYaw, patPitch] = pat[shotIdx];

      const crouchMult = this.lastCrouching ? bal.crouch_recoil : 1.0;
      const adsRecMult = 1.0 - (isHeavy ? 0.30 : 0.42) * this.adsT;
      const stanceRec = crouchMult * adsRecMult;

      const totalSpread = this.computeCurrentSpread();
      const adsSprayScale = 1.0 - 0.35 * this.adsT;
      const sprayDx = this.sprayYaw * bal.bullet_scale * adsSprayScale;
      const sprayDy = this.sprayPitch * bal.bullet_scale * adsSprayScale;

      const isMp5Sil = this.isMP5S && this.mp5Suppressed;
      const isBoltSn = Boolean(spec.bolt_cycle);
      this.sfxQueue.push(this.isMP5S ? (this.mp5Suppressed ? (spec.sfx_fire || "fire_mp5_silenced") : (spec.sfx_fire_unsupp || "fire_mp5_loud")) : (spec.sfx_fire || "fire"));
      if (isMp5Sil) {
        this.muzzleTimer *= 0.35;
        this.flashLightTimer *= 0.16;
      }

      let muzzleWorldPos = null;
      if (vmToWorld) {
        const gunM = this.computeGunViewMatrix();
        let muzzleLocal = (this.isMP5S && !this.mp5Suppressed && spec.muzzle_local_unsupp)
          ? spec.muzzle_local_unsupp
          : (spec.muzzle_local || this.MUZZLE_LOCAL);
        if (isDb) muzzleLocal = [this.dbLastBarrel === 0 ? -0.0138 : 0.0138, muzzleLocal[1], muzzleLocal[2]];
        muzzleWorldPos = mat4TransformPt(vmToWorld, mat4TransformPt(gunM, muzzleLocal));
        this.flashWorldPos = muzzleWorldPos;
      }

      if (isSg || isBoltSn) {
        this.pumpActive = true;
        this.pumpTimer = 0.0;
        this.pumpDuration = spec.pump_duration || (isBoltSn ? 0.92 : 0.76);
        this.pumpEvBack = false;
        this.pumpEvEject = false;
        this.pumpEvFwd = false;
      } else if (!isDb && !isRev) {
        this._spawnEjectedShell(vmToWorld, "pistol");
      }

      const pellets = spec.pellets || 1;
      const fLen = Math.hypot(aimDir[0], aimDir[1], aimDir[2]) || 1.0;
      const fwd = [aimDir[0] / fLen, aimDir[1] / fLen, aimDir[2] / fLen];
      const upRef = Math.abs(fwd[1]) < 0.96 ? [0, 1, 0] : [1, 0, 0];
      const rRaw = [
        fwd[1]*upRef[2] - fwd[2]*upRef[1],
        fwd[2]*upRef[0] - fwd[0]*upRef[2],
        fwd[0]*upRef[1] - fwd[1]*upRef[0],
      ];
      const rLen = Math.hypot(rRaw[0], rRaw[1], rRaw[2]) || 1.0;
      const rightV = [rRaw[0] / rLen, rRaw[1] / rLen, rRaw[2] / rLen];
      const upV = [
        rightV[1]*fwd[2] - rightV[2]*fwd[1],
        rightV[2]*fwd[0] - rightV[0]*fwd[2],
        rightV[0]*fwd[1] - rightV[1]*fwd[0],
      ];

      // v14: Rich muzzle smoke, brake blast & suppressor steam FX
      const fxOrigin = muzzleWorldPos || [
        eyePos[0] + fwd[0] * 0.45 + rightV[0] * 0.12 - upV[0] * 0.08,
        eyePos[1] + fwd[1] * 0.45 + rightV[1] * 0.12 - upV[1] * 0.08,
        eyePos[2] + fwd[2] * 0.45 + rightV[2] * 0.12 - upV[2] * 0.08,
      ];
      this._spawnMuzzleFx(fxOrigin, fwd, rightV, upV);

      let hitAny = false;
      let firstRay = fwd;this.netRays=[];
      const sparksPerHit = isHeavy ? 2 : (isRev ? 6 : 5);
      for (let pi = 0; pi < pellets; pi++) {
        let dx = sprayDx, dy = sprayDy;
        if (pellets === 1) {
          const ang = Math.random() * Math.PI * 2;
          const rad = totalSpread * Math.pow(Math.random(), 0.62);
          dx += Math.cos(ang) * rad;
          dy += Math.sin(ang) * rad;
        } else if (pi === 0) {
          const ang = Math.random() * Math.PI * 2;
          const rad = Math.random() * totalSpread * 0.28;
          dx += Math.cos(ang) * rad;
          dy += Math.sin(ang) * rad;
        } else {
          const ang = ((pi - 1) / Math.max(1, pellets - 1)) * Math.PI * 2 + (Math.random() * 0.64 - 0.32);
          const rad = totalSpread * Math.sqrt(0.24 + Math.random() * 0.76);
          dx += Math.cos(ang) * rad;
          dy += Math.sin(ang) * rad;
        }
        const rx = fwd[0] + rightV[0]*dx + upV[0]*dy;
        const ry = fwd[1] + rightV[1]*dx + upV[1]*dy;
        const rz = fwd[2] + rightV[2]*dx + upV[2]*dy;
        const rl = Math.hypot(rx, ry, rz) || 1.0;
        const rayD = [rx / rl, ry / rl, rz / rl];
        this.netRays.push(rayD);if (pi === 0) firstRay = rayD;

        let bestT = 80.0;
        let hitBox = null;
        for (let i = 0; i < colliders.length; i++) {
          const box = colliders[i];
          const tHit = raycastAABB(eyePos, rayD, box, bestT);
          if (tHit !== null && tHit > 0.12 && tHit < bestT) {
            bestT = tHit;
            hitBox = box;
          }
        }
        const endPt = [
          eyePos[0] + rayD[0] * Math.min(bestT, 45.0),
          eyePos[1] + rayD[1] * Math.min(bestT, 45.0),
          eyePos[2] + rayD[2] * Math.min(bestT, 45.0),
        ];
        // v14: Weapon-specific 3D bullet tracers & M110 supersonic vapor trails!
        if (this.isM110) {
          this._spawnTracer(fxOrigin, endPt, [1.0, 0.94, 0.68], 0.013, 0.12);
          this._spawnTracer(fxOrigin, endPt, [0.84, 0.88, 0.94], 0.022, 0.42);
        } else if (isMp5Sil) {
          this._spawnTracer(fxOrigin, endPt, [0.76, 0.84, 0.92], 0.006, 0.055);
        } else if (isHeavy) {
          if (pi % 2 === 0) {
            this._spawnTracer(fxOrigin, endPt, [1.0, 0.72, 0.34], 0.0085, 0.068);
          }
        } else if (isRev) {
          this._spawnTracer(fxOrigin, endPt, [1.0, 0.88, 0.46], 0.012, 0.095);
        } else {
          this._spawnTracer(fxOrigin, endPt, [1.0, 0.84, 0.44], 0.0085, 0.072);
        }

        if (hitBox !== null) {
          hitAny = true;
          const hp = [
            eyePos[0] + rayD[0]*bestT,
            eyePos[1] + rayD[1]*bestT,
            eyePos[2] + rayD[2]*bestT,
          ];
          const eps = 0.035;
          let nx = 0, ny = 0, nz = 0;
          if (Math.abs(hp[0] - hitBox[0]) < eps) nx = -1;
          else if (Math.abs(hp[0] - hitBox[3]) < eps) nx = 1;
          else if (Math.abs(hp[1] - hitBox[1]) < eps) ny = -1;
          else if (Math.abs(hp[1] - hitBox[4]) < eps) ny = 1;
          else if (Math.abs(hp[2] - hitBox[2]) < eps) nz = -1;
          else nz = 1;

          if (pellets === 1 || (pi % 2 === 0)) {
            this._spawnImpactFx(hp, nx, ny, nz, isHeavy);
          }

          for (let k = 0; k < sparksPerHit; k++) {
            const sp = 1.4 + Math.random() * 2.4;
            const life = (isHeavy ? 0.20 : 0.16) + Math.random() * 0.14;
            this.sparks.push({
              pos: [hp[0] + nx*0.025, hp[1] + ny*0.025, hp[2] + nz*0.025],
              vel: [
                nx*sp + (Math.random()*2.6 - 1.3),
                ny*sp + (Math.random()*2.1 - 0.3),
                nz*sp + (Math.random()*2.6 - 1.3),
              ],
              roll: Math.random() * Math.PI * 2,
              life,
              maxLife: life,
            });
          }
          if (pellets === 1 || (pi % 2 === 0)) {
            this.bulletHoles.push({
              pos: [hp[0] + nx*0.018, hp[1] + ny*0.018, hp[2] + nz*0.018],
              life: 6.5,
              maxLife: 6.5,
              roll: Math.random() * Math.PI * 2,
            });
            if (this.bulletHoles.length > 48) this.bulletHoles.shift();
          }
        }
      }
      this.lastShotDir = firstRay;
      if (this.sparks.length > 52) this.sparks.splice(0, this.sparks.length - 52);
      if (hitAny) this.sfxQueue.push("impact");

      // Advance CS2 spray pattern, fire bloom & camera recoil viewpunch AFTER shot!
      this.timeSinceShot = 0.0;
      this.sprayIndex = Math.min(pat.length, this.sprayIndex + 1.0);
      this.sprayPitch = Math.min(0.22, this.sprayPitch + patPitch * stanceRec);
      this.sprayYaw = Math.max(-0.15, Math.min(0.15, this.sprayYaw + patYaw * stanceRec));
      this.fireBloom = Math.min(bal.max_bloom, this.fireBloom + bal.inacc_fire * (this.lastCrouching ? 0.82 : 1.0) * (1.0 - 0.35 * this.adsT));

      const camPitch = (patPitch * bal.cam_scale + (spec.cam_pitch || 0.034) * 0.24) * stanceRec;
      const rK = (spec.cam_roll || 0.012);
      const camYaw = (patYaw * bal.cam_scale * 0.85 + ((Math.random() * 0.44 - 0.22) * rK)) * stanceRec;
      this.pendingCamPitch += camPitch;
      this.pendingCamRoll += camYaw;
      this.camRecoilPitch = Math.min(0.28, this.camRecoilPitch + camPitch);
      this.camRecoilYaw = Math.max(-0.16, Math.min(0.16, this.camRecoilYaw + camYaw));
      return true;
    }

    tryFire(eyePos, aimDir, vmToWorld, colliders, isAutoHold = false) {
      if (!this.hasWeapon || !this.equipped || this.equipT < 0.78) return false;
      this.cancelInspect();
      this.cancelMagCheck();
      const spec = this.activeSpec;
      if (this.reloading) {
        if (!isAutoHold && spec.shell_reload && this.magAmmo > 0 && (this.shellReloadPhase === "chamber" || this.shellReloadPhase === "enter" || this.shellReloadPhase === "insert")) {
          this.reloadInterrupt = true;
          if (this.shellReloadPhase === "chamber" && this.reloadTimer < this.reloadDuration * 0.78) {
            this.chamberThenLoad = false;
          }
        } else if (!isAutoHold && spec.revolver_reload && this.magAmmo > 0 && this.revReloadPhase === "load") {
          this.reloadInterrupt = true;
        }
        return false;
      }
      if (this.pumpActive || this.revCocking || this.suppAnimActive || this.fireTimer > 0.0) return false;
      if (isAutoHold && !spec.full_auto) return false;
      if (this.magAmmo <= 0) {
        if (isAutoHold) return false;
        this.fireTimer = 0.24;
        this.triggerPull = 1.0;
        this.sfxQueue.push("dry_fire");
        return false;
      }

      // Single-action 8-shot Revolver: first visibly COCK THE HAMMER with the thumb & index cylinder
      // (0.24s) — the actual magnum shot fires automatically when cocking completes in update()!
      if (spec.weapon_type === "revolver") {
        this.revCocking = true;
        this.revCockTimer = 0.0;
        this.revCockDuration = spec.cock_duration || 0.24;
        this.revCylTarget += Math.PI / 4.0;
        this.sfxQueue.push("rev_cock");
        return false;
      }
      return this._executeFireShot(eyePos, aimDir, vmToWorld, colliders);
    }

    _spawnWorldMag(vmToWorld, hasBullet, colliders = null) {
      if (!vmToWorld) return;
      const gunM = this.computeGunViewMatrix();
      const ma = this.isAK47 ? this.AK_MAG_AXIS
        : (this.isM4A1 ? this.M4_MAG_AXIS
        : (this.isM110 ? this.M110_MAG_AXIS
        : (this.isMP5S ? this.MP5S_MAG_AXIS
        : (this.isMP7 ? this.MP7_MAG_AXIS : this.MAG_AXIS))));
      const mc = this.isAK47 ? [0.0, -0.065, -0.195]
        : (this.isM4A1 ? this.M4_MAG_CENTER
        : (this.isM110 ? this.M110_MAG_CENTER
        : (this.isMP5S ? this.MP5S_MAG_CENTER
        : (this.isMP7 ? this.MP7_MAG_CENTER : this.MAG_MESH_CENTER))));
      const dropD = this.isRifle ? (hasBullet ? 0.14 : 0.10) : (this.isMP7 ? (hasBullet ? 0.12 : 0.085) : (hasBullet ? 0.11 : 0.075));
      const pView = mat4TransformPt(gunM, [
        mc[0] + ma[0] * dropD,
        mc[1] + ma[1] * dropD,
        mc[2] + ma[2] * dropD,
      ]);
      const pRaw = mat4TransformPt(vmToWorld, pView);
      const eyeW = [vmToWorld[3], vmToWorld[7], vmToWorld[11]];
      const pSafe = safePropSpawn(eyeW, pRaw, colliders, 0.055, 0.10);

      const rightW = [vmToWorld[0], vmToWorld[4], vmToWorld[8]];
      const upW = [vmToWorld[1], vmToWorld[5], vmToWorld[9]];
      const fwdW = [-vmToWorld[2], -vmToWorld[6], -vmToWorld[10]];

      const rx = -0.35 + (Math.random() * 0.40 - 0.20);
      const uy = -0.55 + (Math.random() * 0.25 - 0.12);
      const fz = 0.22 + (Math.random() * 0.36 - 0.18);
      const vel = [
        rightW[0] * rx + upW[0] * uy + fwdW[0] * fz,
        rightW[1] * rx + upW[1] * uy + fwdW[1] * fz,
        rightW[2] * rx + upW[2] * uy + fwdW[2] * fz,
      ];
      const baseYaw = Math.atan2(-fwdW[0], -fwdW[2]) + (Math.random() * 0.8 - 0.4);
      this.worldMags.push({
        weaponId: this.activeWeapon,
        pos: [pSafe[0], pSafe[1], pSafe[2]],
        vel,
        rot: [0.35, baseYaw, -0.65],
        avel: [
          Math.random() * 4.5 + 3.5,
          Math.random() * 5.0 - 2.5,
          -(Math.random() * 5.0 + 3.0),
        ],
        hasBullet: Boolean(hasBullet),
        grounded: false,
        hasBounced: false,
      });
      if (this.worldMags.length > 16) this.worldMags.shift();
    }

    update(dt, mouseDx, mouseDy, horizSpeed, onGround, sprinting, adsHeld, crouching, sliding, kicking, vmToWorld = null, colliders = null, eyePos = null, aimDir = null, inspectHeld = false, velY = 0.0, magCheckHeld = false) {
      this.elapsed += dt;
      if (this.fireTimer > 0.0) this.fireTimer = Math.max(0.0, this.fireTimer - dt);
      if (this.pickupCd > 0.0) this.pickupCd = Math.max(0.0, this.pickupCd - dt);
      if (this.muzzleTimer > 0.0) this.muzzleTimer = Math.max(0.0, this.muzzleTimer - dt);
      if (this.flashLightTimer > 0.0) this.flashLightTimer = Math.max(0.0, this.flashLightTimer - dt);

      // Track live movement/stance state for CS2 spread & landing settle penalty
      if (!this.wasOnGround && onGround) this.landingPenalty = 0.18;
      else if (this.landingPenalty > 0.0) this.landingPenalty = Math.max(0.0, this.landingPenalty - dt);
      this.wasOnGround = Boolean(onGround);
      this.lastHorizSpeed = horizSpeed;
      this.lastVelY = velY;
      this.lastOnGround = Boolean(onGround);
      this.lastCrouching = Boolean(crouching);
      this.lastSliding = Boolean(sliding);

      // CS2 spray pattern, fire bloom & camera viewpunch recovery (manual mouse pull-down cancels recovery!)
      this.timeSinceShot += dt;
      const bal = this._getBallistics();
      if (mouseDy > 0.0 && this.camRecoilPitch > 0.0) {
        this.camRecoilPitch = Math.max(0.0, this.camRecoilPitch - mouseDy * 0.0022);
      }
      if (Math.abs(mouseDx) > 0.0 && Math.abs(this.camRecoilYaw) > 1e-5) {
        if ((mouseDx < 0.0 && this.camRecoilYaw > 0.0) || (mouseDx > 0.0 && this.camRecoilYaw < 0.0)) {
          const stepC = Math.abs(mouseDx) * 0.0020;
          this.camRecoilYaw = (this.camRecoilYaw > 0.0)
            ? Math.max(0.0, this.camRecoilYaw - stepC)
            : Math.min(0.0, this.camRecoilYaw + stepC);
        }
      }

      const resetDelay = Math.max(0.085, (this.activeSpec.fire_cooldown || 0.10) * 1.18);
      const tau = bal.recovery_tau * (crouching ? 0.80 : 1.0) * (1.0 - 0.20 * this.adsT);
      if (this.timeSinceShot >= resetDelay) {
        const kR = 1.0 - Math.exp(-dt / Math.max(tau * 0.55, 0.06));
        this.fireBloom *= (1.0 - kR);
        this.sprayPitch *= (1.0 - kR);
        this.sprayYaw *= (1.0 - kR);
        this.sprayIndex = Math.max(0.0, this.sprayIndex - dt / Math.max(tau * 0.16, 0.025));
        if (this.fireBloom < 1e-4) this.fireBloom = 0.0;
        if (Math.abs(this.sprayPitch) < 1e-4) this.sprayPitch = 0.0;
        if (Math.abs(this.sprayYaw) < 1e-4) this.sprayYaw = 0.0;
        if (Math.abs(this.camRecoilPitch) > 1e-5 || Math.abs(this.camRecoilYaw) > 1e-5) {
          const kCam = 1.0 - Math.exp(-dt * 9.5);
          const dpBack = this.camRecoilPitch * kCam;
          const dyBack = this.camRecoilYaw * kCam;
          this.camRecoilPitch -= dpBack;
          this.camRecoilYaw -= dyBack;
          this.pendingCamPitch -= dpBack;
          this.pendingCamRoll -= dyBack;
        }
      } else {
        this.fireBloom *= Math.exp(-dt * 0.65);
      }

      this.currentSpread = this.computeCurrentSpread();
      const visRad = this.currentSpread + Math.hypot(this.sprayYaw, this.sprayPitch) * 0.42;
      const tgtPx = (this.hasWeapon && this.equipped) ? Math.min(46.0, Math.max(0.0, (visRad - 0.0012) * 340.0)) : 0.0;
      this.crosshairSpreadPx += (tgtPx - this.crosshairSpreadPx) * (1.0 - Math.exp(-dt * 18.0));

      // CS2-style Weapon Inspect [F] update (holding [F] pauses gracefully at showcase angle!)
      this.inspectHold = Boolean(inspectHeld);
      if (this.inspecting) {
        if (!this.equipped || adsHeld || this.reloading || this.pumpActive || this.revCocking || sliding || kicking || (sprinting && horizSpeed > 5.2)) {
          this.cancelInspect();
        } else {
          const uIn = this.inspectTimer / Math.max(this.inspectDuration, 1e-4);
          if (!this.evInsTurn && uIn >= 0.42) {
            this.evInsTurn = true;
            this.sfxQueue.push("inspect_turn");
          }
          if (!this.evInsEnd && uIn >= 0.78) {
            this.evInsEnd = true;
            this.sfxQueue.push("inspect");
          }
          if (!(inspectHeld && uIn >= 0.42 && uIn <= 0.50)) {
            this.inspectTimer += dt * 0.86;
          }
          if (this.inspectTimer >= this.inspectDuration) {
            this.inspecting = false;
            this.inspectTimer = 0.0;
          }
        }
      }
      const inTgt = this.inspecting ? 1.0 : 0.0;
      this.inspectBlend += (inTgt - this.inspectBlend) * (1.0 - Math.exp(-dt * (this.inspecting ? 10.0 : 16.0)));
      if (!this.inspecting && this.inspectBlend < 0.005) {
        this.inspectBlend = 0.0;
        this.inspectTimer = 0.0;
      }

      // v14: Weapon-specific 5-mode Bodycam Mag Check ([Hold R]):
      //  - mag_empty:     Act 1 pull out empty 3D mag & re-insert -> Act 2 rack bolt/slide 3x!
      //  - double_barrel: Break/lower barrels open to reveal twin 12ga chambers, then snap shut!
      //  - revolver:      Swing out cylinder, inspect, dynamically spin (3x rev_cock) & flick shut!
      //  - mag_loaded:    Pull mag out, tilt & inspect brass, slap home!
      //  - shotgun:       Check receiver port / tube (rack bolt 2x if empty)!
      if (this.magCheckLabelTimer > 0.0) {
        this.magCheckLabelTimer = Math.max(0.0, this.magCheckLabelTimer - dt);
        if (this.magCheckLabelTimer <= 0.0) this.magCheckLabel = "";
      }
      if (this.magChecking) {
        if (!this.equipped || adsHeld || this.reloading || this.pumpActive || this.revCocking || this.suppAnimActive || sliding || kicking || (sprinting && horizSpeed > 5.2)) {
          this.cancelMagCheck();
        } else {
          const uMc = this.magCheckTimer / Math.max(this.magCheckDuration, 1e-4);
          const modeMc = this.magCheckMode || "mag_loaded";
          if (modeMc === "double_barrel") {
            if (!this.mcEv1 && uMc >= 0.14) {
              this.mcEv1 = true; this.sfxQueue.push("db_open");
              this._addCamJolt(-0.12, 0.06, -0.14);
              this._spawnBreechSmoke(vmToWorld, 6);
            }
            if (!this.mcEv2 && uMc >= 0.44) { this.mcEv2 = true; this.sfxQueue.push("inspect_turn"); }
            if (!this.mcEv3 && uMc >= 0.78) {
              this.mcEv3 = true; this.sfxQueue.push("db_close");
              this.recoilPitch += 0.045; this.recoilY += 0.012;
              this._addCamJolt(0.20, -0.06, 0.18);
            }
          } else if (modeMc === "revolver") {
            if (!this.mcEv1 && uMc >= 0.12) {
              this.mcEv1 = true; this.sfxQueue.push("rev_open");
              this._addCamJolt(0.08, -0.08, -0.16);
              this._spawnBreechSmoke(vmToWorld, 4);
            }
            if (!this.mcEv2 && uMc >= 0.46) { this.mcEv2 = true; this.sfxQueue.push("rev_cock"); this._addCamJolt(0.03, 0.04, 0.08); }
            if (!this.mcEv3 && uMc >= 0.56) { this.mcEv3 = true; this.sfxQueue.push("rev_cock"); }
            if (!this.mcEv4 && uMc >= 0.65) { this.mcEv4 = true; this.sfxQueue.push("rev_cock"); }
            if (!this.mcEv5 && uMc >= 0.78) {
              this.mcEv5 = true; this.sfxQueue.push("rev_close");
              this.recoilRoll -= 0.058;
              this._addCamJolt(0.14, 0.10, -0.28);
            }
            const spinSpd = s5(0.42, 0.52, uMc) * (1.0 - s5(0.72, 0.80, uMc));
            this.mcRevSpin += dt * 19.5 * spinSpd;
          } else if (modeMc === "mag_empty") {
            // ACT 1: Pull out empty mag & re-seat it; ACT 2: Rack bolt/slide 3x!
            if (!this.mcEv1 && uMc >= 0.09) {
              this.mcEv1 = true; this.sfxQueue.push("mag_out");
              this._addCamJolt(-0.10, 0.06, -0.14);
            }
            if (!this.mcEv2 && uMc >= 0.35) {
              this.mcEv2 = true; this.sfxQueue.push("mag_in");
              this.recoilY += 0.012;
              this._addCamJolt(0.15, -0.05, 0.14);
            }
            if (!this.mcEv3 && uMc >= 0.48) { this.mcEv3 = true; this.sfxQueue.push(this.isM110 ? "bolt_cycle" : "slide_back"); this._addCamJolt(0.07, -0.06, 0.11); }
            if (!this.mcEv4 && uMc >= 0.58) { this.mcEv4 = true; if (!this.isM110) this.sfxQueue.push("slide_fwd"); }
            if (!this.mcEv5 && uMc >= 0.65) { this.mcEv5 = true; this.sfxQueue.push(this.isM110 ? "bolt_cycle" : "slide_back"); this._addCamJolt(0.07, -0.06, 0.11); }
            if (!this.mcEv6 && uMc >= 0.75) { this.mcEv6 = true; if (!this.isM110) this.sfxQueue.push("slide_fwd"); }
            if (!this.mcEv7 && uMc >= 0.81) { this.mcEv7 = true; this.sfxQueue.push(this.isM110 ? "bolt_cycle" : "slide_back"); this._addCamJolt(0.08, -0.06, 0.12); }
          } else if (modeMc === "shotgun") {
            if (this.magCheckEmpty) {
              if (!this.mcEv1 && uMc >= 0.16) { this.mcEv1 = true; this.sfxQueue.push("pump_back"); this._addCamJolt(0.08, -0.06, 0.12); }
              if (!this.mcEv2 && uMc >= 0.34) { this.mcEv2 = true; this.sfxQueue.push("pump_fwd"); this._addCamJolt(-0.07, 0.04, -0.08); }
              if (!this.mcEv3 && uMc >= 0.54) { this.mcEv3 = true; this.sfxQueue.push("pump_back"); this._addCamJolt(0.08, -0.06, 0.12); }
              if (!this.mcEv4 && uMc >= 0.74) { this.mcEv4 = true; this.sfxQueue.push("pump_fwd"); this._addCamJolt(-0.07, 0.04, -0.08); }
            } else {
              if (!this.mcEv1 && uMc >= 0.24) { this.mcEv1 = true; this.sfxQueue.push("pump_back"); this._addCamJolt(-0.025, 0.018, -0.035); }
              if (!this.mcEv2 && uMc >= 0.59) { this.mcEv2 = true; this.sfxQueue.push("pump_fwd"); this._addCamJolt(0.022, -0.016, 0.030); }
            }
          } else {
            // mag_loaded
            if (!this.mcEv1 && uMc >= 0.14) { this.mcEv1 = true; this.sfxQueue.push("mag_out"); this._addCamJolt(-0.10, 0.06, -0.14); }
            if (!this.mcEv2 && uMc >= 0.44) { this.mcEv2 = true; this.sfxQueue.push("inspect_turn"); }
            if (!this.mcEv3 && uMc >= 0.78) {
              this.mcEv3 = true; this.sfxQueue.push("mag_in");
              this.recoilY += 0.014;
              this._addCamJolt(0.18, -0.05, 0.15);
            }
          }
          const holdWindow = (modeMc === "double_barrel")
            ? (uMc >= 0.42 && uMc <= 0.50)
            : ((modeMc === "revolver") ? (uMc >= 0.30 && uMc <= 0.38) : (modeMc === "mag_loaded" && uMc >= 0.50 && uMc <= 0.54));
          if (!(magCheckHeld && holdWindow)) {
            this.magCheckTimer += dt * 0.82;
          }
          if (this.magCheckTimer >= this.magCheckDuration) {
            this.magChecking = false;
            this.magCheckTimer = 0.0;
          }
        }
      }
      const mcTgt = this.magChecking ? 1.0 : 0.0;
      this.magCheckBlend += (mcTgt - this.magCheckBlend) * (1.0 - Math.exp(-dt * (this.magChecking ? 13.0 : 16.0)));
      if (!this.magChecking && this.magCheckBlend < 0.005) {
        this.magCheckBlend = 0.0;
        this.magCheckTimer = 0.0;
        this.mcRevSpin = 0.0;
      }

      // v14: Non-accumulating damped-spring camera jolts + continuous dynamic camera motion
      // during reloads, Bodycam mag-checks, suppressor threading & bolt/pump cycling!
      const kSp = 92.0, cSp = 15.0;
      this.camJoltPitchV += (-kSp * this.camJoltPitch - cSp * this.camJoltPitchV) * dt;
      this.camJoltYawV   += (-kSp * this.camJoltYaw   - cSp * this.camJoltYawV)   * dt;
      this.camJoltRollV  += (-kSp * this.camJoltRoll  - cSp * this.camJoltRollV)  * dt;
      this.camJoltPitch += this.camJoltPitchV * dt;
      this.camJoltYaw   += this.camJoltYawV   * dt;
      this.camJoltRoll  += this.camJoltRollV  * dt;

      let bcR = 0.0, bcP = 0.0, bcY = 0.0;
      if (this.magCheckBlend > 0.01) {
        const uMc = clamp01(this.magCheckTimer / Math.max(this.magCheckDuration, 1e-4));
        const wMc = this.magCheckBlend;
        const modeMc = this.magCheckMode || "mag_loaded";
        if (modeMc === "double_barrel") {
          const env = s5(0.0, 0.22, uMc) * (1.0 - s5(0.78, 1.0, uMc));
          bcR += -0.044 * env * wMc;
          bcP += (-0.058 * env + 0.010 * Math.sin(this.elapsed * 3.8) * env) * wMc;
          bcY += 0.026 * env * wMc;
        } else if (modeMc === "revolver") {
          const env = s5(0.0, 0.18, uMc) * (1.0 - s5(0.82, 1.0, uMc));
          const spinVib = Math.sin(uMc * Math.PI * 10.0) * s5(0.44, 0.54, uMc) * (1.0 - s5(0.70, 0.80, uMc));
          const snapF = Math.sin(s5(0.74, 0.88, uMc) * Math.PI);
          bcR += (-0.052 * env + 0.014 * spinVib - 0.068 * snapF) * wMc;
          bcP += (-0.028 * env + 0.024 * snapF) * wMc;
          bcY += (-0.032 * env + 0.038 * snapF) * wMc;
        } else if (modeMc === "mag_empty") {
          const act1 = s5(0.0, 0.12, uMc) * (1.0 - s5(0.36, 0.44, uMc));
          const act2 = s5(0.42, 0.50, uMc) * (1.0 - s5(0.88, 1.0, uMc));
          const jolt = Math.sin(uMc * Math.PI * 14.0) * act2;
          bcR += (-0.056 * act1 + 0.052 * act2 + 0.026 * jolt) * wMc;
          bcP += (-0.038 * act1 - 0.024 * act2 + 0.016 * Math.abs(jolt)) * wMc;
          bcY += (0.034 * act1 - 0.030 * act2) * wMc;
        } else if (modeMc === "shotgun" && this.magCheckEmpty) {
          const env = s5(0.0, 0.18, uMc) * (1.0 - s5(0.84, 1.0, uMc));
          const jolt = Math.sin(uMc * Math.PI * 8.0) * env;
          bcR += (0.044 * env + 0.022 * jolt) * wMc;
          bcP += (-0.028 * env + 0.014 * Math.abs(jolt)) * wMc;
        } else {
          const swBc = Math.sin(this.elapsed * 4.8) * 0.009 + Math.sin(this.elapsed * 2.3) * 0.006;
          bcR += (-0.052 * Math.sin(uMc * Math.PI) + swBc) * wMc;
          bcP += (-0.036 * Math.sin(uMc * Math.PI) + 0.008 * Math.cos(this.elapsed * 3.9)) * wMc;
          bcY += (0.028 * Math.sin(uMc * Math.PI)) * wMc;
        }
      }
      if (this.reloading) {
        const uR = clamp01(this.reloadTimer / Math.max(this.reloadDuration, 1e-4));
        if (this.reloadType === "empty") {
          const tilt = s5(0.0, 0.16, uR) * (1.0 - s5(0.86, 1.0, uR));
          const magOutW = Math.sin(s5(0.10, 0.28, uR) * Math.PI);
          const magSlap = Math.sin(s5(0.56, 0.69, uR) * Math.PI);
          const rackW = Math.sin(s5(0.70, 0.87, uR) * Math.PI);
          bcR += -0.068 * tilt - 0.032 * magOutW + 0.048 * magSlap + 0.052 * rackW;
          bcP += -0.042 * tilt - 0.018 * magOutW + 0.046 * magSlap - 0.024 * rackW;
          bcY += 0.040 * tilt - 0.034 * rackW;
        } else if (this.reloadType === "tactical") {
          const tilt = s5(0.0, 0.18, uR) * (1.0 - s5(0.86, 1.0, uR));
          const magOutW = Math.sin(s5(0.16, 0.34, uR) * Math.PI);
          const magSlap = Math.sin(s5(0.68, 0.83, uR) * Math.PI);
          bcR += -0.060 * tilt - 0.028 * magOutW + 0.046 * magSlap;
          bcP += -0.036 * tilt + 0.042 * magSlap;
          bcY += 0.034 * tilt;
        } else if (this.reloadType === "break_action") {
          if (this.dbReloadPhase === "open") {
            const t = s5(0.0, 0.65, uR);
            bcR += -0.056 * t; bcP += -0.052 * t; bcY += 0.032 * t;
          } else if (this.dbReloadPhase === "insert") {
            const ins = Math.sin(s5(0.45, 0.78, uR) * Math.PI);
            bcR += -0.052 + 0.024 * ins; bcP += -0.048 - 0.018 * ins; bcY += 0.028;
          } else {
            const t = 1.0 - s5(0.20, 0.90, uR);
            const snap = Math.sin(s5(0.44, 0.72, uR) * Math.PI);
            bcR += -0.052 * t + 0.058 * snap; bcP += -0.048 * t + 0.056 * snap; bcY += 0.028 * t - 0.025 * snap;
          }
        } else if (this.reloadType === "revolver") {
          if (this.revReloadPhase === "open_dump") {
            const dmp = Math.sin(s5(0.24, 0.60, uR) * Math.PI);
            bcR += -0.058 * s5(0.0, 0.35, uR) - 0.035 * dmp;
            bcP += 0.058 * dmp - 0.025 * s5(0.60, 1.0, uR);
            bcY += -0.034 * s5(0.0, 0.40, uR);
          } else if (this.revReloadPhase === "load") {
            const ins = Math.sin(s5(0.42, 0.74, uR) * Math.PI);
            bcR += -0.054 + 0.022 * ins; bcP += -0.034 - 0.016 * ins; bcY += -0.028;
          } else {
            const t = 1.0 - s5(0.18, 0.88, uR);
            const flk = Math.sin(s5(0.28, 0.60, uR) * Math.PI);
            bcR += -0.054 * t - 0.072 * flk; bcP += -0.034 * t + 0.042 * flk; bcY += -0.028 * t + 0.048 * flk;
          }
        } else if (this.reloadType === "shell") {
          const ins = Math.sin(s5(0.42, 0.74, uR) * Math.PI);
          bcR += 0.054 + 0.022 * ins; bcP += -0.032 - 0.016 * ins; bcY += -0.026;
        }
      }
      if (this.pumpActive) {
        const puC = clamp01(this.pumpTimer / Math.max(this.pumpDuration, 1e-4));
        const pw = Math.sin(puC * Math.PI);
        bcR += (this.isSniper ? 0.038 : 0.028) * pw;
        bcP += (this.isSniper ? -0.018 : -0.012) * pw;
      }
      if (this.suppAnimActive) {
        const suC = clamp01(this.suppAnimTimer / Math.max(this.suppAnimDuration, 1e-4));
        const envS = s5(0.0, 0.22, suC) * (1.0 - s5(0.80, 1.0, suC));
        const twS = Math.sin(suC * Math.PI * 6.0) * envS;
        bcR += -0.046 * envS + 0.016 * twS;
        bcP += -0.030 * envS;
        bcY += 0.028 * envS;
      }

      const kBc = 1.0 - Math.exp(-dt * 16.0);
      this.bodycamCamRoll  += (bcR + this.camJoltRoll  - this.bodycamCamRoll)  * kBc;
      this.bodycamCamPitch += (bcP + this.camJoltPitch - this.bodycamCamPitch) * kBc;
      this.bodycamCamYaw   += (bcY + this.camJoltYaw   - this.bodycamCamYaw)   * kBc;

      // v14: Screen-space VFX pulse decay & ULTRAKILL speed-lines during slide / high sprint!
      this.fxImpactPulse = Math.max(0.0, this.fxImpactPulse - dt * 2.8);
      const speedTgt = sliding ? 0.92 : ((sprinting && horizSpeed > 5.6 && onGround) ? 0.34 : 0.0);
      this.fxSpeedLines += (speedTgt - this.fxSpeedLines) * (1.0 - Math.exp(-dt * (speedTgt > this.fxSpeedLines ? 12.0 : 7.0)));

      // MP5-S Detachable Suppressor Screw / Unscrew Animation ([P])!
      if (this.suppAnimActive) {
        if (!this.equipped || !this.isMP5S) {
          this.suppAnimActive = false;
        } else {
          this.suppAnimTimer += dt;
          const su = this.suppAnimTimer / Math.max(this.suppAnimDuration, 1e-4);
          if (!this.suppEvSfx && su >= 0.14) {
            this.suppEvSfx = true;
            this.sfxQueue.push(this.suppTargetState ? "supp_attach" : "supp_detach");
          }
          if (su >= 0.56 && this.mp5Suppressed !== this.suppTargetState) {
            this.mp5Suppressed = this.suppTargetState;
          }
          if (su >= 1.0) {
            this.mp5Suppressed = this.suppTargetState;
            this.suppAnimActive = false;
            this.suppAnimTimer = 0.0;
          }
        }
      }

      const eqTarget = (this.hasWeapon && this.equipped) ? 1.0 : 0.0;
      const eqStep = dt / Math.max(this.DRAW_DUR, 0.05);
      if (this.equipT < eqTarget) this.equipT = Math.min(1.0, this.equipT + eqStep);
      else if (this.equipT > eqTarget) this.equipT = Math.max(0.0, this.equipT - eqStep);

      const adsBlock = this.reloading || sprinting || sliding || kicking || this.magChecking || this.suppAnimActive || (this.isSniper && this.pumpActive);
      const adsTarget = (adsHeld && this.hasWeapon && this.equipped && !adsBlock) ? 1.0 : 0.0;
      this.adsT += (adsTarget - this.adsT) * (1.0 - Math.exp(-dt * 14.0));

      // M110 SASS Optical Sniper Scope + walk-induced blur!
      const scTgt = (this.hasWeapon && this.equipped && this.isSniper && !adsBlock) ? s5(0.56, 0.96, this.adsT) : 0.0;
      this.scopeAmount += (scTgt - this.scopeAmount) * (1.0 - Math.exp(-dt * 18.0));
      if (this.scopeAmount < 0.004) this.scopeAmount = 0.0;
      let blurTgt = 0.0;
      if (this.scopeAmount > 0.01) {
        let wb = onGround ? clamp01((horizSpeed - 0.35) / 3.4) : 1.0;
        if (crouching && onGround) wb *= 0.65;
        blurTgt = clamp01(wb * 0.92 + (this.landingPenalty > 0.0 ? 0.45 : 0.0));
      }
      this.scopeBlur += (blurTgt - this.scopeBlur) * (1.0 - Math.exp(-dt * 10.0));

      const spTarget = (sprinting && !this.reloading && !adsHeld && !crouching && !sliding) ? 1.0 : 0.0;
      this.sprintBlend += (spTarget - this.sprintBlend) * (1.0 - Math.exp(-dt * 9.5));

      const kRec = Math.exp(-dt * 12.5);
      this.recoilPitch *= kRec;
      this.recoilRoll *= Math.exp(-dt * 14.0);
      this.recoilZ *= Math.exp(-dt * 16.0);
      this.recoilY *= Math.exp(-dt * 14.0);
      this.slideKick *= Math.exp(-dt * 24.0);
      this.triggerPull *= Math.exp(-dt * 19.0);
      this.hammerDrop *= Math.exp(-dt * 16.0);

      // Smoothly index revolver cylinder toward target chamber angle!
      this.revCylAngle += (this.revCylTarget - this.revCylAngle) * (1.0 - Math.exp(-dt * 22.0));

      // Single-action revolver pre-shot hammer cocking -> fires magnum round when cock completes!
      if (this.revCocking) {
        this.revCockTimer += dt;
        const cu = clamp01(this.revCockTimer / Math.max(this.revCockDuration, 1e-4));
        if (cu >= 1.0) {
          this.revCocking = false;
          this.revCockTimer = 0.0;
          if (this.hasWeapon && this.equipped && this.isRevolver && this.magAmmo > 0) {
            const ep = eyePos || (vmToWorld ? [vmToWorld[3], vmToWorld[7], vmToWorld[11]] : [0, 1.6, 0]);
            const ad = aimDir || (vmToWorld ? [-vmToWorld[2], -vmToWorld[6], -vmToWorld[10]] : [0, 0, -1]);
            this._executeFireShot(ep, ad, vmToWorld, colliders || []);
          }
        }
      }

      const swayLim = 0.026 * (1.0 - 0.65 * this.adsT);
      const tSx = Math.max(-swayLim, Math.min(swayLim, -mouseDx * 0.00055));
      const tSy = Math.max(-swayLim, Math.min(swayLim, mouseDy * 0.00048));
      const kSway = 1.0 - Math.exp(-dt * 11.0);
      this.swayX += (tSx - this.swayX) * kSway;
      this.swayY += (tSy - this.swayY) * kSway;

      if (onGround && horizSpeed > 0.35 && !sliding) {
        const sp = Math.min(1.0, Math.max(0.0, (horizSpeed - 4.5) / 2.6));
        const freq = crouching ? 6.2 : (7.8 + 2.8 * sp);
        this.bobPhase += dt * freq;
        const tgtAmp = (0.009 + 0.008 * sp) * (1.0 - 0.72 * this.adsT) * (crouching ? 0.65 : 1.0);
        this.bobAmp += (tgtAmp - this.bobAmp) * (1.0 - Math.exp(-dt * 10.0));
      } else {
        this.bobAmp *= Math.exp(-dt * 10.0);
      }

      // Shotgun post-shot bolt-racking / pump cooldown cycle!
      // On last shell (magAmmo === 0), bolt retracts & locks open so the empty chamber stays visibly exposed!
      if (this.pumpActive) {
        this.pumpTimer += dt;
        const pu = clamp01(this.pumpTimer / Math.max(this.pumpDuration, 1e-4));
        const isBoltSn = Boolean(this.activeSpec && this.activeSpec.bolt_cycle);
        if (!this.pumpEvBack && pu >= 0.22) {
          this.pumpEvBack = true;
          this.sfxQueue.push(isBoltSn ? "bolt_cycle" : "pump_back");
        }
        if (!this.pumpEvEject && pu >= 0.44) {
          this.pumpEvEject = true;
          this._spawnEjectedShell(vmToWorld, isBoltSn ? "pistol" : "shotgun");
        }
        if (this.magAmmo > 0) {
          if (!this.pumpEvFwd && pu >= 0.58) {
            this.pumpEvFwd = true;
            if (!isBoltSn) this.sfxQueue.push("pump_fwd");
            this.recoilZ += 0.010;
            this.recoilPitch -= 0.024;
          }
        } else {
          if (pu >= 0.50) this.slideLocked = true;
        }
        if (this.pumpTimer >= this.pumpDuration) {
          this.pumpActive = false;
          this.pumpTimer = 0.0;
        }
      }

      if (this.reloading) {
        this.reloadTimer += dt * 0.80;
        const u = clamp01(this.reloadTimer / Math.max(this.reloadDuration, 1e-4));
        if (this.reloadType === "break_action") {
          const phase = this.dbReloadPhase;
          if (phase === "open") {
            if (!this.dbEvOpen && u >= 0.24) {
              this.dbEvOpen = true;
              this.sfxQueue.push("db_open");
              this._addCamJolt(-0.14, 0.05, -0.16);
              this._spawnBreechSmoke(vmToWorld, 7);
            }
            if (!this.dbEvEject && u >= 0.62) {
              this.dbEvEject = true;
              this.sfxQueue.push("shell_eject");
              this._addCamJolt(0.10, -0.04, 0.08);
              for (let ei = 0; ei < this.dbEjectCount; ei++) {
                this._spawnDbEjectedShell(vmToWorld, ei);
              }
              this.spentShells = 0;
              this._saveToActive();
            }
            if (u >= 1.0) {
              if (this.dbReloadTarget > 0 && this.reserveAmmo > 0) {
                this.dbReloadPhase = "insert";
                this.reloadTimer = 0.0;
                this.reloadDuration = 0.54;
                this.evShellIn = false;
              } else {
                this.dbReloadPhase = "close";
                this.reloadTimer = 0.0;
                this.reloadDuration = 0.38;
                this.dbEvClose = false;
              }
            }
          } else if (phase === "insert") {
            if (!this.evShellIn && u >= 0.64) {
              this.evShellIn = true;
              if (this.magAmmo < this.MAG_CAPACITY && this.reserveAmmo > 0) {
                this.magAmmo += 1;
                this.reserveAmmo -= 1;
                this.dbInsertedCount += 1;
                this._saveToActive();
                this.sfxQueue.push("shell_insert");
                this.recoilPitch -= 0.020;
                this._addCamJolt(-0.08, 0.04, 0.08);
              }
            }
            if (u >= 1.0) {
              if (this.dbInsertedCount < this.dbReloadTarget && this.magAmmo < this.MAG_CAPACITY && this.reserveAmmo > 0) {
                this.reloadTimer = 0.0;
                this.evShellIn = false;
              } else {
                this.dbReloadPhase = "close";
                this.reloadTimer = 0.0;
                this.reloadDuration = 0.38;
                this.dbEvClose = false;
              }
            }
          } else {
            if (!this.dbEvClose && u >= 0.56) {
              this.dbEvClose = true;
              this.sfxQueue.push("db_close");
              this.recoilPitch += 0.052;
              this.recoilY += 0.012;
              this._addCamJolt(0.24, -0.06, 0.20);
            }
            if (u >= 1.0) {
              this.reloading = false;
              this.dbReloadPhase = "";
              this.reloadTimer = 0.0;
            }
          }
        } else if (this.reloadType === "revolver") {
          if (this.revReloadPhase === "open_dump") {
            if (!this.revEvOpen && u >= 0.14) {
              this.revEvOpen = true;
              this.sfxQueue.push("rev_open");
              this._addCamJolt(0.10, -0.06, -0.18);
              this._spawnBreechSmoke(vmToWorld, 5);
            }
            if (!this.revEvDump && u >= 0.42) {
              this.revEvDump = true;
              const spentIndices = [];
              for (let ci = 0; ci < 8; ci++) {
                if (this.revChambers[ci] === "spent") spentIndices.push(ci);
              }
              if (spentIndices.length > 0) {
                this._spawnRevDumpedShells(vmToWorld, spentIndices);
                for (const ci of spentIndices) this.revChambers[ci] = "empty";
                this.spentShells = 0;
                this.sfxQueue.push("shell_eject");
                this.recoilPitch += 0.045;
                this._addCamJolt(0.18, 0.04, -0.14);
                this._saveToActive();
              }
            }
            if (u >= 1.0) {
              const nextEmpty = this.revChambers.indexOf("empty");
              if (nextEmpty >= 0 && this.reserveAmmo > 0) {
                this.revReloadPhase = "load";
                this.revLoadChamber = nextEmpty;
                this.reloadTimer = 0.0;
                this.reloadDuration = 0.28;
                this.revEvRoundIn = false;
              } else {
                this.revReloadPhase = "close";
                this.reloadTimer = 0.0;
                this.reloadDuration = 0.40;
                this.revEvClose = false;
              }
            }
          } else if (this.revReloadPhase === "load") {
            if (!this.revEvRoundIn && u >= 0.58) {
              this.revEvRoundIn = true;
              const ci = this.revLoadChamber;
              if (ci >= 0 && ci < 8 && this.revChambers[ci] === "empty" && this.reserveAmmo > 0) {
                this.revChambers[ci] = "live";
                this.magAmmo = Math.min(this.MAG_CAPACITY, this.magAmmo + 1);
                this.reserveAmmo -= 1;
                this.sfxQueue.push("rev_load");
                this.recoilY += 0.008;
                this.recoilPitch -= 0.018;
                this._addCamJolt(-0.07, 0.03, 0.07);
                this._saveToActive();
              }
            }
            if (u >= 1.0) {
              const nextEmpty = this.revChambers.indexOf("empty");
              if (!this.reloadInterrupt && nextEmpty >= 0 && this.reserveAmmo > 0) {
                this.revLoadChamber = nextEmpty;
                this.reloadTimer = 0.0;
                this.reloadDuration = 0.28;
                this.revEvRoundIn = false;
              } else {
                this.revReloadPhase = "close";
                this.revLoadChamber = -1;
                this.reloadTimer = 0.0;
                this.reloadDuration = 0.40;
                this.revEvClose = false;
              }
            }
          } else {
            if (!this.revEvClose && u >= 0.42) {
              this.revEvClose = true;
              this.sfxQueue.push("rev_close");
              this.recoilRoll -= 0.068;
              this.revCylTarget += Math.PI * 0.5;
              this._addCamJolt(0.16, 0.12, -0.30);
            }
            if (u >= 1.0) {
              this.reloading = false;
              this.revReloadPhase = "";
              this.revLoadChamber = -1;
              this.reloadInterrupt = false;
              this.reloadTimer = 0.0;
              this._saveToActive();
            }
          }
        } else if (this.reloadType === "shell") {
          if (this.shellReloadPhase === "chamber") {
            if (!this.evShellIn && u >= 0.51) {
              this.evShellIn = true;
              if (this.magAmmo < this.MAG_CAPACITY && this.reserveAmmo > 0) {
                this.magAmmo += 1;
                this.reserveAmmo -= 1;
                this._saveToActive();
                this.sfxQueue.push("shell_insert");
              }
            }
            if (!this.relEvFwd && u >= 0.76) {
              this.relEvFwd = true;
              this.slideLocked = false;
              this._saveToActive();
              this.sfxQueue.push("pump_fwd");
            }
            if (u >= 1.0) {
              if (this.chamberThenLoad && !this.reloadInterrupt && this.magAmmo < this.MAG_CAPACITY && this.reserveAmmo > 0) {
                this.shellReloadPhase = "insert";
                this.reloadTimer = 0.0;
                this.reloadDuration = 0.72;
                this.evShellIn = false;
              } else {
                this.reloading = false;
                this.shellReloadPhase = "";
                this.reloadInterrupt = false;
                this.reloadTimer = 0.0;
              }
            }
          } else if (this.shellReloadPhase === "enter") {
            if (u >= 1.0) {
              if (this.reloadInterrupt && this.magAmmo > 0) {
                this.shellReloadPhase = "exit";
                this.reloadDuration = 0.48;
              } else {
                this.shellReloadPhase = "insert";
                this.reloadDuration = 0.72;
                this.evShellIn = false;
              }
              this.reloadTimer = 0.0;
            }
          } else if (this.shellReloadPhase === "insert") {
            if (!this.evShellIn && u >= 0.62) {
              this.evShellIn = true;
              if (this.magAmmo < this.MAG_CAPACITY && this.reserveAmmo > 0) {
                this.magAmmo += 1;
                this.reserveAmmo -= 1;
                this._saveToActive();
                this.sfxQueue.push("shell_insert");
              }
            }
            if (u >= 1.0) {
              const canMore = !this.reloadInterrupt && this.magAmmo < this.MAG_CAPACITY && this.reserveAmmo > 0;
              if (canMore) {
                this.reloadTimer = 0.0;
                this.evShellIn = false;
              } else {
                this.shellReloadPhase = "exit";
                this.reloadTimer = 0.0;
                this.reloadDuration = 0.46;
              }
            }
          } else {
            if (u >= 1.0) {
              this.reloading = false;
              this.shellReloadPhase = "";
              this.reloadInterrupt = false;
              this.reloadTimer = 0.0;
            }
          }
        } else if (this.reloadType === "empty") {
          if (!this.relEvOut && u >= 0.17) {
            this.relEvOut = true;
            this.sfxQueue.push("mag_out");
            this._spawnWorldMag(vmToWorld, false, colliders);
            this._addCamJolt(-0.14, 0.08, -0.20);
          }
          if (!this.relEvIn && u >= 0.61) {
            this.relEvIn = true;
            this.sfxQueue.push("mag_in");
            this.recoilY += 0.022;
            this.recoilPitch += 0.095;
            this._addCamJolt(0.24, -0.07, 0.22);
          }
          if (!this.relEvBack && u >= 0.72) {
            this.relEvBack = true;
            this.sfxQueue.push("slide_back");
            this._addCamJolt(0.11, -0.08, 0.16);
          }
          if (!this.relEvFwd && u >= 0.81) {
            this.relEvFwd = true;
            this.slideLocked = false;
            this.sfxQueue.push("slide_fwd");
            this.recoilZ += 0.018;
            this.recoilPitch -= 0.055;
            this._addCamJolt(-0.16, 0.06, -0.14);
            if (!this.relCommitted) {
              const take = Math.min(this.MAG_CAPACITY - this.magAmmo, this.reserveAmmo);
              this._applyMagReload(take);
              this.relCommitted = true;
            }
          }
        } else {
          if (!this.relEvOut && u >= 0.24) {
            this.relEvOut = true;
            this.sfxQueue.push("mag_out");
            this._spawnWorldMag(vmToWorld, true, colliders);
            this._addCamJolt(-0.13, 0.07, -0.18);
          }
          if (!this.relEvIn && u >= 0.74) {
            this.relEvIn = true;
            this.sfxQueue.push("mag_in");
            this.recoilY += 0.022;
            this.recoilPitch += 0.098;
            this._addCamJolt(0.22, -0.06, 0.20);
            if (!this.relCommitted) {
              const take = Math.min(this.MAG_CAPACITY - this.magAmmo, this.reserveAmmo);
              this._applyMagReload(take);
              this.relCommitted = true;
            }
          }
        }
        if (this.reloadType !== "shell" && this.reloadType !== "break_action" && this.reloadType !== "revolver") {
          if (this.reloadTimer >= this.reloadDuration) {
            if (!this.relCommitted) {
              const take = Math.min(this.MAG_CAPACITY - this.magAmmo, this.reserveAmmo);
              this._applyMagReload(take);
              this.relCommitted = true;
            }
            this.slideLocked = false;
            this.reloading = false;
            this.reloadTimer = 0.0;
            this._saveToActive();
          } else if (this.relCommitted) {
            this._saveToActive();
          }
        }
      }

      // Bulletproof physics for all thrown pistols (Colt M1911 & Glock-17)
      for (let i = 0; i < this.worldPickups.length; i++) {
        const wp = this.worldPickups[i];
        if (!wp.grounded) {
          if (stepWorldProp(wp, dt, colliders, 0.028, 0.085, [0.028, 0.095, 0.145], 0.26, 0.55)) {
            this.sfxQueue.push("mag_drop");
          }
        }
      }

      // Bulletproof physics for persistent ejected 3D magazines lying in the arena
      for (let i = 0; i < this.worldMags.length; i++) {
        const wm = this.worldMags[i];
        if (!wm.grounded) {
          if (stepWorldProp(wm, dt, colliders, 0.017, 0.048, [0.017, 0.078, 0.045], 0.30, 0.58)) {
            this.sfxQueue.push("mag_drop");
          }
        }
      }

      for (let i = this.shells.length - 1; i >= 0; i--) {
        const sh = this.shells[i];
        sh.life -= dt;
        if (sh.life <= 0.0) { this.shells.splice(i, 1); continue; }
        if (sh.shellType === "rev_spent") {
          if (!sh.grounded) {
            sh.vel[1] -= 11.8 * dt;
            sh.pos[0] += sh.vel[0] * dt;
            sh.pos[1] += sh.vel[1] * dt;
            sh.pos[2] += sh.vel[2] * dt;
            sh.rot[0] += sh.avel[0] * dt;
            sh.rot[1] += sh.avel[1] * dt;
            sh.rot[2] += sh.avel[2] * dt;
            const [sx, sy, sz] = sh.pos;
            let supp = 0.0;
            if (colliders) {
              for (let c = 0; c < colliders.length; c++) {
                const box = colliders[c];
                if (sx >= box[0] && sx <= box[3] && sz >= box[2] && sz <= box[5]) {
                  if (box[4] > supp && box[4] <= sy + 0.09) supp = box[4];
                }
              }
            }
            if (sh.pos[1] <= supp + 0.012) {
              sh.pos[1] = supp + 0.012;
              if (sh.vel[1] < -0.65 && (sh.bounces || 0) < 3) {
                sh.bounces = (sh.bounces || 0) + 1;
                sh.vel[1] = -sh.vel[1] * 0.36;
                sh.vel[0] *= 0.55; sh.vel[2] *= 0.55;
                sh.avel[0] *= 0.52; sh.avel[1] *= 0.52; sh.avel[2] *= 0.52;
              } else {
                sh.vel = [0, 0, 0]; sh.avel = [0, 0, 0];
                sh.rot[0] = 0.0;
                sh.grounded = true;
              }
            }
          }
          continue;
        }
        sh.vel[1] -= 9.8 * dt;
        sh.pos[0] += sh.vel[0] * dt;
        sh.pos[1] += sh.vel[1] * dt;
        sh.pos[2] += sh.vel[2] * dt;
        if (sh.pos[1] < 0.015) {
          sh.pos[1] = 0.015;
          sh.vel[1] = -sh.vel[1] * 0.38;
          sh.vel[0] *= 0.68;
          sh.vel[2] *= 0.68;
          sh.avel[0] *= 0.55; sh.avel[1] *= 0.55; sh.avel[2] *= 0.55;
        }
        sh.rot[0] += sh.avel[0] * dt;
        sh.rot[1] += sh.avel[1] * dt;
        sh.rot[2] += sh.avel[2] * dt;
      }

      for (let i = this.sparks.length - 1; i >= 0; i--) {
        const sp = this.sparks[i];
        sp.life -= dt;
        if (sp.life <= 0.0) { this.sparks.splice(i, 1); continue; }
        sp.vel[1] -= 7.2 * dt;
        sp.pos[0] += sp.vel[0] * dt;
        sp.pos[1] += sp.vel[1] * dt;
        sp.pos[2] += sp.vel[2] * dt;
        sp.roll += dt * 12.0;
      }

      for (let i = this.bulletHoles.length - 1; i >= 0; i--) {
        const bh = this.bulletHoles[i];
        bh.life -= dt;
        if (bh.life <= 0.0) this.bulletHoles.splice(i, 1);
      }

      // v14: Update 3D bullet tracers & 3D particles (muzzle smoke, concrete debris, embers, dust, sparks!)
      for (let i = this.tracers.length - 1; i >= 0; i--) {
        const tr = this.tracers[i];
        tr.life -= dt;
        if (tr.life <= 0.0) this.tracers.splice(i, 1);
      }
      for (let i = this.particles.length - 1; i >= 0; i--) {
        const pt = this.particles[i];
        pt.life -= dt;
        if (pt.life <= 0.0) { this.particles.splice(i, 1); continue; }
        pt.vel[1] -= pt.gravity * dt;
        const damp = Math.exp(-pt.drag * dt);
        pt.vel[0] *= damp;
        pt.vel[1] *= damp;
        pt.vel[2] *= damp;
        pt.pos[0] += pt.vel[0] * dt;
        pt.pos[1] += pt.vel[1] * dt;
        pt.pos[2] += pt.vel[2] * dt;
        if (pt.pos[1] < 0.025) {
          pt.pos[1] = 0.025;
          if (pt.vel[1] < 0.0) {
            pt.vel[1] = -pt.vel[1] * 0.28;
            pt.vel[0] *= 0.65;
            pt.vel[2] *= 0.65;
          }
        }
        pt.rot += pt.avel * dt;
      }
    }

    _pumpPoseOffsets() {
      if (!this.pumpActive) {
        return { gOff: [0, 0, 0], gRot: [0, 0, 0], slideExtra: 0.0 };
      }
      const pu = clamp01(this.pumpTimer / Math.max(this.pumpDuration, 1e-4));
      const back = s5(0.16, 0.48, pu);
      let fwd = 0.0;
      let slideExtra = 0.0;
      if (this.magAmmo === 0) {
        // Last shot: bolt retracts and locks open so the empty chamber stays visibly exposed!
        slideExtra = this.SLIDE_TRAVEL * back;
      } else {
        fwd = s5(0.54, 0.88, pu);
        slideExtra = this.SLIDE_TRAVEL * back * (1.0 - fwd);
      }
      const tilt = s5(0.10, 0.34, pu) * (1.0 - s5(0.76, 1.0, pu));
      const pullPulse = back * (1.0 - fwd) * (1.0 - s5(0.76, 1.0, pu));
      return {
        gOff: [-0.014 * tilt, -0.012 * tilt - 0.006 * pullPulse, 0.018 * pullPulse],
        gRot: [-0.05 * tilt, 0.06 * tilt - 0.04 * pullPulse, -0.18 * tilt - 0.06 * pullPulse],
        slideExtra,
      };
    }


    _reloadPoseOffsets() {
      if (!this.reloading) {
        return { gOff: [0, 0, 0], gRot: [0, 0, 0], slideExtra: 0.0 };
      }
      const u = clamp01(this.reloadTimer / Math.max(this.reloadDuration, 1e-4));
      if (this.reloadType === "break_action") {
        const phase = this.dbReloadPhase;
        const openPos = [-0.012, -0.010, -0.056];
        const openRot = [0.24, -0.04, -0.28];
        if (phase === "open") {
          const t = s5(0.0, 0.42, u);
          const ejectKick = Math.sin(s5(0.56, 0.84, u) * Math.PI);
          return {
            gOff: [openPos[0] * t, openPos[1] * t - 0.010 * ejectKick, openPos[2] * t + 0.014 * ejectKick],
            gRot: [openRot[0] * t, openRot[1] * t + 0.09 * ejectKick, openRot[2] * t - 0.05 * ejectKick],
            slideExtra: 0.0,
          };
        } else if (phase === "insert") {
          const bump = Math.sin(s5(0.54, 0.80, u) * Math.PI);
          return {
            gOff: [openPos[0], openPos[1] - 0.006 * bump, openPos[2] - 0.008 * bump],
            gRot: [openRot[0], openRot[1] - 0.035 * bump, openRot[2] + 0.03 * bump],
            slideExtra: 0.0,
          };
        } else {
          const t = s5(0.24, 0.92, u);
          const snapBump = Math.sin(s5(0.46, 0.72, u) * Math.PI);
          return {
            gOff: [openPos[0] * (1.0 - t), openPos[1] * (1.0 - t) + 0.016 * snapBump, openPos[2] * (1.0 - t) + 0.012 * snapBump],
            gRot: [openRot[0] * (1.0 - t), openRot[1] * (1.0 - t) + 0.14 * snapBump, openRot[2] * (1.0 - t) + 0.06 * snapBump],
            slideExtra: 0.0,
          };
        }
      }
      if (this.reloadType === "revolver") {
        const loadPos = [-0.012, 0.074, -0.008];
        const loadRot = [-0.26, -0.06, -0.34];
        if (this.revReloadPhase === "open_dump") {
          if (u < 0.64) {
            const op = s5(0.0, 0.26, u);
            const dumpUp = s5(0.24, 0.44, u) * (1.0 - s5(0.50, 0.64, u));
            return {
              gOff: [-0.022 * op, 0.048 * op + 0.044 * dumpUp, -0.028 * op + 0.022 * dumpUp],
              gRot: [-0.26 * op, 0.14 * op + 0.56 * dumpUp, -0.34 * op - 0.14 * dumpUp],
              slideExtra: 0.0,
            };
          }
          const t = s5(0.64, 1.0, u);
          return {
            gOff: lerp3([-0.022, 0.048, -0.028], loadPos, t),
            gRot: lerp3([-0.26, 0.14, -0.34], loadRot, t),
            slideExtra: 0.0,
          };
        }
        if (this.revReloadPhase === "load") {
          const seat = Math.sin(s5(0.46, 0.72, u) * Math.PI);
          return {
            gOff: [loadPos[0], loadPos[1] - 0.005 * seat, loadPos[2] - 0.006 * seat],
            gRot: [loadRot[0] + 0.02 * seat, loadRot[1] - 0.03 * seat, loadRot[2] + 0.03 * seat],
            slideExtra: 0.0,
          };
        }
        const t = s5(0.0, 1.0, u);
        const snap = Math.sin(s5(0.18, 0.56, u) * Math.PI);
        const p = lerp3(loadPos, [0, 0, 0], t);
        const r = lerp3(loadRot, [0, 0, 0], t);
        return {
          gOff: [p[0] + 0.018 * snap, p[1] + 0.016 * snap, p[2]],
          gRot: [r[0], r[1] + 0.10 * snap, r[2] + 0.28 * snap],
          slideExtra: 0.0,
        };
      }
      if (this.reloadType === "shell") {
        const st = this.SLIDE_TRAVEL;
        const phase = this.shellReloadPhase;
        const loadPos = [-0.032, 0.088, 0.010];
        const loadRot = [0.18, 0.24, -1.02];
        const chPos   = [-0.018, -0.022, 0.016];
        const chRot   = [-0.28, 0.12, 1.18];

        if (phase === "chamber") {
          if (u < 0.22) {
            const t = s5(0.0, 0.22, u);
            return { gOff: lerp3([0, 0, 0], chPos, t), gRot: lerp3([0, 0, 0], chRot, t), slideExtra: st };
          } else if (u < 0.80) {
            const dropBump = Math.sin(s5(0.45, 0.58, u) * Math.PI);
            const closeT = s5(0.68, 0.80, u);
            const closeBump = Math.sin(s5(0.73, 0.80, u) * Math.PI);
            return {
              gOff: [chPos[0], chPos[1] - 0.006 * dropBump - 0.004 * closeBump, chPos[2] + 0.007 * closeBump],
              gRot: [chRot[0], chRot[1] - 0.022 * dropBump - 0.020 * closeBump, chRot[2] + 0.025 * dropBump],
              slideExtra: st * (1.0 - closeT),
            };
          } else {
            const t = s5(0.80, 1.0, u);
            const targetP = this.chamberThenLoad ? loadPos : [0, 0, 0];
            const targetR = this.chamberThenLoad ? loadRot : [0, 0, 0];
            return { gOff: lerp3(chPos, targetP, t), gRot: lerp3(chRot, targetR, t), slideExtra: 0.0 };
          }
        } else if (phase === "enter") {
          const t = s5(0.0, 1.0, u);
          return { gOff: lerp3([0, 0, 0], loadPos, t), gRot: lerp3([0, 0, 0], loadRot, t), slideExtra: 0.0 };
        } else if (phase === "insert") {
          const bump = Math.sin(s5(0.46, 0.78, u) * Math.PI);
          return {
            gOff: [loadPos[0], loadPos[1] + 0.012 * bump, loadPos[2] - 0.004 * bump],
            gRot: [loadRot[0], loadRot[1] + 0.042 * bump, loadRot[2] - 0.045 * bump],
            slideExtra: 0.0,
          };
        } else {
          const t = s5(0.0, 1.0, u);
          return { gOff: lerp3(loadPos, [0, 0, 0], t), gRot: lerp3(loadRot, [0, 0, 0], t), slideExtra: 0.0 };
        }
      }
      if (this.reloadType === "empty") {
        const cant = sstep(0.0, 0.14, u) * (1.0 - sstep(0.86, 1.0, u));
        const ejectFlick = Math.sin(sstep(0.13, 0.28, u) * Math.PI);
        const slamBump = Math.sin(sstep(0.58, 0.68, u) * Math.PI);
        const rackPose = sstep(0.64, 0.72, u) * (1.0 - sstep(0.84, 0.94, u));
        const rackPull = sstep(0.72, 0.80, u) * (1.0 - sstep(0.80, 0.84, u));

        const gx = -0.026 * cant + 0.018 * ejectFlick + 0.024 * rackPose;
        const gy =  0.102 * cant + 0.022 * ejectFlick + 0.026 * slamBump - 0.052 * rackPose - 0.014 * rackPull;
        const gz = -0.022 * cant - 0.032 * rackPose + 0.020 * rackPull;

        const gyaw   = -0.32 * cant - 0.10 * rackPose;
        const gpitch =  0.24 * cant + 0.14 * ejectFlick + 0.12 * slamBump - 0.17 * rackPose - 0.12 * rackPull;
        const groll  = -0.62 * cant - 0.22 * ejectFlick + 0.34 * rackPose;

        let slideExtra = 0.0;
        if (this.isAK47) {
          if (u >= 0.72 && u < 0.80) slideExtra = this.SLIDE_TRAVEL * s5(0.72, 0.80, u);
          else if (u >= 0.80 && u < 0.85) slideExtra = this.SLIDE_TRAVEL * (1.0 - s5(0.80, 0.85, u));
        } else {
          if (u < 0.72) slideExtra = this.SLIDE_TRAVEL;
          else if (u < 0.80) slideExtra = this.SLIDE_TRAVEL * (1.0 + 0.14 * sstep(0.72, 0.80, u));
          else if (u < 0.84) slideExtra = this.SLIDE_TRAVEL * 1.14 * (1.0 - sstep(0.80, 0.84, u));
        }

        return { gOff: [gx, gy, gz], gRot: [gyaw, gpitch, groll], slideExtra };
      } else {
        const cant = sstep(0.0, 0.15, u) * (1.0 - sstep(0.82, 1.0, u));
        const ejectFlick = Math.sin(sstep(0.20, 0.34, u) * Math.PI);
        const slamBump = Math.sin(sstep(0.70, 0.82, u) * Math.PI);

        const gx = -0.024 * cant + 0.014 * ejectFlick;
        const gy =  0.098 * cant + 0.018 * ejectFlick + 0.026 * slamBump;
        const gz = -0.020 * cant;

        const gyaw   = -0.30 * cant;
        const gpitch =  0.22 * cant + 0.10 * ejectFlick + 0.12 * slamBump;
        const groll  = -0.58 * cant - 0.16 * ejectFlick;

        return { gOff: [gx, gy, gz], gRot: [gyaw, gpitch, groll], slideExtra: 0.0 };
      }
    }

    _inspectPoseOffsets() {
      if (this.inspectBlend <= 0.002) return { gOff: [0, 0, 0], gRot: [0, 0, 0] };
      const u = clamp01(this.inspectTimer / Math.max(this.inspectDuration, 1e-4));
      const w = this.inspectBlend;
      const isLg = this.isLongGun;
      const sway = Math.sin(this.elapsed * 1.7) * 0.018;
      const swayY = Math.cos(this.elapsed * 1.3) * 0.006;
      let pLeft, rLeft, pRight, rRight, pTop, rTop;
      if (isLg) {
        pLeft  = [-0.092, 0.046 + swayY, -0.025]; rLeft  = [ 0.62 + sway,  0.14, -0.42];
        pRight = [-0.048, 0.058 + swayY,  0.010]; rRight = [-0.56 + sway,  0.20,  0.64];
        pTop   = [-0.075, 0.024 + swayY, -0.010]; rTop   = [ 0.24,        -0.18, -0.22];
      } else {
        pLeft  = [-0.108, 0.068 + swayY,  0.055]; rLeft  = [ 0.72 + sway,  0.18, -0.46];
        pRight = [-0.062, 0.072 + swayY,  0.060]; rRight = [-0.78 + sway,  0.22,  0.72];
        pTop   = [-0.088, 0.042 + swayY,  0.045]; rTop   = [ 0.28,        -0.22, -0.18];
      }
      const z0 = [0, 0, 0];
      let pos, rot;
      if (u < 0.22) { const t = s5(0.0, 0.22, u); pos = lerp3(z0, pLeft, t); rot = lerp3(z0, rLeft, t); }
      else if (u < 0.36) { pos = pLeft; rot = rLeft; }
      else if (u < 0.52) { const t = s5(0.36, 0.52, u); pos = lerp3(pLeft, pRight, t); rot = lerp3(rLeft, rRight, t); }
      else if (u < 0.70) { pos = pRight; rot = rRight; }
      else if (u < 0.84) { const t = s5(0.70, 0.84, u); pos = lerp3(pRight, pTop, t); rot = lerp3(rRight, rTop, t); }
      else { const t = s5(0.84, 1.0, u); pos = lerp3(pTop, z0, t); rot = lerp3(rTop, z0, t); }
      return {
        gOff: [pos[0] * w, pos[1] * w, pos[2] * w],
        gRot: [rot[0] * w, rot[1] * w, rot[2] * w],
      };
    }

    _magCheckPoseOffsets() {
      const w = this.magCheckBlend || 0.0;
      if (w <= 0.002) return { gOff: [0, 0, 0], gRot: [0, 0, 0], slideExtra: 0.0 };
      const u = clamp01(this.magCheckTimer / Math.max(this.magCheckDuration, 1e-4));
      const st = this.SLIDE_TRAVEL;
      const jitX = Math.sin(this.elapsed * 6.4) * 0.0045 + Math.cos(this.elapsed * 11.2) * 0.0020;
      const jitY = Math.cos(this.elapsed * 5.2) * 0.0040 + Math.sin(this.elapsed * 9.7) * 0.0018;
      const jitR = Math.sin(this.elapsed * 5.8) * 0.032;
      const modeMc = this.magCheckMode || "mag_loaded";

      if (modeMc === "double_barrel") {
        const env = s5(0.0, 0.20, u) * (1.0 - s5(0.80, 1.0, u));
        const snap = Math.sin(s5(0.74, 0.90, u) * Math.PI);
        return {
          gOff: [(-0.022 * env + jitX * env) * w, (0.012 * env + 0.018 * snap + jitY * env) * w, (-0.048 * env + 0.016 * snap) * w],
          gRot: [(0.26 * env + jitR * 0.4 * env) * w, (-0.08 * env + 0.16 * snap) * w, (-0.28 * env + 0.08 * snap + jitR * env) * w],
          slideExtra: 0.0,
        };
      }
      if (modeMc === "revolver") {
        const env = s5(0.0, 0.18, u) * (1.0 - s5(0.84, 1.0, u));
        const spinVib = Math.sin(u * Math.PI * 12.0) * s5(0.44, 0.54, u) * (1.0 - s5(0.70, 0.80, u));
        const flick = Math.sin(s5(0.73, 0.88, u) * Math.PI);
        return {
          gOff: [(-0.036 * env + 0.024 * flick + jitX * env) * w, (0.056 * env + 0.020 * flick + jitY * env) * w, (0.012 * env) * w],
          gRot: [(-0.34 * env + jitR * 0.5 * env) * w, (0.14 * env + 0.12 * flick) * w, (-0.54 * env + 0.08 * spinVib + 0.42 * flick + jitR * env) * w],
          slideExtra: 0.0,
        };
      }
      if (modeMc === "mag_empty") {
        // ACT 1 (0.00..0.42): Pull out empty 3D magazine & re-insert it
        // ACT 2 (0.42..1.00): Rotate weapon to inspect chamber & rack bolt/slide 3 times!
        const act1 = s5(0.0, 0.10, u) * (1.0 - s5(0.37, 0.45, u));
        const slap1 = Math.sin(s5(0.32, 0.41, u) * Math.PI);
        const act2 = s5(0.41, 0.49, u) * (1.0 - s5(0.88, 1.0, u));
        let rack = 0.0;
        for (const [t0, t1] of [[0.47, 0.60], [0.63, 0.76], [0.78, 0.91]]) {
          if (u >= t0 && u <= t1) {
            const lu = (u - t0) / (t1 - t0);
            rack = Math.max(rack, s5(0.0, 0.46, lu) * (1.0 - s5(0.56, 1.0, lu)));
          }
        }
        const env = Math.max(act1, act2);
        return {
          gOff: [(-0.072 * act1 - 0.058 * act2 - 0.014 * rack + jitX * env) * w, (0.044 * act1 + 0.014 * slap1 + 0.036 * act2 - 0.010 * rack + jitY * env) * w, (0.022 * act1 + 0.026 * act2 + 0.024 * rack) * w],
          gRot: [(0.34 * act1 - 0.28 * act2 - 0.08 * rack + jitR * 0.4 * env) * w, (0.22 * act1 + 0.06 * slap1 + 0.20 * act2 + 0.09 * rack) * w, (-0.58 * act1 + 0.68 * act2 + 0.14 * rack + jitR * env) * w],
          slideExtra: st * rack * w,
        };
      }
      if (modeMc === "shotgun" && this.magCheckEmpty) {
        const env = s5(0.0, 0.16, u) * (1.0 - s5(0.86, 1.0, u));
        let rack = 0.0;
        for (const [t0, t1] of [[0.16, 0.44], [0.52, 0.80]]) {
          if (u >= t0 && u <= t1) {
            const lu = (u - t0) / (t1 - t0);
            rack = Math.max(rack, s5(0.0, 0.46, lu) * (1.0 - s5(0.56, 1.0, lu)));
          }
        }
        return {
          gOff: [(-0.056 * env - 0.012 * rack + jitX * env) * w, (0.034 * env - 0.010 * rack + jitY * env) * w, (0.026 * env + 0.024 * rack) * w],
          gRot: [(-0.28 * env - 0.08 * rack) * w, (0.20 * env + 0.09 * rack + jitR * 0.5 * env) * w, (0.68 * env + 0.14 * rack + jitR * env) * w],
          slideExtra: st * rack * w,
        };
      }
      if (modeMc === "shotgun" && !this.magCheckEmpty) {
        // Loaded Remington: cant the ejection port toward the eye and make a short, gentle bolt rack.
        const env = s5(0.04, 0.18, u) * (1.0 - s5(0.84, 0.97, u));
        const rack = s5(0.26, 0.38, u) * (1.0 - s5(0.56, 0.70, u));
        return {
          gOff: [(-0.024 * env + jitX * env) * w, (0.022 * env + 0.008 * rack + jitY * env) * w, (0.018 * env + 0.012 * rack) * w],
          gRot: [(-0.20 * env - 0.035 * rack) * w, (0.14 * env + 0.035 * rack) * w, (0.32 * env + jitR * env) * w],
          slideExtra: st * 0.25 * rack * w,
        };
      }
      // mag_loaded
      const env = s5(0.0, 0.20, u) * (1.0 - s5(0.82, 1.0, u));
      const slap = Math.sin(s5(0.75, 0.87, u) * Math.PI);
      return {
        gOff: [(-0.068 * env + jitX * env) * w, (0.044 * env + 0.014 * slap + jitY * env) * w, (0.022 * env - 0.008 * slap) * w],
        gRot: [(0.34 * env + jitR * 0.5 * env) * w, (0.22 * env + 0.075 * slap) * w, (-0.58 * env + jitR * env) * w],
        slideExtra: 0.0,
      };
    }

    _suppPoseOffsets() {
      if (!this.suppAnimActive) return { gOff: [0, 0, 0], gRot: [0, 0, 0] };
      const u = clamp01(this.suppAnimTimer / Math.max(this.suppAnimDuration, 1e-4));
      const env = s5(0.0, 0.18, u) * (1.0 - s5(0.82, 1.0, u));
      const tw = Math.sin(u * Math.PI * 6.0) * s5(0.16, 0.24, u) * (1.0 - s5(0.74, 0.84, u));
      return {
        gOff: [-0.072 * env, 0.038 * env + 0.005 * tw, 0.055 * env],
        gRot: [0.42 * env, 0.16 * env + 0.025 * tw, -0.44 * env + 0.06 * tw],
      };
    }

    _computeGunPose(kickProgress = 0.0, sliding = false) {
      const isSg = this.isShotgun;
      const isDb = this.isDoubleBarrel;
      const isRev = this.isRevolver;
      const isAk = this.isAK47;
      const isM4 = this.isM4A1;
      const isMp7 = this.isMP7;
      const isM110 = this.isM110;
      const isMp5s = this.isMP5S;
      const isLg = this.isLongGun;
      let bx, by, bz;
      if (isAk) {
        bx = lerp(0.168, 0.000, this.adsT);
        by = lerp(-0.172, -0.1195, this.adsT);
        bz = lerp(-0.258, -0.168, this.adsT);
      } else if (isM4) {
        bx = lerp(0.165, 0.000, this.adsT);
        by = lerp(-0.174, -0.1300, this.adsT);
        bz = lerp(-0.256, -0.268, this.adsT);
      } else if (isM110) {
        bx = lerp(0.174, 0.000, this.adsT);
        by = lerp(-0.182, -0.1480, this.adsT);
        bz = lerp(-0.250, -0.165, this.adsT);
      } else if (isMp5s) {
        bx = lerp(0.160, 0.000, this.adsT);
        by = lerp(-0.168, -0.1165, this.adsT);
        bz = lerp(-0.268, -0.185, this.adsT);
      } else if (isMp7) {
        // ADS aligns eye straight through open red-dot collimator window at y = 0.136!
        bx = lerp(0.158, 0.000, this.adsT);
        by = lerp(-0.164, -0.1360, this.adsT);
        bz = lerp(-0.275, -0.192, this.adsT);
      } else if (isDb) {
        bx = lerp(0.154, 0.000, this.adsT);
        by = lerp(-0.144, -0.090, this.adsT);
        bz = lerp(-0.258, -0.218, this.adsT);
      } else if (isSg) {
        bx = lerp(0.152, 0.000, this.adsT);
        by = lerp(-0.148, -0.098, this.adsT);
        bz = lerp(-0.260, -0.215, this.adsT);
      } else if (isRev) {
        bx = lerp(0.154, 0.000, this.adsT);
        by = lerp(-0.158, -0.0985, this.adsT);
        bz = lerp(-0.340, -0.278, this.adsT);
      } else {
        bx = lerp(0.148, 0.000, this.adsT);
        by = lerp(-0.148, -0.079, this.adsT);
        bz = lerp(-0.345, -0.258, this.adsT);
      }

      const breathY = Math.sin(this.elapsed * 2.1) * 0.0022 * (1.0 - 0.65 * this.adsT);
      const bobX = -Math.cos(this.bobPhase) * this.bobAmp * 0.95;
      const bobY = Math.abs(Math.sin(this.bobPhase)) * this.bobAmp * 0.85;

      const sp = this.sprintBlend;
      const drawDrop = 1.0 - sstep(0.0, 1.0, this.equipT);
      const slRoll = sliding ? -0.24 : 0.0;
      const slY = sliding ? -0.028 : 0.0;
      const kSin = (kickProgress > 0.001) ? Math.sin(kickProgress * Math.PI) : 0.0;

      // Subtle steady-hold torque while thumb cocks the revolver hammer before firing!
      let cockPitch = 0.0, cockRoll = 0.0, cockZ = 0.0;
      if (this.revCocking) {
        const cu = clamp01(this.revCockTimer / Math.max(this.revCockDuration, 1e-4));
        const cp = Math.sin(cu * Math.PI);
        cockPitch = 0.058 * cp * (1.0 - 0.55 * this.adsT);
        cockRoll = -0.095 * cp * (1.0 - 0.55 * this.adsT);
        cockZ = 0.008 * cp;
      }

      const rp = this._reloadPoseOffsets();
      const pp = this._pumpPoseOffsets();
      const ip = this._inspectPoseOffsets();
      const mcp = this._magCheckPoseOffsets();
      const sup = this._suppPoseOffsets();
      const tx = bx + bobX + this.swayX + (isLg ? -0.090 : -0.065) * sp + 0.065 * kSin + rp.gOff[0] + pp.gOff[0] + ip.gOff[0] + mcp.gOff[0] + sup.gOff[0];
      const ty = by + bobY + breathY + this.swayY + (isLg ? -0.056 : -0.048) * sp - 0.34 * drawDrop + slY + 0.035 * kSin + this.recoilY + rp.gOff[1] + pp.gOff[1] + ip.gOff[1] + mcp.gOff[1] + sup.gOff[1];
      const tz = bz + 0.032 * sp + this.recoilZ + cockZ + rp.gOff[2] + pp.gOff[2] + ip.gOff[2] + mcp.gOff[2] + sup.gOff[2];

      const yaw = (isLg ? 0.50 : 0.42) * sp - this.swayX * 2.2 + rp.gRot[0] + pp.gRot[0] + ip.gRot[0] + mcp.gRot[0] + sup.gRot[0];
      const pitch = this.recoilPitch + cockPitch + (isLg ? -0.20 : -0.24) * sp - 0.75 * drawDrop + this.swayY * 1.8 + rp.gRot[1] + pp.gRot[1] + ip.gRot[1] + mcp.gRot[1] + sup.gRot[1];
      const roll = this.recoilRoll + cockRoll + 0.28 * sp + slRoll - 0.22 * kSin - this.swayX * 3.0 + rp.gRot[2] + pp.gRot[2] + ip.gRot[2] + mcp.gRot[2] + sup.gRot[2];
      const gunM = mat4Mul(mat4Translate(tx, ty, tz), rotYXZ(yaw, pitch, roll));
      return { gunPos: [tx, ty, tz], gunRot: [yaw, pitch, roll], gunM, rp, pp, mcp };
    }

    computeGunViewMatrix(kickProgress = 0.0, sliding = false) {
      return this._computeGunPose(kickProgress, sliding).gunM;
    }

    _build11BoneHandMats(handM, pose, isLeft, sc) {
      const mcpTbl = isLeft ? this.FP_HAND_MCP_L : this.FP_HAND_MCP_R;
      const out = { palm: handM };
      for (const fname of this.FP_FINGER_NAMES) {
        const mcp = mcpTbl[fname];
        const [mcpCurl, mcpYaw, mcpRoll, pipCurl] = pose[fname];
        const mP = mat4Mul(
          handM,
          mat4Mul(mat4Translate(mcp[0] * sc, mcp[1] * sc, mcp[2] * sc), rotYXZ(mcpYaw, -mcpCurl, mcpRoll))
        );
        const mD = mat4Mul(
          mP,
          mat4Mul(mat4Translate(0.0, 0.0, -this.FP_HAND_PIP_LEN[fname] * sc), mat4RotX(-pipCurl))
        );
        out[fname + "_p"] = mP;
        out[fname + "_d"] = mD;
      }
      return out;
    }

    computePartsViewMatrices(kickProgress = 0.0, sliding = false, gender = "male") {
      const { gunRot, gunM, rp, pp, mcp } = this._computeGunPose(kickProgress, sliding);
      const isSg = this.isShotgun;
      const isDb = this.isDoubleBarrel;
      const isRev = this.isRevolver;
      const isAk = this.isAK47;
      const isM4 = this.isM4A1;
      const isMp7 = this.isMP7;
      const isM110 = this.isM110;
      const isMp5s = this.isMP5S;
      const isRifle = this.isRifle;
      const isLg = this.isLongGun;

      // ---- 1. Articulated Weapon Mechanical Parts ----
      const st = this.SLIDE_TRAVEL;
      const mcW = this.magCheckBlend || 0.0;
      const mcU = mcW > 0.01 ? clamp01(this.magCheckTimer / Math.max(this.magCheckDuration, 1e-4)) : 0.0;
      const slideBack = this.reloading
        ? rp.slideExtra
        : ((mcW > 0.01 && (this.magCheckEmpty || this.magCheckMode === "shotgun")) ? mcp.slideExtra : (this.pumpActive ? pp.slideExtra : (this.slideLocked ? st : (st * Math.pow(this.slideKick, 0.75)))));

      const insW = this.inspectBlend || 0.0;
      const insU = insW > 0.01 ? clamp01(this.inspectTimer / Math.max(this.inspectDuration, 1e-4)) : 0.0;

      let mSlide, mHammer;
      if (isDb) {
        const dbAng = this._dbBreakAngle();
        const [hx, hy, hz] = this.DB_HINGE_PIVOT;
        mSlide = mat4Mul(
          gunM,
          mat4Mul(mat4Translate(hx, hy, hz), mat4Mul(mat4RotX(-dbAng), mat4Translate(-hx, -hy, -hz)))
        );
        const lvAng = this._dbLeverAngle();
        const [lx, ly, lz] = this.DB_LEVER_PIVOT;
        mHammer = mat4Mul(
          gunM,
          mat4Mul(mat4Translate(lx, ly, lz), mat4Mul(mat4RotY(lvAng), mat4Translate(-lx, -ly, -lz)))
        );
      } else if (isRev) {
        const insSpin = 1.57 * Math.sin(s5(0.24, 0.76, insU) * Math.PI) * insW;
        const mcSpin = (mcW > 0.01 && this.magCheckMode === "revolver") ? (this.mcRevSpin || 0.0) : 0.0;
        const [cx, cy, cz] = this.REV_CYL_PIVOT;
        const spinM = mat4Mul(
          mat4Translate(cx, cy, cz),
          mat4Mul(mat4RotZ(-(this.revCylAngle + insSpin + mcSpin)), mat4Translate(-cx, -cy, -cz))
        );
        const crAng = this._revCraneAngle();
        if (Math.abs(crAng) > 1e-5) {
          const [kx, ky, kz] = this.REV_CRANE_PIVOT;
          const craneM = mat4Mul(
            mat4Translate(kx, ky, kz),
            mat4Mul(mat4RotZ(crAng), mat4Translate(-kx, -ky, -kz))
          );
          mSlide = mat4Mul(gunM, mat4Mul(craneM, spinM));
        } else {
          mSlide = mat4Mul(gunM, spinM);
        }
        const hmAng = this._revHammerAngle();
        const [hx, hy, hz] = this.REV_HAMMER_PIVOT;
        mHammer = mat4Mul(
          gunM,
          mat4Mul(mat4Translate(hx, hy, hz), mat4Mul(mat4RotX(hmAng), mat4Translate(-hx, -hy, -hz)))
        );
      } else if (isM4 || isMp7) {
        mSlide = mat4Mul(gunM, mat4Translate(0.0, 0.0, slideBack));
        let chPull = 0.0;
        if (this.reloading && this.reloadType === "empty") {
          const ru = clamp01(this.reloadTimer / Math.max(this.reloadDuration, 1e-4));
          chPull = (isM4 ? 0.068 : 0.048) * s5(0.71, 0.80, ru) * (1.0 - s5(0.80, 0.86, ru));
        } else if (mcW > 0.01 && this.magCheckMode === "mag_empty" && mcp.slideExtra > 0.001) {
          chPull = Math.min(isM4 ? 0.068 : 0.048, mcp.slideExtra * 1.15);
        }
        mHammer = mat4Mul(gunM, mat4Translate(0.0, 0.0, chPull));
      } else {
        mSlide = mat4Mul(gunM, mat4Translate(0.0, 0.0, slideBack));
        mHammer = mat4Mul(
          gunM,
          mat4Mul(
            mat4Translate(0.0, 0.056, 0.036),
            mat4Mul(mat4RotX(-0.45 * this.hammerDrop), mat4Translate(0.0, -0.056, -0.036))
          )
        );
      }

      let trigAmt = this.triggerPull;
      if (isRev && this.revCocking) {
        const cuT = clamp01(this.revCockTimer / Math.max(this.revCockDuration, 1e-4));
        trigAmt = Math.max(trigAmt, 0.48 * s5(0.55, 1.0, cuT));
      }
      const mTrigger = mat4Mul(gunM, mat4Translate(0.0, 0.0, 0.008 * trigAmt));

      // ---- 2. Realistic Anatomical Right Hand Wrist Placement (Wrist Origin = (0,0,0)!) ----
      const insRTwist = 0.08 * Math.sin(insU * Math.PI * 2.0) * insW;
      let rPos, rRot, elbowR;
      if (isSg || isDb) {
        rPos = [0.016, 0.044, 0.072];
        rRot = [-0.18, -0.34, -1.56 + insRTwist];
        elbowR = [0.34, -0.34, 0.24];
      } else if (isAk) {
        rPos = [0.016, 0.024, 0.068];
        rRot = [-0.14, -0.28, -1.56 + insRTwist];
        elbowR = [0.34, -0.34, 0.24];
      } else if (isM4) {
        rPos = [0.016, 0.024, 0.068];
        rRot = [-0.14, -0.28, -1.56 + insRTwist];
        elbowR = [0.34, -0.34, 0.24];
      } else if (isRev) {
        rPos = [0.016, 0.020, 0.072];
        rRot = [-0.14, -0.24, -1.56 + insRTwist];
        elbowR = [0.32, -0.36, 0.22];
      } else {
        rPos = [0.016, 0.018, 0.070];
        rRot = [-0.14, -0.24, -1.56 + insRTwist];
        elbowR = [0.32, -0.36, 0.22];
      }
      const mHandR = mat4Mul(gunM, mat4Mul(mat4Translate(rPos[0], rPos[1], rPos[2]), rotYXZ(rRot[0], rRot[1], rRot[2])));
      const wristR = [mHandR[3], mHandR[7], mHandR[11]];
      const mForearmR = aimForearmMatrix(wristR, elbowR, [0, 1, 0]);

      // ---- 3. Magazine + Anatomically Accurate Left Support Hand Placement & Choreography ----
      const baseLhPos = [-0.024, 0.002, 0.056];
      const baseLhRot = [-0.24, -0.16, 1.48];
      const sgForendPos = [-0.018, 0.018, -0.278];
      const sgForendRot = [-0.14, 0.08, 2.70];
      const dbForendPos = [-0.018, 0.022, -0.228];
      const dbForendRot = [-0.14, 0.08, 2.74];
      const akForendPos = [-0.018, 0.032, -0.268];
      const akForendRot = [-0.16, 0.10, 2.66];
      const m4ForendPos = [-0.018, 0.006, -0.274];
      const m4ForendRot = [-0.28, 0.08, 1.78];
      const mp7ForendPos = [-0.016, -0.014, -0.150];
      const mp7ForendRot = [-0.26, 0.06, 1.72];
      const m110ForendPos = [-0.020, 0.028, -0.318];
      const m110ForendRot = [-0.16, 0.10, 2.66];
      const mp5sForendPos = [-0.018, 0.024, -0.228];
      const mp5sForendRot = [-0.16, 0.08, 2.64];

      let ma, magGripOff, magGripRot, entryDist;
      if (isAk) {
        ma = this.AK_MAG_AXIS;
        magGripOff = [-0.036, -0.088, -0.148];
        magGripRot = [-0.22, -0.28, 1.54];
        entryDist = 0.175;
      } else if (isM4) {
        ma = this.M4_MAG_AXIS;
        magGripOff = [-0.034, -0.082, -0.108];
        magGripRot = [-0.20, -0.18, 1.52];
        entryDist = 0.165;
      } else if (isM110) {
        ma = this.M110_MAG_AXIS;
        magGripOff = [-0.034, -0.072, -0.128];
        magGripRot = [-0.20, -0.18, 1.52];
        entryDist = 0.170;
      } else if (isMp5s) {
        ma = this.MP5S_MAG_AXIS;
        magGripOff = [-0.032, -0.094, -0.155];
        magGripRot = [-0.20, -0.16, 1.50];
        entryDist = 0.165;
      } else if (isMp7) {
        ma = this.MP7_MAG_AXIS;
        magGripOff = [-0.030, -0.115, 0.072];
        magGripRot = [-0.18, -0.12, 1.48];
        entryDist = 0.165;
      } else {
        ma = this.MAG_AXIS;
        magGripOff = [-0.030, -0.062, 0.068];
        magGripRot = [-0.18, -0.12, 1.48];
        entryDist = 0.155;
      }

      let elbowL = (isLg || isMp7 || isMp5s) ? [-0.34, -0.38, 0.16] : [-0.30, -0.38, 0.22];
      let magVis = (!isSg && !isDb && !isRev);
      let magHasRound = this.magAmmo > 0;
      let mMag = gunM;
      let mShell12g = null;
      const extraShells12g = [];

      let mHandL;
      if (isSg) {
        mHandL = mat4Mul(mSlide, mat4Mul(mat4Translate(...sgForendPos), rotYXZ(...sgForendRot)));
      } else if (isDb) {
        mHandL = mat4Mul(mSlide, mat4Mul(mat4Translate(...dbForendPos), rotYXZ(...dbForendRot)));
      } else if (isAk) {
        mHandL = mat4Mul(gunM, mat4Mul(mat4Translate(...akForendPos), rotYXZ(...akForendRot)));
      } else if (isM4) {
        mHandL = mat4Mul(gunM, mat4Mul(mat4Translate(...m4ForendPos), rotYXZ(...m4ForendRot)));
      } else if (isM110) {
        mHandL = mat4Mul(gunM, mat4Mul(mat4Translate(...m110ForendPos), rotYXZ(...m110ForendRot)));
      } else if (isMp5s) {
        mHandL = mat4Mul(gunM, mat4Mul(mat4Translate(...mp5sForendPos), rotYXZ(...mp5sForendRot)));
      } else if (isMp7) {
        mHandL = mat4Mul(gunM, mat4Mul(mat4Translate(...mp7ForendPos), rotYXZ(...mp7ForendRot)));
      } else {
        mHandL = mat4Mul(gunM, mat4Mul(mat4Translate(...baseLhPos), rotYXZ(...baseLhRot)));
      }

      let fMagRelease = 0.0, fDbLever = 0.0, fRevCock = 0.0;
      let fLOpen = 0.0, fLMagHold = 0.0, fLShellPush = 0.0, fLPalmSlap = 0.0, fLSlidePinch = 0.0, fLSlideFlick = 0.0;
      if (isSg && this.pumpActive) fLSlidePinch = 0.72;
      if (isRev && this.revCocking) {
        const cuC = clamp01(this.revCockTimer / Math.max(this.revCockDuration, 1e-4));
        fRevCock = s5(0.0, 0.72, cuC) * (1.0 - 0.15 * s5(0.84, 1.0, cuC));
      }

      const offscreenMagPos = [-0.10, -0.46, -0.26];
      const offscreenHandPos = [-0.15, -0.52, -0.22];
      const entryWorld = mat4TransformPt(gunM, [ma[0] * entryDist, ma[1] * entryDist, ma[2] * entryDist]);

      const akMagTransform = (d, rock) => {
        const [px, py, pz] = this.AK_MAG_PIVOT;
        return mat4Mul(
          gunM,
          mat4Mul(
            mat4Translate(ma[0] * d, ma[1] * d, ma[2] * d),
            mat4Mul(mat4Translate(px, py, pz), mat4Mul(mat4RotX(-rock), mat4Translate(-px, -py, -pz)))
          )
        );
      };

      if (this.reloading) {
        const u = clamp01(this.reloadTimer / Math.max(this.reloadDuration, 1e-4));
        if (this.reloadType === "break_action") {
          const dbAng = this._dbBreakAngle();
          if (dbAng > 0.16) {
            const loadedSlots = (this.dbReloadPhase === "open")
              ? (this.magAmmo >= 1 ? [0] : [])
              : (this.dbReloadPhase === "insert"
                ? (this.magAmmo >= 2 ? [0, 1] : (this.magAmmo === 1 ? [0] : []))
                : (this.magAmmo >= 2 ? [0, 1] : (this.magAmmo === 1 ? [0] : [])));
            for (const slIdx of loadedSlots) {
              const sx = slIdx === 0 ? -0.0138 : 0.0138;
              extraShells12g.push(mat4Mul(mSlide, mat4Translate(sx, 0.0725, -0.158)));
            }
            if (this.dbReloadPhase === "open" && !this.dbEvEject) {
              const ext = s5(0.24, 0.44, u);
              if (this.dbEjectCount >= 1) extraShells12g.push(mat4Mul(mSlide, mat4Translate(0.0138, 0.0725, -0.156 + 0.032 * ext)));
              if (this.dbEjectCount >= 2) extraShells12g.push(mat4Mul(mSlide, mat4Translate(-0.0138, 0.0725, -0.156 + 0.032 * ext)));
            }
          }
          const dbPouchPos = [-0.078, -0.245, -0.095];
          const dbPouchRot = [-0.16, -0.18, 1.90];
          const dbPouchElb = [-0.64, -0.82, -0.06];
          const dbForendElb = [-0.56, -0.54, -0.06];
          const phase = this.dbReloadPhase;

          if (phase === "open") {
            fDbLever = Math.sin(s5(0.02, 0.34, u) * Math.PI);
            if (u < 0.52) {
              mHandL = mat4Mul(mSlide, mat4Mul(mat4Translate(...dbForendPos), rotYXZ(...dbForendRot)));
              elbowL = dbForendElb;
              fLSlidePinch = 0.55 * s5(0.18, 0.48, u);
            } else {
              const t = s5(0.52, 1.0, u);
              const mStart = mat4Mul(mSlide, mat4Mul(mat4Translate(...dbForendPos), rotYXZ(...dbForendRot)));
              const pStart = [mStart[3], mStart[7], mStart[11]];
              const pEnd = mat4TransformPt(gunM, dbPouchPos);
              const lhW = lerp3(pStart, pEnd, t);
              const lhR = lerp3(
                [gunRot[0] - 0.14, gunRot[1] - 0.52, gunRot[2] + 2.74],
                [gunRot[0] + dbPouchRot[0], gunRot[1] + dbPouchRot[1], gunRot[2] + dbPouchRot[2]],
                t
              );
              mHandL = mat4Mul(mat4Translate(...lhW), rotYXZ(...lhR));
              elbowL = lerp3(dbForendElb, dbPouchElb, t);
              fLOpen = 0.75 * t;
            }
          } else if (phase === "insert") {
            const slotIdx = (this.magAmmo === 0) ? 0 : 1;
            const sx = slotIdx === 0 ? -0.0138 : 0.0138;
            const pPouchW = mat4TransformPt(gunM, dbPouchPos);
            const pReadySh = mat4TransformPt(mSlide, [sx, 0.0725, -0.095]);
            const pReadyH = mat4TransformPt(mSlide, [sx - 0.056, 0.028, -0.040]);
            const pPushH = mat4TransformPt(mSlide, [sx - 0.044, 0.040, -0.086]);
            const rotPouchW = [gunRot[0] + dbPouchRot[0], gunRot[1] + dbPouchRot[1], gunRot[2] + dbPouchRot[2]];
            const rotLoadW = [gunRot[0] - 0.12, gunRot[1] - 0.56, gunRot[2] + 1.85];
            const loadElb = [-0.64, -0.64, -0.06];
            if (u < 0.38) {
              const t = s5(0.0, 0.38, u);
              const arc = Math.sin(t * Math.PI);
              const lhW = lerp3(pPouchW, pReadyH, t);
              lhW[0] -= 0.018 * arc; lhW[1] += 0.022 * arc;
              const lhR = lerp3(rotPouchW, rotLoadW, t);
              mHandL = mat4Mul(mat4Translate(...lhW), rotYXZ(...lhR));
              elbowL = lerp3(dbPouchElb, loadElb, t);
              fLMagHold = 0.45 + 0.55 * t;
              const shW = lerp3([pPouchW[0] + 0.02, pPouchW[1] + 0.03, pPouchW[2] - 0.02], pReadySh, t);
              mShell12g = mat4Mul(mat4Translate(...shW), rotYXZ(gunRot[0], gunRot[1] - dbAng * t, gunRot[2]));
            } else if (u < 0.64) {
              const t = s5(0.38, 0.64, u);
              const lhW = lerp3(pReadyH, pPushH, t);
              mHandL = mat4Mul(mat4Translate(...lhW), rotYXZ(...rotLoadW));
              elbowL = loadElb;
              fLMagHold = 1.0 - 0.70 * t;
              fLShellPush = Math.sin(t * Math.PI);
              if (u < 0.60) {
                const shT = s5(0.38, 0.59, u);
                mShell12g = mat4Mul(mSlide, mat4Translate(sx, 0.0725, lerp(-0.066, -0.152, shT)));
              }
            } else {
              const t = s5(0.64, 1.0, u);
              const lhW = lerp3(pPushH, pPouchW, t);
              const lhR = lerp3(rotLoadW, rotPouchW, t);
              mHandL = mat4Mul(mat4Translate(...lhW), rotYXZ(...lhR));
              elbowL = lerp3(loadElb, dbPouchElb, t);
              fLOpen = 0.70 * t;
            }
          } else {
            if (u < 0.22) {
              const t = s5(0.0, 0.22, u);
              const pStart = mat4TransformPt(gunM, dbPouchPos);
              const mEnd = mat4Mul(mSlide, mat4Mul(mat4Translate(...dbForendPos), rotYXZ(...dbForendRot)));
              const pEnd = [mEnd[3], mEnd[7], mEnd[11]];
              const lhW = lerp3(pStart, pEnd, t);
              const lhR = lerp3(
                [gunRot[0] + dbPouchRot[0], gunRot[1] + dbPouchRot[1], gunRot[2] + dbPouchRot[2]],
                [gunRot[0] - 0.14, gunRot[1] - 0.52, gunRot[2] + 2.74],
                t
              );
              mHandL = mat4Mul(mat4Translate(...lhW), rotYXZ(...lhR));
              elbowL = lerp3(dbPouchElb, dbForendElb, t);
              fLOpen = 0.65 * (1.0 - t);
            } else {
              mHandL = mat4Mul(mSlide, mat4Mul(mat4Translate(...dbForendPos), rotYXZ(...dbForendRot)));
              elbowL = dbForendElb;
              fLSlidePinch = 0.75 * (1.0 - s5(0.60, 1.0, u));
            }
          }
        } else if (this.reloadType === "revolver") {
          const revCylPush = [-0.056, 0.032, -0.012];
          const revRodTap = [-0.064, 0.044, -0.092];
          const revPouchPos = [-0.110, -0.185, 0.055];
          const revSnapPos = [-0.056, 0.034, -0.016];
          if (this.revReloadPhase === "open_dump") {
            fMagRelease = Math.sin(s5(0.04, 0.26, u) * Math.PI);
            if (u < 0.30) {
              const t = s5(0.0, 0.30, u);
              const lhP = lerp3(baseLhPos, revCylPush, t);
              const lhR = lerp3(baseLhRot, [-0.28, -0.10, 1.85], t);
              mHandL = mat4Mul(gunM, mat4Mul(mat4Translate(...lhP), rotYXZ(...lhR)));
              elbowL = lerp3([-0.34, -0.44, 0.22], [-0.42, -0.44, 0.14], t);
              fLOpen = 0.45 * t;
            } else if (u < 0.64) {
              const t = s5(0.30, 0.46, u);
              const lhP = lerp3(revCylPush, revRodTap, t);
              const lhR = lerp3([-0.28, -0.10, 1.85], [-0.34, -0.18, 2.05], t);
              mHandL = mat4Mul(gunM, mat4Mul(mat4Translate(...lhP), rotYXZ(...lhR)));
              elbowL = [-0.44, -0.40, 0.12];
              fLPalmSlap = Math.sin(s5(0.34, 0.56, u) * Math.PI);
            } else {
              const t = s5(0.64, 1.0, u);
              const lhP = lerp3(revRodTap, revPouchPos, t);
              const lhR = lerp3([-0.34, -0.18, 2.05], [-0.20, 0.08, 1.65], t);
              mHandL = mat4Mul(gunM, mat4Mul(mat4Translate(...lhP), rotYXZ(...lhR)));
              elbowL = lerp3([-0.44, -0.40, 0.12], [-0.44, -0.74, 0.06], t);
              fLOpen = 0.65 * (1.0 - t);
              fLMagHold = t;
            }
          } else if (this.revReloadPhase === "load") {
            const ciL = Math.max(0, Math.min(7, this.revLoadChamber));
            const [cxR, cyR] = this.REV_CYL_PIVOT;
            const angC = Math.PI * 0.5 + ciL * (Math.PI * 2.0 / 8.0);
            const chX = cxR + Math.cos(angC) * this.REV_CHAMBER_RING_R;
            const chY = cyR + Math.sin(angC) * this.REV_CHAMBER_RING_R;
            const pReadyV = mat4TransformPt(mSlide, [chX, chY, this.REV_CHAMBER_REAR_Z + 0.052]);
            const pPushV  = mat4TransformPt(mSlide, [chX, chY, this.REV_CHAMBER_REAR_Z + 0.010]);
            const pPouchV = mat4TransformPt(gunM, revPouchPos);
            const wReady = [pReadyV[0] - 0.042, pReadyV[1] - 0.026, pReadyV[2] + 0.036];
            const wPush  = [pPushV[0]  - 0.036, pPushV[1]  - 0.022, pPushV[2]  + 0.028];
            if (u < 0.42) {
              const t = s5(0.0, 0.42, u);
              const lhW = lerp3(pPouchV, wReady, t);
              const lhR = lerp3([-0.20, 0.08, 1.65], [gunRot[0] - 0.14, gunRot[1] + 0.14, gunRot[2] + 1.68], t);
              mHandL = mat4Mul(mat4Translate(...lhW), rotYXZ(...lhR));
              elbowL = lerp3([-0.44, -0.74, 0.06], [-0.42, -0.44, 0.12], t);
              fLMagHold = 1.0;
            } else if (u < 0.72) {
              const t = s5(0.42, 0.66, u);
              const lhW = lerp3(wReady, wPush, t);
              const lhR = [gunRot[0] - 0.14, gunRot[1] + 0.16, gunRot[2] + 1.72];
              mHandL = mat4Mul(mat4Translate(...lhW), rotYXZ(...lhR));
              elbowL = [-0.40, -0.42, 0.10];
              fLMagHold = 1.0 - 0.75 * s5(0.56, 0.72, u);
              fLShellPush = Math.sin(s5(0.44, 0.72, u) * Math.PI);
            } else {
              const t = s5(0.72, 1.0, u);
              const lhW = lerp3(wPush, pPouchV, t);
              const lhR = lerp3([gunRot[0] - 0.14, gunRot[1] + 0.16, gunRot[2] + 1.72], [-0.20, 0.08, 1.65], t);
              mHandL = mat4Mul(mat4Translate(...lhW), rotYXZ(...lhR));
              elbowL = lerp3([-0.40, -0.42, 0.10], [-0.44, -0.74, 0.06], t);
              fLOpen = 0.45 * (1.0 - t);
            }
          } else {
            if (u < 0.56) {
              const t = s5(0.0, 0.56, u);
              const lhP = lerp3(revPouchPos, revSnapPos, t);
              const lhR = lerp3([-0.20, 0.08, 1.65], [-0.22, -0.05, 1.72], t);
              mHandL = mat4Mul(gunM, mat4Mul(mat4Translate(...lhP), rotYXZ(...lhR)));
              elbowL = lerp3([-0.44, -0.74, 0.06], [-0.40, -0.44, 0.14], t);
              fLPalmSlap = Math.sin(t * Math.PI);
            } else {
              const t = s5(0.56, 1.0, u);
              const lhP = lerp3(revSnapPos, baseLhPos, t);
              const lhR = lerp3([-0.22, -0.05, 1.72], baseLhRot, t);
              mHandL = mat4Mul(gunM, mat4Mul(mat4Translate(...lhP), rotYXZ(...lhR)));
              elbowL = lerp3([-0.40, -0.44, 0.14], [-0.34, -0.44, 0.22], t);
              fLOpen = Math.sin(t * Math.PI) * 0.40;
            }
          }
        } else if (this.reloadType === "shell") {
          const pouchPos = [-0.036, -0.220, -0.088];
          const pouchRot = [-0.16, -0.10, 2.25];
          const pouchElb = [-0.48, -0.72, 0.05];
          const portReadyPos = [-0.022, -0.072, -0.030];
          const portReadyRot = [-0.12, 0.20, 2.42];
          const portPushPos = [-0.018, -0.015, -0.034];
          const portPushRot = [-0.12, 0.26, 2.48];
          const portElb = [-0.42, -0.47, 0.10];
          const forendElb = [-0.56, -0.54, -0.06];
          const sgCurPos = [sgForendPos[0], sgForendPos[1], sgForendPos[2] + slideBack];
          const sgOpenPos = [sgForendPos[0], sgForendPos[1], sgForendPos[2] + st];
          const phase = this.shellReloadPhase;

          if (phase === "chamber") {
            const chAbovePos = [0.048, 0.148, -0.052];
            const chAboveRot = [0.10, -0.14, 0.65];
            const chDropPos = [0.026, 0.118, -0.054];
            const chDropRot = [0.08, -0.18, 0.72];
            const chElb = [-0.48, -0.26, 0.04];
            let lhP = sgForendPos, lhR = sgForendRot;
            if (u < 0.22) {
              const t = s5(0.0, 0.22, u);
              lhP = lerp3(sgOpenPos, pouchPos, t); lhR = lerp3(sgForendRot, pouchRot, t);
              elbowL = lerp3(forendElb, pouchElb, t); fLOpen = 0.65 * t;
            } else if (u < 0.43) {
              const t = s5(0.22, 0.43, u); const arc = Math.sin(t * Math.PI);
              lhP = lerp3(pouchPos, chAbovePos, t);
              lhP[0] += 0.012 * arc; lhP[1] += 0.024 * arc; lhP[2] -= 0.008 * arc;
              lhR = lerp3(pouchRot, chAboveRot, t); elbowL = lerp3(pouchElb, chElb, t);
              fLMagHold = 1.0;
              mHandL = mat4Mul(gunM, mat4Mul(mat4Translate(...lhP), rotYXZ(...lhR)));
              mShell12g = mat4Mul(mHandL, mat4Mul(mat4Translate(0.016, -0.016, -0.072), rotYXZ(-0.08, 0.14, -0.65)));
            } else if (u < 0.56) {
              const t = s5(0.43, 0.56, u);
              lhP = lerp3(chAbovePos, chDropPos, t); lhR = lerp3(chAboveRot, chDropRot, t);
              elbowL = chElb; fLMagHold = 1.0 - 0.70 * t; fLOpen = 0.50 * t;
              mHandL = mat4Mul(gunM, mat4Mul(mat4Translate(...lhP), rotYXZ(...lhR)));
              if (u < 0.53) {
                const mHa = mat4Mul(mat4Translate(...chAbovePos), rotYXZ(...chAboveRot));
                const shStart = mat4TransformPt(mHa, [0.016, -0.016, -0.072]);
                const shT = s5(0.43, 0.52, u);
                const shLoc = lerp3(shStart, [0.006, 0.077, -0.124], shT);
                mShell12g = mat4Mul(gunM, mat4Mul(mat4Translate(...shLoc), rotYXZ(-0.04 * (1.0 - shT), 0.04 * (1.0 - shT), 0.0)));
              }
            } else if (u < 0.68) {
              const t = s5(0.56, 0.68, u);
              lhP = lerp3(chDropPos, sgOpenPos, t); lhR = lerp3(chDropRot, sgForendRot, t);
              elbowL = lerp3(chElb, forendElb, t); fLOpen = 0.50 * (1.0 - t); fLSlidePinch = 0.65 * t;
            } else if (u < 0.80) {
              lhP = sgCurPos; lhR = sgForendRot; elbowL = forendElb; fLSlidePinch = 0.65;
            } else {
              const t = s5(0.80, 1.0, u);
              if (this.chamberThenLoad) {
                lhP = lerp3(sgForendPos, pouchPos, t); lhR = lerp3(sgForendRot, pouchRot, t);
                elbowL = lerp3(forendElb, pouchElb, t); fLSlidePinch = 0.65 * (1.0 - t); fLOpen = 0.55 * t;
              } else {
                lhP = sgForendPos; lhR = sgForendRot; elbowL = forendElb; fLSlidePinch = 0.65 * (1.0 - t);
              }
            }
            if (!mShell12g || u >= 0.56 || u < 0.22) {
              mHandL = mat4Mul(gunM, mat4Mul(mat4Translate(...lhP), rotYXZ(...lhR)));
            }
          } else if (phase === "enter") {
            const t = s5(0.0, 1.0, u);
            const lhP = lerp3(sgForendPos, pouchPos, t), lhR = lerp3(sgForendRot, pouchRot, t);
            mHandL = mat4Mul(gunM, mat4Mul(mat4Translate(...lhP), rotYXZ(...lhR)));
            elbowL = lerp3(forendElb, pouchElb, t); fLOpen = 0.55 * t;
          } else if (phase === "insert") {
            if (u < 0.38) {
              const t = s5(0.0, 0.38, u);
              const lhP = lerp3(pouchPos, portReadyPos, t), lhR = lerp3(pouchRot, portReadyRot, t);
              elbowL = lerp3(pouchElb, portElb, t); fLMagHold = 0.45 + 0.55 * t;
              mHandL = mat4Mul(gunM, mat4Mul(mat4Translate(...lhP), rotYXZ(...lhR)));
              mShell12g = mat4Mul(mHandL, mat4Mul(mat4Translate(0.012, -0.024, -0.088), rotYXZ(-0.08, 0.14, -2.42)));
            } else if (u < 0.68) {
              const t = s5(0.38, 0.68, u);
              const lhP = lerp3(portReadyPos, portPushPos, t), lhR = lerp3(portReadyRot, portPushRot, t);
              elbowL = portElb; fLMagHold = 1.0 - 0.65 * t; fLShellPush = Math.sin(t * Math.PI);
              mHandL = mat4Mul(gunM, mat4Mul(mat4Translate(...lhP), rotYXZ(...lhR)));
              if (u < 0.62) {
                const mHr = mat4Mul(mat4Translate(...portReadyPos), rotYXZ(...portReadyRot));
                const shStart = mat4TransformPt(mHr, [0.012, -0.024, -0.088]);
                const pushT = s5(0.38, 0.61, u);
                const shLoc = lerp3(shStart, [0.0, 0.032, -0.120], pushT);
                mShell12g = mat4Mul(gunM, mat4Mul(mat4Translate(...shLoc), rotYXZ(0.0, 0.14 * (1.0 - pushT), 0.0)));
              }
            } else {
              const t = s5(0.68, 1.0, u);
              const lhP = lerp3(portPushPos, pouchPos, t), lhR = lerp3(portPushRot, pouchRot, t);
              elbowL = lerp3(portElb, pouchElb, t); fLOpen = 0.55 * t;
              mHandL = mat4Mul(gunM, mat4Mul(mat4Translate(...lhP), rotYXZ(...lhR)));
            }
          } else {
            const t = s5(0.0, 1.0, u);
            const lhP = lerp3(pouchPos, sgForendPos, t), lhR = lerp3(pouchRot, sgForendRot, t);
            mHandL = mat4Mul(gunM, mat4Mul(mat4Translate(...lhP), rotYXZ(...lhR)));
            elbowL = lerp3(pouchElb, forendElb, t); fLOpen = 0.55 * (1.0 - t);
          }
        } else if (this.reloadType === "empty") {
          const restP = isAk ? akForendPos : (isM4 ? m4ForendPos : (isMp7 ? mp7ForendPos : baseLhPos));
          const restR = isAk ? akForendRot : (isM4 ? m4ForendRot : (isMp7 ? mp7ForendRot : baseLhRot));
          const restElb = (isRifle || isMp7) ? [-0.56, -0.52, -0.05] : [-0.34, -0.44, 0.22];
          fMagRelease = Math.sin(sstep(0.05, 0.21, u) * Math.PI);
          if (u < 0.17) {
            magVis = true; magHasRound = false;
            const d = 0.075 * sstep(0.11, 0.17, u);
            mMag = isAk ? akMagTransform(d, 0.34 * sstep(0.08, 0.17, u)) : mat4Mul(gunM, mat4Translate(ma[0] * d, ma[1] * d, ma[2] * d));
            const t = sstep(0.0, 0.17, u);
            const pStart = mat4TransformPt(gunM, restP);
            const lhW = lerp3(pStart, [-0.15, -0.34, -0.24], t);
            const lhR = lerp3([gunRot[0] + restR[0], gunRot[1] + restR[1], gunRot[2] + restR[2]], [-0.22, -0.25, 1.35], t);
            mHandL = mat4Mul(mat4Translate(...lhW), rotYXZ(...lhR));
            elbowL = lerp3(restElb, [-0.44, -0.68, 0.06], t); fLOpen = 0.85 * t;
          } else if (u < 0.30) {
            magVis = false; magHasRound = true;
            const t = sstep(0.17, 0.30, u);
            const lhW = lerp3([-0.15, -0.34, -0.24], offscreenHandPos, t);
            mHandL = mat4Mul(mat4Translate(...lhW), rotYXZ(-0.20, -0.12, 1.48));
            elbowL = [-0.46, -0.84, 0.04]; fLOpen = 0.85 * (1.0 - t); fLMagHold = t;
          } else if (u < 0.52) {
            magVis = true; magHasRound = true;
            const t = sstep(0.30, 0.52, u);
            const mp = lerp3(offscreenMagPos, entryWorld, t);
            mp[0] -= 0.022 * Math.sin(t * Math.PI); mp[2] -= 0.028 * Math.sin(t * Math.PI);
            const mr = lerp3([gunRot[0] + 0.18, gunRot[1] + 0.24, gunRot[2] + 0.32], gunRot, sstep(0.15, 0.96, t));
            mMag = mat4Mul(mat4Translate(...mp), rotYXZ(...mr));
            mHandL = mat4Mul(mMag, mat4Mul(mat4Translate(...magGripOff), rotYXZ(...magGripRot)));
            elbowL = lerp3([-0.46, -0.84, 0.04], [-0.44, -0.48, 0.06], t); fLMagHold = 1.0;
          } else if (u < 0.64) {
            magVis = true; magHasRound = true;
            const insT = sstep(0.52, 0.615, u);
            const d = lerp(entryDist, 0.0, insT * insT);
            mMag = isAk ? akMagTransform(d, 0.30 * (1.0 - insT)) : mat4Mul(gunM, mat4Translate(ma[0] * d, ma[1] * d, ma[2] * d));
            const slap = Math.sin(sstep(0.59, 0.64, u) * Math.PI);
            const hOff = [magGripOff[0], magGripOff[1] + 0.014 * slap, magGripOff[2] - 0.004 * slap];
            mHandL = mat4Mul(mMag, mat4Mul(mat4Translate(...hOff), rotYXZ(...magGripRot)));
            elbowL = [-0.44, -0.46, 0.06]; fLMagHold = 1.0 - slap; fLPalmSlap = slap;
          } else if (u < 0.72) {
            magVis = true; magHasRound = true;
            const t = sstep(0.64, 0.72, u);
            const pGrab = isAk ? [0.015, 0.055, -0.118 + slideBack] : ((isM4 || isMp7) ? [-0.036, 0.095, 0.082] : [-0.044, 0.058, 0.048 + slideBack]);
            const rGrab = isAk ? [-0.48, 0.12, 2.38] : ((isM4 || isMp7) ? [-0.18, -0.06, 1.24] : [-0.28, -0.14, 1.02]);
            const lhP = lerp3(magGripOff, pGrab, t), lhR = lerp3(magGripRot, rGrab, t);
            mHandL = mat4Mul(gunM, mat4Mul(mat4Translate(...lhP), rotYXZ(...lhR)));
            elbowL = lerp3([-0.44, -0.46, 0.06], [-0.46, -0.44, 0.14], t);
            fLOpen = Math.sin(t * Math.PI) * 0.75; fLSlidePinch = sstep(0.55, 1.0, t);
          } else if (u < 0.85) {
            magVis = true; magHasRound = true;
            const relT = sstep(0.80, 0.85, u);
            let lhP, lhR;
            if (isAk) {
              lhP = [0.015 - 0.045 * relT, 0.055 + 0.012 * relT, -0.118 + slideBack + 0.025 * relT];
              lhR = [-0.48, 0.12, 2.38 - 0.25 * relT];
            } else if (isM4 || isMp7) {
              const chP = (isM4 ? 0.068 : 0.048) * s5(0.71, 0.80, u) * (1.0 - s5(0.80, 0.86, u));
              lhP = [-0.036 - 0.040 * relT, 0.095 + 0.014 * relT, 0.082 + chP + 0.025 * relT];
              lhR = [-0.18, -0.06, 1.24 + 0.24 * relT];
            } else {
              lhP = [-0.044 - 0.044 * relT, 0.058 + 0.018 * relT, 0.048 + slideBack + 0.036 * relT];
              lhR = [-0.28, -0.14, 1.02 + 0.28 * relT];
            }
            mHandL = mat4Mul(gunM, mat4Mul(mat4Translate(...lhP), rotYXZ(...lhR)));
            elbowL = [-0.46, -0.44, 0.14]; fLSlidePinch = 1.0 - relT; fLSlideFlick = relT;
          } else {
            magVis = true; magHasRound = true;
            const t = sstep(0.85, 1.0, u);
            const pFrom = isAk ? [-0.030, 0.067, -0.093] : ((isM4 || isMp7) ? [-0.076, 0.109, 0.107] : [-0.088, 0.076, 0.084]);
            const rFrom = isAk ? [-0.48, 0.12, 2.13] : ((isM4 || isMp7) ? [-0.18, -0.06, 1.48] : [-0.28, -0.14, 1.30]);
            const lhP = lerp3(pFrom, restP, t), lhR = lerp3(rFrom, restR, t);
            mHandL = mat4Mul(gunM, mat4Mul(mat4Translate(...lhP), rotYXZ(...lhR)));
            elbowL = lerp3([-0.46, -0.44, 0.14], restElb, t); fLSlideFlick = 1.0 - t;
          }
        } else {
          const restP = isAk ? akForendPos : (isM4 ? m4ForendPos : (isMp7 ? mp7ForendPos : baseLhPos));
          const restR = isAk ? akForendRot : (isM4 ? m4ForendRot : (isMp7 ? mp7ForendRot : baseLhRot));
          const restElb = (isRifle || isMp7) ? [-0.56, -0.52, -0.05] : [-0.34, -0.44, 0.22];
          fMagRelease = Math.sin(sstep(0.08, 0.22, u) * Math.PI);
          if (u < 0.15) {
            const t = sstep(0.0, 0.15, u);
            magVis = true; magHasRound = true;
            const lhP = lerp3(restP, magGripOff, t), lhR = lerp3(restR, magGripRot, t);
            mHandL = mat4Mul(gunM, mat4Mul(mat4Translate(...lhP), rotYXZ(...lhR)));
            elbowL = lerp3(restElb, [-0.44, -0.46, 0.06], t);
            fLOpen = Math.sin(t * Math.PI) * 0.55; fLMagHold = t;
          } else if (u < 0.24) {
            const t = sstep(0.15, 0.24, u);
            magVis = true; magHasRound = true;
            const d = 0.105 * t;
            mMag = isAk ? akMagTransform(d, 0.34 * t) : mat4Mul(gunM, mat4Translate(ma[0] * d, ma[1] * d, ma[2] * d));
            mHandL = mat4Mul(mMag, mat4Mul(mat4Translate(...magGripOff), rotYXZ(...magGripRot)));
            elbowL = [-0.44, -0.48, 0.06]; fLMagHold = 1.0;
          } else if (u < 0.40) {
            magVis = false; magHasRound = true;
            const t = sstep(0.24, 0.40, u);
            const pRel = mat4TransformPt(gunM, [
              ma[0] * 0.105 + magGripOff[0], ma[1] * 0.105 + magGripOff[1], ma[2] * 0.105 + magGripOff[2],
            ]);
            const lhW = lerp3(pRel, offscreenHandPos, t);
            const lhR = lerp3([gunRot[0] + magGripRot[0], gunRot[1] + magGripRot[1], gunRot[2] + magGripRot[2]], [-0.20, -0.14, 1.48], t);
            mHandL = mat4Mul(mat4Translate(...lhW), rotYXZ(...lhR));
            elbowL = lerp3([-0.44, -0.48, 0.06], [-0.46, -0.84, 0.04], t);
            fLOpen = Math.sin(t * Math.PI) * 0.90; fLMagHold = sstep(0.65, 1.0, t);
          } else if (u < 0.62) {
            magVis = true; magHasRound = true;
            const t = sstep(0.40, 0.62, u);
            const mp = lerp3(offscreenMagPos, entryWorld, t);
            mp[0] -= 0.022 * Math.sin(t * Math.PI); mp[2] -= 0.028 * Math.sin(t * Math.PI);
            const mr = lerp3([gunRot[0] + 0.18, gunRot[1] + 0.24, gunRot[2] + 0.32], gunRot, sstep(0.15, 0.96, t));
            mMag = mat4Mul(mat4Translate(...mp), rotYXZ(...mr));
            mHandL = mat4Mul(mMag, mat4Mul(mat4Translate(...magGripOff), rotYXZ(...magGripRot)));
            elbowL = lerp3([-0.46, -0.84, 0.04], [-0.44, -0.48, 0.06], t); fLMagHold = 1.0;
          } else if (u < 0.77) {
            magVis = true; magHasRound = true;
            const insT = sstep(0.62, 0.74, u);
            const d = lerp(entryDist, 0.0, insT * insT);
            mMag = isAk ? akMagTransform(d, 0.28 * (1.0 - insT)) : mat4Mul(gunM, mat4Translate(ma[0] * d, ma[1] * d, ma[2] * d));
            const slap = Math.sin(sstep(0.71, 0.77, u) * Math.PI);
            const hOff = [magGripOff[0], magGripOff[1] + 0.014 * slap, magGripOff[2] - 0.004 * slap];
            mHandL = mat4Mul(mMag, mat4Mul(mat4Translate(...hOff), rotYXZ(...magGripRot)));
            elbowL = [-0.44, -0.46, 0.06]; fLMagHold = 1.0 - slap; fLPalmSlap = slap;
          } else {
            magVis = true; magHasRound = true;
            const t = sstep(0.77, 1.0, u);
            const lhP = lerp3(magGripOff, restP, t), lhR = lerp3(magGripRot, restR, t);
            mHandL = mat4Mul(gunM, mat4Mul(mat4Translate(...lhP), rotYXZ(...lhR)));
            elbowL = lerp3([-0.44, -0.46, 0.06], restElb, t);
            fLOpen = Math.sin(t * Math.PI) * 0.45;
          }
        }
      } else if (this.suppAnimActive && isMp5s) {
        const su = clamp01(this.suppAnimTimer / Math.max(this.suppAnimDuration, 1e-4));
        const att = this.suppTargetState;
        const reach = s5(0.0, 0.22, su) * (1.0 - s5(0.80, 1.0, su));
        const tw = Math.sin(su * Math.PI * 6.0) * s5(0.20, 0.28, su) * (1.0 - s5(0.72, 0.80, su));
        const pullZ = (!att) ? (-0.095 * s5(0.52, 0.78, su)) : (-0.095 * (1.0 - s5(0.24, 0.56, su)));
        const lhP = lerp3(mp5sForendPos, [-0.022, 0.056, -0.435 + pullZ], reach);
        const lhR = lerp3(mp5sForendRot, [-0.22, 0.14, 1.72 + 0.48 * tw], reach);
        mHandL = mat4Mul(gunM, mat4Mul(mat4Translate(...lhP), rotYXZ(...lhR)));
        elbowL = lerp3([-0.34, -0.38, 0.16], [-0.42, -0.44, -0.02], reach);
        fLMagHold = 0.88 * reach;
      } else if (mcW > 0.01) {
        const restP = isSg ? sgForendPos : (isDb ? dbForendPos : (isAk ? akForendPos : (isM4 ? m4ForendPos : (isM110 ? m110ForendPos : (isMp5s ? mp5sForendPos : (isMp7 ? mp7ForendPos : baseLhPos))))));
        const restR = isSg ? sgForendRot : (isDb ? dbForendRot : (isAk ? akForendRot : (isM4 ? m4ForendRot : (isM110 ? m110ForendRot : (isMp5s ? mp5sForendRot : (isMp7 ? mp7ForendRot : baseLhRot))))));
        const modeMc = this.magCheckMode || "mag_loaded";
        if (modeMc === "double_barrel") {
          // Break/lower barrels open to reveal twin 12ga chambers, hold forend with left hand, then snap shut!
          const dbAngMc = this._dbBreakAngle();
          if (dbAngMc > 0.14) {
            if (this.magAmmo >= 1) extraShells12g.push(mat4Mul(mSlide, mat4Translate(-0.0138, 0.0725, -0.158)));
            if (this.magAmmo >= 2) extraShells12g.push(mat4Mul(mSlide, mat4Translate(0.0138, 0.0725, -0.158)));
          }
          fDbLever = Math.sin(s5(0.04, 0.28, mcU) * Math.PI);
          mHandL = mat4Mul(mSlide, mat4Mul(mat4Translate(...dbForendPos), rotYXZ(...dbForendRot)));
          elbowL = [-0.56, -0.54, -0.06];
          fLSlidePinch = 0.50 * s5(0.10, 0.30, mcU) * (1.0 - s5(0.75, 0.95, mcU));
        } else if (modeMc === "revolver") {
          // Swing cylinder out -> inspect -> left hand sweeps down to spin cylinder -> snaps shut!
          const reachSpin = s5(0.32, 0.44, mcU) * (1.0 - s5(0.68, 0.80, mcU));
          const spinSweep = Math.sin(s5(0.42, 0.68, mcU) * Math.PI * 2.0);
          const clearT = s5(0.06, 0.20, mcU) * (1.0 - s5(0.80, 0.96, mcU));
          const inspectP = [-0.11, -0.085, 0.055];
          const inspectR = [-0.28, -0.22, 1.85];
          const spinP = [-0.072, 0.046 + 0.022 * spinSweep, -0.068];
          const spinR = [-0.24, 0.18 + 0.28 * spinSweep, 1.48];
          const midP = lerp3(restP, inspectP, clearT);
          const midR = lerp3(restR, inspectR, clearT);
          const lhP = lerp3(midP, spinP, reachSpin);
          const lhR = lerp3(midR, spinR, reachSpin);
          mHandL = mat4Mul(gunM, mat4Mul(mat4Translate(...lhP), rotYXZ(...lhR)));
          elbowL = lerp3([-0.34, -0.38, 0.16], [-0.46, -0.42, 0.02], Math.max(clearT, reachSpin));
          fLOpen = 0.62 * clearT * (1.0 - 0.55 * reachSpin);
          fLSlideFlick = 0.88 * reachSpin;
          fMagRelease = Math.sin(s5(0.06, 0.22, mcU) * Math.PI);
        } else if (modeMc === "mag_empty" && magVis) {
          // ACT 1 (0.00..0.42): Pull out real empty 3D mag & re-seat it!
          // ACT 2 (0.42..1.00): Rack bolt / slide / charging handle 3x!
          magHasRound = false;
          if (mcU < 0.43) {
            const pull = s5(0.08, 0.20, mcU) * (1.0 - s5(0.28, 0.38, mcU));
            const view = s5(0.14, 0.23, mcU) * (1.0 - s5(0.26, 0.34, mcU));
            const reach = s5(0.01, 0.09, mcU) * (1.0 - s5(0.37, 0.43, mcU));
            const mdX = ma[0] * (0.140 * pull) - 0.076 * view;
            const mdY = ma[1] * (0.140 * pull) + 0.088 * view;
            const mdZ = ma[2] * (0.140 * pull) + 0.050 * view;
            mMag = mat4Mul(gunM, mat4Mul(mat4Translate(mdX, mdY, mdZ), rotYXZ(0.50 * view, 0.30 * view, -0.42 * view)));
            if (pull > 0.01) {
              mHandL = mat4Mul(mMag, mat4Mul(mat4Translate(...magGripOff), rotYXZ(...magGripRot)));
            } else {
              const lhP = lerp3(restP, magGripOff, reach), lhR = lerp3(restR, magGripRot, reach);
              mHandL = mat4Mul(gunM, mat4Mul(mat4Translate(...lhP), rotYXZ(...lhR)));
            }
            elbowL = lerp3([-0.34, -0.38, 0.16], [-0.48, -0.46, 0.04], Math.max(reach, pull));
            fLMagHold = Math.max(reach, pull) * 0.92;
            fMagRelease = Math.sin(s5(0.05, 0.14, mcU) * Math.PI);
          } else {
            const reach = s5(0.43, 0.49, mcU) * (1.0 - s5(0.88, 0.98, mcU));
            const boltZ = (isM110 || isM4 || isAk) ? -0.095 : (isMp5s ? -0.245 : -0.045);
            const boltY = (isM110 || isM4 || isAk || isMp5s) ? 0.088 : 0.062;
            const lhP = lerp3(restP, [-0.026, boltY, boltZ + mcp.slideExtra], reach);
            const lhR = lerp3(restR, [-0.24, -0.18, 1.56], reach);
            mHandL = mat4Mul(gunM, mat4Mul(mat4Translate(...lhP), rotYXZ(...lhR)));
            elbowL = lerp3([-0.34, -0.38, 0.16], [-0.42, -0.34, 0.08], reach);
            fLSlidePinch = 0.94 * reach;
          }
        } else if (modeMc === "shotgun" && this.magCheckEmpty) {
          const reach = s5(0.06, 0.18, mcU) * (1.0 - s5(0.84, 0.98, mcU));
          const lhP = lerp3(restP, [sgForendPos[0], sgForendPos[1], sgForendPos[2] + mcp.slideExtra], reach);
          mHandL = mat4Mul(gunM, mat4Mul(mat4Translate(...lhP), rotYXZ(...sgForendRot)));
          fLSlidePinch = 0.90 * reach;
        } else if (modeMc === "shotgun") {
          const chT = s5(0.12, 0.30, mcU) * (1.0 - s5(0.72, 0.90, mcU));
          const rackT = s5(0.26, 0.38, mcU) * (1.0 - s5(0.56, 0.70, mcU));
          const lhP = lerp3(restP, [-0.028, 0.070, -0.122 + mcp.slideExtra], Math.max(chT, rackT));
          const lhR = lerp3(restR, [-0.18, -0.14, 1.72], Math.max(chT, rackT));
          mHandL = mat4Mul(gunM, mat4Mul(mat4Translate(...lhP), rotYXZ(...lhR)));
          fLOpen = 0.55 * chT;
          fLSlidePinch = 0.72 * rackT;
          if (this.magAmmo > 0 && mcU >= 0.32 && mcU <= 0.72) {
            mShell12g = mat4Mul(gunM, mat4Mul(mat4Translate(0.006, 0.077, -0.124), rotYXZ(-0.04, 0.02, 0.0)));
          }
        } else if (magVis) {
          // mag_loaded: pull out loaded mag, view brass, slap back in!
          const pull = s5(0.14, 0.34, mcU) * (1.0 - s5(0.70, 0.84, mcU));
          const view = s5(0.28, 0.44, mcU) * (1.0 - s5(0.64, 0.78, mcU));
          const reach = s5(0.02, 0.16, mcU) * (1.0 - s5(0.84, 0.98, mcU));
          const tiltSw = Math.sin(this.elapsed * 4.5) * 0.16 * view;
          const mdX = ma[0] * (0.145 * pull) - 0.082 * view;
          const mdY = ma[1] * (0.145 * pull) + 0.095 * view;
          const mdZ = ma[2] * (0.145 * pull) + 0.055 * view;
          mMag = mat4Mul(gunM, mat4Mul(mat4Translate(mdX, mdY, mdZ), rotYXZ(0.55 * view + tiltSw, 0.32 * view, -0.45 * view)));
          if (pull > 0.01) {
            mHandL = mat4Mul(mMag, mat4Mul(mat4Translate(...magGripOff), rotYXZ(...magGripRot)));
          } else {
            const lhP = lerp3(restP, magGripOff, reach), lhR = lerp3(restR, magGripRot, reach);
            mHandL = mat4Mul(gunM, mat4Mul(mat4Translate(...lhP), rotYXZ(...lhR)));
          }
          elbowL = lerp3([-0.34, -0.38, 0.16], [-0.48, -0.46, 0.04], Math.max(reach, pull));
          fLMagHold = Math.max(reach, pull) * 0.92;
          fMagRelease = Math.sin(s5(0.08, 0.24, mcU) * Math.PI);
        } else {
          const chT = s5(0.12, 0.32, mcU) * (1.0 - s5(0.72, 0.92, mcU));
          mHandL = mat4Mul(gunM, mat4Mul(mat4Translate(...lerp3(restP, [-0.095, -0.065, 0.055], chT)), rotYXZ(...lerp3(restR, [-0.28, -0.22, 1.80], chT))));
          fLOpen = 0.55 * chT;
        }
      } else if (insW > 0.01) {
        if (isLg) {
          const shift = Math.sin(insU * Math.PI) * insW;
          const baseP = isSg ? sgForendPos : (isDb ? dbForendPos : (isAk ? akForendPos : m4ForendPos));
          const baseR = isSg ? sgForendRot : (isDb ? dbForendRot : (isAk ? akForendRot : m4ForendRot));
          const parM = (isSg || isDb) ? mSlide : gunM;
          const lhP = [baseP[0] - 0.004 * shift, baseP[1] - 0.004 * shift, baseP[2] + 0.022 * shift];
          const lhR = [baseR[0] + 0.08 * shift, baseR[1] + 0.06 * shift, baseR[2] - 0.14 * shift];
          mHandL = mat4Mul(parM, mat4Mul(mat4Translate(...lhP), rotYXZ(...lhR)));
          fLOpen = 0.22 * shift;
        } else {
          const clearT = s5(0.08, 0.30, insU) * (1.0 - s5(0.78, 0.96, insU)) * insW;
          const lhP = lerp3(baseLhPos, [-0.115, -0.105, 0.082], clearT);
          const lhR = lerp3(baseLhRot, [-0.32, -0.28, 1.88], clearT);
          mHandL = mat4Mul(gunM, mat4Mul(mat4Translate(...lhP), rotYXZ(...lhR)));
          fLOpen = 0.68 * clearT;
        }
      } else if (this.sprintBlend > 0.01 || kickProgress > 0.001) {
        const kSin = (kickProgress > 0.001) ? Math.sin(kickProgress * Math.PI) : 0.0;
        if (isLg) {
          fLOpen = Math.max(this.sprintBlend * 0.18, kSin * 0.45);
          if (kSin > 0.01) {
            const fpB = isSg ? sgForendPos : (isDb ? dbForendPos : (isAk ? akForendPos : m4ForendPos));
            const frB = isSg ? sgForendRot : (isDb ? dbForendRot : (isAk ? akForendRot : m4ForendRot));
            const parM = (isSg || isDb) ? mSlide : gunM;
            const lhP = lerp3(fpB, [fpB[0] - 0.04, fpB[1] - 0.04, fpB[2] + 0.05], kSin);
            const lhR = lerp3(frB, [frB[0] - 0.15, frB[1] - 0.12, frB[2] - 0.25], kSin);
            mHandL = mat4Mul(parM, mat4Mul(mat4Translate(...lhP), rotYXZ(...lhR)));
          }
        } else {
          const relax = Math.max(this.sprintBlend * 0.65, kSin);
          const lhP = lerp3(baseLhPos, [-0.13, -0.12, 0.06], relax);
          const lhR = lerp3(baseLhRot, [-0.30, -0.18, 1.82], relax);
          mHandL = mat4Mul(gunM, mat4Mul(mat4Translate(...lhP), rotYXZ(...lhR)));
          fLOpen = relax * 0.78;
        }
      }

      const wristL = [mHandL[3], mHandL[7], mHandL[11]];
      const mForearmL = aimForearmMatrix(wristL, elbowL, [0, 1, 0]);

      // ---- 4. High-Mobility 20-Joint Anatomical Finger Kinematics (11 Bones per Hand!) ----
      const sc = (gender === "female") ? 0.90 : 1.00;
      const tEl = this.elapsed;
      const kSinF = (kickProgress > 0.001) ? Math.sin(kickProgress * Math.PI) : 0.0;
      const drawOpenI = 1.0 - sstep(0.00, 0.74, this.equipT);
      const drawOpenM = 1.0 - sstep(0.06, 0.80, this.equipT);
      const drawOpenR = 1.0 - sstep(0.12, 0.86, this.equipT);
      const drawOpenP = 1.0 - sstep(0.18, 0.94, this.equipT);

      const disc = Math.max(
        this.reloading ? 1.0 : 0.0,
        this.sprintBlend,
        sliding ? 0.90 : 0.0,
        insW * 0.96,
        drawOpenI,
        kSinF * 0.85
      );

      const idleW = (1.0 - 0.50 * this.adsT) * 1.20;
      const walkW = Math.min(1.0, this.bobAmp * 60.0);
      const insFiddle = insW * 0.16;
      const bTh  = (Math.sin(tEl * 2.2 + 0.4) * 0.085 + Math.sin(this.bobPhase) * 0.055 * walkW + Math.sin(tEl * 3.1) * insFiddle) * idleW;
      const bIdx = (Math.sin(tEl * 2.5 + 0.1) * 0.075 + Math.cos(this.bobPhase) * 0.045 * walkW + Math.cos(tEl * 2.7) * insFiddle * 0.6) * idleW;
      const bMid = (Math.sin(tEl * 2.0 + 1.1) * 0.095 + Math.sin(this.bobPhase + 0.4) * 0.065 * walkW + Math.sin(tEl * 2.9 + 0.8) * insFiddle) * idleW;
      const bRng = (Math.sin(tEl * 1.8 + 2.0) * 0.125 + Math.sin(this.bobPhase + 0.9) * 0.080 * walkW + Math.sin(tEl * 3.3 + 1.7) * insFiddle * 1.2) * idleW;
      const bPnk = (Math.sin(tEl * 1.6 + 2.9) * 0.165 + Math.sin(this.bobPhase + 1.5) * 0.105 * walkW + Math.sin(tEl * 3.6 + 2.5) * insFiddle * 1.4) * idleW;

      const recSq = Math.min(1.0, this.recoilPitch * 3.6);
      const tension = Math.min(1.0, recSq + 0.45 * fRevCock);
      const sgWrap = (isSg || isDb) ? 0.84 : (isRifle ? 0.96 : 1.00);

      const rThMcp = 0.24 - 0.18 * drawOpenI + 0.28 * fMagRelease - 0.36 * fDbLever - 0.56 * fRevCock + 0.10 * tension + bTh * 0.8;
      const rThYaw = 0.48 - 0.22 * drawOpenI + 0.32 * fMagRelease + 0.28 * fDbLever + 0.22 * fRevCock - ((isSg || isDb) ? 0.14 : 0.0) + bTh * 0.5;
      const rThRol = 0.62 - 0.24 * fRevCock + 0.20 * fMagRelease;
      const rThPip = 0.42 - 0.26 * drawOpenI + 0.36 * fMagRelease + 0.24 * fRevCock + 0.16 * fDbLever + 0.10 * tension + bTh;

      const rIdxMcp = lerp(0.36 + 0.36 * trigAmt, 0.06, disc) + bIdx * 0.55;
      const rIdxYaw = lerp(0.10 + 0.08 * trigAmt, -0.06, disc);
      const rIdxPip = lerp(0.62 + 0.48 * trigAmt, 0.10, disc) + bIdx * 0.65;

      const rMidMcp = Math.max(0.14, 1.44 * sgWrap + 0.12 * tension - 0.72 * drawOpenM - 0.14 * fMagRelease + bMid * 0.65);
      const rMidPip = Math.max(0.16, 1.46 * sgWrap + 0.14 * tension - 0.76 * drawOpenM + bMid * 0.85);
      const rRngMcp = Math.max(0.14, 1.48 * sgWrap + 0.14 * tension - 0.80 * drawOpenR - 0.18 * fMagRelease + bRng * 0.75);
      const rRngPip = Math.max(0.16, 1.50 * sgWrap + 0.16 * tension - 0.84 * drawOpenR + bRng * 0.95);
      const rPnkMcp = Math.max(0.12, 1.50 * sgWrap + 0.16 * tension - 0.88 * drawOpenP - 0.22 * fMagRelease + bPnk * 0.85);
      const rPnkPip = Math.max(0.14, 1.52 * sgWrap + 0.18 * tension - 0.92 * drawOpenP + bPnk * 1.05);

      const poseR = {
        thumb:  [rThMcp,  rThYaw,  rThRol, rThPip],
        index:  [rIdxMcp, rIdxYaw, 0.04,   rIdxPip],
        middle: [rMidMcp, 0.04,    0.00,   rMidPip],
        ring:   [rRngMcp, 0.00,   -0.03,   rRngPip],
        pinky:  [rPnkMcp, -0.06,  -0.07,   rPnkPip],
      };
      const fingersR = this._build11BoneHandMats(mHandR, poseR, false, sc);

      const lOpenI = Math.max(fLOpen * 0.82, fLPalmSlap * 0.88, fLSlideFlick * 0.92, drawOpenI * 0.80);
      const lOpenM = Math.max(fLOpen * 0.92, fLPalmSlap * 0.94, fLSlideFlick * 0.98, drawOpenM * 0.88);
      const lOpenR = Math.max(fLOpen * 1.02, fLPalmSlap * 0.98, fLSlideFlick * 1.05, drawOpenR * 0.95);
      const lOpenP = Math.max(fLOpen * 1.12, fLPalmSlap * 1.05, fLSlideFlick * 1.14, drawOpenP * 1.05);

      let lBaseMcp, lBasePip, lThBMcp, lThBYaw, lThBPip;
      if (isSg || isDb || isAk) {
        lBaseMcp = 0.68; lBasePip = 0.74;
        lThBMcp = 0.18; lThBYaw = -0.26; lThBPip = 0.20;
      } else if (isM4 || isMp7) {
        lBaseMcp = 1.32; lBasePip = 1.38;
        lThBMcp = 0.14; lThBYaw = -0.20; lThBPip = 0.18;
      } else {
        lBaseMcp = 0.94; lBasePip = 1.02;
        lThBMcp = 0.08; lThBYaw = -0.16; lThBPip = 0.12;
      }

      const lGripAdd = 0.18 * fLMagHold + 0.24 * fLSlidePinch + 0.16 * recSq;
      const lThMcp = Math.max(0.04, lThBMcp + 0.24 * fLMagHold + 0.34 * fLSlidePinch + 0.42 * fLShellPush - 0.18 * lOpenI + bTh * 0.6);
      const lThYaw = lThBYaw - 0.24 * fLMagHold - 0.28 * fLSlidePinch + 0.22 * lOpenI + bTh * 0.4;
      const lThRol = -0.58 - 0.20 * fLSlidePinch - 0.16 * fLShellPush;
      const lThPip = Math.max(0.06, lThBPip + 0.32 * fLMagHold + 0.42 * fLSlidePinch + 0.48 * fLShellPush - 0.16 * lOpenI + bTh * 0.7);

      const lIdxMcp = Math.max(0.08, lBaseMcp * (1.0 - 0.78 * lOpenI) + lGripAdd + 0.20 * fLShellPush + bIdx * 0.7);
      const lIdxPip = Math.max(0.10, lBasePip * (1.0 - 0.80 * lOpenI) + lGripAdd + 0.24 * fLShellPush + bIdx * 0.85);
      const lMidMcp = Math.max(0.08, (lBaseMcp + 0.06) * (1.0 - 0.82 * lOpenM) + lGripAdd + bMid * 0.75);
      const lMidPip = Math.max(0.10, (lBasePip + 0.06) * (1.0 - 0.84 * lOpenM) + lGripAdd + bMid * 0.90);
      const lRngMcp = Math.max(0.08, (lBaseMcp + 0.08) * (1.0 - 0.86 * lOpenR) + lGripAdd * 0.8 - 0.24 * fLShellPush + bRng * 0.85);
      const lRngPip = Math.max(0.10, (lBasePip + 0.08) * (1.0 - 0.88 * lOpenR) + lGripAdd * 0.8 - 0.24 * fLShellPush + bRng * 1.00);
      const lPnkMcp = Math.max(0.08, (lBaseMcp + 0.10) * (1.0 - 0.90 * lOpenP) + lGripAdd * 0.7 - 0.32 * fLShellPush + bPnk * 0.95);
      const lPnkPip = Math.max(0.10, (lBasePip + 0.10) * (1.0 - 0.92 * lOpenP) + lGripAdd * 0.7 - 0.34 * fLShellPush + bPnk * 1.10);

      const poseL = {
        thumb:  [lThMcp,  lThYaw,  lThRol, lThPip],
        index:  [lIdxMcp, -0.04,  -0.03,   lIdxPip],
        middle: [lMidMcp,  0.00,   0.00,   lMidPip],
        ring:   [lRngMcp,  0.03,   0.03,   lRngPip],
        pinky:  [lPnkMcp,  0.08,   0.07,   lPnkPip],
      };
      const fingersL = this._build11BoneHandMats(mHandL, poseL, true, sc);

      let mFlash = null;
      if (this.muzzleTimer > 0.0) {
        const spec = this.activeSpec;
        const baseScale = (isMp5s && !this.mp5Suppressed && spec.flash_scale_unsupp)
          ? spec.flash_scale_unsupp
          : (spec.flash_scale || 1.0);
        const durF = (isSg || isDb) ? 0.078 : (isRev ? 0.062 : 0.048);
        const scF = baseScale * (0.85 + 0.45 * (this.muzzleTimer / durF));
        let ml = (isMp5s && !this.mp5Suppressed && spec.muzzle_local_unsupp)
          ? spec.muzzle_local_unsupp
          : (spec.muzzle_local || this.MUZZLE_LOCAL);
        if (isDb) ml = [this.dbLastBarrel === 0 ? -0.0138 : 0.0138, ml[1], ml[2]];
        mFlash = mat4Mul(
          gunM,
          mat4Mul(mat4Translate(ml[0], ml[1], ml[2]), mat4Mul(mat4RotZ(this.muzzleRoll), mat4Scale(scF, scF, scF)))
        );
      }

      // MP5-S Detachable Suppressor animated spin & thread-off!
      let suppressorVis = false;
      let mSuppressor = gunM;
      if (isMp5s) {
        if (this.suppAnimActive) {
          const su = clamp01(this.suppAnimTimer / Math.max(this.suppAnimDuration, 1e-4));
          const att = this.suppTargetState;
          let angZ = 0.0, offZ = 0.0;
          if (!att) {
            suppressorVis = (su < 0.80);
            angZ = s5(0.18, 0.72, su) * Math.PI * 2.0 * 2.4;
            offZ = -0.115 * s5(0.42, 0.80, su);
          } else {
            suppressorVis = (su >= 0.20);
            angZ = -(1.0 - s5(0.24, 0.78, su)) * Math.PI * 2.0 * 2.4;
            offZ = -0.115 * (1.0 - s5(0.20, 0.58, su));
          }
          mSuppressor = mat4Mul(
            gunM,
            mat4Mul(mat4Translate(0.0, 0.075, offZ), mat4Mul(mat4RotZ(angZ), mat4Translate(0.0, -0.075, 0.0)))
          );
        } else {
          suppressorVis = this.mp5Suppressed;
          mSuppressor = gunM;
        }
      }

      // Revolver per-chamber 3D .357 rounds + visible fresh cartridge slid into open chamber during "load"!
      const revChamberRounds = [];
      let mRevHandRound = null;
      if (isRev) {
        const [cxR, cyR] = this.REV_CYL_PIVOT;
        for (let ci = 0; ci < 8; ci++) {
          const stC = this.revChambers[ci];
          if (stC === "empty") continue;
          const angC = Math.PI * 0.5 + ci * (Math.PI * 2.0 / 8.0);
          const chX = cxR + Math.cos(angC) * this.REV_CHAMBER_RING_R;
          const chY = cyR + Math.sin(angC) * this.REV_CHAMBER_RING_R;
          revChamberRounds.push({
            m: mat4Mul(mSlide, mat4Translate(chX, chY, this.REV_CHAMBER_REAR_Z)),
            live: stC === "live",
          });
        }
        if (this.reloading && this.reloadType === "revolver" && this.revReloadPhase === "load") {
          const uRl = clamp01(this.reloadTimer / Math.max(this.reloadDuration, 1e-4));
          if (uRl < 0.60) {
            const ciL = Math.max(0, Math.min(7, this.revLoadChamber));
            const angC = Math.PI * 0.5 + ciL * (Math.PI * 2.0 / 8.0);
            const chX = cxR + Math.cos(angC) * this.REV_CHAMBER_RING_R;
            const chY = cyR + Math.sin(angC) * this.REV_CHAMBER_RING_R;
            if (uRl < 0.42) {
              const tR = s5(0.0, 0.42, uRl);
              const pFrom = mat4TransformPt(gunM, [-0.095, -0.145, 0.045]);
              const pTo = mat4TransformPt(mSlide, [chX, chY, this.REV_CHAMBER_REAR_Z + 0.058]);
              const pCur = lerp3(pFrom, pTo, tR);
              mRevHandRound = mat4Mul(
                mat4Translate(pCur[0], pCur[1], pCur[2]),
                mat4Mul(rotYXZ(gunRot[0], gunRot[1] + 0.35 * (1.0 - tR), gunRot[2]), mat4Scale(1.18, 1.18, 1.18))
              );
            } else {
              const tIn = s5(0.42, 0.60, uRl);
              const zSl = this.REV_CHAMBER_REAR_Z + 0.058 * (1.0 - tIn);
              mRevHandRound = mat4Mul(mSlide, mat4Translate(chX, chY, zSl));
            }
          }
        }
      }

      const glbSc = 0.87 * sc;
      const handScM = mat4Scale(glbSc, glbSc, glbSc);
      return {
        gun: gunM, slide: mSlide, trigger: mTrigger, hammer: mHammer,
        suppressor: mSuppressor, suppressorVis,
        magazine: mMag, magVisible: magVis, magHasRound, shell12g: mShell12g,
        extraShells12g,
        revChamberRounds,
        revHandRound: mRevHandRound,
        handR: mHandR, forearmR: mForearmR, handL: mHandL, forearmL: mForearmL,
        handRGlb: mat4Mul(mHandR, handScM),
        handLGlb: mat4Mul(mHandL, handScM),
        wristR, wristL, elbowR, elbowL, poseR, poseL, handScale: sc,
        fingersR, fingersL,
        flash: mFlash,
      };
    }
  }

  async function boot() {
    const banner = document.getElementById("loading-banner");
    const canvas = document.getElementById("gl-canvas");
    const gl = canvas.getContext("webgl2", { antialias: false, alpha: false, depth: false });
    if (!gl) {
      banner.textContent = "Ошибка: браузер не поддерживает WebGL2.";
      return;
    }

    const resp = await fetch("/api/scene", { cache: "no-store" });
    if (!resp.ok) {
      banner.textContent = "Не удалось загрузить /api/scene: " + resp.status;
      return;
    }
    const scene = await resp.json();

    const sceneProg = createProgram(gl, scene.shaders.sceneVert, scene.shaders.sceneFrag);
    const postProg = createProgram(gl, scene.shaders.postVert, scene.shaders.postFrag);

    // Uniform locations
    const su = {
      gaze:uLoc(gl,sceneProg,"u_gaze"),
      acting: uLoc(gl, sceneProg, "u_acting"),
      model: uLoc(gl, sceneProg, "u_model"),
      view: uLoc(gl, sceneProg, "u_view"),
      proj: uLoc(gl, sceneProg, "u_proj"),
      jitterEn: uLoc(gl, sceneProg, "u_jitter_enabled"),
      jitterGrid: uLoc(gl, sceneProg, "u_jitter_grid"),
      tex: uLoc(gl, sceneProg, "u_tex"),
      emissive: uLoc(gl, sceneProg, "u_emissive"),
      colorTint: uLoc(gl, sceneProg, "u_color_tint"),
      alpha: uLoc(gl, sceneProg, "u_alpha"),
      softParticle: uLoc(gl, sceneProg, "u_soft_particle"),
      camPos: uLoc(gl, sceneProg, "u_cam_pos"),
      camDir: uLoc(gl, sceneProg, "u_cam_dir"),
      ambient: uLoc(gl, sceneProg, "u_ambient_boost"),
      numLights: uLoc(gl, sceneProg, "u_num_lights"),
      lightPos: uLoc(gl, sceneProg, "u_light_pos"),
      lightCol: uLoc(gl, sceneProg, "u_light_color"),
      lightInt: uLoc(gl, sceneProg, "u_light_intensity"),
      lightRad: uLoc(gl, sceneProg, "u_light_radius"),
      flashOn: uLoc(gl, sceneProg, "u_flash_on"),
      fogColor: uLoc(gl, sceneProg, "u_fog_color"),
      mistClock:uLoc(gl, sceneProg,"u_mist_clock"),
      fogStart: uLoc(gl, sceneProg, "u_fog_start"),
      fogEnd: uLoc(gl, sceneProg, "u_fog_end"),
    };
    const pu = {
      sceneTex: uLoc(gl, postProg, "u_scene_tex"),
      lowRes: uLoc(gl, postProg, "u_low_res"),
      screenRes: uLoc(gl, postProg, "u_screen_res"),
      time: uLoc(gl, postProg, "u_time"),
      ditherEn: uLoc(gl, postProg, "u_dither_enabled"),
      ditherStr: uLoc(gl, postProg, "u_dither_strength"),
      colorLvls: uLoc(gl, postProg, "u_color_levels"),
      scanEn: uLoc(gl, postProg, "u_scanlines_enabled"),
      scanStr: uLoc(gl, postProg, "u_scanline_strength"),
      vigStr: uLoc(gl, postProg, "u_vignette_strength"),
      grainStr: uLoc(gl, postProg, "u_grain_strength"),
      contrast: uLoc(gl, postProg, "u_contrast"),
      saturation: uLoc(gl, postProg, "u_saturation"),
      tint: uLoc(gl, postProg, "u_tint"),
      crosshair: uLoc(gl, postProg, "u_show_crosshair"),
      crosshairSpread: uLoc(gl, postProg, "u_crosshair_spread"),
      fadeBlack: uLoc(gl, postProg, "u_fade_black"),
      scopeAmount: uLoc(gl, postProg, "u_scope_amount"),
      scopeBlur: uLoc(gl, postProg, "u_scope_blur"),
      fxImpact: uLoc(gl, postProg, "u_fx_impact"),
      fxSpeed: uLoc(gl, postProg, "u_fx_speed"),
    };

    // Upload procedural PS1 textures (bottom-up RGB bytes from Python textures.py)
    const textures = {};
    gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, false);
    gl.pixelStorei(gl.UNPACK_ALIGNMENT, 1);
    for (const [name, tinfo] of Object.entries(scene.textures)) {
      const tex = gl.createTexture();
      gl.bindTexture(gl.TEXTURE_2D, tex);
      gl.texImage2D(
        gl.TEXTURE_2D, 0, gl.RGB8,
        tinfo.width, tinfo.height, 0,
        gl.RGB, gl.UNSIGNED_BYTE,
        b64ToUint8(tinfo.rgbB64)
      );
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.REPEAT);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.REPEAT);
      textures[name] = tex;
    }

    // Upload arena + 3D menu + cigarette + leg meshes
    const worldBatches = uploadBatches(gl, scene.worldBatches);
    const menuBatches = uploadBatches(gl, scene.menuScene.batches);
    const flatMeshes = uploadBatches(gl, scene.prologue.batches);
    const glassPanes = flatMeshes.filter(b=>b.texture==='apt_glass');
    const prologueBatches = flatMeshes.filter(b=>b.texture!=='apt_glass');
    const menuCfg = scene.menuScene.config;
    const menuLights = scene.menuScene.lights;
    const cigBatches = scene.menuScene.cigarette ? {
      body: uploadBatches(gl, scene.menuScene.cigarette.body),
      ember: uploadBatches(gl, scene.menuScene.cigarette.ember),
      tipLocal: scene.menuScene.cigarette.tipLocal || [0.017, -0.068, -0.082],
      mouthLocal: scene.menuScene.cigarette.mouthLocal || [0.0, 0.068, 0.116],
    } : null;

    // Detailed 7-model classic pre-2008 passing winter traffic cars + spinning 16-sided 3D wheels!
    const tcSrc = scene.menuScene.trafficCars || null;
    const DEFAULT_CAR_SPEC = { wheel_x: 0.78, wheel_r: 0.33, wheel_w: 0.23, wheel_zf: 1.24, wheel_zr: -1.28, z_front: 2.02, z_rear: -2.05, hl_y: 0.68, hl_x: 0.54, tl_y: 0.65, ex_x: -0.52 };
    const trafficCarGpu = tcSrc ? {
      styles: tcSrc.styles || ["sedan", "hatch", "wagon", "niva", "volga", "buhanka", "e34"],
      specs: tcSrc.specs || {},
      tints: tcSrc.tints || [[0.56, 0.16, 0.18]],
      wheelL: uploadBatches(gl, tcSrc.wheelL),
      wheelR: uploadBatches(gl, tcSrc.wheelR),
      bodies: Object.fromEntries(
        Object.entries(tcSrc.bodies || {}).map(([st, d]) => [
          st,
          { tinted: uploadBatches(gl, d.tinted), fixed: uploadBatches(gl, d.fixed) },
        ])
      ),
    } : null;

    // 5 winter passerby handheld 3D props (bag, book, coffee, briefcase, box)
    const npSrc = scene.menuScene.npcProps || null;
    const npcPropsGpu = npSrc ? {
      tints: npSrc.tints || {},
      meshes: Object.fromEntries(
        Object.entries(npSrc.meshes || {}).map(([k, d]) => [
          k,
          { tinted: uploadBatches(gl, d.tinted), fixed: uploadBatches(gl, d.fixed) },
        ])
      ),
    } : null;

    function drawNpcProp(propType, npcBones, tint = [1, 1, 1]) {
      if (!npcPropsGpu || !propType || propType === "none") return;
      const pm = npcPropsGpu.meshes[propType];
      if (!pm) return;
      const mBone = (propType === "box") ? npcBones.torso : npcBones.hand_r;
      if (!mBone) return;
      gl.uniformMatrix4fv(su.model, false, toGlMat4(mBone));
      gl.uniform3f(su.colorTint, tint[0], tint[1], tint[2]);
      gl.uniform1f(su.emissive, 0.0);
      for (const b of pm.tinted) {
        gl.bindTexture(gl.TEXTURE_2D, textures[b.texture] || textures["char_fabric"]);
        gl.bindVertexArray(b.vao);
        gl.drawArrays(gl.TRIANGLES, 0, b.vertexCount);
      }
      gl.uniform3f(su.colorTint, 1.0, 1.0, 1.0);
      for (const b of pm.fixed) {
        gl.bindTexture(gl.TEXTURE_2D, textures[b.texture] || textures["crate"]);
        gl.uniform1f(su.emissive, b.emissive || 0.0);
        gl.bindVertexArray(b.vao);
        gl.drawArrays(gl.TRIANGLES, 0, b.vertexCount);
      }
      gl.uniform1f(su.emissive, 0.0);
      gl.uniform3f(su.colorTint, 1.0, 1.0, 1.0);
    }

    function drawTrafficCars(cars, yOffset = 0.0) {
      if (!trafficCarGpu || !cars || !cars.length) return;
      for (let i = 0; i < cars.length; i++) {
        const car = cars[i];
        const bm = trafficCarGpu.bodies[car.style] || trafficCarGpu.bodies.sedan;
        if (!bm) continue;
        const spec = trafficCarGpu.specs[car.style] || DEFAULT_CAR_SPEC;
        const sc = car.scale || 1.18;
        const mCar = mat4Mul(
          mat4Translate(car.pos[0], car.pos[1] + yOffset, car.pos[2]),
          mat4Mul(mat4RotY(car.yaw), mat4Scale(sc, sc, sc))
        );
        gl.uniformMatrix4fv(su.model, false, toGlMat4(mCar));
        gl.uniform3f(su.colorTint, car.tint[0], car.tint[1], car.tint[2]);
        for (const b of bm.tinted) {
          gl.bindTexture(gl.TEXTURE_2D, textures[b.texture] || textures["car_body_white"]);
          gl.uniform1f(su.emissive, 0.0);
          gl.bindVertexArray(b.vao);
          gl.drawArrays(gl.TRIANGLES, 0, b.vertexCount);
        }
        gl.uniform3f(su.colorTint, 1.0, 1.0, 1.0);
        for (const b of bm.fixed) {
          gl.bindTexture(gl.TEXTURE_2D, textures[b.texture] || textures["lada_chrome"]);
          gl.uniform1f(su.emissive, b.emissive || 0.0);
          gl.bindVertexArray(b.vao);
          gl.drawArrays(gl.TRIANGLES, 0, b.vertexCount);
        }
        const mSpin = mat4RotX(car.wheelSpin || 0.0);
        const wScR = (spec.wheel_r || 0.33) / 0.33;
        const wScW = (spec.wheel_w || 0.23) / 0.23;
        const mWScale = mat4Scale(wScW, wScR, wScR);
        const wxAbs = spec.wheel_x || 0.78;
        const wr = spec.wheel_r || 0.33;
        const wPositions = [
          [-wxAbs, spec.wheel_zf, false],
          [ wxAbs, spec.wheel_zf, true],
          [-wxAbs, spec.wheel_zr, false],
          [ wxAbs, spec.wheel_zr, true],
        ];
        gl.uniform1f(su.emissive, 0.0);
        for (let w = 0; w < wPositions.length; w++) {
          const [wx, wz, isR] = wPositions[w];
          const mWh = mat4Mul(mCar, mat4Mul(mat4Translate(wx, wr, wz), mat4Mul(mSpin, mWScale)));
          gl.uniformMatrix4fv(su.model, false, toGlMat4(mWh));
          const wList = isR ? trafficCarGpu.wheelR : trafficCarGpu.wheelL;
          for (const b of wList) {
            gl.bindTexture(gl.TEXTURE_2D, textures[b.texture] || textures["lada_tire"]);
            gl.bindVertexArray(b.vao);
            gl.drawArrays(gl.TRIANGLES, 0, b.vertexCount);
          }
        }
      }
      gl.uniform1f(su.emissive, 0.0);
      gl.uniform3f(su.colorTint, 1.0, 1.0, 1.0);
    }

    // Reusable dynamic particle VBO/VAO for dense winter snow, smoke, exhaust & headlight halos!
    const MAX_PARTICLE_QUADS = 1024;
    const particleData = new Float32Array(MAX_PARTICLE_QUADS * 6 * 8);
    const particleVbo = gl.createBuffer();
    const particleVao = gl.createVertexArray();
    gl.bindVertexArray(particleVao);
    gl.bindBuffer(gl.ARRAY_BUFFER, particleVbo);
    gl.bufferData(gl.ARRAY_BUFFER, particleData.byteLength, gl.DYNAMIC_DRAW);
    gl.enableVertexAttribArray(0);
    gl.vertexAttribPointer(0, 3, gl.FLOAT, false, 32, 0);
    gl.enableVertexAttribArray(1);
    gl.vertexAttribPointer(1, 3, gl.FLOAT, false, 32, 12);
    gl.enableVertexAttribArray(2);
    gl.vertexAttribPointer(2, 2, gl.FLOAT, false, 32, 24);
    gl.bindVertexArray(null);

    function createMenuLifeController(residential=false) {
      const SNOW_COUNT = residential?1000:820;
      const HERO_POS = menuCfg.heroPos || [-12.92, 0.14, 11.18];
      const HERO_YAW = (typeof menuCfg.heroYaw === "number") ? menuCfg.heroYaw : 2.08;
      const NPC_LANES = residential ? [[-1.53,.13,-48,18.2,0],[9.35,.13,32,-48,Math.PI],[-2.36,.13,18.0,-46,Math.PI]] : [
        [-1.55, 0.14,  15.5, -44.0, Math.PI],
        [-1.55, 0.14, -44.0,  15.5, 0.0],
        [-16.05, 0.14, -15.5, -46.0, Math.PI],
        [-16.05, 0.14, -46.0, -15.5, 0.0],
      ];
      const TRAFFIC_LANES = residential ? [[1.64,-79,48,0],[5.45,48,-79,Math.PI]] : [
        [-6.45, -106.0,  26.0, 0.0],      // Approaching camera along open right lane (+Z)
        [-8.45,   26.0, -106.0, Math.PI], // Driving away down the snowy avenue (-Z)
      ];
      const TC_STYLES = trafficCarGpu ? trafficCarGpu.styles : ["sedan", "hatch", "wagon"];
      const TC_TINTS = trafficCarGpu ? trafficCarGpu.tints : [[0.56, 0.16, 0.18]];
      // Deterministic lightweight LCG PRNG so snowfall, traffic & NPC rhythm feel balanced
      let rngState = residential ? 12072007 : 20261117;
      function rand() {
        rngState = (Math.imul(rngState, 1664525) + 1013904223) >>> 0;
        return rngState / 4294967296;
      }
      function randRange(a, b) { return a + (b - a) * rand(); }
      function pick(arr) { return arr[Math.floor(rand() * arr.length) % arr.length]; }

      function makeWinterNpcAppearance() {
        const gender = pick(["male", "female"]);
        return {
          gender,
          skin: Math.floor(rand() * 6),
          face_style: gender === "male" ? pick([0, 1, 2, 4, 6]) : pick([0, 1, 3, 5, 7]),
          hair_style: gender === "male"
            ? pick([0, 3, 7, 8, 9, 10, 11, 14, 15, 16, 17, 18, 20])
            : pick([1, 2, 3, 4, 5, 6, 12, 13, 17, 18, 19, 20]),
          hair_color: Math.floor(rand() * 8),
          top_style: pick([0, 1, 3, 4, 5, 5, 6, 6, 7, 8, 9, 10]),
          top_color: Math.floor(rand() * 8),
          bottom_style: gender === "male" ? pick([0, 1, 2, 5]) : pick([0, 1, 2, 3, 5]),
          bottom_color: Math.floor(rand() * 7),
          shoe_style: pick([0, 0, 1, 2, 3, 3, 4]),
          shoe_color: Math.floor(rand() * 6),
          accessory: pick([0, 0, 1, 1, 5, 5, 6, 6, 7, 8, 10, 2, 3]),
          accessory_color: Math.floor(rand() * 5),
          bust_size: gender === "female" ? Math.round(randRange(0.25, 0.85) * 100) / 100 : 0.5,
        };
      }

      // Four outdoor volumes around the shell. No emitter intersects the loggia or rooms.
      const snowBands=[[-36,14,-33,20.48],[-31,16,34.17,62.5],[-31,-14,20.48,34.17],[.02,15,20.48,34.17]];
      const band=i=>snowBands[i<370?0:i<865?1:i<935?2:3];
      const snow = [];
      for (let i = 0; i < SNOW_COUNT; i++) {
        const isNear = (i < 300);
        snow.push([
          randRange(-24.0, 3.0),
          randRange(0.15, 13.5),
          randRange(-34.0, 16.5),
          isNear ? randRange(0.85, 1.55) : randRange(1.15, 2.05),
          isNear ? randRange(0.062, 0.112) : randRange(0.040, 0.072),
          randRange(0.0, Math.PI * 2),
          isNear ? randRange(0.68, 0.98) : randRange(0.45, 0.85),
        ]);
      }

      if(residential)snow.forEach((fl,i)=>{const [a,b,c,d]=band(i);fl[0]=randRange(a,b);fl[2]=randRange(c,d);fl[1]=randRange(.05,22);fl[4]*=.74;});
      const state = {
        heroPos: HERO_POS,
        heroYaw: HERO_YAW,
        snow,
        smoke: [],
        npcs: [],
        trafficCars: [],
        carPassEvent: null,
        emberGlow: 0.0,
        cigTipWorld: [HERO_POS[0], HERO_POS[1] + 1.35, HERO_POS[2]],
        _exhaleAccum: 0.0,
        _emberAccum: 0.0,
        _nextNpcTimer: 3.2,
        _nextCarTimer: 4.6,
      };

      function spawnTrafficCar() {
        // Pick a lane whose entrance is clear (>= 55m from any existing car on that lane)
        const candidates = [];
        for (let idx = 0; idx < TRAFFIC_LANES.length; idx++) {
          const z0 = TRAFFIC_LANES[idx][1];
          let blocked = false;
          for (let j = 0; j < state.trafficCars.length; j++) {
            const c = state.trafficCars[j];
            if (c.lane === idx && Math.abs(c.pos[2] - z0) < 55.0) {
              blocked = true;
              break;
            }
          }
          if (!blocked) candidates.push(idx);
        }
        if (candidates.length === 0) return;
        const laneIdx = pick(candidates);
        const [lx, z0, z1, yaw] = TRAFFIC_LANES[laneIdx];
        state.trafficCars.push({
          lane: laneIdx,
          style: pick(TC_STYLES),
          tint: pick(TC_TINTS),
          pos: [lx, residential ? .085 : 0.0, z0],
          zEnd: z1,
          dirZ: z1 > z0 ? 1.0 : -1.0,
          yaw,
          scale: 1.18,
          speed: randRange(9.8, 13.6),
          wheelSpin: 0.0,
          exhaustAccum: 0.0,
        });
      }

      state.getDynamicLights = function(baseLights, yOffset = 0.0) {
        const out = (Math.abs(yOffset) > 1e-4)
          ? baseLights.map(lt => ({
              ...lt,
              position: [lt.position[0], lt.position[1] + yOffset, lt.position[2]],
            }))
          : baseLights.slice();
        if (!state.trafficCars.length) return out;
        let best = null, bestDist = 999.0;
        for (let i = 0; i < state.trafficCars.length; i++) {
          const car = state.trafficCars[i];
          const cz = car.pos[2];
          if (cz >= -58.0 && cz <= 20.0) {
            const d = Math.abs(cz + 4.0);
            if (d < bestDist) { bestDist = d; best = car; }
          }
        }
        if (!best) return out;
        if (out.length > 2) {
          out[2] = {
            position: [best.pos[0], 1.35 + yOffset, best.pos[2] + best.dirZ * 5.2],
            color: [1.0, 0.86, 0.58],
            intensity: 4.6,
            radius: 22.0,
            flickerAmount: 0.0,
            flickerSpeed: 0.0,
            pulseAmount: 0.0,
            pulseSpeed: 0.0,
            phase: 0.0,
          };
        }
        return out;
      };

      // Permanent residents use footpaths that go around the footprint, including its rear.
      // Seed them in a blind alley during the opening cut; neither the body nor its bag is recycled.
      const walks=[[[36.85,-5.8],[36.85,2.86],[35.8,3.69],[10.75,3.69],[9.06,4.40],[8.15,3.62],[8.14,-8.65],[9.34,-9.06],[35.70,-9.06],[36.85,-8.56]],[[33.42,21.3],[33.42,27.45],[32.4,27.87],[10.66,27.87],[10.16,26.85],[10.16,18.05],[10.48,17.28],[32.12,17.28],[33.42,18.37]]];
      function pathStep(n,remaining){const path=walks[n.path];while(remaining>.00001){const target=path[n.next],dx=target[0]-n.pos[0],dz=target[1]-n.pos[2],d=Math.hypot(dx,dz);if(d<.0001){n.next=(n.next+1)%path.length;continue;}const moved=Math.min(d,remaining);n.pos[0]+=dx*moved/d;n.pos[2]+=dz*moved/d;remaining-=moved;const a=Math.atan2(dx,dz)-n.yaw;n.yaw+=Math.atan2(Math.sin(a),Math.cos(a))*Math.min(1,moved*5);}}
      function inView(x,y,z){if(!state.view)return false;const [eye,d]=state.view,dx=x-eye[0],dy=y+.9-eye[1],dz=z-eye[2];return dx*d[0]+dy*d[1]+dz*d[2]>-.08*Math.hypot(dx,dy,dz);}
      function spawnNpc() {
        const used = new Set(state.npcs.map(n => n.lane));
        const avail = [];
        for (let i = 0; i < NPC_LANES.length; i++) {
          if (!used.has(i) && (residential||!inView(NPC_LANES[i][0],NPC_LANES[i][1],NPC_LANES[i][2]))) avail.push(i);
        }
        if (avail.length === 0) return;
        const weighted = [];
        for (const i of avail) {
          weighted.push(i);
          if (i < 2) weighted.push(i);
        }
        const laneIdx = pick(weighted);
        const [lx, ly, z0, z1, yaw] = NPC_LANES[laneIdx];
        const prop = pick(["none", "none", "none", "bag", "bag", "book", "coffee", "coffee", "briefcase", "box"]);
        const tintPool = (npcPropsGpu && npcPropsGpu.tints && npcPropsGpu.tints[prop]) ? npcPropsGpu.tints[prop] : [[1, 1, 1]];
        state.npcs.push({
          lane: laneIdx,
          appearance: makeWinterNpcAppearance(),
          pos: [lx + randRange(-0.22, 0.22), ly, z0],
          zEnd: z1,
          dirZ: z1 < z0 ? -1.0 : 1.0,
          yaw,
          speed: randRange(1.18, 1.62),
          walkPhase: randRange(0.0, Math.PI * 2),
          blinkOffset: randRange(0.0, 18.0),
          prop,
          propTint: pick(tintPool),
        });
      }

      state.stageStreet = function(t){
        state.npcs.length=0;spawnNpc();spawnNpc();
        state.npcs.forEach((n,i)=>{n.path=i;n.next=1;const p=walks[i][0];n.pos=[p[0],.132,p[1]];n.yaw=0;});
        state.trafficCars.length=0;spawnTrafficCar();const c=state.trafficCars[0];
        Object.assign(c,{lane:0,pos:[1.64,.085,-29.6+Math.min(t,13)*5.6],dirZ:1,yaw:0,zEnd:48,speed:5.6});
        state._nextCarTimer=12;state.smoke.length=0;
      };
      state.update = function(dt, elapsed, heroChar, updateHeroAndNpcs, updateTraffic = true) {
        const cdt = Math.min(Math.max(dt, 0.0), 0.05);
        state.carPassEvent = null;
        // 1. Dense multi-layered 3D Falling winter snow
        const windX = 0.48 + 0.24 * Math.sin(elapsed * 0.35);
        const windZ = -0.26 + 0.14 * Math.cos(elapsed * 0.27);
        for (let i = 0; i < state.snow.length; i++) {
          const fl = state.snow[i];
          fl[0] += (windX + 0.34 * Math.sin(elapsed * 1.25 + fl[5])) * cdt;
          fl[1] -= fl[3] * cdt;
          fl[2] += (windZ + 0.26 * Math.cos(elapsed * 0.98 + fl[5])) * cdt;
          if(residential){const [a,b,c,d]=band(i);if(fl[1]<.045){fl[1]=22;fl[0]=randRange(a,b);fl[2]=randRange(c,d);}if(fl[0]>b)fl[0]-=b-a;if(fl[0]<a)fl[0]+=b-a;if(fl[2]>d)fl[2]-=d-c;if(fl[2]<c)fl[2]+=d-c;continue;}
          if (fl[1] < 0.05) {
            fl[1] = 13.5;
            fl[0] = randRange(-24.0, 3.0);
            fl[2] = randRange(-34.0, 16.5);
          }
          if (fl[0] > 3.5) fl[0] -= 28.0;
          else if (fl[0] < -24.5) fl[0] += 28.0;
          if (fl[2] > 17.0) fl[2] -= 51.5;
          else if (fl[2] < -34.5) fl[2] += 51.5;
        }

        // 1b. Occasional passing winter traffic cars (COMPLETELY FROZEN when updateTraffic === false in creator!)
        if (updateTraffic) {
          state._nextCarTimer -= cdt;
          if (state._nextCarTimer <= 0.0) {
            if (state.trafficCars.length < 1) spawnTrafficCar();
            state._nextCarTimer = randRange(10.5, 19.6);
          }
          const aliveCars = [];
          for (let i = 0; i < state.trafficCars.length; i++) {
            const car = state.trafficCars[i];
            const spec = (trafficCarGpu && trafficCarGpu.specs[car.style]) ? trafficCarGpu.specs[car.style] : DEFAULT_CAR_SPEC;
            const step = car.speed * cdt;
            const oldZ = car.pos[2];
            car.pos[2] = residential && car.lane===0 && mode==='cinema' && storyTime<13 ? -29.6+storyTime*5.6 : car.pos[2]+car.dirZ*step;
            const triggerZ = residential ? 4.5 : (car.dirZ > 0.0 ? -22.0 : 24.0);
            if ((car.dirZ > 0.0 && oldZ < triggerZ && car.pos[2] >= triggerZ) ||
                (car.dirZ < 0.0 && oldZ > triggerZ && car.pos[2] <= triggerZ)) {
              state.carPassEvent = (car.dirZ > 0.0) ? "far" : "near";
            }
            car.wheelSpin += step / ((spec.wheel_r || 0.33) * car.scale);
            if (car.pos[2] >= -45.0 && car.pos[2] <= 20.0) {
              car.exhaustAccum += cdt * 11.0;
              while (car.exhaustAccum >= 1.0) {
                car.exhaustAccum -= 1.0;
                const dz = car.dirZ;
                const exX = car.pos[0] + ((spec.ex_x || -0.52) * car.scale * (dz > 0 ? 1.0 : -1.0));
                const exZ = car.pos[2] - dz * ((Math.abs(spec.z_rear || -2.05) + 0.18) * car.scale);
                state.smoke.push({
                  pos: [exX, 0.28, exZ],
                  vel: [randRange(-0.18, 0.18), randRange(0.32, 0.55), -dz * randRange(0.8, 1.4)],
                  age: 0.0,
                  life: randRange(1.0, 1.6),
                  size0: 0.10,
                  size1: randRange(0.45, 0.72),
                  alpha: 0.36,
                });
              }
            }
            const done = (car.dirZ < 0 && car.pos[2] <= car.zEnd) || (car.dirZ > 0 && car.pos[2] >= car.zEnd);
            if (!done) aliveCars.push(car);
          }
          state.trafficCars = aliveCars;
        }

        if (!updateHeroAndNpcs) return null;

        // 2. Smoking protagonist pose + glowing cherry + exhale smoke puffs
        const gender = (heroChar && heroChar.gender) ? heroChar.gender : "male";
        const heroBustSz = (heroChar && typeof heroChar.bust_size === "number") ? heroChar.bust_size : 0.5;
        const heroBones = evaluateCharacterBones(
          charJoints, gender, storyActive()?actor(storyTime).pos:HERO_POS, storyActive()?actor(storyTime).yaw:HERO_YAW, { elapsed:storyActive()?storyTime-9:elapsed, delivery:mode==='cinema'?linePerformance('hero',storyTime,'balcony'):null,listen:mode==='cinema'?phase(storyTime,17.6,18.5)*(1-phase(storyTime,22,23)):0, throwAmount:mode==='cinema'?phase(storyTime,22.82,23.15)*(1-phase(storyTime,23.22,24.08)):0, smokingMenu: !storyActive()||storyTime<24.08, horizSpeed:0, walkPhase:0, onGround:true, bustSize: heroBustSz }
        );
        const smk = evalSmokingCycle(storyActive()?storyTime-9:elapsed);
        state.emberGlow = smk.emberGlow;

        const mHandR = heroBones.hand_r;
        const mHead = heroBones.head;
        const tipLoc = cigBatches ? cigBatches.tipLocal : [-0.010, -0.070, 0.114];
        const mouthLoc = cigBatches ? cigBatches.mouthLocal : [0.0, 0.068, 0.116];
        state.cigTipWorld = mat4TransformPt(mHandR, tipLoc);
        const mouthW = mat4TransformPt(mHead, mouthLoc);
        const hFwd = normalize([mHead[2], mHead[6], mHead[10]]);

        state._emberAccum += (storyActive()&&storyTime>23?0:cdt) * (2.6 + 2.4 * smk.emberGlow);
        while (state._emberAccum >= 1.0) {
          state._emberAccum -= 1.0;
          state.smoke.push({
            pos: [state.cigTipWorld[0], state.cigTipWorld[1] + 0.015, state.cigTipWorld[2]],
            vel: [randRange(-0.04, 0.08), randRange(0.18, 0.28), randRange(-0.06, 0.04)],
            age: 0.0,
            life: randRange(1.5, 2.3),
            size0: 0.022,
            size1: randRange(0.10, 0.16),
            alpha: 0.28,
          });
        }

        if (smk.exhaleAmt > 0.08) {
          state._exhaleAccum += cdt * (7.5 * smk.exhaleAmt);
          while (state._exhaleAccum >= 1.0) {
            state._exhaleAccum -= 1.0;
            const spd = randRange(0.45, 0.78);
            state.smoke.push({
              pos: [
                mouthW[0] + hFwd[0] * 0.04 + randRange(-0.015, 0.015),
                mouthW[1] + hFwd[1] * 0.04 + randRange(-0.01, 0.015),
                mouthW[2] + hFwd[2] * 0.04 + randRange(-0.015, 0.015),
              ],
              vel: [
                hFwd[0] * spd + randRange(-0.10, 0.12),
                Math.max(0.16, hFwd[1] * spd + randRange(0.20, 0.34)),
                hFwd[2] * spd + randRange(-0.10, 0.10),
              ],
              age: 0.0,
              life: randRange(2.6, 3.8),
              size0: randRange(0.048, 0.068),
              size1: randRange(0.24, 0.36),
              alpha: randRange(0.46, 0.62) * (0.65 + 0.35 * smk.exhaleAmt),
            });
          }
        } else {
          state._exhaleAccum = 0.0;
        }

        const drag = Math.exp(-cdt * 1.45);
        const yDrag = Math.exp(-cdt * 0.85);
        const aliveSmoke = [];
        for (let i = 0; i < state.smoke.length; i++) {
          const p = state.smoke[i];
          p.age += cdt;
          if (p.age < p.life) {
            p.vel[0] = p.vel[0] * drag + 0.16 * cdt;
            p.vel[1] = p.vel[1] * yDrag + 0.19 * cdt;
            p.vel[2] = p.vel[2] * drag - 0.09 * cdt;
            p.pos[0] += p.vel[0] * cdt;
            p.pos[1] += p.vel[1] * cdt;
            p.pos[2] += p.vel[2] * cdt;
            aliveSmoke.push(p);
          }
        }
        state.smoke = aliveSmoke.length > 96 ? aliveSmoke.slice(-96) : aliveSmoke;

        // 3. Occasional random winter sidewalk passersby
        state._nextNpcTimer -= cdt;
        if (state._nextNpcTimer <= 0.0) {
          if (state.npcs.length < 2) spawnNpc();
          state._nextNpcTimer = randRange(8.5, 17.5);
        }
        const aliveNpcs = [];
        for (let i = 0; i < state.npcs.length; i++) {
          const npc = state.npcs[i];
          const step = npc.speed * cdt;
          if(residential&&npc.path!==undefined){pathStep(npc,step);npc.walkPhase+=step*Math.PI/.68;aliveNpcs.push(npc);continue;}
          npc.pos[2] += npc.dirZ * step;
          npc.walkPhase += step * (Math.PI / 0.68);
          const done = (npc.dirZ < 0 && npc.pos[2] <= npc.zEnd) || (npc.dirZ > 0 && npc.pos[2] >= npc.zEnd);
          if(done&&!residential&&inView(...npc.pos)){const lane=NPC_LANES[npc.lane];npc.pos[2]=npc.zEnd;npc.zEnd=npc.zEnd===lane[3]?lane[2]:lane[3];npc.dirZ*=-1;npc.yaw=npc.dirZ>0?0:Math.PI;aliveNpcs.push(npc);}else if (!done) aliveNpcs.push(npc);
        }
        state.npcs = aliveNpcs;

        return heroBones;
      };

      function writeQuad(off, cx, cy, cz, halfS, rx, ry, rz, ux, uy, uz, alpha) {
        const a = Math.max(0.015, Math.min(0.995, alpha));
        const ny = Math.sqrt(Math.max(0.0, 1.0 - a * a));
        const hxr = rx * halfS, hyr = ry * halfS, hzr = rz * halfS;
        const hxu = ux * halfS, hyu = uy * halfS, hzu = uz * halfS;

        const x0 = cx - hxr - hxu, y0 = cy - hyr - hyu, z0 = cz - hzr - hzu;
        const x1 = cx + hxr - hxu, y1 = cy + hyr - hyu, z1 = cz + hzr - hzu;
        const x2 = cx + hxr + hxu, y2 = cy + hyr + hyu, z2 = cz + hzr + hzu;
        const x3 = cx - hxr + hxu, y3 = cy - hyr + hyu, z3 = cz - hzr + hzu;

        let k = off;
        particleData[k++]=x0; particleData[k++]=y0; particleData[k++]=z0; particleData[k++]=a; particleData[k++]=ny; particleData[k++]=0; particleData[k++]=0; particleData[k++]=0;
        particleData[k++]=x1; particleData[k++]=y1; particleData[k++]=z1; particleData[k++]=a; particleData[k++]=ny; particleData[k++]=0; particleData[k++]=1; particleData[k++]=0;
        particleData[k++]=x2; particleData[k++]=y2; particleData[k++]=z2; particleData[k++]=a; particleData[k++]=ny; particleData[k++]=0; particleData[k++]=1; particleData[k++]=1;
        particleData[k++]=x0; particleData[k++]=y0; particleData[k++]=z0; particleData[k++]=a; particleData[k++]=ny; particleData[k++]=0; particleData[k++]=0; particleData[k++]=0;
        particleData[k++]=x2; particleData[k++]=y2; particleData[k++]=z2; particleData[k++]=a; particleData[k++]=ny; particleData[k++]=0; particleData[k++]=1; particleData[k++]=1;
        particleData[k++]=x3; particleData[k++]=y3; particleData[k++]=z3; particleData[k++]=a; particleData[k++]=ny; particleData[k++]=0; particleData[k++]=0; particleData[k++]=1;
        return k;
      }

      state.renderParticles = function(camPos, camFwd, camUp, includeSmoke, yOffset = 0.0) {
        const camR = normalize(vec3Cross(camFwd, camUp));
        const camU = normalize(vec3Cross(camR, camFwd));
        const rx = camR[0], ry = camR[1], rz = camR[2];
        const ux = camU[0], uy = camU[1], uz = camU[2];

        gl.enable(gl.BLEND);
        gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);
        gl.depthMask(false);
        gl.disable(gl.CULL_FACE);
        gl.uniformMatrix4fv(su.model, false, idMat4);
        gl.uniform1f(su.emissive, 0.0);
        gl.bindTexture(gl.TEXTURE_2D, textures["smoke_soft"]);
        gl.bindVertexArray(particleVao);
        gl.bindBuffer(gl.ARRAY_BUFFER, particleVbo);

        // 1. Falling winter snowflakes (soft_mode = 2.0)
        if (state.snow.length > 0) {
          let off = 0;
          const cnt = Math.min(state.snow.length, MAX_PARTICLE_QUADS);
          for (let i = 0; i < cnt; i++) {
            const fl = state.snow[i];
            off = writeQuad(off, fl[0], fl[1] + yOffset, fl[2], fl[4], rx, ry, rz, ux, uy, uz, fl[6]);
          }
          gl.bufferSubData(gl.ARRAY_BUFFER, 0, particleData.subarray(0, off));
          gl.uniform3f(su.colorTint, 0.94, 0.97, 1.0);
          gl.uniform1f(su.alpha, 0.92);
          gl.uniform1f(su.softParticle, 2.0);
          gl.drawArrays(gl.TRIANGLES, 0, cnt * 6);
        }

        // 2. Cigarette smoke puffs sorted back-to-front (soft_mode = 1.0)
        if (includeSmoke && state.smoke.length > 0) {
          const cx = camPos[0], cy = camPos[1], cz = camPos[2];
          const ordered = state.smoke.slice().sort((a, b) => {
            const da = (a.pos[0]-cx)**2 + (a.pos[1]+yOffset-cy)**2 + (a.pos[2]-cz)**2;
            const db = (b.pos[0]-cx)**2 + (b.pos[1]+yOffset-cy)**2 + (b.pos[2]-cz)**2;
            return db - da;
          });
          let off = 0;
          const cnt = Math.min(ordered.length, MAX_PARTICLE_QUADS);
          for (let i = 0; i < cnt; i++) {
            const p = ordered[i];
            const u = p.age / Math.max(p.life, 1e-4);
            const fadeIn = Math.min(1.0, u / 0.14);
            const fadeOut = Math.pow(Math.max(0.0, 1.0 - Math.max(0.0, (u - 0.25) / 0.75)), 1.35);
            const a = p.alpha * fadeIn * fadeOut;
            const rad = p.size0 + (p.size1 - p.size0) * Math.pow(u, 0.72);
            off = writeQuad(off, p.pos[0], p.pos[1] + yOffset, p.pos[2], rad, rx, ry, rz, ux, uy, uz, a);
          }
          gl.bufferSubData(gl.ARRAY_BUFFER, 0, particleData.subarray(0, off));
          gl.uniform3f(su.colorTint, 0.86, 0.89, 0.94);
          gl.uniform1f(su.alpha, 0.95);
          gl.uniform1f(su.softParticle, 1.0);
          gl.drawArrays(gl.TRIANGLES, 0, cnt * 6);
        }

        // 3. Soft warm headlight & red taillight glow halos aligned per classic car model!
        if (state.trafficCars.length > 0) {
          let offH = 0;
          for (let i = 0; i < state.trafficCars.length; i++) {
            const car = state.trafficCars[i];
            const spec = (trafficCarGpu && trafficCarGpu.specs[car.style]) ? trafficCarGpu.specs[car.style] : DEFAULT_CAR_SPEC;
            const sc = car.scale || 1.18;
            const hz = car.pos[2] + car.dirZ * (((spec.z_front || 2.02) + 0.06) * sc);
            const hlx = (spec.hl_x || 0.54) * sc;
            const hly = (spec.hl_y || 0.68) * sc + yOffset;
            offH = writeQuad(offH, car.pos[0] - hlx, hly, hz, 0.62, rx, ry, rz, ux, uy, uz, 0.72);
            offH = writeQuad(offH, car.pos[0] + hlx, hly, hz, 0.62, rx, ry, rz, ux, uy, uz, 0.72);
          }
          gl.bufferSubData(gl.ARRAY_BUFFER, 0, particleData.subarray(0, offH));
          gl.uniform3f(su.colorTint, 1.0, 0.88, 0.62);
          gl.uniform1f(su.alpha, 0.92);
          gl.uniform1f(su.softParticle, 1.0);
          gl.drawArrays(gl.TRIANGLES, 0, state.trafficCars.length * 12);

          let offT = 0;
          for (let i = 0; i < state.trafficCars.length; i++) {
            const car = state.trafficCars[i];
            const spec = (trafficCarGpu && trafficCarGpu.specs[car.style]) ? trafficCarGpu.specs[car.style] : DEFAULT_CAR_SPEC;
            const sc = car.scale || 1.18;
            const tz = car.pos[2] - car.dirZ * ((Math.abs(spec.z_rear || -2.05) + 0.06) * sc);
            const hlx = (spec.hl_x || 0.54) * sc;
            const tly = (spec.tl_y || 0.65) * sc + yOffset;
            offT = writeQuad(offT, car.pos[0] - hlx, tly, tz, 0.44, rx, ry, rz, ux, uy, uz, 0.64);
            offT = writeQuad(offT, car.pos[0] + hlx, tly, tz, 0.44, rx, ry, rz, ux, uy, uz, 0.64);
          }
          gl.bufferSubData(gl.ARRAY_BUFFER, 0, particleData.subarray(0, offT));
          gl.uniform3f(su.colorTint, 1.0, 0.16, 0.12);
          gl.uniform1f(su.alpha, 0.88);
          gl.uniform1f(su.softParticle, 1.0);
          gl.drawArrays(gl.TRIANGLES, 0, state.trafficCars.length * 12);
        }

        gl.uniform3f(su.colorTint, 1.0, 1.0, 1.0);
        gl.uniform1f(su.alpha, 1.0);
        gl.uniform1f(su.softParticle, 0.0);
        gl.depthMask(true);
        gl.disable(gl.BLEND);
        gl.enable(gl.CULL_FACE);
      };

      return state;
    }
    var menuLife = null, storyLife=null;
    const legBatches = {
      thigh: uploadBatches(gl, scene.leg.segments.thigh),
      shin: uploadBatches(gl, scene.leg.segments.shin),
      foot: uploadBatches(gl, scene.leg.segments.foot),
    };

    // Upload articulated 3D character customization library (male + female)
    const rawCat = scene.characterLib.catalog;
    const charCatalog = {
      skinPalette: rawCat.palettes.skin,
      hairPalette: rawCat.palettes.hair,
      topPalette: rawCat.palettes.top,
      bottomPalette: rawCat.palettes.bottom,
      shoePalette: rawCat.palettes.shoes,
      accessoryPalette: rawCat.palettes.accessory,
      faceStyles: rawCat.options.faceStyles,
      hairStyles: rawCat.options.hairStyles,
      topStyles: rawCat.options.topStyles,
      bottomStyles: rawCat.options.bottomStyles,
      shoeStyles: rawCat.options.shoeStyles,
      accessoryStyles: rawCat.options.accessories,
    };
    const charJoints = scene.characterLib.joints;
    const charGpu = {
      necks: {
        male: uploadBatches(gl, scene.characterLib.necks.male),
        female: uploadBatches(gl, scene.characterLib.necks.female),
      },
      heads: {
        male: uploadBatches(gl, scene.characterLib.heads.male),
        female: uploadBatches(gl, scene.characterLib.heads.female),
      },
      hairs: {},
      accessories: {},
      tops: { male: {}, female: {} },
      bottoms: { male: {}, female: {} },
      shoes: { male: {}, female: {} },
    };
    for (const [sid, list] of Object.entries(scene.characterLib.hairs)) {
      charGpu.hairs[sid] = uploadBatches(gl, list);
    }
    for (const [aid, bmap] of Object.entries(scene.characterLib.accessories)) {
      charGpu.accessories[aid] = uploadBoneBatchMap(gl, bmap);
    }
    for (const g of ["male", "female"]) {
      for (const [tid, bmap] of Object.entries(scene.characterLib.tops[g])) {
        charGpu.tops[g][tid] = uploadBoneBatchMap(gl, bmap);
      }
      for (const [bid, bmap] of Object.entries(scene.characterLib.bottoms[g])) {
        charGpu.bottoms[g][bid] = uploadBoneBatchMap(gl, bmap);
      }
      for (const [sid, bmap] of Object.entries(scene.characterLib.shoes[g])) {
        charGpu.shoes[g][sid] = uploadBoneBatchMap(gl, bmap);
      }
    }

    // Upload all 10 weapons + 23-bone skinned GLB 1st-person hands (low-poly_fps_hands.glb) & 6 sculpted clothing sleeves!
    const wLib = scene.weaponLib;
    const weaponOrderInit = (wLib.constants && wLib.constants.weaponOrder) || [
      "m1911", "glock17", "remington11", "double_barrel", "revolver", "ak47", "m4a1", "mp7", "m110", "mp5s",
    ];
    const doorCfg = scene.door || null;
    const doorBatches = (doorCfg && doorCfg.batches) ? uploadBatches(gl, doorCfg.batches) : [];
    const doorCtrl = {
      enabled: Boolean(doorCfg && doorBatches.length),
      hingePos: doorCfg ? doorCfg.hingePos : [4.80, 0.0, 10.52],
      width: doorCfg ? doorCfg.width : 1.26,
      height: doorCfg ? doorCfg.height : 2.36,
      openAngle: doorCfg ? doorCfg.openAngle : 1.70,
      kickMaxAngle: doorCfg ? doorCfg.kickMaxAngle : 2.04,
      quietHoldDelay: doorCfg ? doorCfg.quietHoldDelay : 0.24,
      angle: 0.0,
      targetAngle: 0.0,
      angVel: 0.0,
      mode: "closed",
      eHolding: false,
      eHoldTime: 0.0,
      quietDir: 1.0,
      quietActive: false,
      quietSfxCd: 0.0,
      nearPlayer: false,
      kickShake: 0.0,
      sfxQueue: [],
      centerPos() {
        const half = this.width * 0.5;
        return [this.hingePos[0] + half * Math.sin(this.angle), this.hingePos[1] + this.height * 0.5, this.hingePos[2] + half * Math.cos(this.angle)];
      },
      doorwayCenter() { return [this.hingePos[0], 1.2, 11.15]; },
      canInteract(pPos) {
        if (!this.enabled) return false;
        for (const [cx, , cz] of [this.centerPos(), this.doorwayCenter()]) {
          if (Math.hypot(pPos[0] - cx, pPos[2] - cz) <= 2.35 && Math.abs(pPos[1]) < 2.2) return true;
        }
        return false;
      },
      getActiveCollider() {
        if (!this.enabled) return null;
        if (this.angle < 0.52) {
          return [this.hingePos[0] - 0.14, 0.0, 10.45, this.hingePos[0] + 0.14, 2.40, 11.85];
        }
        const ex = this.hingePos[0] + this.width * Math.sin(this.angle);
        const ez = this.hingePos[2] + this.width * Math.cos(this.angle);
        const pad = 0.09;
        return [Math.min(this.hingePos[0], ex) - pad, 0.0, Math.min(this.hingePos[2], ez) - pad, Math.max(this.hingePos[0], ex) + pad, this.height, Math.max(this.hingePos[2], ez) + pad];
      },
      toggleNormal() {
        if (this.angle < 0.62) {
          this.targetAngle = this.openAngle;
          this.mode = "opening";
          this.sfxQueue.push("door_open");
        } else {
          this.targetAngle = 0.0;
          this.mode = "closing";
          this.sfxQueue.push("door_close");
        }
      },
      tryKick(eyePos, fwd) {
        if (!this.enabled) return false;
        let hit = false;
        for (const [cx, cy, cz] of [this.centerPos(), this.doorwayCenter()]) {
          const vx = cx - eyePos[0], vy = (cy - 0.18) - eyePos[1], vz = cz - eyePos[2];
          const d = Math.hypot(vx, vy, vz);
          if (d <= 2.45 && d > 0.05 && (vx * fwd[0] + vy * fwd[1] + vz * fwd[2]) / d > 0.34) { hit = true; break; }
        }
        if (!hit) return false;
        if (this.angle < 1.65) {
          this.mode = "kicked";
          this.targetAngle = this.kickMaxAngle;
          this.angVel = 13.8;
          this.kickShake = 1.0;
          this.sfxQueue.push("door_kick");
        } else {
          this.mode = "kicked";
          this.targetAngle = 0.0;
          this.angVel = -11.5;
          this.kickShake = 0.75;
          this.sfxQueue.push("door_kick");
        }
        this.eHolding = false; this.quietActive = false;
        return true;
      },
      handleEInput(dt, eDown, canUse) {
        this.quietActive = false;
        if (eDown && canUse) {
          if (!this.eHolding) {
            this.eHolding = true;
            this.eHoldTime = 0.0;
            this.quietDir = (this.angle < this.openAngle * 0.85) ? 1.0 : -1.0;
          } else {
            this.eHoldTime += dt;
            if (this.eHoldTime >= this.quietHoldDelay) {
              this.quietActive = true;
              this.mode = "quiet";
              this.angVel = 0.0;
              const prev = this.angle;
              this.angle = Math.max(0.0, Math.min(this.openAngle, this.angle + this.quietDir * 0.62 * dt));
              this.targetAngle = this.angle;
              if (Math.abs(this.angle - prev) > 1e-4 && this.quietSfxCd <= 0.0) {
                this.sfxQueue.push("door_quiet");
                this.quietSfxCd = 0.42;
              }
            }
          }
          return true;
        }
        if (this.eHolding) {
          const wasQuick = this.eHoldTime < this.quietHoldDelay;
          this.eHolding = false;
          this.eHoldTime = 0.0;
          if (wasQuick && canUse) { this.toggleNormal(); return true; }
        }
        return false;
      },
      update(dt, pPos) {
        if (!this.enabled) return;
        const cdt = Math.min(Math.max(dt, 0.0), 0.05);
        this.nearPlayer = this.canInteract(pPos);
        this.quietSfxCd = Math.max(0.0, this.quietSfxCd - cdt);
        if (this.kickShake > 0.0) this.kickShake = Math.max(0.0, this.kickShake - cdt * 4.5);
        if (this.mode === "kicked") {
          this.angle += this.angVel * cdt;
          const damp = (this.angVel < 0.0) ? 6.2 : 2.2;
          this.angVel *= Math.exp(-cdt * damp);
          if (this.angle >= this.kickMaxAngle) {
            this.angle = this.kickMaxAngle;
            if (this.angVel > 1.2) this.angVel = -Math.min(2.4, this.angVel * 0.24);
            else { this.angVel = 0.0; this.targetAngle = 1.88; this.mode = "open"; }
          } else if (this.angVel < 0.0 && Math.abs(this.angVel) < 0.18) {
            this.angVel = 0.0; this.targetAngle = this.angle; this.mode = "open";
          } else if (this.angle < 0.0) {
            this.angle = 0.0; this.angVel = 0.0; this.mode = "closed";
          }
        } else if (this.mode === "opening" || this.mode === "closing") {
          const diff = this.targetAngle - this.angle;
          const spd = (this.mode === "opening") ? 4.6 : 5.2;
          this.angle += diff * (1.0 - Math.exp(-cdt * spd));
          if (Math.abs(diff) < 0.008) {
            this.angle = this.targetAngle;
            this.mode = (this.angle < 0.05) ? "closed" : "open";
          }
        }
      },
      modelMatrix() {
        return mat4Mul(mat4Translate(this.hingePos[0], this.hingePos[1], this.hingePos[2]), mat4RotY(this.angle));
      },
      popSfx() {
        if (!this.sfxQueue.length) return [];
        const o = this.sfxQueue.slice();
        this.sfxQueue.length = 0;
        return o;
      },
    };
    window.__doorCtrl = doorCtrl;
    const TOP_STYLE_INDEX = (wLib.constants && wLib.constants.topStyleIndex) || {
      jacket: 0, sweater: 1, tshirt: 2, hoodie: 3, tracksuit: 4, coat: 5,
      puffer: 6, flannel: 7, tactical: 8, vest: 9, windbreaker: 10, tanktop: 11,
    };

    function b64ToBytes(b64) {
      const bin = atob(b64);
      const out = new Uint8Array(bin.length);
      for (let i = 0; i < bin.length; i++) out[i] = bin.charCodeAt(i);
      return out;
    }

    function createGlbSubmeshGpu(subJson) {
      const pos = new Float32Array(b64ToBytes(subJson.posB64).buffer);
      const nrm = new Float32Array(b64ToBytes(subJson.nrmB64).buffer);
      const uv = new Float32Array(b64ToBytes(subJson.uvB64).buffer);
      const jts = b64ToBytes(subJson.jtsB64);
      const wts = new Float32Array(b64ToBytes(subJson.wtsB64).buffer);
      const idx = new Uint16Array(b64ToBytes(subJson.idxB64).buffer);
      const vCount = subJson.vCount;
      const iCount = subJson.iCount;
      const sPos = new Float32Array(vCount * 3);
      const sNrm = new Float32Array(vCount * 3);
      const outInter = new Float32Array(iCount * 8);
      const vbo = gl.createBuffer();
      const vao = gl.createVertexArray();
      gl.bindVertexArray(vao);
      gl.bindBuffer(gl.ARRAY_BUFFER, vbo);
      gl.bufferData(gl.ARRAY_BUFFER, outInter.byteLength, gl.DYNAMIC_DRAW);
      gl.enableVertexAttribArray(0);
      gl.vertexAttribPointer(0, 3, gl.FLOAT, false, 32, 0);
      gl.enableVertexAttribArray(1);
      gl.vertexAttribPointer(1, 3, gl.FLOAT, false, 32, 12);
      gl.enableVertexAttribArray(2);
      gl.vertexAttribPointer(2, 2, gl.FLOAT, false, 32, 24);
      gl.bindVertexArray(null);
      return {
        texture: subJson.texture,
        slot: subJson.slot,
        vCount, iCount, pos, nrm, uv, jts, wts, idx,
        sPos, sNrm, outInter, vbo, vao,
      };
    }

    function skinSubmeshIntoGpu(sm, skinMats) {
      const { vCount, iCount, pos, nrm, uv, jts, wts, idx, sPos, sNrm, outInter } = sm;
      for (let vi = 0; vi < vCount; vi++) {
        const v3 = vi * 3, v4 = vi * 4;
        const px = pos[v3], py = pos[v3 + 1], pz = pos[v3 + 2];
        const nx = nrm[v3], ny = nrm[v3 + 1], nz = nrm[v3 + 2];
        let ox = 0, oy = 0, oz = 0, onx = 0, ony = 0, onz = 0;
        for (let k = 0; k < 4; k++) {
          const w = wts[v4 + k];
          if (w <= 1e-6) continue;
          const m = skinMats[jts[v4 + k]];
          ox += (m[0] * px + m[1] * py + m[2] * pz + m[3]) * w;
          oy += (m[4] * px + m[5] * py + m[6] * pz + m[7]) * w;
          oz += (m[8] * px + m[9] * py + m[10] * pz + m[11]) * w;
          onx += (m[0] * nx + m[1] * ny + m[2] * nz) * w;
          ony += (m[4] * nx + m[5] * ny + m[6] * nz) * w;
          onz += (m[8] * nx + m[9] * ny + m[10] * nz) * w;
        }
        const nLen = Math.hypot(onx, ony, onz);
        if (nLen > 1e-6) { onx /= nLen; ony /= nLen; onz /= nLen; }
        sPos[v3] = ox; sPos[v3 + 1] = oy; sPos[v3 + 2] = oz;
        sNrm[v3] = onx; sNrm[v3 + 1] = ony; sNrm[v3 + 2] = onz;
      }
      let o = 0;
      for (let k = 0; k < iCount; k++) {
        const vi = idx[k], v3 = vi * 3, v2 = vi * 2;
        outInter[o++] = sPos[v3]; outInter[o++] = sPos[v3 + 1]; outInter[o++] = sPos[v3 + 2];
        outInter[o++] = sNrm[v3]; outInter[o++] = sNrm[v3 + 1]; outInter[o++] = sNrm[v3 + 2];
        outInter[o++] = uv[v2];   outInter[o++] = uv[v2 + 1];
      }
      gl.bindBuffer(gl.ARRAY_BUFFER, sm.vbo);
      gl.bufferSubData(gl.ARRAY_BUFFER, 0, outInter);
    }

    function glbBoneLocalRot(curl, twist, splay) {
      let m = mat4Identity();
      if (Math.abs(splay) > 1e-6) m = mat4Mul(m, mat4RotZ(splay));
      if (Math.abs(curl) > 1e-6)  m = mat4Mul(m, mat4RotX(curl));
      if (Math.abs(twist) > 1e-6) m = mat4Mul(m, mat4RotY(twist));
      return m;
    }

    function glbMat4RotAxisAngle(axis, angle) {
      if (Math.abs(angle) < 1e-6) return mat4Identity();
      const [ax, ay, az] = normalize(axis);
      const c = Math.cos(angle), s = Math.sin(angle), t = 1.0 - c;
      return [
        t*ax*ax + c,      t*ax*ay - s*az, t*ax*az + s*ay, 0,
        t*ax*ay + s*az,   t*ay*ay + c,    t*ay*az - s*ax, 0,
        t*ax*az - s*ay,   t*ay*az + s*ax, t*az*az + c,    0,
        0, 0, 0, 1,
      ];
    }

    function glbMat4RotFromTo(uIn, vIn, maxAngle = 1.65) {
      const u = normalize(uIn), v = normalize(vIn);
      const d = Math.max(-1.0, Math.min(1.0, u[0]*v[0] + u[1]*v[1] + u[2]*v[2]));
      if (d > 0.99995) return mat4Identity();
      const cr = vec3Cross(u, v);
      const crLen = Math.hypot(cr[0], cr[1], cr[2]);
      const axis = crLen > 1e-6 ? [cr[0]/crLen, cr[1]/crLen, cr[2]/crLen] : [0, 1, 0];
      const ang = Math.min(Math.acos(d), maxAngle);
      return glbMat4RotAxisAngle(axis, ang);
    }

    function buildGlb23BonePose(handM, wristPos, elbowPos, pose, isLeft) {
      const sgn = isLeft ? -1.0 : 1.0;
      const toHandLocal = (vx, vy, vz) => normalize([
        handM[0]*vx + handM[4]*vy + handM[8]*vz,
        handM[1]*vx + handM[5]*vy + handM[9]*vz,
        handM[2]*vx + handM[6]*vy + handM[10]*vz,
      ]);
      const eUnit = normalize(vec3Sub(elbowPos, wristPos));
      const elbowDir = toHandLocal(eUnit[0], eUnit[1], eUnit[2]);
      const realElb = [
        wristPos[0] + eUnit[0] * 0.165,
        wristPos[1] + eUnit[1] * 0.165,
        wristPos[2] + eUnit[2] * 0.165,
      ];
      const shPos = [isLeft ? -0.38 : 0.38, -0.36, 0.34];
      const shoulderDir = toHandLocal(shPos[0] - realElb[0], shPos[1] - realElb[1], shPos[2] - realElb[2]);
      const cl = (v, lo, hi) => Math.max(lo, Math.min(hi, v));
      const f4 = (name, ms, ps, ds) => {
        const [mcp, yaw, _r, pip] = pose[name];
        return [cl(mcp*ms, -0.22, 1.18), cl(pip*ps, -0.10, 1.24), cl(pip*ds, -0.08, 0.96), cl(yaw*0.85*sgn, -0.34, 0.34)];
      };
      const [tm, ty, _tr, tp] = pose.thumb;
      return {
        thumb: [cl(tm*0.78 + 0.22, -0.28, 0.92), cl(tp*0.74 + 0.12, -0.22, 0.88), cl(tp*0.58 + 0.06, -0.18, 0.76), cl(ty*0.55*sgn - 0.06, -0.46, 0.52)],
        index: f4("index", 0.76, 0.82, 0.60),
        middle: f4("middle", 0.72, 0.78, 0.58),
        ring: f4("ring", 0.74, 0.80, 0.60),
        pinky: f4("pinky", 0.76, 0.82, 0.62),
        palmCup: 0.14,
        forearmTwist: 0.0,
        elbowDir,
        shoulderDir,
      };
    }

    function createGlbHandGpuSide(sideJson, parents, evalOrder) {
      return {
        isLeft: !!sideJson.isLeft,
        parents,
        evalOrder,
        bindWorld: sideJson.bindWorld,
        invBind: sideJson.invBind,
        relBind: sideJson.relBind,
        hand: createGlbSubmeshGpu(sideJson.hand),
        sleeves: sideJson.sleeves.map(st => st.map(sm => createGlbSubmeshGpu(sm))),
      };
    }

    function evalGlbHandSkinMatrices(sideGpu, pose23) {
      const sM = sideGpu.isLeft ? -1.0 : 1.0;
      const deltas = new Array(23);
      for (let i = 0; i < 23; i++) deltas[i] = mat4Identity();

      const pc = pose23.palmCup || 0.0;
      deltas[4]  = glbBoneLocalRot(pc * 0.22, 0.0, 0.0);
      deltas[11] = glbBoneLocalRot(pc * 0.35, 0.0, 0.0);
      deltas[15] = glbBoneLocalRot(pc * 0.62, 0.0, -pc * 0.18 * sM);
      deltas[19] = glbBoneLocalRot(pc * 0.95, 0.0, -pc * 0.34 * sM);

      const { thumb, index, middle, ring, pinky } = pose23;
      deltas[5]  = glbBoneLocalRot(index[0], 0.0, index[3] * sM);
      deltas[6]  = glbBoneLocalRot(index[1], 0.0, 0.0);
      deltas[7]  = glbBoneLocalRot(index[2], 0.0, 0.0);
      deltas[8]  = glbBoneLocalRot(thumb[0], thumb[3] * 0.65 * sM, thumb[3] * 0.85 * sM);
      deltas[9]  = glbBoneLocalRot(thumb[1], 0.0, 0.0);
      deltas[10] = glbBoneLocalRot(thumb[2], 0.0, 0.0);
      deltas[12] = glbBoneLocalRot(middle[0], 0.0, middle[3] * sM);
      deltas[13] = glbBoneLocalRot(middle[1], 0.0, 0.0);
      deltas[14] = glbBoneLocalRot(middle[2], 0.0, 0.0);
      deltas[16] = glbBoneLocalRot(ring[0], 0.0, ring[3] * sM);
      deltas[17] = glbBoneLocalRot(ring[1], 0.0, 0.0);
      deltas[18] = glbBoneLocalRot(ring[2], 0.0, 0.0);
      deltas[20] = glbBoneLocalRot(pinky[0], 0.0, pinky[3] * sM);
      deltas[21] = glbBoneLocalRot(pinky[1], 0.0, 0.0);
      deltas[22] = glbBoneLocalRot(pinky[2], 0.0, 0.0);

      const pw = new Array(23);
      for (let i = 0; i < 23; i++) pw[i] = mat4Identity();
      pw[3] = mat4Mul(sideGpu.relBind[3], deltas[3]);
      for (let k = 0; k < sideGpu.evalOrder.length; k++) {
        const bi = sideGpu.evalOrder[k];
        if (bi <= 3) continue;
        const pi = sideGpu.parents[bi];
        pw[bi] = mat4Mul(pw[pi], mat4Mul(sideGpu.relBind[bi], deltas[bi]));
      }

      const bw2 = sideGpu.bindWorld[2], bw0 = sideGpu.bindWorld[0];
      const pElbB = [bw2[3], bw2[7], bw2[11]];
      const pShB  = [bw0[3], bw0[7], bw0[11]];
      const uForeBind = normalize(pElbB);

      const rAimFore = glbMat4RotFromTo(uForeBind, pose23.elbowDir, 1.35);
      pw[2] = mat4Mul(rAimFore, bw2);

      const m2 = pw[2];
      const pe = [m2[3], m2[7], m2[11]];
      const m2Delta = mat4Mul(m2, sideGpu.invBind[2]);
      const pShRigid = [
        m2Delta[0]*pShB[0] + m2Delta[1]*pShB[1] + m2Delta[2]*pShB[2] + m2Delta[3],
        m2Delta[4]*pShB[0] + m2Delta[5]*pShB[1] + m2Delta[6]*pShB[2] + m2Delta[7],
        m2Delta[8]*pShB[0] + m2Delta[9]*pShB[1] + m2Delta[10]*pShB[2] + m2Delta[11],
      ];
      const uUpperCur = normalize(vec3Sub(pShRigid, pe));
      const rElb = glbMat4RotFromTo(uUpperCur, pose23.shoulderDir, 1.55);
      const tPiv = mat4Mul(mat4Translate(pe[0], pe[1], pe[2]), mat4Mul(rElb, mat4Translate(-pe[0], -pe[1], -pe[2])));
      const mUpperDelta = mat4Mul(tPiv, m2Delta);
      pw[1] = mat4Mul(mUpperDelta, sideGpu.bindWorld[1]);
      pw[0] = mat4Mul(mUpperDelta, sideGpu.bindWorld[0]);

      const skinMats = new Array(23);
      for (let i = 0; i < 23; i++) skinMats[i] = mat4Mul(pw[i], sideGpu.invBind[i]);
      return skinMats;
    }

    function drawGlbSkinnedHand(sideGpu, worldMat, skinMats, topStyleIdx, slotCols) {
      if (!sideGpu) return;
      const topRgb = slotCols.top || [0.5, 0.5, 0.5];
      const skinRgb = slotCols.skin || [0.9, 0.78, 0.68];
      const colMap = {
        skin: skinRgb,
        top: topRgb,
        top_dark: [topRgb[0] * 0.72, topRgb[1] * 0.72, topRgb[2] * 0.72],
        fixed: [1.0, 1.0, 1.0],
      };
      gl.uniformMatrix4fv(su.model, false, toGlMat4(worldMat));
      gl.uniform1f(su.emissive, 0.0);

      // 1. Bare rigged GLB hand
      skinSubmeshIntoGpu(sideGpu.hand, skinMats);
      gl.uniform3f(su.colorTint, skinRgb[0], skinRgb[1], skinRgb[2]);
      gl.bindTexture(gl.TEXTURE_2D, textures[sideGpu.hand.texture] || textures["fp_skin_detailed"] || textures["char_skin"]);
      gl.bindVertexArray(sideGpu.hand.vao);
      gl.drawArrays(gl.TRIANGLES, 0, sideGpu.hand.iCount);

      // 2. Sculpted 3D clothing sleeve matching player's selected top style (0..5)!
      const stIdx = Math.max(0, Math.min(sideGpu.sleeves.length - 1, topStyleIdx | 0));
      const slList = sideGpu.sleeves[stIdx] || [];
      for (let i = 0; i < slList.length; i++) {
        const sm = slList[i];
        skinSubmeshIntoGpu(sm, skinMats);
        const c = colMap[sm.slot] || [1.0, 1.0, 1.0];
        gl.uniform3f(su.colorTint, c[0], c[1], c[2]);
        gl.bindTexture(gl.TEXTURE_2D, textures[sm.texture] || textures["char_top_fabric"]);
        gl.bindVertexArray(sm.vao);
        gl.drawArrays(gl.TRIANGLES, 0, sm.iCount);
      }
      gl.uniform3f(su.colorTint, 1.0, 1.0, 1.0);
    }

    const weaponGpu = {
      models: {},
      m1911: {},
      fpHandsGlb: null,
    };
    for (const wid of weaponOrderInit) {
      weaponGpu.models[wid] = {};
      const src = wLib[wid] || wLib.m1911;
      for (const [part, list] of Object.entries(src)) {
        weaponGpu.models[wid][part] = uploadBatches(gl, list);
      }
    }
    weaponGpu.m1911 = weaponGpu.models.m1911;
    if (wLib.fpHandsGlb) {
      const glbSrc = wLib.fpHandsGlb;
      weaponGpu.fpHandsGlb = {
        right: createGlbHandGpuSide(glbSrc.right, glbSrc.parents, glbSrc.evalOrder),
        left: createGlbHandGpuSide(glbSrc.left, glbSrc.parents, glbSrc.evalOrder),
      };
    }
    const weapon = new WeaponController(wLib.constants, false);

    function drawWeaponBatches(list, modelMat, slotColors = null, tintOverride = null) {
      if (!list || !list.length) return;
      gl.uniformMatrix4fv(su.model, false, toGlMat4(modelMat));
      for (const batch of list) {
        let tint = tintOverride || [1.0, 1.0, 1.0];
        if (!tintOverride && slotColors && batch.tintSlot && slotColors[batch.tintSlot]) {
          tint = slotColors[batch.tintSlot];
        }
        const isFx = (batch.texture === "muzzle_flash" || batch.texture === "sight_red_dot");
        gl.uniform3f(su.colorTint, tint[0], tint[1], tint[2]);
        gl.bindTexture(gl.TEXTURE_2D, textures[batch.texture] || textures["char_skin"]);
        gl.uniform1f(su.emissive, isFx ? 2.8 : 0.0);
        gl.bindVertexArray(batch.vao);
        gl.drawArrays(gl.TRIANGLES, 0, batch.vertexCount);
      }
      gl.uniform1f(su.emissive, 0.0);
    }

    const BLINK_CYCLE_LEN = 18.4;
    const BLINK_EVENTS = [2.40, 5.95, 8.60, 8.92, 12.80, 16.35];
    function computeBlinkSuffix(elapsed) {
      const t = ((elapsed % BLINK_CYCLE_LEN) + BLINK_CYCLE_LEN) % BLINK_CYCLE_LEN;
      for (let i = 0; i < BLINK_EVENTS.length; i++) {
        const dt = t - BLINK_EVENTS[i];
        if (dt >= 0.0 && dt < 0.170) {
          if (dt < 0.036) return "_half";
          if (dt < 0.118) return "_blink";
          return "_half";
        }
      }
      return "";
    }

    function drawCharacterModel(ap, bones, elapsed = 0, talkMouth = "", actingFace=null, gaze=null) {
      const colors = getCharacterSlotColors(charCatalog, ap);
      const g = (ap.gender === "female") ? "female" : "male";
      const faceStyleObj = charCatalog.faceStyles[(ap.face_style || 0) % charCatalog.faceStyles.length];
      const blinkSfx = computeBlinkSuffix(elapsed);
      const faceSfx = ((talkMouth === "_talk1" || talkMouth === "_talk2") && blinkSfx === "") ? talkMouth : blinkSfx;
      const faceTexName = `char_face_${g === "female" ? "f" : "m"}_${faceStyleObj.id}${faceSfx}`;
      const hairId = charCatalog.hairStyles[ap.hair_style % charCatalog.hairStyles.length].id;
      const topId = charCatalog.topStyles[ap.top_style % charCatalog.topStyles.length].id;
      const bottomId = charCatalog.bottomStyles[ap.bottom_style % charCatalog.bottomStyles.length].id;
      const shoeId = charCatalog.shoeStyles[ap.shoe_style % charCatalog.shoeStyles.length].id;
      const accId = charCatalog.accessoryStyles[ap.accessory % charCatalog.accessoryStyles.length].id;

      const drawBoneList = (boneName, list) => {
        if (!list || !list.length) return;
        const m = bones[boneName];
        if (!m) return;
        gl.uniformMatrix4fv(su.model, false, toGlMat4(m));
        for (const batch of list) {
          const tint = colors[batch.tintSlot] || [1, 1, 1];
          const texName = (batch.texture === "char_face_slot") ? faceTexName : batch.texture;
          gl.uniform3f(su.colorTint, tint[0], tint[1], tint[2]);
          gl.bindTexture(gl.TEXTURE_2D, textures[texName] || textures["char_skin"]);
          gl.uniform1f(su.emissive, 0.0);
          gl.bindVertexArray(batch.vao);
          const perform=actingFace && batch.texture==='char_face_slot';
          if(perform){gl.uniform4fv(su.acting,actingFace);gl.uniform4f(su.gaze,gaze?.[0]??0,gaze?.[1]??0,g==='female'?1:0,blinkSfx?0:1);}
          gl.drawArrays(gl.TRIANGLES, 0, batch.vertexCount);
          if(perform){gl.uniform4f(su.acting,0,0,0,0);gl.uniform4f(su.gaze,0,0,0,0);}
        }
      };

      drawBoneList("neck", charGpu.necks[g]);
      drawBoneList("head", charGpu.heads[g]);
      drawBoneList("head", charGpu.hairs[hairId]);
      for (const [bone, list] of Object.entries(charGpu.tops[g][topId] || {})) drawBoneList(bone, list);
      for (const [bone, list] of Object.entries(charGpu.bottoms[g][bottomId] || {})) drawBoneList(bone, list);
      for (const [bone, list] of Object.entries(charGpu.shoes[g][shoeId] || {})) drawBoneList(bone, list);
      for (const [bone, list] of Object.entries(charGpu.accessories[accId] || {})) drawBoneList(bone, list);
      gl.uniform3f(su.colorTint, 1.0, 1.0, 1.0);
    }

    // Fullscreen post-process quad
    const quadVao = gl.createVertexArray();
    gl.bindVertexArray(quadVao);
    const quadVbo = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, quadVbo);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([
      -1, -1, 0, 0,  1, -1, 1, 0,  1, 1, 1, 1,
      -1, -1, 0, 0,  1,  1, 1, 1, -1, 1, 0, 1,
    ]), gl.STATIC_DRAW);
    gl.enableVertexAttribArray(0);
    gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 16, 0);
    gl.enableVertexAttribArray(1);
    gl.vertexAttribPointer(1, 2, gl.FLOAT, false, 16, 8);
    gl.bindVertexArray(null);

    // State (single Buckshot render style; auto-opens 3D Character Creator on first launch!)
    let preset = structuredClone(scene.preset);
    let showHud = true;
    let mode = "menu"; // "creator" | "menu" | "playing" | "paused"
    let thirdPerson = false;
    let dragging = false;
    let creatorDragging = false;
    let lastMouseX = 0, lastMouseY = 0;

    // 3D Character Creator stage camera & turntable state
    const CREATOR_STAGE_POS = [-6.8, 0.0, 5.2];
    const CREATOR_BODY_CAM = [-8.1, 1.18, 7.15];
    const CREATOR_BODY_TGT = [-6.24, 0.96, 4.90];
    const CREATOR_FACE_CAM = [-7.28, 1.56, 5.95];
    const CREATOR_FACE_TGT = [-6.54, 1.53, 5.10];
    const CREATOR_BASE_YAW = Math.atan2(CREATOR_BODY_CAM[0] - CREATOR_STAGE_POS[0], CREATOR_BODY_CAM[2] - CREATOR_STAGE_POS[2]);
    let creatorYawOffset = -0.15;
    let creatorTargetYawOffset = -0.15;
    let creatorZoomTarget = 0.0; // 0 = full body, 1 = face close-up
    let creatorZoom = 0.0;
    const creatorBustPhys = new BustPhysics();
    const playerBustPhys = new BustPhysics();

    // Smooth cursor-tracking camera state for the 3D main menu
    let menuTargetX = 0, menuTargetY = 0;
    let menuCursorX = 0, menuCursorY = 0;

    const player = new Player(scene.spawnPos, scene.spawnYaw, scene.playerConfig, scene.leg);
    weapon.spawnWorldPickup(scene.weaponSpawns);

    // Low-res FBO management
    let lowW = 0, lowH = 0;
    let fbo = null, fboColor = null, fboDepth = null;

    function rebuildFbo() {
      const idx = preset.resolution_index;
      let targetW, targetH;
      if (idx >= 0 && idx < scene.pixelResolutions.length) {
        const baseH = scene.pixelResolutions[idx][1];
        const aspect = canvas.width / Math.max(canvas.height, 1);
        targetW = Math.max(64, Math.round(baseH * aspect));
        targetH = baseH;
      } else {
        targetW = Math.max(64, canvas.width);
        targetH = Math.max(64, canvas.height);
      }
      if (fbo && targetW === lowW && targetH === lowH) return;

      if (fbo) gl.deleteFramebuffer(fbo);
      if (fboColor) gl.deleteTexture(fboColor);
      if (fboDepth) gl.deleteRenderbuffer(fboDepth);

      lowW = targetW;
      lowH = targetH;

      fboColor = gl.createTexture();
      gl.bindTexture(gl.TEXTURE_2D, fboColor);
      gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA8, lowW, lowH, 0, gl.RGBA, gl.UNSIGNED_BYTE, null);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);

      fboDepth = gl.createRenderbuffer();
      gl.bindRenderbuffer(gl.RENDERBUFFER, fboDepth);
      gl.renderbufferStorage(gl.RENDERBUFFER, gl.DEPTH_COMPONENT24, lowW, lowH);

      fbo = gl.createFramebuffer();
      gl.bindFramebuffer(gl.FRAMEBUFFER, fbo);
      gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, fboColor, 0);
      gl.framebufferRenderbuffer(gl.FRAMEBUFFER, gl.DEPTH_ATTACHMENT, gl.RENDERBUFFER, fboDepth);
      gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    }

    function resizeCanvas() {
      const dpr = Math.min(window.devicePixelRatio || 1, 2);
      const w = Math.max(1, Math.floor(window.innerWidth * dpr));
      const h = Math.max(1, Math.floor(window.innerHeight * dpr));
      if (canvas.width !== w || canvas.height !== h) {
        canvas.width = w;
        canvas.height = h;
      }
      rebuildFbo();
    }
    window.addEventListener("resize", resizeCanvas);
    resizeCanvas();

    // UI elements
    const mainMenuEl = document.getElementById("main-menu");
    const settingsCard = document.getElementById("settings-card");
    const tabBtnGeneral = document.getElementById("tab-btn-general");
    const tabBtnControls = document.getElementById("tab-btn-controls");
    const tabPaneGeneral = document.getElementById("tab-pane-general");
    const tabPaneControls = document.getElementById("tab-pane-controls");
    const setDitherToggle = document.getElementById("set-dither-toggle");
    const setDitherLevels = document.getElementById("set-dither-levels");
    const setDitherStrength = document.getElementById("set-dither-strength");
    const setFovSlider = document.getElementById("set-fov");
    const setSensSlider = document.getElementById("set-sens");
    const setSfxSlider = document.getElementById("set-sfx");
    const valDitherLevels = document.getElementById("val-dither-levels");
    const valDitherStrength = document.getElementById("val-dither-strength");
    const valFov = document.getElementById("val-fov");
    const valSens = document.getElementById("val-sens");
    const valSfx = document.getElementById("val-sfx");
    const bindListEl = document.getElementById("bind-list");
    const settingsStatus = document.getElementById("settings-status");
    const overlay = document.getElementById("overlay");
    const overlayTitle = document.getElementById("overlay-title");
    const hudEl = document.getElementById("hud");
    const hudMain = document.getElementById("hud-main");
    const hudFlags = document.getElementById("hud-flags");
    const hudHelp = hudEl.querySelector(".row-help");
    const hudBadge = document.getElementById("hud-badge");
    const quickBar = document.getElementById("quick-bar");
    const ammoHudEl = document.getElementById("ammo-hud");
    const ammoTitle = document.getElementById("ammo-title");
    const ammoMag = document.getElementById("ammo-mag");
    const ammoRes = document.getElementById("ammo-res");
    const ammoPips = document.getElementById("ammo-pips");
    const ammoState = document.getElementById("ammo-state");
    const pickupPromptEl = document.getElementById("pickup-prompt");
    const btnPickup = document.getElementById("btn-pickup");
    const btnDrop = document.getElementById("btn-drop");
    const btnReload = document.getElementById("btn-reload");
    const btnInspect = document.getElementById("btn-inspect");
    const btnChat = document.getElementById("btn-chat");
    const btnHolster = document.getElementById("btn-holster");
    const hitMarker = document.getElementById("hit-marker");
    const mpCard = document.getElementById("mp-card");
    const mpNickInput = document.getElementById("mp-nick");
    const mpRoomListEl = document.getElementById("mp-room-list");
    const mpNewRoomInput = document.getElementById("mp-new-room-name");
    const mpPasswordInput = document.getElementById("mp-password");
    const mpServerUrlInput = document.getElementById("mp-server-url");
    const mpStatusEl = document.getElementById("mp-status");
    const mpNameplatesEl = document.getElementById("mp-nameplates");
    const chatBoxEl = document.getElementById("chat-box");
    const chatFeedEl = document.getElementById("chat-feed");
    const chatInputEl = document.getElementById("chat-input");
    const chatSendBtn = document.getElementById("chat-send-btn");
    const creatorPanel = document.getElementById("creator-panel");
    const creatorStageBar = document.getElementById("creator-stage-bar");
    const creatorHeading = document.getElementById("creator-heading");
    const creatorSubtitle = document.getElementById("creator-subtitle");
    const btnGenderMale = document.getElementById("btn-gender-male");
    const btnGenderFemale = document.getElementById("btn-gender-female");
    const btnCamBody = document.getElementById("btn-cam-body");
    const btnCamFace = document.getElementById("btn-cam-face");
    const btnCamMode = document.getElementById("btn-cam-mode");
    let hitHideTimer = null;

    // ---------- 3D Character Customization State & Persistence ----------
    const CHAR_KEY = "psx_sandbox_character_v1";
    const DEFAULT_CHARACTER = {
      gender: "male",
      skin: 1,
      face_style: 1,
      hair_style: 0,
      hair_color: 0,
      top_style: 0,
      top_color: 0,
      bottom_style: 0,
      bottom_color: 0,
      shoe_style: 0,
      shoe_color: 0,
      accessory: 0,
      accessory_color: 0,
      bust_size: 0.5,
    };
    let character = { ...DEFAULT_CHARACTER };
    let hasSavedCharacter = false;

    function clampChar(ap) {
      ap.gender = (ap.gender === "female") ? "female" : "male";
      const wrap = (v, n) => ((Math.trunc(Number(v) || 0) % n) + n) % n;
      ap.skin = wrap(ap.skin, charCatalog.skinPalette.length);
      ap.face_style = wrap(ap.face_style, charCatalog.faceStyles.length);
      ap.hair_style = wrap(ap.hair_style, charCatalog.hairStyles.length);
      ap.hair_color = wrap(ap.hair_color, charCatalog.hairPalette.length);
      ap.top_style = wrap(ap.top_style, charCatalog.topStyles.length);
      ap.top_color = wrap(ap.top_color, charCatalog.topPalette.length);
      ap.bottom_style = wrap(ap.bottom_style, charCatalog.bottomStyles.length);
      ap.bottom_color = wrap(ap.bottom_color, charCatalog.bottomPalette.length);
      ap.shoe_style = wrap(ap.shoe_style, charCatalog.shoeStyles.length);
      ap.shoe_color = wrap(ap.shoe_color, charCatalog.shoePalette.length);
      ap.accessory = wrap(ap.accessory, charCatalog.accessoryStyles.length);
      ap.accessory_color = wrap(ap.accessory_color, charCatalog.accessoryPalette.length);
      const rawBs = Number(ap.bust_size);
      ap.bust_size = Number.isFinite(rawBs) ? Math.max(0.0, Math.min(1.0, Math.round(rawBs * 100) / 100)) : 0.5;
      return ap;
    }

    function loadCharacter() {
      try {
        const raw = window.localStorage.getItem(CHAR_KEY);
        if (!raw) return false;
        const parsed = JSON.parse(raw);
        if (!parsed || typeof parsed !== "object") return false;
        character = clampChar({ ...character, ...parsed });
        return true;
      } catch (_) {
        return false;
      }
    }

    function saveCharacter() {
      hasSavedCharacter = true;
      try {
        window.localStorage.setItem(CHAR_KEY, JSON.stringify(character));
      } catch (_) {}
      if (typeof mp !== "undefined" && mp.connected) mp.sendAppearance();
    }

    function rgbToCss(rgb) {
      return `rgb(${Math.round(rgb[0] * 255)}, ${Math.round(rgb[1] * 255)}, ${Math.round(rgb[2] * 255)})`;
    }

    function buildSwatchRow(containerId, palette, fieldKey) {
      const box = document.getElementById(containerId);
      box.innerHTML = "";
      palette.forEach((item, idx) => {
        const btn = document.createElement("button");
        btn.className = "swatch";
        btn.title = item.name;
        btn.style.backgroundColor = rgbToCss(item.rgb);
        btn.dataset.idx = String(idx);
        btn.addEventListener("click", () => {
          character[fieldKey] = idx;
          syncCreatorUi();
        });
        box.appendChild(btn);
      });
    }

    buildSwatchRow("sw-skin", charCatalog.skinPalette, "skin");
    buildSwatchRow("sw-hair", charCatalog.hairPalette, "hair_color");
    buildSwatchRow("sw-top", charCatalog.topPalette, "top_color");
    buildSwatchRow("sw-bottom", charCatalog.bottomPalette, "bottom_color");
    buildSwatchRow("sw-shoe", charCatalog.shoePalette, "shoe_color");
    buildSwatchRow("sw-acc", charCatalog.accessoryPalette, "accessory_color");

    function syncSwatchActive(containerId, activeIdx) {
      const nodes = document.getElementById(containerId).children;
      for (let i = 0; i < nodes.length; i++) {
        nodes[i].classList.toggle("active", i === activeIdx);
      }
    }

    function syncCreatorUi() {
      clampChar(character);
      const isF = character.gender === "female";
      btnGenderMale.classList.toggle("active", !isF);
      btnGenderFemale.classList.toggle("active", isF);
      document.getElementById("val-gender").textContent = isF ? "Женщина" : "Мужчина";
      const secBust = document.getElementById("sec-bust-size");
      const rngBust = document.getElementById("rng-bust-size");
      const valBust = document.getElementById("val-bust-size");
      if (secBust) secBust.style.display = isF ? "block" : "none";
      const bustPct = Math.round((character.bust_size ?? 0.5) * 100);
      if (rngBust) rngBust.value = String(bustPct);
      if (valBust) valBust.textContent = `${bustPct}%`;
      document.getElementById("val-skin").textContent = charCatalog.skinPalette[character.skin].name;
      document.getElementById("lbl-face-style").textContent = charCatalog.faceStyles[character.face_style].name;
      document.getElementById("lbl-hair-style").textContent = charCatalog.hairStyles[character.hair_style].name;
      document.getElementById("val-hair-color").textContent = charCatalog.hairPalette[character.hair_color].name;
      document.getElementById("lbl-top-style").textContent = charCatalog.topStyles[character.top_style].name;
      document.getElementById("val-top-color").textContent = charCatalog.topPalette[character.top_color].name;
      document.getElementById("lbl-bottom-style").textContent = charCatalog.bottomStyles[character.bottom_style].name;
      document.getElementById("val-bottom-color").textContent = charCatalog.bottomPalette[character.bottom_color].name;
      document.getElementById("lbl-shoe-style").textContent = charCatalog.shoeStyles[character.shoe_style].name;
      document.getElementById("val-shoe-color").textContent = charCatalog.shoePalette[character.shoe_color].name;
      document.getElementById("lbl-acc-style").textContent = charCatalog.accessoryStyles[character.accessory].name;
      document.getElementById("val-acc-color").textContent = charCatalog.accessoryPalette[character.accessory_color].name;

      syncSwatchActive("sw-skin", character.skin);
      syncSwatchActive("sw-hair", character.hair_color);
      syncSwatchActive("sw-top", character.top_color);
      syncSwatchActive("sw-bottom", character.bottom_color);
      syncSwatchActive("sw-shoe", character.shoe_color);
      syncSwatchActive("sw-acc", character.accessory_color);
    }

    hasSavedCharacter = loadCharacter();
    syncCreatorUi();

    // ---------- Settings (Dithering, FOV, Sensitivity, Rebindable Controls) ----------
    const SETTINGS_KEY = "psx_sandbox_settings_v1";
    const BASE_MOUSE_SENS = scene.playerConfig.mouse_sensitivity;
    const BIND_DEFS = [
      { id: "forward",    label: "Вперёд" },
      { id: "backward",   label: "Назад" },
      { id: "left",       label: "Влево" },
      { id: "right",      label: "Вправо" },
      { id: "jump",       label: "Прыжок" },
      { id: "sprint",     label: "Спринт" },
      { id: "crouch",     label: "Присед / Подкат" },
      { id: "kick",       label: "Удар ногой" },
      { id: "inspect",    label: "Осмотр оружия (CS2)" },
      { id: "flashlight", label: "Фонарик" },
      { id: "chat",       label: "Чат" },
      { id: "pickup",     label: "Подобрать оружие" },
      { id: "drop",       label: "Выбросить оружие" },
      { id: "reload",     label: "Перезарядка" },
      { id: "holster",    label: "Слот 1 / Убрать" },
    ];
    const DEFAULT_BINDS = {
      forward:    "KeyW",
      backward:   "KeyS",
      left:       "KeyA",
      right:      "KeyD",
      jump:       "Space",
      sprint:     "ShiftLeft",
      crouch:     "ControlLeft",
      kick:       "KeyV",
      inspect:    "KeyF",
      flashlight: "KeyL",
      chat:       "KeyU",
      pickup:     "KeyE",
      drop:       "KeyG",
      reload:     "KeyR",
      holster:    "Digit1",
    };

    const userSettings = {
      ditherEnabled: Boolean(scene.preset.dither_enabled),
      colorLevels: Math.round(scene.preset.color_levels),
      ditherStrength: Math.round(scene.preset.dither_strength * 100),
      fovDeg: Math.round(scene.playerConfig.fov_deg),
      sensMult: 1.0,
      sfxVolume: 85,
      binds: { ...DEFAULT_BINDS },
    };

    let listeningBindAction = null;

    function formatKeyCode(code) {
      if (!code) return "—";
      if (code.startsWith("Key")) return code.slice(3);
      if (code.startsWith("Digit")) return code.slice(5);
      const map = {
        Space: "SPACE",
        ShiftLeft: "SHIFT",
        ShiftRight: "R-SHIFT",
        ControlLeft: "CTRL",
        ControlRight: "R-CTRL",
        AltLeft: "ALT",
        AltRight: "R-ALT",
        Tab: "TAB",
        ArrowUp: "↑",
        ArrowDown: "↓",
        ArrowLeft: "←",
        ArrowRight: "→",
        Backquote: "~",
        BracketLeft: "[",
        BracketRight: "]",
      };
      return map[code] || code.toUpperCase();
    }

    function codeMatchesAction(code, action) {
      const bound = userSettings.binds[action];
      if (!bound) return false;
      if (code === bound) return true;
      // Allow either Shift when bound to ShiftLeft, and either Ctrl when bound to ControlLeft
      if (bound === "ShiftLeft" && code === "ShiftRight") return true;
      if (bound === "ControlLeft" && code === "ControlRight") return true;
      return false;
    }

    function isActionDown(action) {
      const bound = userSettings.binds[action];
      if (keysDown.has(bound)) return true;
      if (bound === "ShiftLeft" && keysDown.has("ShiftRight")) return true;
      if (bound === "ControlLeft" && keysDown.has("ControlRight")) return true;
      return false;
    }

    function saveUserSettings() {
      try {
        window.localStorage.setItem(SETTINGS_KEY, JSON.stringify(userSettings));
      } catch (_) {}
    }

    function loadUserSettings() {
      try {
        const raw = window.localStorage.getItem(SETTINGS_KEY);
        if (!raw) return;
        const parsed = JSON.parse(raw);
        if (typeof parsed.ditherEnabled === "boolean") userSettings.ditherEnabled = parsed.ditherEnabled;
        if (typeof parsed.colorLevels === "number") {
          userSettings.colorLevels = Math.max(6, Math.min(32, Math.round(parsed.colorLevels)));
        }
        if (typeof parsed.ditherStrength === "number") {
          userSettings.ditherStrength = Math.max(0, Math.min(150, Math.round(parsed.ditherStrength)));
        }
        if (typeof parsed.fovDeg === "number") {
          userSettings.fovDeg = Math.max(60, Math.min(110, Math.round(parsed.fovDeg)));
        }
        if (typeof parsed.sensMult === "number") {
          userSettings.sensMult = Math.max(0.2, Math.min(3.0, parsed.sensMult));
        }
        if (typeof parsed.sfxVolume === "number") {
          userSettings.sfxVolume = Math.max(0, Math.min(100, Math.round(parsed.sfxVolume)));
        }
        if (parsed.binds && typeof parsed.binds === "object") {
          for (const def of BIND_DEFS) {
            if (typeof parsed.binds[def.id] === "string" && parsed.binds[def.id].length > 0) {
              userSettings.binds[def.id] = parsed.binds[def.id];
            }
          }
          // Migrate legacy binds: crouch strictly Ctrl (remove C), F -> CS2 inspect, flashlight -> L!
          let migrated = false;
          if (userSettings.binds.crouch === "KeyC") {
            userSettings.binds.crouch = "ControlLeft";
            migrated = true;
          }
          if (!parsed.binds.inspect && userSettings.binds.flashlight === "KeyF") {
            userSettings.binds.flashlight = "KeyL";
            userSettings.binds.inspect = "KeyF";
            migrated = true;
          }
          if (!parsed.binds.chat) {
            userSettings.binds.chat = "KeyU";
            migrated = true;
          }
          if (migrated) saveUserSettings();
        }
      } catch (_) {}
    }

    function renderBindList() {
      bindListEl.innerHTML = "";
      for (const def of BIND_DEFS) {
        const row = document.createElement("div");
        row.className = "bind-row";
        const nameSpan = document.createElement("span");
        nameSpan.className = "bind-name";
        nameSpan.textContent = def.label;

        const btn = document.createElement("button");
        btn.className = "bind-btn" + (listeningBindAction === def.id ? " listening" : "");
        btn.textContent = listeningBindAction === def.id
          ? "НАЖМИТЕ КЛАВИШУ..."
          : formatKeyCode(userSettings.binds[def.id]);
        btn.addEventListener("click", (ev) => {
          ev.stopPropagation();
          listeningBindAction = (listeningBindAction === def.id) ? null : def.id;
          renderBindList();
          settingsStatus.textContent = listeningBindAction
            ? "Ожидание клавиши (Esc — отмена)..."
            : "Сохраняется автоматически";
        });

        row.appendChild(nameSpan);
        row.appendChild(btn);
        bindListEl.appendChild(row);
      }
    }

    function applySettingsToEngine() {
      preset.dither_enabled = userSettings.ditherEnabled;
      preset.color_levels = userSettings.colorLevels;
      preset.dither_strength = userSettings.ditherStrength / 100.0;
      player.cfg.fov_deg = userSettings.fovDeg;
      player.cfg.mouse_sensitivity = BASE_MOUSE_SENS * userSettings.sensMult;

      // Sync UI controls
      setDitherToggle.textContent = userSettings.ditherEnabled ? "ВКЛ" : "ВЫКЛ";
      setDitherToggle.classList.toggle("on", userSettings.ditherEnabled);
      setDitherLevels.value = String(userSettings.colorLevels);
      valDitherLevels.textContent = String(userSettings.colorLevels);
      setDitherStrength.value = String(userSettings.ditherStrength);
      valDitherStrength.textContent = `${userSettings.ditherStrength}%`;
      setFovSlider.value = String(userSettings.fovDeg);
      valFov.textContent = `${userSettings.fovDeg}°`;
      setSensSlider.value = userSettings.sensMult.toFixed(2);
      valSens.textContent = `${userSettings.sensMult.toFixed(2)}×`;
      setSfxSlider.value = String(userSettings.sfxVolume);
      valSfx.textContent = `${userSettings.sfxVolume}%`;

      // Sync dynamic key labels on HUD & quick bar
      const kF = formatKeyCode(userSettings.binds.forward);
      const kL = formatKeyCode(userSettings.binds.left);
      const kB = formatKeyCode(userSettings.binds.backward);
      const kR = formatKeyCode(userSettings.binds.right);
      const kSp = formatKeyCode(userSettings.binds.sprint);
      const kCr = formatKeyCode(userSettings.binds.crouch);
      const kJm = formatKeyCode(userSettings.binds.jump);
      const kKk = formatKeyCode(userSettings.binds.kick);
      const kIn = formatKeyCode(userSettings.binds.inspect);
      const kFl = formatKeyCode(userSettings.binds.flashlight);
      const kPk = formatKeyCode(userSettings.binds.pickup);
      const kDr = formatKeyCode(userSettings.binds.drop);
      const kRl = formatKeyCode(userSettings.binds.reload);
      const kHl = formatKeyCode(userSettings.binds.holster);
      const kCh = formatKeyCode(userSettings.binds.chat);
      if (hudHelp) {
        hudHelp.textContent =
          `${kF}${kL}${kB}${kR} Ходьба | ${kCr} Присед/Подкат | ${kIn} Осмотр | ${kFl} Фонарик | ${kCh} Чат | ${kPk} Взять | ${kDr} Бросить | ` +
          `1–0 Оружие · Q Смена | ЛКМ Огонь | ПКМ Прицел | ${kRl} Перезарядка | ${kKk} Удар`;
      }
      if (pickupPromptEl) {
        const nearWp = weapon.nearestPickup(player.pos) || weapon.worldPickup;
        const pSpec = nearWp ? (weapon.SPECS[nearWp.weaponId || "m1911"] || weapon.SPECS.m1911) : weapon.SPECS.m1911;
        const wpM = nearWp ? nearWp.magAmmo : pSpec.mag_capacity;
        const wpR = nearWp ? nearWp.reserveAmmo : pSpec.start_reserve;
        pickupPromptEl.innerHTML = `<span class="pkey">${kPk}</span>ПОДОБРАТЬ ${pSpec.name} (${pSpec.caliber} · ${wpM}/${wpR})`;
      }
      if (btnPickup) btnPickup.textContent = `${kPk}: ПОДОБРАТЬ`;
      if (btnDrop) btnDrop.textContent = `${kDr}: БРОСИТЬ`;
      if (btnReload) btnReload.textContent = `${kRl}: ПЕРЕЗАРЯДКА`;
      document.getElementById("btn-kick").textContent = `${kKk}: УДАР НОГОЙ`;
      document.getElementById("btn-flash").textContent = `${kFl}: ФОНАРИК`;
      renderBindList();
      if (typeof updateMenuMusicVolume === "function") updateMenuMusicVolume();
    }

    function setSettingsTab(tab) {
      const isGen = (tab === "general");
      tabBtnGeneral.classList.toggle("active", isGen);
      tabBtnControls.classList.toggle("active", !isGen);
      tabPaneGeneral.classList.toggle("hidden", !isGen);
      tabPaneControls.classList.toggle("hidden", isGen);
      if (isGen) listeningBindAction = null;
      renderBindList();
    }

    function openSettings() {
      listeningBindAction = null;
      applySettingsToEngine();
      settingsCard.classList.remove("hidden");
    }

    function closeSettings() {
      listeningBindAction = null;
      settingsCard.classList.add("hidden");
      settingsStatus.textContent = "Сохраняется автоматически";
    }

    loadUserSettings();
    applySettingsToEngine();

    // ---------- Full Web Audio API SFX Engine (real 11-slice concrete footsteps + PS1 action SFX!) ----------
    const FOOTSTEP_WINDOWS = [
      [0.00, 0.25], [0.33, 0.60], [0.68, 0.95], [1.03, 1.30],
      [1.38, 1.65], [1.73, 1.98], [2.12, 2.38], [2.48, 2.72],
      [2.83, 3.08], [3.22, 3.48], [3.60, 3.85],
    ];
    var audioCtx = null;
    var concreteAudioBuf = null;
    var concreteFetchStarted = false;
    var concreteRawBytesPromise = null;
    var lastFootstepIdx = -1;

    // Seamless main-menu background music (menu_theme.mp3, cut & crossfaded cleanly at 1:21)
    var MENU_MUSIC_GAIN = 0.28;
    var menuMusicRawPromise = null;
    var menuMusicBuf = null;
    var menuMusicLoopStart = 0.0;
    var menuMusicLoopEnd = 0.0;
    var menuMusicSrc = null;
    var menuMusicGainNode = null;
    var menuMusicDesired = true;
    var menuMusicStopTimer = null;

    // Prefetch raw MP3 bytes immediately so decoding is instantaneous on first user interaction!
    function prefetchConcreteMp3() {
      if (concreteFetchStarted) return;
      concreteFetchStarted = true;
      concreteRawBytesPromise = fetch("/assets/sfx/fast-walking-on-a-concrete-surface.mp3", { cache: "force-cache" })
        .then(r => r.ok ? r.arrayBuffer() : null)
        .catch(() => null);
      menuMusicRawPromise = fetch("/assets/sfx/menu_theme.mp3", { cache: "force-cache" })
        .then(r => r.ok ? r.arrayBuffer() : null)
        .catch(() => null);
    }
    prefetchConcreteMp3();

    function getSfxVol() {
      return Math.max(0.0, Math.min(1.0, (userSettings.sfxVolume ?? 85) / 100.0));
    }

    function computeGaplessBounds(buf) {
      // Automatically trim any browser-added MP3 decoder leading/trailing zero padding for 0ms seamless loop!
      const sr = buf.sampleRate || 44100;
      const len = buf.length;
      if (len <= 0) return [0.0, buf.duration || 0.0];
      const ch0 = buf.getChannelData(0);
      const ch1 = buf.numberOfChannels > 1 ? buf.getChannelData(1) : ch0;
      const thr = 1.5e-4;
      let i0 = 0;
      const maxLead = Math.min(len, Math.floor(sr * 0.12));
      while (i0 < maxLead && Math.abs(ch0[i0]) < thr && Math.abs(ch1[i0]) < thr) i0++;
      let i1 = len - 1;
      const minTrail = Math.max(i0 + 1024, len - Math.floor(sr * 0.12));
      while (i1 > minTrail && Math.abs(ch0[i1]) < thr && Math.abs(ch1[i1]) < thr) i1--;
      return [i0 / sr, Math.min(buf.duration, (i1 + 1) / sr)];
    }

    function updateMenuMusicVolume() {
      if (!menuMusicGainNode || !audioCtx || !menuMusicDesired) return;
      const t0 = audioCtx.currentTime;
      const target = Math.max(0.0001, getSfxVol() * MENU_MUSIC_GAIN);
      try {
        menuMusicGainNode.gain.cancelScheduledValues(t0);
        menuMusicGainNode.gain.setTargetAtTime(target, t0, 0.06);
      } catch (_) {}
    }

    function startMenuMusicIfReady() {
      if (!menuMusicDesired || !audioCtx || !menuMusicBuf) return;
      if (audioCtx.state === "suspended") return;
      if (menuMusicStopTimer) {
        clearTimeout(menuMusicStopTimer);
        menuMusicStopTimer = null;
      }
      if (menuMusicSrc) {
        updateMenuMusicVolume();
        return;
      }
      try {
        const t0 = audioCtx.currentTime;
        const g = audioCtx.createGain();
        const target = Math.max(0.0001, getSfxVol() * MENU_MUSIC_GAIN);
        g.gain.setValueAtTime(0.0001, t0);
        g.gain.linearRampToValueAtTime(target, t0 + 0.65);
        g.connect(audioCtx.destination);

        const src = audioCtx.createBufferSource();
        src.buffer = menuMusicBuf;
        src.loop = true;
        src.loopStart = menuMusicLoopStart;
        src.loopEnd = menuMusicLoopEnd > menuMusicLoopStart + 1.0 ? menuMusicLoopEnd : menuMusicBuf.duration;
        src.connect(g);
        src.start(t0, menuMusicLoopStart);
        menuMusicSrc = src;
        menuMusicGainNode = g;
      } catch (_) {}
    }

    function syncMenuMusic(active) {
      menuMusicDesired = Boolean(active);
      if (menuMusicDesired) {
        if (audioCtx && audioCtx.state === "running") {
          startMenuMusicIfReady();
        }
      } else if (menuMusicSrc && menuMusicGainNode && audioCtx) {
        const oldSrc = menuMusicSrc;
        const oldGain = menuMusicGainNode;
        menuMusicSrc = null;
        menuMusicGainNode = null;
        try {
          const t0 = audioCtx.currentTime;
          oldGain.gain.cancelScheduledValues(t0);
          oldGain.gain.setValueAtTime(Math.max(0.0001, oldGain.gain.value), t0);
          oldGain.gain.linearRampToValueAtTime(0.0001, t0 + 0.60);
          menuMusicStopTimer = setTimeout(() => {
            try { oldSrc.stop(); } catch (_) {}
            try { oldSrc.disconnect(); oldGain.disconnect(); } catch (_) {}
          }, 680);
        } catch (_) {
          try { oldSrc.stop(); } catch (__) {}
        }
      }
    }

    function getAudioCtx() {
      if (!audioCtx) {
        const AC = window.AudioContext || window.webkitAudioContext;
        if (AC) audioCtx = new AC();
      }
      if (audioCtx && audioCtx.state === "suspended") {
        audioCtx.resume().then(() => {
          if (menuMusicDesired) startMenuMusicIfReady();
        }).catch(() => {});
      }
      if (audioCtx && !concreteAudioBuf && concreteRawBytesPromise) {
        const p = concreteRawBytesPromise;
        concreteRawBytesPromise = null;
        p.then(buf => {
          if (buf && audioCtx) {
            return audioCtx.decodeAudioData(buf.slice(0));
          }
          return null;
        }).then(decoded => {
          if (decoded) concreteAudioBuf = decoded;
        }).catch(() => {});
      }
      if (audioCtx && !menuMusicBuf && menuMusicRawPromise) {
        const mp = menuMusicRawPromise;
        menuMusicRawPromise = null;
        mp.then(buf => {
          if (buf && audioCtx) {
            return audioCtx.decodeAudioData(buf.slice(0));
          }
          return null;
        }).then(decoded => {
          if (decoded) {
            menuMusicBuf = decoded;
            const [l0, l1] = computeGaplessBounds(decoded);
            menuMusicLoopStart = l0;
            menuMusicLoopEnd = l1;
            if (menuMusicDesired) startMenuMusicIfReady();
          }
        }).catch(() => {});
      }
      if (audioCtx && audioCtx.state === "running" && menuMusicDesired && menuMusicBuf && !menuMusicSrc) {
        startMenuMusicIfReady();
      }
      return audioCtx;
    }

    // Unlock browser WebAudio on the very first user gesture so menu music starts seamlessly!
    function unlockMenuAudioOnGesture() {
      getAudioCtx();
      if (menuMusicDesired) startMenuMusicIfReady();
    }
    window.addEventListener("pointerdown", unlockMenuAudioOnGesture, { passive: true });
    window.addEventListener("keydown", unlockMenuAudioOnGesture, { passive: true });
    // Also attempt immediate start for browsers/environments where autoplay is allowed
    try { getAudioCtx(); } catch (_) {}

    function playFootstepSfx(stepMode = "walk", stairDistance = null) {
      try {
        const vol = getSfxVol()*(stairDistance===null?1:1-.81*stairDistance);
        if (vol <= 0.005) return;
        const ac = getAudioCtx();
        if (!ac) return;
        const t0 = ac.currentTime;
        const modeGain = stepMode === "sprint" ? 0.96 : (stepMode === "crouch" ? 0.38 : 0.72);

        function route(src,g,duration){src.connect(g);if(stairDistance===null){g.connect(ac.destination);return;}
          const filter=ac.createBiquadFilter(),echo=ac.createDelay(.25),reflection=ac.createGain();filter.type='lowpass';filter.frequency.value=4400-3300*stairDistance;
          echo.delayTime.value=.076+.057*stairDistance;reflection.gain.value=.27+.10*stairDistance;g.connect(filter).connect(ac.destination);filter.connect(echo).connect(reflection).connect(ac.destination);
          setTimeout(()=>{[src,g,filter,echo,reflection].forEach(n=>n.disconnect());},(duration+.5)*1000);}
        if (concreteAudioBuf) {
          let idx = Math.floor(Math.random() * FOOTSTEP_WINDOWS.length);
          if (idx === lastFootstepIdx) idx = (idx + 1) % FOOTSTEP_WINDOWS.length;
          lastFootstepIdx = idx;
          const [w0, w1] = FOOTSTEP_WINDOWS[idx];
          const dur = Math.max(0.08, w1 - w0);

          const src = ac.createBufferSource();
          src.buffer = concreteAudioBuf;
          src.playbackRate.value = (stepMode === "sprint" ? 1.14 : (stepMode === "crouch" ? 0.93 : 1.0)) * (0.96 + Math.random() * 0.08);

          // Boost quiet MP3 cleanly (3.8x) with gentle fade-in/fade-out to avoid clicks
          const g = ac.createGain();
          const peak = Math.min(4.8, 3.8 * vol * modeGain);
          g.gain.setValueAtTime(0.001, t0);
          g.gain.linearRampToValueAtTime(peak, t0 + 0.006);
          g.gain.setValueAtTime(peak, t0 + Math.max(0.015, dur / src.playbackRate.value - 0.018));
          g.gain.linearRampToValueAtTime(0.001, t0 + dur / src.playbackRate.value);

          route(src,g,dur);
          src.start(t0, w0, dur);
        } else {
          // Instant gritty procedural concrete fallback while MP3 decodes
          const dur = 0.12;
          const len = Math.floor(ac.sampleRate * dur);
          const buf = ac.createBuffer(1, len, ac.sampleRate);
          const ch = buf.getChannelData(0);
          for (let i = 0; i < len; i++) {
            const t = i / ac.sampleRate;
            ch[i] = (Math.sin(2 * Math.PI * 92 * t) * Math.exp(-t * 28) * 0.6 + (Math.random() * 2 - 1) * Math.exp(-t * 36) * 0.45);
          }
          const src = ac.createBufferSource();
          src.buffer = buf;
          const g = ac.createGain();
          g.gain.value = 0.45 * vol * modeGain;
          route(src,g,dur);
          src.start(t0);
        }
      } catch (_) {}
    }

    function playJumpSfx() {
      try {
        const vol = getSfxVol();
        if (vol <= 0.005) return;
        const ac = getAudioCtx();
        if (!ac) return;
        const t0 = ac.currentTime;
        const osc = ac.createOscillator();
        osc.type = "sine";
        osc.frequency.setValueAtTime(110, t0);
        osc.frequency.exponentialRampToValueAtTime(165, t0 + 0.08);
        const gain = ac.createGain();
        gain.gain.setValueAtTime(0.24 * vol, t0);
        gain.gain.exponentialRampToValueAtTime(0.001, t0 + 0.11);
        osc.connect(gain).connect(ac.destination);
        osc.start(t0);
        osc.stop(t0 + 0.12);
      } catch (_) {}
    }

    function playKickWhoosh() {
      try {
        const vol = getSfxVol();
        if (vol <= 0.005) return;
        const ac = getAudioCtx();
        if (!ac) return;
        const t0 = ac.currentTime;
        const len = Math.floor(ac.sampleRate * 0.16);
        const buf = ac.createBuffer(1, len, ac.sampleRate);
        const ch = buf.getChannelData(0);
        for (let i = 0; i < len; i++) ch[i] = (Math.random() * 2 - 1) * (1 - i / len);
        const src = ac.createBufferSource();
        src.buffer = buf;
        const filt = ac.createBiquadFilter();
        filt.type = "bandpass";
        filt.frequency.setValueAtTime(240, t0);
        filt.frequency.exponentialRampToValueAtTime(950, t0 + 0.08);
        filt.frequency.exponentialRampToValueAtTime(180, t0 + 0.16);
        filt.Q.value = 1.4;
        const gain = ac.createGain();
        gain.gain.setValueAtTime(0.001, t0);
        gain.gain.linearRampToValueAtTime(0.28 * vol, t0 + 0.04);
        gain.gain.exponentialRampToValueAtTime(0.001, t0 + 0.16);
        src.connect(filt).connect(gain).connect(ac.destination);
        src.start(t0);
      } catch (_) {}
    }

    function playKickThud() {
      try {
        const vol = getSfxVol();
        if (vol <= 0.005) return;
        const ac = getAudioCtx();
        if (!ac) return;
        const t0 = ac.currentTime;
        const osc = ac.createOscillator();
        osc.type = "triangle";
        osc.frequency.setValueAtTime(150, t0);
        osc.frequency.exponentialRampToValueAtTime(38, t0 + 0.15);
        const gain = ac.createGain();
        gain.gain.setValueAtTime(0.55 * vol, t0);
        gain.gain.exponentialRampToValueAtTime(0.001, t0 + 0.16);
        osc.connect(gain).connect(ac.destination);
        osc.start(t0);
        osc.stop(t0 + 0.17);
      } catch (_) {}
    }

    function playSlideSfx() {
      try {
        const vol = getSfxVol();
        if (vol <= 0.005) return;
        const ac = getAudioCtx();
        if (!ac) return;
        const t0 = ac.currentTime;
        const dur = 0.50;
        const len = Math.floor(ac.sampleRate * dur);
        const buf = ac.createBuffer(1, len, ac.sampleRate);
        const ch = buf.getChannelData(0);
        for (let i = 0; i < len; i++) {
          const u = i / len;
          ch[i] = (Math.random() * 2 - 1) * Math.pow(1 - u, 1.25);
        }
        const src = ac.createBufferSource();
        src.buffer = buf;
        const filt = ac.createBiquadFilter();
        filt.type = "bandpass";
        filt.frequency.setValueAtTime(720, t0);
        filt.frequency.exponentialRampToValueAtTime(195, t0 + dur);
        filt.Q.value = 1.1;
        const gain = ac.createGain();
        gain.gain.setValueAtTime(0.001, t0);
        gain.gain.linearRampToValueAtTime(0.32 * vol, t0 + 0.035);
        gain.gain.exponentialRampToValueAtTime(0.001, t0 + dur);
        src.connect(filt).connect(gain).connect(ac.destination);
        src.start(t0);
      } catch (_) {}
    }

    function playLandSfx(speed) {
      try {
        const vol = getSfxVol();
        if (vol <= 0.005) return;
        const ac = getAudioCtx();
        if (!ac) return;
        const t0 = ac.currentTime;
        const osc = ac.createOscillator();
        osc.type = "sine";
        osc.frequency.setValueAtTime(118, t0);
        osc.frequency.exponentialRampToValueAtTime(36, t0 + 0.11);
        const gain = ac.createGain();
        const amp = Math.min(0.45, 0.16 + speed * 0.035) * vol;
        gain.gain.setValueAtTime(amp, t0);
        gain.gain.exponentialRampToValueAtTime(0.001, t0 + 0.12);
        osc.connect(gain).connect(ac.destination);
        osc.start(t0);
        osc.stop(t0 + 0.13);
        if (speed > 3.2) playFootstepSfx(speed > 6.0 ? "sprint" : "walk");
      } catch (_) {}
    }

    function playFlashlightSfx() {
      try {
        const vol = getSfxVol();
        if (vol <= 0.005) return;
        const ac = getAudioCtx();
        if (!ac) return;
        const t0 = ac.currentTime;
        for (const [dtOff, f] of [[0.0, 1900], [0.022, 2450]]) {
          const osc = ac.createOscillator();
          osc.type = "square";
          osc.frequency.setValueAtTime(f, t0 + dtOff);
          const g = ac.createGain();
          g.gain.setValueAtTime(0.14 * vol, t0 + dtOff);
          g.gain.exponentialRampToValueAtTime(0.001, t0 + dtOff + 0.018);
          osc.connect(g).connect(ac.destination);
          osc.start(t0 + dtOff);
          osc.stop(t0 + dtOff + 0.02);
        }
      } catch (_) {}
    }

    function playUiClickSfx() {
      try {
        const vol = getSfxVol();
        if (vol <= 0.005) return;
        const ac = getAudioCtx();
        if (!ac) return;
        const t0 = ac.currentTime;
        const osc = ac.createOscillator();
        osc.type = "sine";
        osc.frequency.setValueAtTime(860, t0);
        const g = ac.createGain();
        g.gain.setValueAtTime(0.10 * vol, t0);
        g.gain.exponentialRampToValueAtTime(0.001, t0 + 0.035);
        osc.connect(g).connect(ac.destination);
        osc.start(t0);
        osc.stop(t0 + 0.04);
      } catch (_) {}
    }

    let _dlgBlipIdx = 0;
    let _npcVoiceSrc = null;
    function _stopNpcVoice() {
      if (_npcVoiceSrc) {
        try { _npcVoiceSrc.stop(); } catch (_) {}
        try { _npcVoiceSrc.disconnect(); } catch (_) {}
        _npcVoiceSrc = null;
      }
    }
    function _playNpcVoiceMono(evKey,room=null) {
      const meta = WPN_WAV_MAP[evKey];
      if (!meta) return false;
      ensureWeaponWavsLoaded();
      const buf = wpnWavBufs[evKey];
      if (!buf) return false;
      try {
        const vol = getSfxVol();
        if (vol <= 0.005) return true;
        const ac = getAudioCtx();
        if (!ac) return false;
        _stopNpcVoice();
        const src = ac.createBufferSource();
        src.buffer = buf;
        const jit = meta.jitter || 0.0;
        src.playbackRate.value = (room?.rate ?? meta.pitch ?? 1.0) * (1.0 + (Math.random() * 2.0 - 1.0) * jit);
        const g = ac.createGain();
        g.gain.value = Math.min(1.0, vol * meta.gain * (room?.gain ?? 1));
        if(room){const lp=ac.createBiquadFilter();lp.type='lowpass';lp.frequency.value=room.cutoff;src.connect(lp).connect(g);}else src.connect(g);
        g.connect(ac.destination);
        if(evKey==='hero_text')src.start(ac.currentTime,0,Math.min(.095,buf.duration));else src.start(ac.currentTime);
        _npcVoiceSrc = src;
        return true;
      } catch (_) {
        return false;
      }
    }
    function _playUndertaleBlipFallback(idx) {
      try {
        const vol = getSfxVol();
        if (vol <= 0.005) return;
        const ac = getAudioCtx();
        if (!ac) return;
        _stopNpcVoice();
        const t0 = ac.currentTime;
        // Authentic Undertale-style staccato chiptune voice blip (crisp filtered square wave)
        const baseFreqs = [196, 208, 220, 214, 232, 202, 226, 240];
        const f0 = baseFreqs[((idx | 0) & 7)] * (0.97 + Math.random() * 0.06);
        const osc = ac.createOscillator();
        osc.type = "square";
        osc.frequency.setValueAtTime(f0 * 1.06, t0);
        osc.frequency.exponentialRampToValueAtTime(f0 * 0.94, t0 + 0.048);
        const flt = ac.createBiquadFilter();
        flt.type = "lowpass";
        flt.frequency.setValueAtTime(1450, t0);
        const g = ac.createGain();
        g.gain.setValueAtTime(0.001, t0);
        g.gain.linearRampToValueAtTime(0.11 * vol, t0 + 0.003);
        g.gain.setValueAtTime(0.095 * vol, t0 + 0.032);
        g.gain.exponentialRampToValueAtTime(0.001, t0 + 0.052);
        osc.connect(flt).connect(g).connect(ac.destination);
        osc.start(t0);
        osc.stop(t0 + 0.056);
        _npcVoiceSrc = osc;
      } catch (_) {}
    }

    function playDialogueEventSfx(ev) {
      if (ev === "dlg_open") {
        if (_playNpcVoiceMono("npc_greet")) return;
      } else if (ev && ev.startsWith("dlg_voice_")) {
        const idx = parseInt(ev.slice("dlg_voice_".length), 10) || 0;
        if (_playNpcVoiceMono("npc_voice_" + (idx & 7))) return;
        _playUndertaleBlipFallback(idx);
        return;
      } else if (ev === "dlg_blip") {
        _dlgBlipIdx = (_dlgBlipIdx + 1) % 8;
        if (_playNpcVoiceMono("npc_voice_" + _dlgBlipIdx)) return;
        _playUndertaleBlipFallback(_dlgBlipIdx);
        return;
      }
      try {
        const vol = getSfxVol();
        if (vol <= 0.005) return;
        const ac = getAudioCtx();
        if (!ac) return;
        const t0 = ac.currentTime;
        if (ev === "dlg_open") {
          const notes = [[212, 0.0, 0.052], [268, 0.064, 0.068]];
          for (const [freq, off, dur] of notes) {
            const osc = ac.createOscillator();
            osc.type = "square";
            osc.frequency.setValueAtTime(freq, t0 + off);
            const flt = ac.createBiquadFilter();
            flt.type = "lowpass";
            flt.frequency.setValueAtTime(1500, t0 + off);
            const g = ac.createGain();
            g.gain.setValueAtTime(0.001, t0 + off);
            g.gain.linearRampToValueAtTime(0.10 * vol, t0 + off + 0.005);
            g.gain.exponentialRampToValueAtTime(0.001, t0 + off + dur);
            osc.connect(flt).connect(g).connect(ac.destination);
            osc.start(t0 + off);
            osc.stop(t0 + off + dur + 0.01);
          }
        } else if (ev === "dlg_next") {
          const osc = ac.createOscillator();
          osc.type = "square";
          osc.frequency.setValueAtTime(520, t0);
          osc.frequency.linearRampToValueAtTime(680, t0 + 0.04);
          const flt = ac.createBiquadFilter();
          flt.type = "lowpass";
          flt.frequency.setValueAtTime(1700, t0);
          const g = ac.createGain();
          g.gain.setValueAtTime(0.07 * vol, t0);
          g.gain.exponentialRampToValueAtTime(0.001, t0 + 0.068);
          osc.connect(flt).connect(g).connect(ac.destination);
          osc.start(t0);
          osc.stop(t0 + 0.072);
        } else if (ev === "dlg_close") {
          _stopNpcVoice();
          const notes = [[380, 0.0, 0.048], [260, 0.046, 0.088]];
          for (const [freq, off, dur] of notes) {
            const osc = ac.createOscillator();
            osc.type = "square";
            osc.frequency.setValueAtTime(freq, t0 + off);
            const flt = ac.createBiquadFilter();
            flt.type = "lowpass";
            flt.frequency.setValueAtTime(1500, t0 + off);
            const g = ac.createGain();
            g.gain.setValueAtTime(0.07 * vol, t0 + off);
            g.gain.exponentialRampToValueAtTime(0.001, t0 + off + dur);
            osc.connect(flt).connect(g).connect(ac.destination);
            osc.start(t0 + off);
            osc.stop(t0 + off + dur + 0.01);
          }
        }
      } catch (_) {}
    }

    // ---- M1911 .45 ACP WebAudio Procedural SFX (mirrors engine/audio.py 1:1) ----
    function makeNoiseBuf(ac, durSec) {
      const len = Math.max(1, Math.floor(ac.sampleRate * durSec));
      const buf = ac.createBuffer(1, len, ac.sampleRate);
      const data = buf.getChannelData(0);
      for (let i = 0; i < len; i++) data[i] = Math.random() * 2.0 - 1.0;
      return buf;
    }

    function playGunshotSfx() {
      try {
        const vol = getSfxVol();
        if (vol <= 0.005) return;
        const ac = getAudioCtx();
        if (!ac) return;
        const t0 = ac.currentTime;
        // 1) Supersonic muzzle crack + warehouse reverb tail
        const ns = ac.createBufferSource();
        ns.buffer = makeNoiseBuf(ac, 0.42);
        const lp = ac.createBiquadFilter();
        lp.type = "lowpass";
        lp.frequency.setValueAtTime(4800 + Math.random() * 600, t0);
        lp.frequency.exponentialRampToValueAtTime(520, t0 + 0.34);
        const ng = ac.createGain();
        ng.gain.setValueAtTime(0.75 * vol, t0);
        ng.gain.exponentialRampToValueAtTime(0.08 * vol, t0 + 0.085);
        ng.gain.exponentialRampToValueAtTime(0.001, t0 + 0.40);
        ns.connect(lp).connect(ng).connect(ac.destination);
        ns.start(t0);
        // 2) Heavy .45 ACP sub-bass punch
        const osc = ac.createOscillator();
        osc.type = "triangle";
        osc.frequency.setValueAtTime(125 + Math.random() * 16, t0);
        osc.frequency.exponentialRampToValueAtTime(38, t0 + 0.14);
        const og = ac.createGain();
        og.gain.setValueAtTime(0.88 * vol, t0);
        og.gain.exponentialRampToValueAtTime(0.001, t0 + 0.18);
        osc.connect(og).connect(ac.destination);
        osc.start(t0);
        osc.stop(t0 + 0.19);
        // 3) Crisp steel slide blowback clack (+30ms)
        const sOsc = ac.createOscillator();
        sOsc.type = "square";
        sOsc.frequency.setValueAtTime(2480, t0 + 0.030);
        const sg = ac.createGain();
        sg.gain.setValueAtTime(0.14 * vol, t0 + 0.030);
        sg.gain.exponentialRampToValueAtTime(0.001, t0 + 0.068);
        sOsc.connect(sg).connect(ac.destination);
        sOsc.start(t0 + 0.030);
        sOsc.stop(t0 + 0.072);
      } catch (_) {}
    }

    function playGlockShotSfx() {
      try {
        const vol = getSfxVol(); if (vol <= 0.005) return;
        const ac = getAudioCtx(); if (!ac) return;
        const t0 = ac.currentTime;
        // 1) Snappy high-velocity 9mm muzzle crack
        const ns = ac.createBufferSource();
        ns.buffer = makeNoiseBuf(ac, 0.28);
        const lp = ac.createBiquadFilter();
        lp.type = "lowpass";
        lp.frequency.setValueAtTime(6400 + Math.random() * 800, t0);
        lp.frequency.exponentialRampToValueAtTime(780, t0 + 0.22);
        const ng = ac.createGain();
        ng.gain.setValueAtTime(0.68 * vol, t0);
        ng.gain.exponentialRampToValueAtTime(0.06 * vol, t0 + 0.055);
        ng.gain.exponentialRampToValueAtTime(0.001, t0 + 0.26);
        ns.connect(lp).connect(ng).connect(ac.destination);
        ns.start(t0);
        // 2) Tight 9mm polymer punch (higher & faster decay than .45 ACP)
        const osc = ac.createOscillator();
        osc.type = "triangle";
        osc.frequency.setValueAtTime(175 + Math.random() * 24, t0);
        osc.frequency.exponentialRampToValueAtTime(56, t0 + 0.09);
        const og = ac.createGain();
        og.gain.setValueAtTime(0.72 * vol, t0);
        og.gain.exponentialRampToValueAtTime(0.001, t0 + 0.12);
        osc.connect(og).connect(ac.destination);
        osc.start(t0);
        osc.stop(t0 + 0.13);
        // 3) Fast metallic Glock slide cycle (+20ms)
        const sOsc = ac.createOscillator();
        sOsc.type = "square";
        sOsc.frequency.setValueAtTime(3020, t0 + 0.020);
        const sg = ac.createGain();
        sg.gain.setValueAtTime(0.12 * vol, t0 + 0.020);
        sg.gain.exponentialRampToValueAtTime(0.001, t0 + 0.048);
        sOsc.connect(sg).connect(ac.destination);
        sOsc.start(t0 + 0.020);
        sOsc.stop(t0 + 0.052);
      } catch (_) {}
    }

    function playShotgunShotSfx() {
      try {
        const vol = getSfxVol(); if (vol <= 0.005) return;
        const ac = getAudioCtx(); if (!ac) return;
        const t0 = ac.currentTime;
        // 1) Thunderous 12ga blast noise + room tail
        const ns = ac.createBufferSource();
        ns.buffer = makeNoiseBuf(ac, 0.56);
        const lp = ac.createBiquadFilter();
        lp.type = "lowpass";
        lp.frequency.setValueAtTime(5200 + Math.random() * 600, t0);
        lp.frequency.exponentialRampToValueAtTime(360, t0 + 0.45);
        const ng = ac.createGain();
        ng.gain.setValueAtTime(1.05 * vol, t0);
        ng.gain.exponentialRampToValueAtTime(0.24 * vol, t0 + 0.11);
        ng.gain.exponentialRampToValueAtTime(0.001, t0 + 0.54);
        ns.connect(lp).connect(ng).connect(ac.destination);
        ns.start(t0);
        // 2) Heavy 12-gauge chest-thumping sub-boom
        const osc = ac.createOscillator();
        osc.type = "sine";
        osc.frequency.setValueAtTime(138 + Math.random() * 18, t0);
        osc.frequency.exponentialRampToValueAtTime(38, t0 + 0.24);
        const og = ac.createGain();
        og.gain.setValueAtTime(0.98 * vol, t0);
        og.gain.exponentialRampToValueAtTime(0.001, t0 + 0.30);
        osc.connect(og).connect(ac.destination);
        osc.start(t0);
        osc.stop(t0 + 0.32);
      } catch (_) {}
    }

    function playPumpBackSfx() {
      try {
        const vol = getSfxVol(); if (vol <= 0.005) return;
        const ac = getAudioCtx(); if (!ac) return;
        const t0 = ac.currentTime;
        const ns = ac.createBufferSource();
        ns.buffer = makeNoiseBuf(ac, 0.19);
        const bp = ac.createBiquadFilter();
        bp.type = "bandpass";
        bp.frequency.setValueAtTime(1350, t0);
        bp.frequency.linearRampToValueAtTime(2150, t0 + 0.16);
        bp.Q.value = 2.6;
        const ng = ac.createGain();
        ng.gain.setValueAtTime(0.38 * vol, t0);
        ng.gain.exponentialRampToValueAtTime(0.001, t0 + 0.18);
        ns.connect(bp).connect(ng).connect(ac.destination);
        ns.start(t0);
        const osc = ac.createOscillator();
        osc.type = "triangle";
        osc.frequency.setValueAtTime(2280, t0 + 0.10);
        const og = ac.createGain();
        og.gain.setValueAtTime(0.24 * vol, t0 + 0.10);
        og.gain.exponentialRampToValueAtTime(0.001, t0 + 0.16);
        osc.connect(og).connect(ac.destination);
        osc.start(t0 + 0.10); osc.stop(t0 + 0.17);
      } catch (_) {}
    }

    function playPumpFwdSfx() {
      try {
        const vol = getSfxVol(); if (vol <= 0.005) return;
        const ac = getAudioCtx(); if (!ac) return;
        const t0 = ac.currentTime;
        const osc = ac.createOscillator();
        osc.type = "triangle";
        osc.frequency.setValueAtTime(195, t0);
        osc.frequency.exponentialRampToValueAtTime(75, t0 + 0.09);
        const og = ac.createGain();
        og.gain.setValueAtTime(0.45 * vol, t0);
        og.gain.exponentialRampToValueAtTime(0.001, t0 + 0.11);
        osc.connect(og).connect(ac.destination);
        osc.start(t0); osc.stop(t0 + 0.12);
        for (const [dtOff, freq] of [[0.0, 2100], [0.032, 3150]]) {
          const c = ac.createOscillator();
          c.type = "square";
          c.frequency.setValueAtTime(freq, t0 + dtOff);
          const cg = ac.createGain();
          cg.gain.setValueAtTime(0.24 * vol, t0 + dtOff);
          cg.gain.exponentialRampToValueAtTime(0.001, t0 + dtOff + 0.045);
          c.connect(cg).connect(ac.destination);
          c.start(t0 + dtOff); c.stop(t0 + dtOff + 0.05);
        }
      } catch (_) {}
    }

    function playShellInsertSfx() {
      try {
        const vol = getSfxVol(); if (vol <= 0.005) return;
        const ac = getAudioCtx(); if (!ac) return;
        const t0 = ac.currentTime;
        const ns = ac.createBufferSource();
        ns.buffer = makeNoiseBuf(ac, 0.08);
        const bp = ac.createBiquadFilter();
        bp.type = "bandpass"; bp.frequency.value = 1850; bp.Q.value = 3.2;
        const ng = ac.createGain();
        ng.gain.setValueAtTime(0.22 * vol, t0);
        ng.gain.exponentialRampToValueAtTime(0.001, t0 + 0.075);
        ns.connect(bp).connect(ng).connect(ac.destination);
        ns.start(t0);
        const osc = ac.createOscillator();
        osc.type = "triangle";
        osc.frequency.setValueAtTime(2420, t0 + 0.048);
        const og = ac.createGain();
        og.gain.setValueAtTime(0.26 * vol, t0 + 0.048);
        og.gain.exponentialRampToValueAtTime(0.001, t0 + 0.105);
        osc.connect(og).connect(ac.destination);
        osc.start(t0 + 0.048); osc.stop(t0 + 0.11);
      } catch (_) {}
    }

    function playDryFireSfx() {
      try {
        const vol = getSfxVol(); if (vol <= 0.005) return;
        const ac = getAudioCtx(); if (!ac) return;
        const t0 = ac.currentTime;
        const osc = ac.createOscillator();
        osc.type = "square";
        osc.frequency.setValueAtTime(3100, t0);
        const g = ac.createGain();
        g.gain.setValueAtTime(0.24 * vol, t0);
        g.gain.exponentialRampToValueAtTime(0.001, t0 + 0.032);
        osc.connect(g).connect(ac.destination);
        osc.start(t0); osc.stop(t0 + 0.036);
      } catch (_) {}
    }

    function playMagOutSfx() {
      try {
        const vol = getSfxVol(); if (vol <= 0.005) return;
        const ac = getAudioCtx(); if (!ac) return;
        const t0 = ac.currentTime;
        const osc = ac.createOscillator();
        osc.type = "square";
        osc.frequency.setValueAtTime(2350, t0);
        const g = ac.createGain();
        g.gain.setValueAtTime(0.22 * vol, t0);
        g.gain.exponentialRampToValueAtTime(0.001, t0 + 0.025);
        osc.connect(g).connect(ac.destination);
        osc.start(t0); osc.stop(t0 + 0.03);
        const ns = ac.createBufferSource();
        ns.buffer = makeNoiseBuf(ac, 0.14);
        const bp = ac.createBiquadFilter();
        bp.type = "bandpass"; bp.frequency.setValueAtTime(1650, t0 + 0.015);
        bp.frequency.linearRampToValueAtTime(1150, t0 + 0.14); bp.Q.value = 3.0;
        const ng = ac.createGain();
        ng.gain.setValueAtTime(0.22 * vol, t0 + 0.015);
        ng.gain.exponentialRampToValueAtTime(0.001, t0 + 0.15);
        ns.connect(bp).connect(ng).connect(ac.destination);
        ns.start(t0 + 0.015);
      } catch (_) {}
    }

    function playMagDropSfx() {
      try {
        const vol = getSfxVol(); if (vol <= 0.005) return;
        const ac = getAudioCtx(); if (!ac) return;
        const t0 = ac.currentTime;
        for (const [dtOff, amp, fScale] of [[0.0, 0.28, 1.0], [0.095, 0.16, 1.15], [0.165, 0.08, 1.28]]) {
          for (const baseF of [1380, 2690]) {
            const osc = ac.createOscillator();
            osc.type = "triangle";
            osc.frequency.setValueAtTime(baseF * fScale, t0 + dtOff);
            const g = ac.createGain();
            g.gain.setValueAtTime(amp * vol, t0 + dtOff);
            g.gain.exponentialRampToValueAtTime(0.001, t0 + dtOff + 0.085);
            osc.connect(g).connect(ac.destination);
            osc.start(t0 + dtOff); osc.stop(t0 + dtOff + 0.09);
          }
        }
      } catch (_) {}
    }

    function playMagInSfx() {
      try {
        const vol = getSfxVol(); if (vol <= 0.005) return;
        const ac = getAudioCtx(); if (!ac) return;
        const t0 = ac.currentTime;
        // Brief slide-up noise
        const ns = ac.createBufferSource();
        ns.buffer = makeNoiseBuf(ac, 0.07);
        const hp = ac.createBiquadFilter();
        hp.type = "highpass"; hp.frequency.value = 1200;
        const ng = ac.createGain();
        ng.gain.setValueAtTime(0.16 * vol, t0);
        ng.gain.exponentialRampToValueAtTime(0.001, t0 + 0.065);
        ns.connect(hp).connect(ng).connect(ac.destination);
        ns.start(t0);
        // Solid palm-slam thump + metallic latch at +0.06s
        const thump = ac.createOscillator();
        thump.type = "sine";
        thump.frequency.setValueAtTime(155, t0 + 0.058);
        thump.frequency.exponentialRampToValueAtTime(62, t0 + 0.13);
        const tg = ac.createGain();
        tg.gain.setValueAtTime(0.45 * vol, t0 + 0.058);
        tg.gain.exponentialRampToValueAtTime(0.001, t0 + 0.14);
        thump.connect(tg).connect(ac.destination);
        thump.start(t0 + 0.058); thump.stop(t0 + 0.15);

        const cl = ac.createOscillator();
        cl.type = "square";
        cl.frequency.setValueAtTime(1940, t0 + 0.062);
        const cg = ac.createGain();
        cg.gain.setValueAtTime(0.28 * vol, t0 + 0.062);
        cg.gain.exponentialRampToValueAtTime(0.001, t0 + 0.11);
        cl.connect(cg).connect(ac.destination);
        cl.start(t0 + 0.062); cl.stop(t0 + 0.115);
      } catch (_) {}
    }

    function playSlideBackSfx() {
      try {
        const vol = getSfxVol(); if (vol <= 0.005) return;
        const ac = getAudioCtx(); if (!ac) return;
        const t0 = ac.currentTime;
        const ns = ac.createBufferSource();
        ns.buffer = makeNoiseBuf(ac, 0.14);
        const bp = ac.createBiquadFilter();
        bp.type = "bandpass"; bp.frequency.setValueAtTime(1850, t0);
        bp.frequency.linearRampToValueAtTime(2550, t0 + 0.12); bp.Q.value = 2.4;
        const ng = ac.createGain();
        ng.gain.setValueAtTime(0.32 * vol, t0);
        ng.gain.exponentialRampToValueAtTime(0.001, t0 + 0.135);
        ns.connect(bp).connect(ng).connect(ac.destination);
        ns.start(t0);
        const st = ac.createOscillator();
        st.type = "square"; st.frequency.setValueAtTime(2680, t0 + 0.09);
        const sg = ac.createGain();
        sg.gain.setValueAtTime(0.22 * vol, t0 + 0.09);
        sg.gain.exponentialRampToValueAtTime(0.001, t0 + 0.13);
        st.connect(sg).connect(ac.destination);
        st.start(t0 + 0.09); st.stop(t0 + 0.135);
      } catch (_) {}
    }

    function playSlideForwardSfx() {
      try {
        const vol = getSfxVol(); if (vol <= 0.005) return;
        const ac = getAudioCtx(); if (!ac) return;
        const t0 = ac.currentTime;
        const body = ac.createOscillator();
        body.type = "triangle"; body.frequency.setValueAtTime(220, t0);
        body.frequency.exponentialRampToValueAtTime(85, t0 + 0.08);
        const bg = ac.createGain();
        bg.gain.setValueAtTime(0.45 * vol, t0);
        bg.gain.exponentialRampToValueAtTime(0.001, t0 + 0.09);
        body.connect(bg).connect(ac.destination);
        body.start(t0); body.stop(t0 + 0.095);
        for (const [dtOff, f] of [[0.0, 2260], [0.024, 3240]]) {
          const cl = ac.createOscillator();
          cl.type = "square"; cl.frequency.setValueAtTime(f, t0 + dtOff);
          const cg = ac.createGain();
          cg.gain.setValueAtTime(0.28 * vol, t0 + dtOff);
          cg.gain.exponentialRampToValueAtTime(0.001, t0 + dtOff + 0.048);
          cl.connect(cg).connect(ac.destination);
          cl.start(t0 + dtOff); cl.stop(t0 + dtOff + 0.052);
        }
      } catch (_) {}
    }

    function playBulletImpactSfx() {
      try {
        const vol = getSfxVol(); if (vol <= 0.005) return;
        const ac = getAudioCtx(); if (!ac) return;
        const t0 = ac.currentTime;
        const ns = ac.createBufferSource();
        ns.buffer = makeNoiseBuf(ac, 0.09);
        const hp = ac.createBiquadFilter();
        hp.type = "highpass"; hp.frequency.value = 1800;
        const ng = ac.createGain();
        ng.gain.setValueAtTime(0.22 * vol, t0);
        ng.gain.exponentialRampToValueAtTime(0.001, t0 + 0.08);
        ns.connect(hp).connect(ng).connect(ac.destination);
        ns.start(t0);
      } catch (_) {}
    }

    function playWeaponDrawSfx() {
      try {
        const vol = getSfxVol(); if (vol <= 0.005) return;
        const ac = getAudioCtx(); if (!ac) return;
        const t0 = ac.currentTime;
        const ns = ac.createBufferSource();
        ns.buffer = makeNoiseBuf(ac, 0.16);
        const lp = ac.createBiquadFilter();
        lp.type = "lowpass"; lp.frequency.value = 900;
        const ng = ac.createGain();
        ng.gain.setValueAtTime(0.22 * vol, t0);
        ng.gain.exponentialRampToValueAtTime(0.001, t0 + 0.15);
        ns.connect(lp).connect(ng).connect(ac.destination);
        ns.start(t0);
      } catch (_) {}
    }

    function playDoubleBarrelShotSfx() {
      try {
        const vol = getSfxVol(); if (vol <= 0.005) return;
        const ac = getAudioCtx(); if (!ac) return;
        const t0 = ac.currentTime;
        const ns = ac.createBufferSource();
        ns.buffer = makeNoiseBuf(ac, 0.56);
        const lp = ac.createBiquadFilter();
        lp.type = "lowpass"; lp.frequency.setValueAtTime(2400, t0);
        lp.frequency.exponentialRampToValueAtTime(160, t0 + 0.50);
        const ng = ac.createGain();
        ng.gain.setValueAtTime(0.75 * vol, t0);
        ng.gain.exponentialRampToValueAtTime(0.001, t0 + 0.54);
        ns.connect(lp).connect(ng).connect(ac.destination);
        ns.start(t0);
        const sub = ac.createOscillator();
        sub.type = "sine"; sub.frequency.setValueAtTime(125, t0);
        sub.frequency.exponentialRampToValueAtTime(28, t0 + 0.24);
        const sg = ac.createGain();
        sg.gain.setValueAtTime(0.88 * vol, t0);
        sg.gain.exponentialRampToValueAtTime(0.001, t0 + 0.26);
        sub.connect(sg).connect(ac.destination);
        sub.start(t0); sub.stop(t0 + 0.28);
      } catch (_) {}
    }

    function playDbOpenSfx() {
      try {
        const vol = getSfxVol(); if (vol <= 0.005) return;
        const ac = getAudioCtx(); if (!ac) return;
        const t0 = ac.currentTime;
        for (const [dtOff, f, dur] of [[0.0, 2600, 0.045], [0.055, 860, 0.085]]) {
          const osc = ac.createOscillator();
          osc.type = "triangle"; osc.frequency.setValueAtTime(f, t0 + dtOff);
          osc.frequency.exponentialRampToValueAtTime(f * 0.48, t0 + dtOff + dur);
          const g = ac.createGain();
          g.gain.setValueAtTime(0.32 * vol, t0 + dtOff);
          g.gain.exponentialRampToValueAtTime(0.001, t0 + dtOff + dur);
          osc.connect(g).connect(ac.destination);
          osc.start(t0 + dtOff); osc.stop(t0 + dtOff + dur + 0.01);
        }
      } catch (_) {}
    }

    function playDbCloseSfx() {
      try {
        const vol = getSfxVol(); if (vol <= 0.005) return;
        const ac = getAudioCtx(); if (!ac) return;
        const t0 = ac.currentTime;
        const thud = ac.createOscillator();
        thud.type = "triangle"; thud.frequency.setValueAtTime(210, t0);
        thud.frequency.exponentialRampToValueAtTime(52, t0 + 0.10);
        const tg = ac.createGain();
        tg.gain.setValueAtTime(0.62 * vol, t0);
        tg.gain.exponentialRampToValueAtTime(0.001, t0 + 0.11);
        thud.connect(tg).connect(ac.destination);
        thud.start(t0); thud.stop(t0 + 0.12);
        const cl = ac.createOscillator();
        cl.type = "square"; cl.frequency.setValueAtTime(1920, t0 + 0.018);
        const cg = ac.createGain();
        cg.gain.setValueAtTime(0.30 * vol, t0 + 0.018);
        cg.gain.exponentialRampToValueAtTime(0.001, t0 + 0.075);
        cl.connect(cg).connect(ac.destination);
        cl.start(t0 + 0.018); cl.stop(t0 + 0.08);
      } catch (_) {}
    }

    function playShellEjectSfx() {
      try {
        const vol = getSfxVol(); if (vol <= 0.005) return;
        const ac = getAudioCtx(); if (!ac) return;
        const t0 = ac.currentTime;
        for (const [dtOff, f] of [[0.0, 2900], [0.048, 3550], [0.095, 2400]]) {
          const osc = ac.createOscillator();
          osc.type = "sine"; osc.frequency.setValueAtTime(f, t0 + dtOff);
          const g = ac.createGain();
          g.gain.setValueAtTime(0.22 * vol, t0 + dtOff);
          g.gain.exponentialRampToValueAtTime(0.001, t0 + dtOff + 0.065);
          osc.connect(g).connect(ac.destination);
          osc.start(t0 + dtOff); osc.stop(t0 + dtOff + 0.07);
        }
      } catch (_) {}
    }

    function playRevCockSfx() {
      try {
        const vol = getSfxVol(); if (vol <= 0.005) return;
        const ac = getAudioCtx(); if (!ac) return;
        const t0 = ac.currentTime;
        // Iconic 3-stage single-action revolver hammer click + cylinder index latch!
        for (const [dtOff, f, amp] of [[0.0, 2750, 0.24], [0.058, 3550, 0.30], [0.118, 4350, 0.38]]) {
          const osc = ac.createOscillator();
          osc.type = "square"; osc.frequency.setValueAtTime(f, t0 + dtOff);
          osc.frequency.exponentialRampToValueAtTime(f * 0.72, t0 + dtOff + 0.036);
          const g = ac.createGain();
          g.gain.setValueAtTime(amp * vol, t0 + dtOff);
          g.gain.exponentialRampToValueAtTime(0.001, t0 + dtOff + 0.042);
          osc.connect(g).connect(ac.destination);
          osc.start(t0 + dtOff); osc.stop(t0 + dtOff + 0.048);
        }
      } catch (_) {}
    }

    function playRevolverShotSfx() {
      try {
        const vol = getSfxVol() * 0.55; if (vol <= 0.005) return;
        const ac = getAudioCtx(); if (!ac) return;
        const t0 = ac.currentTime;
        const ns = ac.createBufferSource();
        ns.buffer = makeNoiseBuf(ac, 0.44);
        const bp = ac.createBiquadFilter();
        bp.type = "bandpass"; bp.frequency.setValueAtTime(1450, t0);
        bp.Q.value = 0.65;
        const ng = ac.createGain();
        ng.gain.setValueAtTime(0.76 * vol, t0);
        ng.gain.exponentialRampToValueAtTime(0.001, t0 + 0.42);
        ns.connect(bp).connect(ng).connect(ac.destination);
        ns.start(t0);
        const sub = ac.createOscillator();
        sub.type = "sine"; sub.frequency.setValueAtTime(180, t0);
        sub.frequency.exponentialRampToValueAtTime(42, t0 + 0.16);
        const sg = ac.createGain();
        sg.gain.setValueAtTime(0.78 * vol, t0);
        sg.gain.exponentialRampToValueAtTime(0.001, t0 + 0.18);
        sub.connect(sg).connect(ac.destination);
        sub.start(t0); sub.stop(t0 + 0.19);
      } catch (_) {}
    }

    function playRevOpenSfx() {
      try {
        const vol = getSfxVol(); if (vol <= 0.005) return;
        const ac = getAudioCtx(); if (!ac) return;
        const t0 = ac.currentTime;
        for (const [dtOff, f] of [[0.0, 2900], [0.055, 1650]]) {
          const osc = ac.createOscillator();
          osc.type = "square"; osc.frequency.setValueAtTime(f, t0 + dtOff);
          const g = ac.createGain();
          g.gain.setValueAtTime(0.26 * vol, t0 + dtOff);
          g.gain.exponentialRampToValueAtTime(0.001, t0 + dtOff + 0.05);
          osc.connect(g).connect(ac.destination);
          osc.start(t0 + dtOff); osc.stop(t0 + dtOff + 0.055);
        }
      } catch (_) {}
    }

    function playRevLoadSfx() {
      try {
        const vol = getSfxVol(); if (vol <= 0.005) return;
        const ac = getAudioCtx(); if (!ac) return;
        const t0 = ac.currentTime;
        for (const [dtOff, f] of [[0.0, 2400], [0.042, 3100], [0.085, 2850]]) {
          const osc = ac.createOscillator();
          osc.type = "triangle"; osc.frequency.setValueAtTime(f, t0 + dtOff);
          const g = ac.createGain();
          g.gain.setValueAtTime(0.28 * vol, t0 + dtOff);
          g.gain.exponentialRampToValueAtTime(0.001, t0 + dtOff + 0.055);
          osc.connect(g).connect(ac.destination);
          osc.start(t0 + dtOff); osc.stop(t0 + dtOff + 0.06);
        }
      } catch (_) {}
    }

    function playRevCloseSfx() {
      try {
        const vol = getSfxVol(); if (vol <= 0.005) return;
        const ac = getAudioCtx(); if (!ac) return;
        const t0 = ac.currentTime;
        for (const [dtOff, f] of [[0.0, 3200], [0.025, 2600], [0.052, 4100]]) {
          const osc = ac.createOscillator();
          osc.type = "square"; osc.frequency.setValueAtTime(f, t0 + dtOff);
          const g = ac.createGain();
          g.gain.setValueAtTime(0.32 * vol, t0 + dtOff);
          g.gain.exponentialRampToValueAtTime(0.001, t0 + dtOff + 0.048);
          osc.connect(g).connect(ac.destination);
          osc.start(t0 + dtOff); osc.stop(t0 + dtOff + 0.055);
        }
      } catch (_) {}
    }

    function playAK47ShotSfx() {
      try {
        const vol = getSfxVol() * 0.50; if (vol <= 0.005) return;
        const ac = getAudioCtx(); if (!ac) return;
        const t0 = ac.currentTime;
        const pj = 0.96 + Math.random() * 0.08;
        const ns = ac.createBufferSource();
        ns.buffer = makeNoiseBuf(ac, 0.38);
        const bp = ac.createBiquadFilter();
        bp.type = "lowpass"; bp.frequency.setValueAtTime(3400, t0);
        bp.frequency.exponentialRampToValueAtTime(280, t0 + 0.34);
        const ng = ac.createGain();
        ng.gain.setValueAtTime(0.68 * vol, t0);
        ng.gain.exponentialRampToValueAtTime(0.001, t0 + 0.36);
        ns.connect(bp).connect(ng).connect(ac.destination);
        ns.start(t0);
        const th = ac.createOscillator();
        th.type = "triangle"; th.frequency.setValueAtTime(145 * pj, t0);
        th.frequency.exponentialRampToValueAtTime(44, t0 + 0.14);
        const tg = ac.createGain();
        tg.gain.setValueAtTime(0.76 * vol, t0);
        tg.gain.exponentialRampToValueAtTime(0.001, t0 + 0.16);
        th.connect(tg).connect(ac.destination);
        th.start(t0); th.stop(t0 + 0.17);
        const cl = ac.createOscillator();
        cl.type = "square"; cl.frequency.setValueAtTime(2750, t0 + 0.016);
        const cg = ac.createGain();
        cg.gain.setValueAtTime(0.16 * vol, t0 + 0.016);
        cg.gain.exponentialRampToValueAtTime(0.001, t0 + 0.048);
        cl.connect(cg).connect(ac.destination);
        cl.start(t0 + 0.016); cl.stop(t0 + 0.052);
      } catch (_) {}
    }

    function playM4A1ShotSfx() {
      try {
        const vol = getSfxVol() * 0.48; if (vol <= 0.005) return;
        const ac = getAudioCtx(); if (!ac) return;
        const t0 = ac.currentTime;
        const pj = 0.96 + Math.random() * 0.08;
        const ns = ac.createBufferSource();
        ns.buffer = makeNoiseBuf(ac, 0.28);
        const hp = ac.createBiquadFilter();
        hp.type = "bandpass"; hp.frequency.setValueAtTime(1650 * pj, t0); hp.Q.value = 0.7;
        const ng = ac.createGain();
        ng.gain.setValueAtTime(0.62 * vol, t0);
        ng.gain.exponentialRampToValueAtTime(0.001, t0 + 0.26);
        ns.connect(hp).connect(ng).connect(ac.destination);
        ns.start(t0);
        const p = ac.createOscillator();
        p.type = "sine"; p.frequency.setValueAtTime(210 * pj, t0);
        p.frequency.exponentialRampToValueAtTime(68, t0 + 0.09);
        const pg = ac.createGain();
        pg.gain.setValueAtTime(0.62 * vol, t0);
        pg.gain.exponentialRampToValueAtTime(0.001, t0 + 0.11);
        p.connect(pg).connect(ac.destination);
        p.start(t0); p.stop(t0 + 0.12);
        const sp = ac.createOscillator();
        sp.type = "sine"; sp.frequency.setValueAtTime(1580, t0);
        const spg = ac.createGain();
        spg.gain.setValueAtTime(0.07 * vol, t0);
        spg.gain.exponentialRampToValueAtTime(0.001, t0 + 0.08);
        sp.connect(spg).connect(ac.destination);
        sp.start(t0); sp.stop(t0 + 0.085);
      } catch (_) {}
    }

    const WPN_WAV_MAP = {
      fire:               { file: "wpn_fire_m1911.wav",    gain: 0.94, jitter: 0.035 },
      fire_glock:         { file: "wpn_fire_glock.wav",    gain: 0.90, jitter: 0.040 },
      fire_shotgun:       { file: "wpn_fire_shotgun.wav",  gain: 1.00, jitter: 0.030 },
      fire_double_barrel: { file: "wpn_fire_db.wav",       gain: 1.00, jitter: 0.030 },
      fire_revolver:      { file: "wpn_fire_revolver.wav", gain: 0.54, jitter: 0.030 },
      fire_ak47:          { file: "wpn_fire_ak47.wav",     gain: 0.48, jitter: 0.040 },
      fire_m4a1:          { file: "wpn_fire_m4a1.wav",         gain: 0.46, jitter: 0.040 },
      fire_mp7:           { file: "wpn_fire_mp7.wav",          gain: 0.90, jitter: 0.045 },
      fire_m110:          { file: "wpn_fire_m110.wav",         gain: 0.98, jitter: 0.028 },
      bolt_cycle:         { file: "wpn_bolt_cycle.wav",        gain: 0.88, jitter: 0.025 },
      menu_car_pass:      { file: "menu_car_pass.wav",          gain: 0.34, jitter: 0.012 },
      menu_car_pass_near: { file: "menu_car_pass_near.wav",     gain: 0.34, jitter: 0.012 },
      fire_mp5_silenced:  { file: "wpn_fire_mp5_silenced.wav", gain: 0.36, jitter: 0.045 },
      fire_mp5_loud:      { file: "wpn_fire_mp5_loud.wav",     gain: 0.94, jitter: 0.040 },
      supp_detach:        { file: "wpn_supp_detach.wav",       gain: 0.86, jitter: 0.020 },
      supp_attach:        { file: "wpn_supp_attach.wav",       gain: 0.88, jitter: 0.020 },
      door_open:          { file: "door_open.wav",             gain: 0.88, jitter: 0.025 },
      door_close:         { file: "door_close.wav",            gain: 0.90, jitter: 0.025 },
      door_quiet:         { file: "door_quiet.wav",            gain: 0.42, jitter: 0.040 },
      door_kick:          { file: "door_kick.wav",             gain: 0.98, jitter: 0.020 },
      npc_greet:          { file: "npc_greet.wav",             gain: 0.72, pitch: 0.78, jitter: 0.025 },
      hero_text: { file:'hero_text.mp3', gain:.83, jitter:0 },
      waking_breath:{file:'motel_breath_take.wav',gain:1,jitter:0},window_report:{file:'window_report.wav',gain:1,jitter:0},
      flat_chime: { file:'flat_chime.wav', gain:.86, jitter:0 },
      spouse_toriel:       { file: "voice_toriel.mp3", gain:.66, pitch:1.2*2**(-11/12), jitter:0 },
      npc_voice_0:        { file: "npc_voice_0.wav",           gain: 0.56, pitch: 0.78, jitter: 0.032 },
      npc_voice_1:        { file: "npc_voice_1.wav",           gain: 0.56, pitch: 0.78, jitter: 0.032 },
      npc_voice_2:        { file: "npc_voice_2.wav",           gain: 0.56, pitch: 0.78, jitter: 0.032 },
      npc_voice_3:        { file: "npc_voice_3.wav",           gain: 0.56, pitch: 0.78, jitter: 0.032 },
      npc_voice_4:        { file: "npc_voice_4.wav",           gain: 0.56, pitch: 0.78, jitter: 0.032 },
      npc_voice_5:        { file: "npc_voice_5.wav",           gain: 0.56, pitch: 0.78, jitter: 0.032 },
      npc_voice_6:        { file: "npc_voice_6.wav",           gain: 0.56, pitch: 0.78, jitter: 0.032 },
      npc_voice_7:        { file: "npc_voice_7.wav",           gain: 0.56, pitch: 0.78, jitter: 0.032 },
      mag_out:            { file: "wpn_mag_out.wav",       gain: 0.78, jitter: 0.020 },
      mag_in:             { file: "wpn_mag_in.wav",        gain: 0.86, jitter: 0.020 },
      mag_drop:           { file: "wpn_mag_drop.wav",      gain: 0.70, jitter: 0.030 },
      slide_back:         { file: "wpn_slide_back.wav",    gain: 0.84, jitter: 0.020 },
      slide_fwd:          { file: "wpn_slide_fwd.wav",     gain: 0.88, jitter: 0.020 },
      pump_back:          { file: "wpn_pump_back.wav",     gain: 0.88, jitter: 0.020 },
      pump_fwd:           { file: "wpn_pump_fwd.wav",      gain: 0.92, jitter: 0.020 },
      shell_insert:       { file: "wpn_shell_insert.wav",  gain: 0.84, jitter: 0.025 },
      db_open:            { file: "wpn_db_open.wav",       gain: 0.88, jitter: 0.020 },
      db_close:           { file: "wpn_db_close.wav",      gain: 0.94, jitter: 0.020 },
      shell_eject:        { file: "wpn_shell_eject.wav",   gain: 0.82, jitter: 0.025 },
      rev_cock:           { file: "wpn_rev_cock.wav",      gain: 0.88, jitter: 0.020 },
      rev_open:           { file: "wpn_rev_open.wav",      gain: 0.82, jitter: 0.020 },
      rev_load:           { file: "wpn_rev_load.wav",      gain: 0.84, jitter: 0.025 },
      rev_close:          { file: "wpn_rev_close.wav",     gain: 0.92, jitter: 0.020 },
      dry_fire:           { file: "wpn_dry_fire.wav",      gain: 0.74, jitter: 0.020 },
      draw:               { file: "wpn_draw.wav",          gain: 0.70, jitter: 0.020 },
      inspect:            { file: "wpn_inspect.wav",       gain: 0.74, jitter: 0.025 },
      inspect_turn:       { file: "wpn_inspect_turn.wav",  gain: 0.78, jitter: 0.025 },
    };
    const wpnWavBufs = {};
    let wpnWavsLoading = false;
    let pendingMenuCarPass = null;

    function ensureWeaponWavsLoaded() {
      if (wpnWavsLoading) return;
      const ac = getAudioCtx();
      if (!ac) return;
      wpnWavsLoading = true;
      for (const [evKey, meta] of Object.entries(WPN_WAV_MAP).sort(([a],[b])=>(b==='waking_breath')-(a==='waking_breath'))) {
        fetch(`/assets/sfx/${meta.file}`)
          .then((r) => (r.ok ? r.arrayBuffer() : null))
          .then((ab) => (ab ? ac.decodeAudioData(ab) : null))
          .then((buf) => {
            if (buf) {
              // The button contains one 50-ms syllable, followed by silence, at low amplitude.
              if(evKey==='hero_text')for(let ch=0;ch<buf.numberOfChannels;ch++){const data=buf.getChannelData(ch);for(let i=0;i<data.length;i++)data[i]*=7.0;}
              wpnWavBufs[evKey] = buf;
              if (evKey === pendingMenuCarPass && evKey.startsWith("menu_car_pass")) {
                const pending = pendingMenuCarPass;
                pendingMenuCarPass = null;
                playWeaponWavIfReady(pending);
              }
            }
          })
          .catch(() => {});
      }
    }

    function playWeaponWavIfReady(ev) {
      const meta = WPN_WAV_MAP[ev];
      if (!meta) return false;
      ensureWeaponWavsLoaded();
      const buf = wpnWavBufs[ev];
      if (!buf) return false;
      try {
        const vol = getSfxVol();
        if (vol <= 0.005) return true;
        const ac = getAudioCtx();
        if (!ac) return false;
        const src = ac.createBufferSource();
        src.buffer = buf;
        const jit = meta.jitter || 0.0;
        if (jit > 0.0) src.playbackRate.value = 1.0 + (Math.random() * 2.0 - 1.0) * jit;
        const g = ac.createGain();
        g.gain.value = Math.min(1.0, vol * meta.gain);
        src.connect(g).connect(ac.destination);
        src.start(ac.currentTime);
        return true;
      } catch (_) {
        return false;
      }
    }

    function playWeaponEvent(ev, remote=false) {
      if(!remote&&mode==='playing'&&mp.connected&&!/^(fire|door_|menu_|impact)/.test(ev)&&Object.hasOwn(WPN_WAV_MAP,ev))mp.sendJson({type:'sfx',event:ev});
      if (ev.startsWith("menu_car_pass") && !wpnWavBufs[ev]) {
        pendingMenuCarPass = ev;
        ensureWeaponWavsLoaded();
        return;
      }
      if (playWeaponWavIfReady(ev)) return;
      if (ev === "fire") playGunshotSfx();
      else if (ev === "fire_glock" || ev === "fire_mp7") playGlockShotSfx();
      else if (ev === "fire_shotgun") playShotgunShotSfx();
      else if (ev === "fire_double_barrel") playDoubleBarrelShotSfx();
      else if (ev === "fire_ak47") playAK47ShotSfx();
      else if (ev === "fire_m4a1") playM4A1ShotSfx();
      else if (ev === "db_open") playDbOpenSfx();
      else if (ev === "db_close") playDbCloseSfx();
      else if (ev === "shell_eject") playShellEjectSfx();
      else if (ev === "rev_cock") playRevCockSfx();
      else if (ev === "fire_revolver") playRevolverShotSfx();
      else if (ev === "rev_open") playRevOpenSfx();
      else if (ev === "rev_load") playRevLoadSfx();
      else if (ev === "rev_close") playRevCloseSfx();
      else if (ev === "pump_back") playPumpBackSfx();
      else if (ev === "pump_fwd") playPumpFwdSfx();
      else if (ev === "shell_insert") playShellInsertSfx();
      else if (ev === "dry_fire") playDryFireSfx();
      else if (ev === "impact") playBulletImpactSfx();
      else if (ev === "mag_out") playMagOutSfx();
      else if (ev === "mag_drop") playMagDropSfx();
      else if (ev === "mag_in") playMagInSfx();
      else if (ev === "slide_back" || ev === "inspect_turn") playSlideBackSfx();
      else if (ev === "slide_fwd") playSlideForwardSfx();
      else if (ev === "draw" || ev === "inspect") playWeaponDrawSfx();
    }

    player.kick.onStart = () => { playKickWhoosh(); };
    player.kick.onHit = () => {
      playKickThud();if(mp.connected)mp.sendJson({type:'sfx',event:'step_kick'});
      hitMarker.classList.add("show");
      if (hitHideTimer) clearTimeout(hitHideTimer);
      hitHideTimer = setTimeout(() => hitMarker.classList.remove("show"), 150);
    };
    player.onSlideStart = () => { playSlideSfx();if(mp.connected)mp.sendJson({type:'sfx',event:'slide'}); };
    player.onLand = (spd) => { playLandSfx(spd);if(mp.connected&&spd>2)mp.sendJson({type:'sfx',event:'land'}); };
    player.onJump = () => { playJumpSfx();if(mp.connected)mp.sendJson({type:'sfx',event:'jump'}); };
    player.onFootstep = (stepMode) => { playFootstepSfx(stepMode);if(mode==='playing'&&mp.connected)mp.sendJson({type:'sfx',event:'step_'+stepMode}); };

    let wasPointerLocked = false;
    let chatOpen = false;
    document.addEventListener("pointerlockchange", () => {
      const locked = document.pointerLockElement === canvas;
      if (locked) {
        wasPointerLocked = true;
      } else if (wasPointerLocked && mode === "playing" && !chatOpen) {
        wasPointerLocked = false;
        enterPause();
      }
    });

    // ---------------- In-Game Chat (U key, Singleplayer + Multiplayer) ----------------
    const CHAT_FADE_SEC = 11.0;
    const chatMessages = [];
    let chatRenderDirty = false;
    let chatVisibleCount = 0;

    function pushChatMessage(nick, text, isSystem = false) {
      const txt = String(text || "").trim().slice(0, 140);
      if (!txt) return;
      chatMessages.push({
        nick: String(nick || "").trim().slice(0, 20),
        text: txt,
        isSystem: Boolean(isSystem),
        ts: performance.now() * 0.001,
      });
      if (chatMessages.length > 40) chatMessages.splice(0, chatMessages.length - 40);
      chatRenderDirty = true;
      renderChatFeed(true);
    }

    function renderChatFeed(force = false) {
      const nowSec = performance.now() * 0.001;
      const inGame = mode === "playing" || mode === "paused";
      if (!inGame) {
        chatBoxEl.style.display = "none";
        return;
      }
      const visible = chatOpen
        ? chatMessages.slice(-14)
        : chatMessages.filter((m) => (nowSec - m.ts) <= CHAT_FADE_SEC).slice(-7);
      if (!force && !chatRenderDirty && visible.length === chatVisibleCount) return;
      chatRenderDirty = false;
      chatVisibleCount = visible.length;
      chatBoxEl.style.display = (chatOpen || visible.length > 0) ? "flex" : "none";
      chatBoxEl.classList.toggle("chat-open", chatOpen);
      chatFeedEl.classList.toggle("empty", visible.length === 0);
      chatFeedEl.innerHTML = "";
      for (const m of visible) {
        const div = document.createElement("div");
        div.className = "chat-line" + (m.isSystem ? " sys" : "");
        if (m.isSystem || !m.nick) {
          div.textContent = m.text;
        } else {
          const sp = document.createElement("span");
          sp.className = "cnick";
          sp.textContent = m.nick + ": ";
          div.appendChild(sp);
          div.appendChild(document.createTextNode(m.text));
        }
        chatFeedEl.appendChild(div);
      }
      chatFeedEl.scrollTop = chatFeedEl.scrollHeight;
    }

    function openChat() {
      if (mode !== "playing" || chatOpen) return;
      chatOpen = true;
      wasPointerLocked = false;
      keysDown.clear();
      adsDown = false;
      lmbHeld = false;
      dragging = false;
      if (document.pointerLockElement && document.exitPointerLock) {
        try { document.exitPointerLock(); } catch (_) {}
      }
      chatInputEl.value = "";
      chatRenderDirty = true;
      renderChatFeed(true);
      setTimeout(() => {
        if (chatOpen) chatInputEl.focus();
      }, 15);
    }

    function closeChat(reLock = true) {
      if (!chatOpen) return;
      chatOpen = false;
      chatInputEl.blur();
      chatRenderDirty = true;
      renderChatFeed(true);
      if (reLock && mode === "playing") {
        canvas.focus();
        if (canvas.requestPointerLock) {
          try {
            const res = canvas.requestPointerLock();
            if (res && typeof res.catch === "function") res.catch(() => {});
          } catch (_) {}
        }
      }
    }

    function submitChatInput() {
      const text = (chatInputEl.value || "").trim().slice(0, 140);
      chatInputEl.value = "";
      if (text) {
        const cmdNorm = text.replace(/^["'`]+|["'`]+$/g, "").trim().toLowerCase().replace(/\s+/g, " ");
        if (cmdNorm === "sudo noclip on") {
          player.noclip = true;
          player.vel = [0, 0, 0];
          hudTimer = 99;
          pushChatMessage("", "NOCLIP ВКЛЮЧЁН — свободный полёт сквозь стены (WASD + Space вверх / Ctrl вниз / Shift быстро)", true);
        } else if (cmdNorm === "sudo noclip off") {
          player.noclip = false;
          player.vel = [0, 0, 0];
          hudTimer = 99;
          pushChatMessage("", "NOCLIP ВЫКЛЮЧЕН — гравитация и коллизии восстановлены", true);
        } else if (mp.connected) {
          mp.sendChat(text);
        } else {
          const localNick = (mpNickInput.value || "").trim() || "Игрок";
          pushChatMessage(localNick, text, false);
        }
      }
      closeChat(true);
    }

    chatInputEl.addEventListener("keydown", (e) => {
      e.stopPropagation();
      if (e.code === "Enter" || e.code === "NumpadEnter") {
        e.preventDefault();
        submitChatInput();
      } else if (e.code === "Escape") {
        e.preventDefault();
        closeChat(true);
      }
    });
    chatSendBtn.addEventListener("click", (e) => {
      e.stopPropagation();
      submitChatInput();
    });
    if (btnChat) {
      btnChat.addEventListener("click", (e) => {
        e.stopPropagation();
        if (mode === "playing") openChat();
      });
    }

    // ---------------- Render.com Multiplayer Client (WebSocket + Room Browser) ----------------
    const DEFAULT_MP_SERVER = window.location.origin; // Same default as the desktop client.
    const MP_PREFS_KEY = "psx_sandbox_mp_v1";
    const MP_DAMAGE = {
      m1911: 28, glock17: 18, revolver: 55,
      remington11: 52, double_barrel: 68, ak47: 34, m4a1: 26, mp7: 22, m110: 88, mp5s: 24,
    };

    function normalizeMpServerUrls(raw) {
      let base = String(raw || "").trim().replace(/\/+$/, "");
      if (!base) base = DEFAULT_MP_SERVER;
      if (!/^(https?|wss?):\/\//i.test(base)) {
        const isLocal = /^(localhost|127\.0\.0\.1|0\.0\.0\.0|192\.168\.|10\.)/i.test(base);
        base = (isLocal ? "http://" : "https://") + base;
      }
      let httpBase = base;
      if (/^wss:\/\//i.test(httpBase)) httpBase = "https://" + httpBase.slice(6);
      else if (/^ws:\/\//i.test(httpBase)) httpBase = "http://" + httpBase.slice(5);
      httpBase = httpBase.replace(/\/ws$/i, "").replace(/\/+$/, "");
      const wsUrl = (httpBase.startsWith("https://") ? "wss://" + httpBase.slice(8) : "ws://" + httpBase.slice(7)) + "/ws";
      return { httpBase, wsUrl };
    }

    function sanitizeRemoteAppearance(raw) {
      // Local and remote avatars use the same catalog keys and bounds.
      const ap = {};
      for (const key of Object.keys(DEFAULT_CHARACTER)) {
        const value = raw?.[key];
        ap[key] = (key === "gender" ? value === "female" || value === "male" : Number.isFinite(value))
          ? value : DEFAULT_CHARACTER[key];
      }
      return clampChar(ap);
    }

    function netItem(it){return {...it,vel:[0,0,0],avel:[0,0,0],grounded:true,hasBounced:true};}
    function spawnOnline(sp){player.reset();const [x,y,z,a]=sp;if([x,y,z,a].every(Number.isFinite)&&!player.collidesAny(player.getAABB(x,y,z),scene.colliders))player.pos=[x,y,z];player.smoothFootY=player.pos[1];player.yaw=player.targetYaw=Number.isFinite(a)?a:0;fireQueued=reloadQueued=kickQueued=slideQueued=pickupQueued=dropQueued=lmbHeld=adsDown=false;keysDown.clear();}
    const mp = {
      ws: null,
      connected: false,
      connecting: false,
      myId: "",
      roomId: "",
      roomName: "",
      hp: 100,
      rooms: [],
      selectedRoomId: "arena-1",
      createMode: false,
      remotes: new Map(),
      stateAccum: 0,
      pingAccum: 0,

      loadPrefs() {
        try {
          const saved = JSON.parse(window.localStorage.getItem(MP_PREFS_KEY) || "{}");
          if (saved.nick) mpNickInput.value = String(saved.nick).slice(0, 20);
          if (saved.serverUrl) mpServerUrlInput.value = String(saved.serverUrl).slice(0, 120);
        } catch (_) {}
        if (!mpServerUrlInput.value.trim()) {
          mpServerUrlInput.value = DEFAULT_MP_SERVER;
        }
      },

      savePrefs() {
        try {
          window.localStorage.setItem(MP_PREFS_KEY, JSON.stringify({
            nick: (mpNickInput.value || "").trim().slice(0, 20) || "Игрок",
            serverUrl: (mpServerUrlInput.value || "").trim().slice(0, 120),
          }));
        } catch (_) {}
      },

      setStatus(msg, isErr = false) {
        mpStatusEl.textContent = msg || "";
        mpStatusEl.classList.toggle("err", Boolean(isErr));
      },

      renderRoomList() {
        mpRoomListEl.innerHTML = "";
        if (!this.rooms.length) {
          const empty = document.createElement("div");
          empty.style.cssText = "padding:10px;font-size:12px;color:#93a694;";
          empty.textContent = "Нет списка серверов. Укажите URL Render.com и нажмите «Обновить».";
          mpRoomListEl.appendChild(empty);
          return;
        }
        for (const r of this.rooms) {
          const btn = document.createElement("button");
          btn.type = "button";
          btn.className = "mp-room-row" + (!this.createMode && r.id === this.selectedRoomId ? " selected" : "");
          const lock = r.hasPassword ? "🔒 " : "";
          const nameSpan = document.createElement("span");
          nameSpan.textContent = lock + (r.name || r.id);
          const metaSpan = document.createElement("span");
          metaSpan.className = "mp-room-meta";
          metaSpan.textContent = `${r.players || 0}/${r.maxPlayers || 16}`;
          btn.appendChild(nameSpan);
          btn.appendChild(metaSpan);
          btn.addEventListener("click", () => {
            this.createMode = false;
            mpNewRoomInput.style.display = "none";
            this.selectedRoomId = r.id;
            this.renderRoomList();
          });
          mpRoomListEl.appendChild(btn);
        }
      },

      async fetchRooms() {
        this.savePrefs();
        const { httpBase } = normalizeMpServerUrls(mpServerUrlInput.value);
        this.setStatus("Загрузка списка серверов...", false);
        try {
          const resp = await fetch(`${httpBase}/api/rooms`, { cache: "no-store" });
          if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
          const data = await resp.json();
          this.rooms = Array.isArray(data.rooms) ? data.rooms : [];
          if (this.rooms.length && !this.rooms.some((r) => r.id === this.selectedRoomId)) {
            this.selectedRoomId = this.rooms[0].id;
          }
          this.renderRoomList();
          this.setStatus(`Найдено серверов: ${this.rooms.length}`, false);
        } catch (err) {
          this.rooms = [];
          this.renderRoomList();
          this.setStatus("Сервер недоступен. Проверьте URL Render.com (бесплатный сервер может просыпаться 20-40с).", true);
        }
      },

      upsertRemote(pdata) {
        const pid = String(pdata.id || "");
        if (!pid || pid === this.myId) return;
        let r = this.remotes.get(pid);
        const st = pdata.state || pdata;
        if (!r) {
          r = {
            id: pid,
            nick: String(pdata.nick || "Игрок").slice(0, 20),
            appearance: sanitizeRemoteAppearance(pdata.appearance),
            hp: Number.isFinite(pdata.hp) ? pdata.hp : 100,
            pos: [st.x || 0, st.y || 0, st.z || 0],
            tPos: [st.x || 0, st.y || 0, st.z || 0],
            yaw: st.yaw || 0,
            tYaw: st.yaw || 0,
            pitch: st.pitch || 0,
            tPitch: st.pitch || 0,
            speed: st.speed || 0,
            crouch: Boolean(st.crouch),
            slide: Boolean(st.slide),
            grounded: st.grounded !== false,
            kickT: st.kickT || 0,
            weapon: String(st.weapon || "m1911"),
            equipped: Boolean(st.equipped),
            walkPhase: 0,
            muzzleTimer: 0,
            recoilPitch: 0,
            bustPhys: new BustPhysics(),
            lastBust: null,
          };
          this.remotes.set(pid, r);
        } else {
          if (pdata.nick) r.nick = String(pdata.nick).slice(0, 20);
          if (pdata.appearance) r.appearance = sanitizeRemoteAppearance(pdata.appearance);
          if (Number.isFinite(pdata.hp)) r.hp = pdata.hp;
          if (Number.isFinite(st.x)) {
            r.tPos = [st.x || 0, st.y || 0, st.z || 0];
            r.tYaw = st.yaw || 0;
            r.tPitch = st.pitch || 0;
            r.speed = st.speed || 0;
            r.crouch = Boolean(st.crouch);
            r.slide = Boolean(st.slide);
            r.grounded = st.grounded !== false;
            r.kickT = st.kickT || 0;
            r.weapon = String(st.weapon || "m1911");
            r.equipped = Boolean(st.equipped);
          }
        }
        if(r.life!==st.life){r.pos=[...r.tPos];r.yaw=r.tYaw;r.pitch=r.tPitch;r.recoilPitch=0;r.muzzleTimer=0;}r.life=st.life;
        r.netPhase=st.walkPhase;r.velY=st.velY||0;r.reload=st.reload;r.reloadU=st.reloadU;r.reloadType=st.reloadType;r.ads=st.ads;r.ins=st.inspect||st.magCheck||0;r.flash=st.flash;r.suppressed=st.suppressed;r.slideKick=st.slideKick||0;
      },

      async connectAndPlay() {
        if (this.connecting) return;
        this.savePrefs();
        const nick = (mpNickInput.value || "").trim().slice(0, 20) || "Игрок";
        const password = mpPasswordInput.value || "";
        const { httpBase, wsUrl } = normalizeMpServerUrls(mpServerUrlInput.value);
        let roomId = this.selectedRoomId || "arena-1";

        this.connecting = true;
        this.setStatus("Подключение к серверу...", false);

        try {
          if (this.createMode) {
            const newName = (mpNewRoomInput.value || "").trim().slice(0, 28) || "Моя комната";
            const resp = await fetch(`${httpBase}/api/rooms`, {
              method: "POST",
              headers: { "Content-Type": "application/json" },
              body: JSON.stringify({ name: newName, password }),
            });
            const data = await resp.json();
            if (!resp.ok || !data.room) {
              throw new Error(data.error || `HTTP ${resp.status}`);
            }
            roomId = data.room.id;
            this.selectedRoomId = roomId;
            this.createMode = false;
            mpNewRoomInput.style.display = "none";
          }
        } catch (err) {
          this.connecting = false;
          this.setStatus(`Не удалось создать комнату: ${err.message || err}`, true);
          return;
        }

        this.disconnect(false);
        this.connecting = true;
        let ws;
        try {
          ws = new WebSocket(wsUrl);
        } catch (err) {
          this.connecting = false;
          this.setStatus("Неверный адрес сервера", true);
          return;
        }
        this.ws = ws;

        ws.addEventListener("open", () => {
          ws.send(JSON.stringify({
            type: "join",
            room: roomId,
            password,
            nick,
            appearance: { ...character },
          }));
        });

        ws.addEventListener("message", (ev) => {
          let msg;
          try { msg = JSON.parse(ev.data); } catch (_) { return; }
          if(this.ws!==ws)return;const mtype = msg.type;
          if (mtype === "welcome") {
            this.connecting = false;
            this.connected = true;
            this.myId = String(msg.id || "");
            this.roomId = String((msg.room && msg.room.id) || roomId);
            this.roomName = String((msg.room && msg.room.name) || roomId);
            this.hp = 100;
            this.remotes.clear();
            for (const p of (msg.players || [])) this.upsertRemote(p);
            chatMessages.length = 0;
            for (const h of (msg.history || [])) {
              pushChatMessage(h.nick || "", h.text || "", Boolean(h.system));
            }
            pushChatMessage("", `Вы вошли на сервер: ${this.roomName} (U — чат)`, true);
            this.setStatus("", false);
            mpCard.classList.add("hidden");
            startNewMatch(true);
            this.life=msg.life||0;spawnOnline(msg.spawn);this.lastDoor=msg.door||0;doorCtrl.targetAngle=doorCtrl.angle=this.lastDoor;doorCtrl.mode='open';doorCtrl.angVel=0;doorCtrl.eHolding=false;doorCtrl.sfxQueue=[];
            if(Array.isArray(msg.items))weapon.worldPickups=msg.items.map(netItem);
          } else if(mtype==='door'){
            this.lastDoor=msg.target;doorCtrl.targetAngle=msg.target;doorCtrl.mode=msg.target>doorCtrl.angle?'opening':'closing';if(msg.sound)playWeaponEvent(msg.sound,true);
          } else if(mtype==='item'){
            weapon.worldPickups=weapon.worldPickups.filter(w=>w.weaponId!==msg.weapon);if(msg.item)weapon.worldPickups.push(netItem(msg.item));if(msg.owner===this.myId&&msg.taken)weapon.tryPickup(player.pos,0,netItem(msg.taken));updateAmmoHud(true);updateQuickBarButtons();
          } else if(mtype==='sfx'){
            if(msg.event==='jump')playJumpSfx();else if(msg.event==='land')playLandSfx(4);else if(msg.event==='slide')playSlideSfx();else if(msg.event==='step_kick')playKickThud();else if(msg.event?.startsWith('step_'))playFootstepSfx(msg.event.slice(5));else if(Object.hasOwn(WPN_WAV_MAP,msg.event))playWeaponEvent(msg.event,true);
          } else if(mtype==='hit_confirm'){
            hitMarker.classList.add('show');clearTimeout(hitHideTimer);hitHideTimer=setTimeout(()=>hitMarker.classList.remove('show'),160);
          } else if (mtype === "error") {
            this.connecting = false;
            this.connected = false;
            this.setStatus(msg.message || "Ошибка подключения", true);
            try { ws.close(); } catch (_) {}
          } else if (mtype === "player_join") {
            if (msg.player) {
              this.upsertRemote(msg.player);
              pushChatMessage("", `${msg.player.nick || "Игрок"} подключился`, true);
            }
          } else if (mtype === "player_leave") {
            const pid = String(msg.id || "");
            const old = this.remotes.get(pid);
            this.remotes.delete(pid);
            pushChatMessage("", `${msg.nick || (old && old.nick) || "Игрок"} вышел`, true);
          } else if (mtype === "player_appearance") {
            const r = this.remotes.get(String(msg.id || ""));
            if (r && msg.appearance) r.appearance = sanitizeRemoteAppearance(msg.appearance);
          } else if (mtype === "snapshot") {
            const alive = new Set();
            for (const p of (msg.players || [])) {
              const pid = String(p.id || "");
              if (!pid || pid === this.myId) continue;
              alive.add(pid);
              this.upsertRemote(p);
            }
            for (const pid of Array.from(this.remotes.keys())) {
              if (!alive.has(pid)) this.remotes.delete(pid);
            }
          } else if (mtype === "chat") {
            pushChatMessage(msg.nick || "", msg.text || "", Boolean(msg.system));
          } else if (mtype === "shoot") {
            const r = this.remotes.get(String(msg.id || ""));
            const wid = String(msg.weapon || (r && r.weapon) || "m1911");
            if (r) {
              r.muzzleTimer = 0.085;
              r.recoilPitch = 0.28;
            }
            const fireMap = {
              m1911: "fire", glock17: "fire_glock", revolver: "fire_revolver",
              remington11: "fire_shotgun", double_barrel: "fire_double_barrel",
              ak47: "fire_ak47", m4a1: "fire_m4a1", mp7: "fire_mp7",
            };
            playWeaponEvent(wid==='mp5s'?(msg.suppressed?'fire_mp5_silenced':'fire_mp5_loud'):wid==='m110'?'fire_m110':fireMap[wid]||'fire',true);
            if(msg.end&&msg.dir){const at=msg.pos.map((v,i)=>v+msg.dir[i]*.45);for(const ray of msg.rays||[{end:msg.end,impact:msg.impact}]){weapon._spawnTracer(at,ray.end,[1,.82,.47],.009,.11);if(ray.impact){const n=ray.normal||msg.dir.map(v=>-v);weapon._spawnImpactFx(ray.end,...n,false);weapon.bulletHoles.push({pos:ray.end.map((v,i)=>v+n[i]*.018),roll:0,life:6.5,maxLife:6.5});if(weapon.bulletHoles.length>48)weapon.bulletHoles.shift();}}}
          } else if (mtype === "damaged") {
            this.hp = Math.max(0, Math.round(Number(msg.hp) || 0));
            pushChatMessage("", `По вам попал ${msg.fromNick || "Игрок"} (-${Math.round(Number(msg.damage) || 0)} HP)`, true);
          } else if (mtype === "respawn") {
            this.hp = Math.round(Number(msg.hp) || 100);
            this.life=msg.life||0;spawnOnline(msg.spawn);this.stateAccum=.05;
            pushChatMessage("", "Возрождение!", true);
          }
        });

        ws.addEventListener("close", () => {
          if(this.ws!==ws)return;const wasConnected = this.connected;
          this.connecting = false;
          this.connected = false;
          this.remotes.clear();
          mpNameplatesEl.innerHTML = "";
          if (wasConnected && (mode === "playing" || mode === "paused")) {
            pushChatMessage("", "Соединение с сервером потеряно", true);
          } else if (!mpCard.classList.contains("hidden") && !mpStatusEl.textContent) {
            this.setStatus("Не удалось подключиться к серверу", true);
          }
        });

        ws.addEventListener("error", () => {
          this.connecting = false;
          if (!this.connected) {
            this.setStatus("Ошибка подключения к серверу Render.com", true);
          }
        });
      },

      disconnect(silent = true) {
        this.connecting = false;
        this.connected = false;
        this.remotes.clear();
        mpNameplatesEl.innerHTML = "";
        if (this.ws) {
          try { this.ws.close(); } catch (_) {}
          this.ws = null;
        }
        if (!silent) this.setStatus("", false);
      },

      sendJson(obj) {
        if (this.ws && this.ws.readyState === WebSocket.OPEN) {
          try { this.ws.send(JSON.stringify(obj)); } catch (_) {}
        }
      },

      sendChat(text) {
        if (!this.connected) return;
        this.sendJson({ type: "chat", text: String(text || "").slice(0, 140) });
      },

      sendAppearance() {
        if (!this.connected) return;
        this.sendJson({ type: "appearance", appearance: { ...character } });
      },

      sendShootAndHitscan(eyePos, aimDir) {
        if (!this.connected) return;
        const wid = weapon.activeWeapon || "m1911";
        this.sendJson({
          type: "shoot",life:this.life,suppressed:weapon.mp5Suppressed,rays:weapon.netRays,
          weapon: wid,
          pos: [Number(eyePos[0].toFixed(3)), Number(eyePos[1].toFixed(3)), Number(eyePos[2].toFixed(3))],
          dir: [Number(aimDir[0].toFixed(3)), Number(aimDir[1].toFixed(3)), Number(aimDir[2].toFixed(3))],
        });
      },

      update(dt, elapsed) {
        if (!this.connected) return;
        const alpha = 1-Math.exp(-dt*19);
        for (const r of this.remotes.values()) {
          const dx = r.tPos[0] - r.pos[0];
          const dy = r.tPos[1] - r.pos[1];
          const dz = r.tPos[2] - r.pos[2];
          if (dx * dx + dy * dy + dz * dz > 64.0) {
            r.pos[0] = r.tPos[0]; r.pos[1] = r.tPos[1]; r.pos[2] = r.tPos[2];
          } else {
            r.pos[0] += dx * alpha;
            r.pos[1] += dy * alpha;
            r.pos[2] += dz * alpha;
          }
          let dYaw = (r.tYaw - r.yaw + Math.PI) % (Math.PI * 2) - Math.PI;
          if (dYaw < -Math.PI) dYaw += Math.PI * 2;
          r.yaw += dYaw * alpha;
          r.pitch += (r.tPitch - r.pitch) * alpha;
          if(Number.isFinite(r.netPhase))r.walkPhase+=(r.netPhase-r.walkPhase)*alpha;else if (r.speed > 0.25 && r.grounded && !r.slide) {
            r.walkPhase += dt * (4.8 + Math.min(r.speed, 9.0) * 0.65);
          }
          if (r.muzzleTimer > 0) r.muzzleTimer = Math.max(0, r.muzzleTimer - dt);
          if (r.recoilPitch > 0) r.recoilPitch = Math.max(0, r.recoilPitch - dt * 1.8);
          if (r.appearance.gender === "female") {
            const rTopId = charCatalog.topStyles[(r.appearance.top_style || 0) % charCatalog.topStyles.length].id;
            r.lastBust = r.bustPhys.update(dt, elapsed, Math.PI - r.yaw, {
              horizSpeed: r.speed,
              walkPhase: r.walkPhase,
              velY: 0,
              onGround: r.grounded,
              crouching: r.crouch,
              sliding: r.slide,
              kickPhase: r.kickT,
              topId: rTopId,
              bustSize: (typeof r.appearance.bust_size === "number") ? r.appearance.bust_size : 0.5,
            });
          } else {
            r.lastBust = null;
          }
        }

        this.stateAccum += dt;
        if (this.stateAccum >= 0.05) {
          this.stateAccum %= .05;
          const horizSpd = mode==='paused'?0:Math.hypot(player.vel[0], player.vel[2]);
          const kickProg = player.kick.active ? (player.kick.timer / Math.max(0.01, player.kick.duration)) : 0;
          this.sendJson({
            type: "state",
            state: {
              life:this.life,walkPhase:player.walkPhase,velY:player.vel[1],flash:player.flashlightOn,reload:weapon.reloading,reloadU:weapon.reloading?weapon._reloadProgress():0,reloadType:weapon.reloadType,inspect:weapon.inspecting?Math.sin(Math.PI*clamp01(weapon.inspectTimer/weapon.inspectDuration)):0,magCheck:weapon.magCheckBlend,suppressed:weapon.mp5Suppressed,slideKick:weapon.slideKick,ammo:weapon.magAmmo,reserve:weapon.reserveAmmo,
              x: Number(player.pos[0].toFixed(3)),
              y: Number(player.pos[1].toFixed(3)),
              z: Number(player.pos[2].toFixed(3)),
              yaw: Number(player.yaw.toFixed(3)),
              pitch: Number(player.pitch.toFixed(3)),
              speed: Number(horizSpd.toFixed(2)),
              crouch: Boolean(player.crouching),
              slide: Boolean(player.sliding),
              grounded: Boolean(player.onGround),
              kickT: Number(kickProg.toFixed(2)),
              weapon: weapon.activeWeapon || "m1911",
              equipped: Boolean(weapon.hasWeapon && weapon.equipped),
              ads: Boolean(adsDown),
            },
          });
        }
        this.pingAccum += dt;
        if (this.pingAccum >= 10.0) {
          this.pingAccum = 0;
          this.sendJson({ type: "ping", t: performance.now() });
        }
      },

      renderNameplates(viewMat, projMat) {
        if (!this.connected || mode !== "playing" || !showHud || !this.remotes.size) {
          if (mpNameplatesEl.childElementCount > 0) mpNameplatesEl.innerHTML = "";
          return;
        }
        const vp = mat4Mul(projMat, viewMat);
        const w = window.innerWidth;
        const h = window.innerHeight;
        let html = "";
        for (const r of this.remotes.values()) {
          const wx = r.pos[0];
          const wy = r.pos[1] + (r.crouch ? 1.52 : 2.02);
          const wz = r.pos[2];
          const cx = vp[0] * wx + vp[1] * wy + vp[2] * wz + vp[3];
          const cy = vp[4] * wx + vp[5] * wy + vp[6] * wz + vp[7];
          const cw = vp[12] * wx + vp[13] * wy + vp[14] * wz + vp[15];
          if (cw <= 0.15) continue;
          const nx = cx / cw;
          const ny = cy / cw;
          if (nx < -1.05 || nx > 1.05 || ny < -1.05 || ny > 1.05) continue;
          const sx = (nx * 0.5 + 0.5) * w;
          const sy = (1.0 - (ny * 0.5 + 0.5)) * h;
          const hpPct = Math.max(0, Math.min(100, Math.round(r.hp)));
          const safeNick = r.nick.replace(/[&<>"']/g, (ch) => (
            ch === "&" ? "&amp;" : ch === "<" ? "&lt;" : ch === ">" ? "&gt;" : ch === '"' ? "&quot;" : "&#39;"
          ));
          html += `<div class="mp-plate" style="left:${sx.toFixed(1)}px;top:${sy.toFixed(1)}px;">` +
            `<div class="mp-plate-nick">${safeNick}</div>` +
            `<div class="mp-plate-hp"><div style="width:${hpPct}%"></div></div></div>`;
        }
        mpNameplatesEl.innerHTML = html;
      },
    };

    mp.loadPrefs();
    function openMpCard() {
      closeSettings();
      mpCard.classList.remove("hidden");
      mp.createMode = false;
      mpNewRoomInput.style.display = "none";
      mp.fetchRooms();
    }
    function closeMpCard() {
      mpCard.classList.add("hidden");
    }
    document.getElementById("btn-menu-mp").addEventListener("click", () => {
      playUiClickSfx();
      openMpCard();
    });
    document.getElementById("btn-mp-close").addEventListener("click", () => {
      playUiClickSfx();
      closeMpCard();
    });
    document.getElementById("btn-mp-back").addEventListener("click", () => {
      playUiClickSfx();
      closeMpCard();
    });
    document.getElementById("btn-mp-refresh").addEventListener("click", () => {
      playUiClickSfx();
      mp.fetchRooms();
    });
    document.getElementById("btn-mp-new-toggle").addEventListener("click", () => {
      playUiClickSfx();
      mp.createMode = !mp.createMode;
      mpNewRoomInput.style.display = mp.createMode ? "block" : "none";
      mp.renderRoomList();
      if (mp.createMode) mpNewRoomInput.focus();
    });
    document.getElementById("btn-mp-play").addEventListener("click", () => {
      playUiClickSfx();
      mp.connectAndPlay();
    });
    window.addEventListener("beforeunload", () => mp.disconnect(true));

    let lastAmmoKey = "";
    function updateAmmoHud(force = false) {
      if (!ammoHudEl) return;
      const ownedList = weapon.ownedWeapons();
      const key = `${weapon.activeWeapon}|${ownedList.join(",")}|${weapon.hasWeapon}|${weapon.equipped}|${weapon.magAmmo}|${weapon.reserveAmmo}|${weapon.reloading}|${weapon.reloadType}|${weapon.shellReloadPhase}|${weapon.dbReloadPhase}|${weapon.pumpActive}|${weapon.revCocking}|${weapon.slideLocked}|${weapon.inspecting}|${weapon.mp5Suppressed}|${weapon.suppAnimActive}`;
      if (!force && key === lastAmmoKey) return;
      lastAmmoKey = key;

      const kPk = formatKeyCode(userSettings.binds.pickup);
      const kRl = formatKeyCode(userSettings.binds.reload);
      const kHl = formatKeyCode(userSettings.binds.holster);
      const kIn = formatKeyCode(userSettings.binds.inspect);

      if (!weapon.hasWeapon) {
        ammoTitle.textContent = "M1911 · GLOCK · REM11 · .357 · АК · M4 · MP7 · M110 · MP5-S · ";
        ammoMag.textContent = "—";
        ammoRes.textContent = "—";
        ammoMag.classList.remove("empty");
        ammoPips.innerHTML = "";
        for (let i = 0; i < 7; i++) {
          const pip = document.createElement("span");
          pip.className = "bullet-pip";
          ammoPips.appendChild(pip);
        }
        ammoState.className = "ammo-state warn";
        ammoState.textContent = `НАЙДИ ОРУЖИЕ НА АРЕНЕ · [${kPk}]`;
        return;
      }

      const spec = weapon.activeSpec;
      const isSg = weapon.isShotgun;
      const isDb = weapon.isDoubleBarrel;
      const isRev = weapon.isRevolver;
      const isLg = isSg || isDb;
      const slotLbl = (spec.slot === 10) ? "0" : String(spec.slot);
      const modeTag = isDb ? "BREAK-12GA"
        : (isSg ? "BUCKSHOT"
        : (isRev ? "SINGLE-ACT"
        : (weapon.isSniper ? "OPTIC BOLT"
        : (weapon.isMP5S ? (weapon.mp5Suppressed ? "ГЛУШ:ВКЛ [P]" : "ГЛУШ:СНЯТ [P]")
        : (spec.full_auto ? "AUTO" : "SEMI")))));
      const slotHint = ownedList.length > 1 ? ` · [1..0/Q]` : ` [${slotLbl}]`;
      ammoTitle.textContent = `${spec.name} · ${spec.caliber} · ${modeTag}${slotHint}`;
      ammoMag.textContent = String(weapon.magAmmo);
      ammoRes.textContent = String(weapon.reserveAmmo);
      ammoMag.classList.toggle("empty", weapon.magAmmo === 0);

      ammoPips.innerHTML = "";
      const isRifleCap = weapon.MAG_CAPACITY > 18;
      const pipTotal = isRifleCap ? Math.ceil(weapon.MAG_CAPACITY / 2) : weapon.MAG_CAPACITY;
      const pipFilled = isRifleCap ? (weapon.magAmmo > 0 ? Math.ceil(weapon.magAmmo / 2) : 0) : weapon.magAmmo;
      const styleCls = isLg ? " shell" : (weapon.MAG_CAPACITY > 10 ? " compact" : "");
      for (let i = 0; i < pipTotal; i++) {
        const pip = document.createElement("span");
        pip.className = "bullet-pip" + styleCls + (i < pipFilled ? " full" : "");
        ammoPips.appendChild(pip);
      }

      ammoState.className = "ammo-state";
      if (!weapon.equipped) {
        ammoState.textContent = `УБРАНО [${kHl}..7/Q]`;
      } else if (weapon.inspecting) {
        ammoState.textContent = `ОСМОТР ОРУЖИЯ (УДЕРЖ. [${kIn}])`;
      } else if (weapon.revCocking) {
        ammoState.classList.add("warn");
        ammoState.textContent = "ВЗВЕДЕНИЕ КУРКА (.357)...";
      } else if (weapon.pumpActive) {
        ammoState.classList.add("warn");
        ammoState.textContent = "ПЕРЕДЁРГИВАНИЕ ЗАТВОРА...";
      } else if (weapon.reloading) {
        ammoState.classList.add("warn");
        ammoState.textContent = weapon.reloadType === "break_action"
          ? "ПЕРЕЛОМ СТВОЛОВ — ГИЛЬЗЫ И ЗАРЯДКА 12GA..."
          : (weapon.reloadType === "revolver"
            ? "ПЕРЕЗАРЯДКА БАРАБАНА (.357 MAG)..."
            : ((weapon.reloadType === "shell" && weapon.shellReloadPhase === "chamber")
              ? "ЗАРЯДКА 1-ГО ПАТРОНА В ПАТРОННИК..."
              : (weapon.reloadType === "shell"
                ? "ДОЗАРЯДКА МАГАЗИНА 12GA (ЛКМ — ПРЕРВАТЬ)..."
                : (weapon.reloadType === "empty" ? "ПОЛНАЯ ПЕРЕЗАРЯДКА..." : "ТАКТИЧЕСКАЯ ПЕРЕЗАРЯДКА..."))));
      } else if (weapon.magAmmo === 0 && weapon.reserveAmmo > 0) {
        ammoState.classList.add("warn");
        ammoState.textContent = isDb
          ? `СТВОЛЫ ПУСТЫ · [${kRl}] ПЕРЕЛОМИТЬ`
          : (isRev
            ? `БАРАБАН ПУСТ · [${kRl}] ПЕРЕЗАРЯДИТЬ`
            : (isSg
              ? `ЗАТВОР ОТКРЫТ · [${kRl}] В ПАТРОННИК`
              : (weapon.isAK47 ? `МАГАЗИН ПУСТ · [${kRl}] РОЖОК + ЗАТВОР` : `ЗАТВОР НА ЗАДЕРЖКЕ · [${kRl}]`)));
      } else if (weapon.magAmmo === 0 && weapon.reserveAmmo === 0) {
        ammoState.classList.add("warn");
        ammoState.textContent = "ПУСТО · BACKSPACE РЕСПАВН";
      } else if (isDb) {
        ammoState.textContent = `ДВУСТВОЛКА 12GA (10 ДРОБИН) · [${kIn}] ОСМОТР`;
      } else if (isRev) {
        ammoState.textContent = `КУРКОВЫЙ .357 · ПКМ ПРИЦЕЛ · [${kIn}] ОСМОТР`;
      } else if (isSg) {
        ammoState.textContent = `КАРТЕЧЬ 12GA (8 ДРОБИН) · [${kIn}] ОСМОТР`;
      } else if (spec.full_auto) {
        ammoState.textContent = `АВТО (${modeTag}) · ЛКМ · ПКМ · [${kIn}] ОСМОТР`;
      } else if (weapon.magAmmo < weapon.MAG_CAPACITY && weapon.reserveAmmo > 0) {
        ammoState.textContent = `[${kRl}] ТАКТИЧ. · ПКМ ПРИЦЕЛ`;
      } else {
        ammoState.textContent = "ОДИНОЧНЫЙ · ЛКМ / ПКМ ПРИЦЕЛ";
      }
    }

    let lastNearPickup = false;
    let lastPickupAmmoKey = "";
    function updatePickupPrompt() {
      const inGame = mode === "playing" || mode === "paused";
      const nearWp = inGame ? weapon.nearestPickup(player.pos) : null;
      const near = nearWp !== null;
      if (pickupPromptEl) {
        pickupPromptEl.style.display = (near && showHud) ? "block" : "none";
        if (near && nearWp) {
          const kPk = formatKeyCode(userSettings.binds.pickup);
          const pSpec = weapon.SPECS[nearWp.weaponId || "m1911"] || weapon.SPECS.m1911;
          const full = weapon.ownedWeapons().length >= 2;
          const key = `${kPk}|${nearWp.weaponId}|${nearWp.magAmmo}|${nearWp.reserveAmmo}|${full}`;
          if (key !== lastPickupAmmoKey) {
            lastPickupAmmoKey = key;
            pickupPromptEl.innerHTML = full
              ? `<span class="pkey">ИНВЕНТАРЬ 2/2</span>СЛОТЫ ЗАНЯТЫ · СНАЧАЛА БРОСЬТЕ ОРУЖИЕ`
              : `<span class="pkey">${kPk}</span>ПОДОБРАТЬ ${pSpec.name} (${pSpec.caliber} · ${nearWp.magAmmo}/${nearWp.reserveAmmo})`;
          }
        }
      }
      if (btnPickup) {
        btnPickup.style.display = near ? "inline-block" : "none";
        btnPickup.classList.toggle("pickup-ready", near);
        if (near && nearWp) {
          const kPk = formatKeyCode(userSettings.binds.pickup);
          const pSpec = weapon.SPECS[nearWp.weaponId || "m1911"] || weapon.SPECS.m1911;
          const full = weapon.ownedWeapons().length >= 2;
          btnPickup.disabled = full;
          btnPickup.textContent = full ? "СЛОТЫ ЗАНЯТЫ · 2/2" : `${kPk}: ${pSpec.name}`;
        }
      }
      lastNearPickup = near;
    }

    function updateQuickBarButtons() {
      const kHl = formatKeyCode(userSettings.binds.holster);
      const kRl = formatKeyCode(userSettings.binds.reload);
      const kDr = formatKeyCode(userSettings.binds.drop);
      document.getElementById("btn-flash").classList.toggle("active", player.flashlightOn);
      btnCamMode.classList.toggle("active", thirdPerson);
      btnCamMode.textContent = thirdPerson ? "F5: 3-Е ЛИЦО" : "F5: 1-Е ЛИЦО";
      if (btnDrop) {
        btnDrop.style.display = weapon.hasWeapon ? "inline-block" : "none";
        btnDrop.textContent = `${kDr}: БРОСИТЬ`;
      }
      if (btnReload) {
        btnReload.style.display = weapon.hasWeapon ? "inline-block" : "none";
        btnReload.textContent = `${kRl}: ПЕРЕЗАРЯДКА`;
      }
      if (btnInspect) {
        const kInBtn = formatKeyCode(userSettings.binds.inspect);
        btnInspect.style.display = (weapon.hasWeapon && weapon.equipped) ? "inline-block" : "none";
        btnInspect.textContent = `${kInBtn}: ОСМОТР`;
      }
      if (btnChat) {
        const kChBtn = formatKeyCode(userSettings.binds.chat);
        btnChat.textContent = `${kChBtn}: ЧАТ`;
      }
      if (btnHolster) {
        const owned = weapon.ownedWeapons();
        btnHolster.style.display = owned.length > 0 ? "inline-block" : "none";
        btnHolster.classList.toggle("active", weapon.hasWeapon && weapon.equipped);
        const wShortMap = { m1911: "M1911", glock17: "GLOCK-17", remington11: "REM-11", double_barrel: "ДВУСТВ.", revolver: ".357", ak47: "АК-47", m4a1: "M4A1", mp7: "MP7", m110: "M110", mp5s: "MP5-S" };
        const wShort = wShortMap[weapon.activeWeapon] || "M1911";
        btnHolster.textContent = weapon.equipped
          ? (owned.length > 1 ? `Q: ${wShort}` : `${kHl}: ${wShort}`)
          : `${kHl}: КОБУРА`;
      }
      updatePickupPrompt();
    }

    function syncUiVisibility() {
      const inCreator = mode === "creator";
      const inMenu = mode === "menu";
      const inPause = mode === "paused";
      const inGame = mode === "playing" || mode === "paused";
      creatorPanel.classList.toggle("hidden", !inCreator);
      creatorStageBar.classList.toggle("hidden", !inCreator);
      mainMenuEl.classList.toggle("hidden", !inMenu);
      if (!inMenu) mpCard.classList.add("hidden");
      overlay.classList.toggle("hidden", !inPause);
      syncMenuMusic(inMenu || inCreator);
      hudEl.style.display = (inGame && showHud) ? "block" : "none";
      if (ammoHudEl) ammoHudEl.style.display = (inGame && showHud) ? "block" : "none";
      quickBar.style.display = (inGame && showHud) ? "flex" : "none";
      if (!inGame) {
        chatOpen = false;
        player.noclip = false;
        chatBoxEl.style.display = "none";
        mpNameplatesEl.innerHTML = "";
      } else {
        renderChatFeed(true);
      }
      updatePickupPrompt();
      canvas.style.cursor = inCreator ? "grab" : (inMenu ? "default" : "crosshair");
    }

    function enterCreator(isFirstLaunch = false) {
      mode = "creator";
      dragging = false;
      creatorDragging = false;
      keysDown.clear();
      closeSettings();
      if (document.pointerLockElement && document.exitPointerLock) {
        try { document.exitPointerLock(); } catch (_) {}
      }
      creatorHeading.textContent = isFirstLaunch ? "СОЗДАНИЕ ПЕРСОНАЖА" : "РЕДАКТОР ПЕРСОНАЖА";
      creatorSubtitle.textContent = isFirstLaunch
        ? "Первый запуск — выберите пол, причёску, одежду и стиль героя"
        : "Измените внешность вашего 3D-героя (видно в 3-м лице по F5)";
      creatorTargetYawOffset = -0.15;
      creatorZoomTarget = 0.0;
      btnCamBody.classList.add("active");
      btnCamFace.classList.remove("active");
      syncCreatorUi();
      syncUiVisibility();
      if (isFirstLaunch && entryFadeEl) {
        entryFadeEl.classList.add("fade-black-on");
        void entryFadeEl.offsetWidth;
        requestAnimationFrame(() => {
          requestAnimationFrame(() => {
            entryFadeEl.classList.remove("fade-black-on");
          });
        });
      }
    }

    // Input handling (uses e.code so Russian & English layouts both work seamlessly!)
    const keysDown = new Set();
    let kickQueued = false;
    let slideQueued = false;
    let fireQueued = false;
    let lmbHeld = false;
    let rHoldActive = false;
    let rHoldTimer = 0.0;
    let rMagCheckFired = false;
    const doorPromptEl = document.getElementById("door-prompt");
    const doorPromptMainEl = document.getElementById("door-prompt-main");
    const doorPromptSubEl = document.getElementById("door-prompt-sub");
    const doorQuietBarWrapEl = document.getElementById("door-quiet-bar-wrap");
    const doorQuietBarEl = document.getElementById("door-quiet-bar");
    const magCheckBadgeEl = document.getElementById("magcheck-badge");
    let reloadQueued = false;
    let pickupQueued = false;
    let dropQueued = false;
    let adsDown = false;
    let frameMouseDx = 0.0;
    let frameMouseDy = 0.0;
    let dragMoved = false;

    let menuReadyTimeout = null;
    let menuFadeRaf = null;

    function triggerMenuEntranceAnim() {
      if (menuReadyTimeout) { clearTimeout(menuReadyTimeout); menuReadyTimeout = null; }
      if (menuFadeRaf) { cancelAnimationFrame(menuFadeRaf); menuFadeRaf = null; }
      mainMenuEl.classList.remove("menu-ready");
      mainMenuEl.classList.add("menu-pre-enter");
      if (entryFadeEl) entryFadeEl.classList.add("fade-black-on");
      void mainMenuEl.offsetWidth;
      if (entryFadeEl) void entryFadeEl.offsetWidth;
      menuFadeRaf = requestAnimationFrame(() => {
        menuFadeRaf = requestAnimationFrame(() => {
          if (entryFadeEl) entryFadeEl.classList.remove("fade-black-on");
          mainMenuEl.classList.remove("menu-pre-enter");
          menuReadyTimeout = setTimeout(() => {
            mainMenuEl.classList.add("menu-ready");
          }, 1450);
        });
      });
    }

    function enterMenu(playIntro = true) {
      mode = "menu";
      chatOpen = false;
      mp.disconnect(true);
      dragging = false;
      creatorDragging = false;
      adsDown = false;
      lmbHeld = false;
      keysDown.clear();
      closeSettings();
      if (document.pointerLockElement && document.exitPointerLock) {
        try { document.exitPointerLock(); } catch (_) {}
      }
      syncUiVisibility();
      if (typeof testNpc !== "undefined") testNpc.close(true);
      if (playIntro) triggerMenuEntranceAnim();
    }

    const entryFadeEl = document.getElementById("entry-fade");
    const npcPromptEl = document.getElementById("npc-prompt");
    const npcDlgEl = document.getElementById("npc-dialogue");
    const dlgSpeakerNameEl = document.getElementById("dlg-speaker-name");
    const dlgSpeakerRoleEl = document.getElementById("dlg-speaker-role");
    const dlgTextEl = document.getElementById("dlg-text");
    const dlgTextSpanEl = document.getElementById("dlg-text-span");
    const dlgCounterEl = document.getElementById("dlg-counter");
    const dlgHintEl = document.getElementById("dlg-hint");
    const dlgPanelEl = document.getElementById("dlg-panel");

    // Interactive Test NPC ("Стас") standing on the sandbox map with animated PS1 dialogue box!
    const testNpcCfg = scene.testNpc || null;
    const testNpc = {
      enabled: Boolean(testNpcCfg),
      name: testNpcCfg ? testNpcCfg.name : "СТАС",
      role: testNpcCfg ? testNpcCfg.role : "ДЕЖУРНЫЙ ПОЛИГОНА",
      pos: testNpcCfg ? testNpcCfg.pos : [-2.65, 0.0, 7.65],
      idleYaw: testNpcCfg ? testNpcCfg.idleYaw : 0.56,
      interactRadius: testNpcCfg ? testNpcCfg.interactRadius : 3.05,
      autoCloseRadius: testNpcCfg ? testNpcCfg.autoCloseRadius : 5.2,
      appearance: testNpcCfg ? clampChar(Object.assign({}, testNpcCfg.appearance)) : null,
      dialogues: (testNpcCfg && Array.isArray(testNpcCfg.dialogues) && testNpcCfg.dialogues.length)
        ? testNpcCfg.dialogues
        : [["Привет! Я тестовый NPC Стас."]],
      yaw: testNpcCfg ? testNpcCfg.idleYaw : 0.56,
      headLookYaw: 0.0,
      talkBlend: 0.0,
      nearPlayer: false,
      active: false,
      openTimer: 0.0,
      convoIdx: 0,
      lineIdx: 0,
      currentLines: [],
      visibleChars: 0.0,
      lastBlipChar: -1,
      mouthOpen: 0.0,
      mouthTimer: 0.0,
      _lastRenderedText: "",
      _lastPromptVis: false,

      get mouthSuffix() {
        if (this.mouthOpen >= 0.58) return "_talk2";
        if (this.mouthOpen >= 0.20) return "_talk1";
        return "";
      },

      canInteract(pPos) {
        if (!this.enabled) return false;
        const dx = pPos[0] - this.pos[0], dy = pPos[1] - this.pos[1], dz = pPos[2] - this.pos[2];
        return (dx*dx + dz*dz) <= (this.interactRadius * this.interactRadius) && Math.abs(dy) < 2.1;
      },
      _triggerLinePop() {
        if (!dlgTextEl) return;
        dlgTextEl.classList.add("line-pop");
        void dlgTextEl.offsetWidth;
        requestAnimationFrame(() => dlgTextEl.classList.remove("line-pop"));
      },
      _syncDom() {
        const showPrompt = this.enabled && mode === "playing" && !chatOpen && this.nearPlayer && !this.active;
        if (npcPromptEl && showPrompt !== this._lastPromptVis) {
          this._lastPromptVis = showPrompt;
          npcPromptEl.textContent = `[E] ПОГОВОРИТЬ — ${this.name} (${this.role})`;
          npcPromptEl.classList.toggle("visible", showPrompt);
        }
        if (!npcDlgEl) return;
        const shouldOpen = this.enabled && mode === "playing" && this.active;
        npcDlgEl.classList.toggle("open", shouldOpen);
        document.body.classList.toggle("dlg-open", shouldOpen);
        if (!shouldOpen) {
          npcDlgEl.classList.remove("typing");
          return;
        }
        const full = this.currentLines[this.lineIdx] || "";
        const n = Math.max(0, Math.min(full.length, Math.floor(this.visibleChars)));
        const shown = full.slice(0, n);
        const typing = n < full.length;
        npcDlgEl.classList.toggle("typing", typing);
        if (shown !== this._lastRenderedText) {
          this._lastRenderedText = shown;
          if (dlgTextSpanEl) dlgTextSpanEl.textContent = shown;
        }
        if (dlgCounterEl) dlgCounterEl.textContent = `${this.lineIdx + 1} / ${this.currentLines.length}`;
        if (dlgHintEl) {
          if (typing) dlgHintEl.textContent = "[E / SPACE / КЛИК] ПРОПУСТИТЬ";
          else if (this.lineIdx >= this.currentLines.length - 1) dlgHintEl.textContent = "■ [E / SPACE / КЛИК] ЗАКРЫТЬ";
          else dlgHintEl.textContent = "▼ [E / SPACE / КЛИК] ДАЛЕЕ";
        }
      },
      startOrAdvance() {
        if (!this.enabled) return false;
        if (!this.active) {
          this.active = true;
          this.openTimer = 0.0;
          this.currentLines = this.dialogues[this.convoIdx % this.dialogues.length].slice();
          this.lineIdx = 0;
          this.visibleChars = 0.0;
          this.lastBlipChar = -1;
          this._lastRenderedText = "\0";
          if (dlgSpeakerNameEl) dlgSpeakerNameEl.textContent = this.name;
          if (dlgSpeakerRoleEl) dlgSpeakerRoleEl.textContent = `· ${this.role}`;
          playDialogueEventSfx("dlg_open");
          this._triggerLinePop();
          this._syncDom();
          return true;
        }
        const full = this.currentLines[this.lineIdx] || "";
        if (Math.floor(this.visibleChars) < full.length) {
          this.visibleChars = full.length;
          this.lastBlipChar = full.length;
          this._syncDom();
          return true;
        }
        if (this.lineIdx + 1 < this.currentLines.length) {
          this.lineIdx += 1;
          this.visibleChars = 0.0;
          this.lastBlipChar = -1;
          this._lastRenderedText = "\0";
          playDialogueEventSfx("dlg_next");
          this._triggerLinePop();
          this._syncDom();
          return true;
        }
        this.convoIdx = (this.convoIdx + 1) % this.dialogues.length;
        this.close();
        return true;
      },
      close(silent = false) {
        if (this.active) {
          this.active = false;
          if (!silent) playDialogueEventSfx("dlg_close");
        }
        this._syncDom();
      },
      update(dt, pPos) {
        if (!this.enabled) return;
        const cdt = Math.min(Math.max(dt, 0.0), 0.05);
        const dx = pPos[0] - this.pos[0], dy = pPos[1] - this.pos[1], dz = pPos[2] - this.pos[2];
        const distXZ = Math.hypot(dx, dz);
        this.nearPlayer = (distXZ <= this.interactRadius && Math.abs(dy) < 2.1);
        if (this.active && (distXZ > this.autoCloseRadius || Math.abs(dy) > 3.2)) {
          this.close();
        }
        const tgtYaw = (this.active || distXZ < 5.5) ? Math.atan2(dx, dz) : this.idleYaw;
        let diff = ((tgtYaw - this.yaw + Math.PI) % (Math.PI * 2) + Math.PI * 2) % (Math.PI * 2) - Math.PI;
        const kTurn = 1.0 - Math.exp(-cdt * (this.active ? 6.2 : 2.8));
        this.yaw += diff * kTurn;
        this.headLookYaw = Math.max(-0.55, Math.min(0.55, diff * 0.65));

        const full = this.currentLines[this.lineIdx] || "";
        const isTyp = this.active && Math.floor(this.visibleChars) < full.length;
        const talkTgt = isTyp ? 1.0 : (this.active ? 0.35 : 0.0);
        this.talkBlend += (talkTgt - this.talkBlend) * (1.0 - Math.exp(-cdt * 7.5));

        // Lip-sync jaw opening driven by currently spoken character (vowels wide, consonants mid, pauses closed)
        this.mouthTimer += cdt;
        let mouthTgt = 0.0;
        if (this.active && this.openTimer >= 0.16 && isTyp && full.length > 0) {
          const ciM = Math.max(0, Math.min(full.length - 1, Math.floor(this.visibleChars)));
          const chM = full[ciM].toLowerCase();
          const osc = 0.5 + 0.5 * Math.sin(this.mouthTimer * 26.0);
          if ("аеёиоуыэюяaeiouy".includes(chM)) mouthTgt = 0.78 + 0.22 * osc;
          else if (" ,.!?;:—-…«»()\n".includes(chM)) mouthTgt = 0.0;
          else mouthTgt = 0.34 + 0.30 * osc;
        }
        const kMouth = 1.0 - Math.exp(-cdt * (mouthTgt > this.mouthOpen ? 24.0 : 16.0));
        this.mouthOpen += (mouthTgt - this.mouthOpen) * kMouth;
        if (this.mouthOpen < 0.02 && mouthTgt === 0.0) this.mouthOpen = 0.0;

        if (this.active) {
          this.openTimer += cdt;
          if (this.openTimer >= 0.14 && this.visibleChars < full.length) {
            let spd = 36.0;
            const pi = Math.floor(this.visibleChars);
            if (pi > 0 && pi <= full.length) {
              const pch = full[pi - 1];
              if (".!?…".includes(pch)) spd *= 0.22;
              else if (",;:—".includes(pch)) spd *= 0.45;
            }
            this.visibleChars = Math.min(full.length, this.visibleChars + spd * cdt);
            const ci = Math.floor(this.visibleChars);
            if (ci > this.lastBlipChar && ci > 0) {
              const ch = full[ci - 1];
              this.lastBlipChar = ci;
              if (!" ,.!?;:—-«»()\n".includes(ch)) {
                const lowC = ch.toLowerCase();
                let vIdx;
                if ("аоуэaou".includes(lowC)) vIdx = (ci % 2 === 0) ? 0 : 2;
                else if ("еёиыюяei".includes(lowC)) vIdx = (ci % 2 === 0) ? 1 : 4;
                else if ("бвгджзлмнр".includes(lowC)) vIdx = (ci % 2 === 0) ? 3 : 6;
                else vIdx = (ci % 2 === 0) ? 5 : 7;
                playDialogueEventSfx("dlg_voice_" + vIdx);
              }
            }
          }
        }
        this._syncDom();
      },
    };
    if (dlgPanelEl) {
      dlgPanelEl.addEventListener("mousedown", (e) => {
        e.stopPropagation();
        if (mode === "playing" && testNpc.active) testNpc.startOrAdvance();
      });
    }
    let entryFadeTimer = null;

    function triggerGameEntryAnim() {
      const hudNodes = [hudEl, ammoHudEl, quickBar, chatBoxEl].filter(Boolean);
      for (const el of hudNodes) el.classList.add("hud-pre-enter");
      if (entryFadeEl) entryFadeEl.classList.add("fade-black-on");
      // Force layout reflow so initial pre-enter positions & black overlay apply instantaneously
      if (entryFadeEl) void entryFadeEl.offsetWidth;
      if (entryFadeTimer) cancelAnimationFrame(entryFadeTimer);
      entryFadeTimer = requestAnimationFrame(() => {
        entryFadeTimer = requestAnimationFrame(() => {
          if (entryFadeEl) entryFadeEl.classList.remove("fade-black-on");
          for (const el of hudNodes) el.classList.remove("hud-pre-enter");
        });
      });
    }

    function enterPlay(playEntryAnim = false) {
      getAudioCtx();
      ensureWeaponWavsLoaded();
      const wasMenu = (mode === "menu" || mode === "creator");
      mode = "playing";
      closeSettings();
      mpCard.classList.add("hidden");
      updateQuickBarButtons();
      updateAmmoHud(true);
      syncUiVisibility();
      if (playEntryAnim || wasMenu) triggerGameEntryAnim();
      canvas.focus();
      if (canvas.requestPointerLock) {
        try {
          const res = canvas.requestPointerLock();
          if (res && typeof res.catch === "function") res.catch(() => {});
        } catch (_) {}
      }
    }

    function startNewMatch(isMp = false) {
      if (!isMp) mp.disconnect(true);
      player.reset();
      testNpc.close(true);
      weapon.spawnWorldPickup(scene.weaponSpawns);
      adsDown = false;
      lmbHeld = false;
      fireQueued = false;
      reloadQueued = false;
      pickupQueued = false;
      dropQueued = false;
      updateQuickBarButtons();
      updateAmmoHud(true);
      enterPlay(true);
    }

    function enterPause() {
      mode = "paused";
      chatOpen = false;
      testNpc.close(true);
      dragging = false;
      adsDown = false;
      lmbHeld = false;
      keysDown.clear();
      closeSettings();
      overlayTitle.textContent = mp.connected
        ? `ПАУЗА — ОНЛАЙН: ${mp.roomName}`
        : "ПАУЗА — PSX 3D SANDBOX";
      if (document.pointerLockElement && document.exitPointerLock) {
        try { document.exitPointerLock(); } catch (_) {}
      }
      syncUiVisibility();
    }

    window.addEventListener("keydown", (e) => {
      if(storyActive()){if(!el('case-panel').hidden){if(!e.repeat&&['Escape','KeyF','KeyE'].includes(e.code))showCase(false);e.preventDefault();return;}if(!e.repeat&&e.code==='KeyF'&&nearCase()){showCase(true);e.preventDefault();return;}if(e.code==='Escape'&&!e.repeat){leaveStory();e.preventDefault();return;} if(dreamT>=0){e.preventDefault();return;}if(hallT>=0){if(!e.repeat&&(e.code==='KeyE'||e.code==='Enter'))nextHall();e.preventDefault();return;}if(mode==='home'&&supperT<0){if(!e.repeat&&e.code==='KeyE'&&toggleInterior()){e.preventDefault();return;}if(!e.repeat&&useStreetDoor(e.code)){e.preventDefault();return;}if(e.code==='KeyE'&&!e.repeat&&atEntry()){e.preventDefault();startHall();return;}if(e.code==='KeyE'&&!e.repeat&&nearGuestSeat()){e.preventDefault();startSupper();return;}keysDown.add(e.code);}if(['Space','ArrowLeft','ArrowRight','ArrowUp','ArrowDown','Enter'].includes(e.code))e.preventDefault();return;}
      // Never hijack typing inside chat input or multiplayer inputs
      if (chatOpen) {
        if (e.code === "Escape") {
          e.preventDefault();
          closeChat(true);
        }
        return;
      }
      if (!mpCard.classList.contains("hidden")) {
        if (e.code === "Escape") {
          e.preventDefault();
          closeMpCard();
        }
        return;
      }
      // Intercept key press while rebinding controls in Settings!
      if (listeningBindAction) {
        e.preventDefault();
        e.stopPropagation();
        if (e.code === "Escape") {
          listeningBindAction = null;
          renderBindList();
          settingsStatus.textContent = "Переназначение отменено";
          return;
        }
        // Swap key if another action already used it
        const prevCode = userSettings.binds[listeningBindAction];
        for (const def of BIND_DEFS) {
          if (def.id !== listeningBindAction && userSettings.binds[def.id] === e.code) {
            userSettings.binds[def.id] = prevCode;
          }
        }
        userSettings.binds[listeningBindAction] = e.code;
        listeningBindAction = null;
        applySettingsToEngine();
        saveUserSettings();
        settingsStatus.textContent = "Клавиша сохранена";
        return;
      }

      if (["Space", "ArrowUp", "ArrowDown", "ArrowLeft", "ArrowRight", "F5"].includes(e.code)) {
        e.preventDefault();
      }

      if (e.code === "Escape") {
        if (!settingsCard.classList.contains("hidden")) {
          closeSettings();
          return;
        }
        if (mode === "playing" && testNpc.active) {
          testNpc.close();
          return;
        }
        if (mode === "creator") {
          saveCharacter();
          enterMenu();
          return;
        }
        if (mode === "playing") enterPause();
        else if (mode === "paused") enterPlay();
        return;
      }

      if (mode === "creator") {
        if (e.code === "ArrowLeft" || e.code === "KeyA") creatorTargetYawOffset -= 0.35;
        else if (e.code === "ArrowRight" || e.code === "KeyD") creatorTargetYawOffset += 0.35;
        else if (e.code === "Enter" || e.code === "NumpadEnter") {
          saveCharacter();
          if(lastSlot)enterMenu();else beginStory();
        }
        return;
      }

      if (mode !== "playing" && settingsCard.classList.contains("hidden") && (e.code === "Enter" || e.code === "NumpadEnter")) {
        e.preventDefault();
        if (mode === "menu" && e.target.closest?.('button,summary,a')) {e.target.click();return;}
        if (mode === "menu") {if(lastSlot)beginStory(lastSlot);else beginStory();}
        else enterPlay();
        return;
      }

      if (mode === "menu" || !settingsCard.classList.contains("hidden")) return;

      if (!e.repeat && mode === "playing" && !chatOpen && testNpc.active) {
        if (codeMatchesAction(e.code, "pickup") || e.code === "Space" || e.code === "Enter" || e.code === "NumpadEnter") {
          e.preventDefault();
          testNpc.startOrAdvance();
          return;
        }
      }

      if (!e.repeat) {
        if (e.code === "F5") {
          thirdPerson = !thirdPerson;
          updateQuickBarButtons();
        } else if (e.code === "BracketLeft") {
          preset.resolution_index = preset.resolution_index < 0
            ? scene.pixelResolutions.length - 1
            : Math.max(0, preset.resolution_index - 1);
          rebuildFbo();
        } else if (e.code === "BracketRight") {
          preset.resolution_index = preset.resolution_index < 0
            ? scene.pixelResolutions.length - 1
            : Math.min(scene.pixelResolutions.length - 1, preset.resolution_index + 1);
          rebuildFbo();
        } else if (e.code === "KeyB") {
          userSettings.ditherEnabled = !userSettings.ditherEnabled;
          applySettingsToEngine();
          saveUserSettings();
        } else if (e.code === "KeyJ") {
          preset.vertex_jitter = !preset.vertex_jitter;
        } else if (e.code === "KeyT") {
          preset.scanlines = !preset.scanlines;
          if (preset.scanlines && preset.scanline_strength <= 0.001) preset.scanline_strength = 0.14;
        } else if (codeMatchesAction(e.code, "flashlight")) {
          player.flashlightOn = !player.flashlightOn;
          playFlashlightSfx();
          updateQuickBarButtons();
        } else if (codeMatchesAction(e.code, "chat") && mode === "playing") {
          e.preventDefault();
          openChat();
        } else if (e.code === "KeyH") {
          showHud = !showHud;
          syncUiVisibility();
        } else if (codeMatchesAction(e.code, "pickup") && mode === "playing") {
          pickupQueued = true;
        } else if (codeMatchesAction(e.code, "drop") && mode === "playing") {
          dropQueued = true;
        } else if (codeMatchesAction(e.code, "reload") && mode === "playing") {
          if (!rHoldActive) {
            rHoldActive = true;
            rHoldTimer = 0.0;
            rMagCheckFired = false;
          }
        } else if (e.code === "KeyP" && mode === "playing") {
          for (const ev of weapon.toggleSuppressor()) playWeaponEvent(ev);
          updateAmmoHud(true);
        } else if (codeMatchesAction(e.code, "holster") && mode === "playing") {
          weapon.selectWeapon("m1911");
          for (const ev of weapon.popSfxEvents()) playWeaponEvent(ev);
          updateQuickBarButtons();
          updateAmmoHud(true);
        } else if (e.code === "Digit2" && mode === "playing") {
          weapon.selectWeapon("glock17");
          for (const ev of weapon.popSfxEvents()) playWeaponEvent(ev);
          updateQuickBarButtons();
          updateAmmoHud(true);
        } else if (e.code === "Digit3" && mode === "playing") {
          weapon.selectWeapon("remington11");
          for (const ev of weapon.popSfxEvents()) playWeaponEvent(ev);
          updateQuickBarButtons();
          updateAmmoHud(true);
        } else if (e.code === "Digit4" && mode === "playing") {
          weapon.selectWeapon("double_barrel");
          for (const ev of weapon.popSfxEvents()) playWeaponEvent(ev);
          updateQuickBarButtons();
          updateAmmoHud(true);
        } else if (e.code === "Digit5" && mode === "playing") {
          weapon.selectWeapon("revolver");
          for (const ev of weapon.popSfxEvents()) playWeaponEvent(ev);
          updateQuickBarButtons();
          updateAmmoHud(true);
        } else if (e.code === "Digit6" && mode === "playing") {
          weapon.selectWeapon("ak47");
          for (const ev of weapon.popSfxEvents()) playWeaponEvent(ev);
          updateQuickBarButtons();
          updateAmmoHud(true);
        } else if (e.code === "Digit7" && mode === "playing") {
          weapon.selectWeapon("m4a1");
          for (const ev of weapon.popSfxEvents()) playWeaponEvent(ev);
          updateQuickBarButtons();
          updateAmmoHud(true);
        } else if (e.code === "Digit8" && mode === "playing") {
          weapon.selectWeapon("mp7");
          for (const ev of weapon.popSfxEvents()) playWeaponEvent(ev);
          updateQuickBarButtons();
          updateAmmoHud(true);
        } else if (e.code === "Digit9" && mode === "playing") {
          weapon.selectWeapon("m110");
          for (const ev of weapon.popSfxEvents()) playWeaponEvent(ev);
          updateQuickBarButtons();
          updateAmmoHud(true);
        } else if (e.code === "Digit0" && mode === "playing") {
          weapon.selectWeapon("mp5s");
          for (const ev of weapon.popSfxEvents()) playWeaponEvent(ev);
          updateQuickBarButtons();
          updateAmmoHud(true);
        } else if (codeMatchesAction(e.code, "inspect") && mode === "playing") {
          for (const ev of weapon.startInspect()) playWeaponEvent(ev);
          updateAmmoHud(true);
        } else if (e.code === "KeyQ" && mode === "playing") {
          weapon.cycleWeapon(1);
          for (const ev of weapon.popSfxEvents()) playWeaponEvent(ev);
          updateQuickBarButtons();
          updateAmmoHud(true);
        } else if (e.code === "Backspace") {
          if(mp.connected)return;player.reset();
          weapon.spawnWorldPickup(scene.weaponSpawns);
          updateQuickBarButtons();
          updateAmmoHud(true);
        } else if (codeMatchesAction(e.code, "kick") && mode === "playing") {
          kickQueued = true;
        } else if (codeMatchesAction(e.code, "crouch") && mode === "playing") {
          slideQueued = true;
        }
      }
      keysDown.add(e.code);
    });

    window.addEventListener("keyup", (e) => {
      if (mode === "playing" && !chatOpen && codeMatchesAction(e.code, "reload") && rHoldActive) {
        if (!rMagCheckFired) reloadQueued = true;
        rHoldActive = false;
        rHoldTimer = 0.0;
        rMagCheckFired = false;
      }
      keysDown.delete(e.code);
    });
    window.addEventListener("blur", () => {
      keysDown.clear();
      adsDown = false;
      lmbHeld = false;
      dragging = false;
    });

    // Pointer Lock + drag fallback during gameplay; 360° drag in Character Creator; parallax in 3D menu!
    canvas.addEventListener("contextmenu", (e) => {
      if (mode === "playing" || mode === "creator") e.preventDefault();
    });
    canvas.addEventListener("mousedown", (e) => {
      if(storyActive()){if(mode==='home'&&dreamT<0&&hallT<0&&supperT<0){canvas.focus();dragging=true;lastMouseX=e.clientX;lastMouseY=e.clientY;try{canvas.requestPointerLock?.()?.catch?.(()=>{});}catch(_){}}return;}
      if (mode === "creator") {
        if (e.button === 0) {
          creatorDragging = true;
          lastMouseX = e.clientX;
          canvas.style.cursor = "grabbing";
        }
        return;
      }
      if (mode === "menu") return;
      if (mode === "paused") {
        enterPlay();
        return;
      }
      if (chatOpen) {
        closeChat(true);
        return;
      }
      canvas.focus();
      if (e.button === 2) {
        adsDown = true;
        return;
      }
      if (document.pointerLockElement === canvas) {
        if (e.button === 0) {
          if (testNpc.active) {
            testNpc.startOrAdvance();
          } else {
            fireQueued = true;
            lmbHeld = true;
          }
        }
        return;
      }
      if (e.button === 0 && testNpc.active) {
        testNpc.startOrAdvance();
        return;
      }
      if (canvas.requestPointerLock) {
        try {
          const p = canvas.requestPointerLock();
          if (p && typeof p.catch === "function") p.catch(() => {});
        } catch (_) {}
      }
      if (e.button === 0) {
        dragging = true;
        dragMoved = false;
        lmbHeld = true;
        if (weapon.isFullAuto && weapon.hasWeapon && weapon.equipped) fireQueued = true;
        lastMouseX = e.clientX;
        lastMouseY = e.clientY;
      }
    });
    canvas.addEventListener("wheel", (e) => {
      if (mode === "creator") {
        e.preventDefault();
        creatorZoomTarget = Math.max(0.0, Math.min(1.0, creatorZoomTarget + (e.deltaY < 0 ? 0.25 : -0.25)));
        btnCamBody.classList.toggle("active", creatorZoomTarget < 0.5);
        btnCamFace.classList.toggle("active", creatorZoomTarget >= 0.5);
        return;
      }
      if (mode === "playing" && !chatOpen && weapon.ownedWeapons().length > 0) {
        e.preventDefault();
        weapon.cycleWeapon(e.deltaY > 0 ? 1 : -1);
        for (const ev of weapon.popSfxEvents()) playWeaponEvent(ev);
        updateQuickBarButtons();
        updateAmmoHud(true);
      }
    }, { passive: false });
    window.addEventListener("mouseup", (e) => {
      if (e.button === 2) adsDown = false;
      if (e.button === 0) {
        if (dragging && mode === "playing" && !dragMoved && !weapon.isFullAuto) {
          fireQueued = true;
        }
        lmbHeld = false;
      }
      dragging = false;
      if (creatorDragging) {
        creatorDragging = false;
        if (mode === "creator") canvas.style.cursor = "grab";
      }
    });
    window.addEventListener("mousemove", (e) => {
      if (chatOpen) return;
      if (mode === "creator") {
        if (creatorDragging) {
          const dx = e.clientX - lastMouseX;
          lastMouseX = e.clientX;
          creatorTargetYawOffset += dx * 0.012;
        }
        return;
      }
      if (mode === "menu") {
        menuTargetX = Math.max(-1, Math.min(1, (e.clientX / Math.max(window.innerWidth, 1)) * 2 - 1));
        menuTargetY = Math.max(-1, Math.min(1, 1 - (e.clientY / Math.max(window.innerHeight, 1)) * 2));
        return;
      }
      if (mode !== "playing" && mode !== "home") return;
      if (document.pointerLockElement === canvas) {
        frameMouseDx += e.movementX;
        frameMouseDy += e.movementY;
        player.rotate(e.movementX, e.movementY);
      } else if (dragging) {
        const dx = e.clientX - lastMouseX;
        const dy = e.clientY - lastMouseY;
        if (Math.abs(dx) + Math.abs(dy) > 3) dragMoved = true;
        lastMouseX = e.clientX;
        lastMouseY = e.clientY;
        frameMouseDx += dx * 1.4;
        frameMouseDy += dy * 1.4;
        player.rotate(dx * 1.4, dy * 1.4);
      }
    });

    document.getElementById("btn-menu-play").addEventListener("click", () => {
      startNewMatch();
    });
    document.getElementById("btn-menu-creator").addEventListener("click", () => {
      enterCreator(false);
    });

    // Character Creator controls
    btnGenderMale.addEventListener("click", () => {
      playUiClickSfx();
      if (character.gender !== "male") {
        character.gender = "male";
        if ([1, 2, 4, 5, 6, 12, 13].includes(character.hair_style)) character.hair_style = 0;
        if (character.face_style === 3) character.face_style = 1;
        if (character.bottom_style === 3) character.bottom_style = 0;
        syncCreatorUi();
      }
    });
    btnGenderFemale.addEventListener("click", () => {
      playUiClickSfx();
      if (character.gender !== "female") {
        character.gender = "female";
        creatorBustPhys.impulse(-0.18, 0.14);
        if ([0, 7, 9, 10, 14].includes(character.hair_style)) character.hair_style = 1;
        if (character.face_style === 2) character.face_style = 3;
        syncCreatorUi();
      }
    });
    document.querySelectorAll(".step-btn").forEach((btn) => {
      btn.addEventListener("click", () => {
        playUiClickSfx();
        const key = btn.dataset.step;
        const dir = Number(btn.dataset.dir) || 1;
        character[key] = (character[key] || 0) + dir;
        syncCreatorUi();
      });
    });
    document.getElementById("btn-char-random").addEventListener("click", () => {
      playUiClickSfx();
      const rand = (n) => Math.floor(Math.random() * n);
      const g = Math.random() < 0.5 ? "male" : "female";
      character.gender = g;
      character.skin = rand(charCatalog.skinPalette.length);
      character.face_style = g === "female"
        ? [0, 1, 3, 5, 6, 7][rand(6)]
        : [0, 1, 2, 4, 5, 6, 7][rand(7)];
      character.hair_style = g === "female"
        ? [1, 2, 3, 4, 5, 6, 8, 12, 13, 17, 18, 19, 20][rand(13)]
        : [0, 3, 7, 8, 9, 10, 11, 14, 15, 16, 17, 18, 20][rand(13)];
      character.hair_color = rand(charCatalog.hairPalette.length);
      character.top_style = rand(charCatalog.topStyles.length);
      character.top_color = rand(charCatalog.topPalette.length);
      character.bottom_style = g === "female"
        ? rand(charCatalog.bottomStyles.length)
        : [0, 1, 2, 4, 5][rand(5)];
      character.bottom_color = rand(charCatalog.bottomPalette.length);
      character.shoe_style = rand(charCatalog.shoeStyles.length);
      character.shoe_color = rand(charCatalog.shoePalette.length);
      character.accessory = rand(charCatalog.accessoryStyles.length);
      character.accessory_color = rand(charCatalog.accessoryPalette.length);
      character.bust_size = g === "female" ? Math.round((0.15 + Math.random() * 0.75) * 20) / 20 : 0.5;
      if (g === "female") creatorBustPhys.impulse(-0.26, 0.20);
      syncCreatorUi();
    });
    const rngBustEl = document.getElementById("rng-bust-size");
    if (rngBustEl) {
      let prevBustVal = Number(rngBustEl.value) || 50;
      rngBustEl.addEventListener("input", () => {
        const v = Number(rngBustEl.value);
        const dv = Math.sign(v - prevBustVal) || 1;
        prevBustVal = v;
        character.bust_size = Math.max(0.0, Math.min(1.0, v / 100.0));
        creatorBustPhys.impulse(-0.30 * dv, 0.22);
        syncCreatorUi();
      });
    }
    document.getElementById("btn-rot-left").addEventListener("click", () => {
      creatorTargetYawOffset -= Math.PI * 0.25;
    });
    document.getElementById("btn-rot-right").addEventListener("click", () => {
      creatorTargetYawOffset += Math.PI * 0.25;
    });
    btnCamBody.addEventListener("click", () => {
      creatorZoomTarget = 0.0;
      btnCamBody.classList.add("active");
      btnCamFace.classList.remove("active");
    });
    btnCamFace.addEventListener("click", () => {
      creatorZoomTarget = 1.0;
      btnCamBody.classList.remove("active");
      btnCamFace.classList.add("active");
    });
    document.getElementById("btn-char-play").addEventListener("click", () => {
      saveCharacter();
      if(lastSlot)enterMenu();else beginStory();
    });
    document.getElementById("btn-char-menu").addEventListener("click", () => {
      saveCharacter();
      enterMenu();
    });
    btnCamMode.addEventListener("click", (e) => {
      e.stopPropagation();
      thirdPerson = !thirdPerson;
      updateQuickBarButtons();
      canvas.focus();
    });
    document.getElementById("btn-menu-settings").addEventListener("click", () => {
      if (settingsCard.classList.contains("hidden")) openSettings();
      else closeSettings();
    });
    document.getElementById("btn-pause-settings").addEventListener("click", () => {
      if (settingsCard.classList.contains("hidden")) openSettings();
      else closeSettings();
    });
    document.getElementById("btn-settings-close").addEventListener("click", closeSettings);
    tabBtnGeneral.addEventListener("click", () => setSettingsTab("general"));
    tabBtnControls.addEventListener("click", () => setSettingsTab("controls"));

    setDitherToggle.addEventListener("click", () => {
      userSettings.ditherEnabled = !userSettings.ditherEnabled;
      applySettingsToEngine();
      saveUserSettings();
    });
    setDitherLevels.addEventListener("input", () => {
      userSettings.colorLevels = Number(setDitherLevels.value);
      applySettingsToEngine();
      saveUserSettings();
    });
    setDitherStrength.addEventListener("input", () => {
      userSettings.ditherStrength = Number(setDitherStrength.value);
      applySettingsToEngine();
      saveUserSettings();
    });
    setFovSlider.addEventListener("input", () => {
      userSettings.fovDeg = Number(setFovSlider.value);
      applySettingsToEngine();
      saveUserSettings();
    });
    setSensSlider.addEventListener("input", () => {
      userSettings.sensMult = Number(setSensSlider.value);
      applySettingsToEngine();
      saveUserSettings();
    });
    setSfxSlider.addEventListener("input", () => {
      userSettings.sfxVolume = Number(setSfxSlider.value);
      applySettingsToEngine();
      saveUserSettings();
    });
    setSfxSlider.addEventListener("change", () => {
      playUiClickSfx();
    });
    document.getElementById("btn-settings-reset").addEventListener("click", () => {
      listeningBindAction = null;
      userSettings.ditherEnabled = Boolean(scene.preset.dither_enabled);
      userSettings.colorLevels = Math.round(scene.preset.color_levels);
      userSettings.ditherStrength = Math.round(scene.preset.dither_strength * 100);
      userSettings.fovDeg = Math.round(scene.playerConfig.fov_deg);
      userSettings.sensMult = 1.0;
      userSettings.sfxVolume = 85;
      userSettings.binds = { ...DEFAULT_BINDS };
      applySettingsToEngine();
      saveUserSettings();
      playUiClickSfx();
      settingsStatus.textContent = "Сброшено по умолчанию";
    });

    document.getElementById("btn-play").addEventListener("click", enterPlay);
    document.getElementById("btn-respawn").addEventListener("click", () => {
      startNewMatch();
    });
    document.getElementById("btn-back-menu").addEventListener("click", enterMenu);
    if (btnPickup) {
      btnPickup.addEventListener("click", (e) => {
        e.stopPropagation();
        if (mode !== "playing") enterPlay();
        pickupQueued = true;
        canvas.focus();
      });
    }
    if (btnDrop) {
      btnDrop.addEventListener("click", (e) => {
        e.stopPropagation();
        if (mode !== "playing") enterPlay();
        dropQueued = true;
        canvas.focus();
      });
    }
    document.getElementById("btn-reload").addEventListener("click", (e) => {
      e.stopPropagation();
      if (mode !== "playing") enterPlay();
      reloadQueued = true;
      canvas.focus();
    });
    if (btnInspect) {
      btnInspect.addEventListener("click", (e) => {
        e.stopPropagation();
        if (mode !== "playing") enterPlay();
        for (const ev of weapon.startInspect()) playWeaponEvent(ev);
        updateAmmoHud(true);
        canvas.focus();
      });
    }
    document.getElementById("btn-holster").addEventListener("click", (e) => {
      e.stopPropagation();
      if (weapon.ownedWeapons().length > 1) weapon.cycleWeapon(1);
      else weapon.toggleEquip();
      for (const ev of weapon.popSfxEvents()) playWeaponEvent(ev);
      updateQuickBarButtons();
      updateAmmoHud(true);
      canvas.focus();
    });
    document.getElementById("btn-kick").addEventListener("click", (e) => {
      e.stopPropagation();
      if (mode !== "playing") enterPlay();
      kickQueued = true;
    });
    document.getElementById("btn-flash").addEventListener("click", (e) => {
      e.stopPropagation();
      player.flashlightOn = !player.flashlightOn;
      playFlashlightSfx();
      updateQuickBarButtons();
      canvas.focus();
    });
    document.getElementById("btn-pause").addEventListener("click", (e) => {
      e.stopPropagation();
      if (mode === "playing") enterPause(); else enterPlay();
    });

    // Chapter 00 is playable. The other eleven chapter slots deliberately have no levels yet.
    const chapters = Array.from({length:12},(_,n)=>({id:n,available:n===0}));
    const storyKey='psx.campaign.v1', roomSpawn=[-10,12.201,24.12], duration=30.6;
    const el=id=>document.getElementById(id);
    const storyUI=el('story-hud'), storyNote=el('story-note'), black=el('story-black');
    const yardLights=[[-.7,5.12,11,1,.67,.36,5.2,20],[9.27,5.12,-15,1,.74,.44,4.7,21],[-16.72,5.12,-4,1,.68,.36,4.6,19],[-.7,5.12,-36,.80,.86,1,4.5,21],[-15,24,9,.56,.68,1,3.2,57]].map(([x,y,z,r,g,b,intensity,radius])=>({position:[x,y,z],color:[r,g,b],intensity,radius}));
    const parkedCars=[[-.95,-21,'wagon',.16], [8.25,17,'volga',3.12],[-19,17.3,'sedan',1.56],[16.3,31.17,'wagon',1.58],[16.25,33.80,'sedan',1.53],[16.32,38.94,'volga',-1.56],[-17.99,-30.10,'sedan',1.59],[-18.04,-24.98,'wagon',-1.55],[-3.8,52.2,'sedan',-1.56]].map(([x,z,style,yaw],i)=>({pos:[x,.10,z],style,yaw,scale:1.03,wheelSpin:0,tint:[[.33,.42,.40],[.31,.24,.22],[.55,.42,.28]][i%3]}));
    const neighbours=scene.prologue.balconyResidents.map((r,i)=>({...r,phase:i*2.37,
      ap:clampChar({...DEFAULT_CHARACTER,gender:i%2?'male':'female',skin:i%4,hair_style:[4,2,11,1,7][i%5],hair_color:i%3,top_style:[5,2,4,7,1][i%5],top_color:[3,6,2,1,5][i%5],bottom_style:i%3}),prop:['coffee','book','none'][r.type]}));
    function drawBalconyLife(eye,fwd,elapsed){
      // Rear courtyard: carry groceries home along the trodden path under the windows.
      if(eye[2]>-12)for(let i=0;i<2;i++){
        const u=(elapsed*.051+i*.94)%2,leg=u<1?u:2-u,xx=-20+15*leg,zz=52.59+.14*Math.sin(leg*Math.PI)+i*.25;
        const n=neighbours[i+2],yaw=(u<1?1:-1)*Math.PI/2+.04*Math.sin(leg*Math.PI);const prop=i?'bag':'briefcase';
        const bones=evaluateCharacterBones(charJoints,n.ap.gender,[xx,.045,zz],yaw,{elapsed:elapsed+i*1.8,onGround:true,horizSpeed:.80,walkPhase:elapsed*4.5+i*3,npcProp:prop});
        drawCharacterModel(n.ap,bones,elapsed+i);drawNpcProp(prop,bones,[.65,.54,.44]);}

      for(const n of neighbours){const dx=n.pos[0]-eye[0],dy=n.pos[1]+.95-eye[1],dz=n.pos[2]-eye[2];
        // Don't skin or submit distant/offscreen figures from the other side of the apartment.
        const d2=dx*dx+dy*dy+dz*dz;if(d2>60*60 || dx*fwd[0]+dy*fwd[1]+dz*fwd[2]<.18*Math.sqrt(d2))continue;
        const t=elapsed+n.phase, bones=evaluateCharacterBones(charJoints,n.ap.gender,n.pos,n.yaw+.07*Math.sin(t*.27),{
          elapsed:t,onGround:true,npcProp:n.prop,balconyIdle:n.type,npcHeadLookYaw:.48*Math.sin(t*.38)});
        drawCharacterModel(n.ap,bones,t);if(n.prop!=='none')drawNpcProp(n.prop,bones,[.87,.74,.58]);
      }
    }
    const kitchenLights=[{position:[-3.95,13.79,30.26],color:[1,.88,.70],intensity:.52,radius:3.8},{position:[-7.86,14.38,31.59],color:[1,.84,.62],intensity:1.70,radius:6.8},{position:[-3.87,14.38,31.38],color:[1,.77,.52],intensity:1.72,radius:6.6},{position:[-5,4.67,43.1],color:[1,.76,.46],intensity:4.5,radius:21},{position:[-22,4.67,44.6],color:[1,.70,.44],intensity:3.9,radius:20}];
    const storyLights=[{position:[-10,14.53,25.6],color:[1,.78,.55],intensity:1.65,radius:7},
      {position:[-7.45,13.86,26.7],color:[1,.61,.32],intensity:1.1,radius:4.5},
      {position:[-11.1,14.15,20.5],color:[.56,.70,1],intensity:1.18,radius:4.8}];
    let storyTime=0,autoAt=0,lastSlot=null,saveOK=true,lastBlip=-1,releaseMatrix=null,storyStamp='',restartUntil=0;
    const storyActive=()=>mode==='cinema'||mode==='home';
    // One household seed, independent of restarts and of the live character editor.
    // The alternate gender uses the same seed. Only the Settings button replaces it.
    const partnerKey='psx.household.seed.v1'; let familySeed=null, partner=null, partnerLook=0;
    const validSeed=n=>Number.isInteger(n)&&n>=0&&n<=0xffffffff;
    function freshSeed(){return crypto.getRandomValues(new Uint32Array(1))[0];}
    function persistSeed(){try{localStorage.setItem(partnerKey,JSON.stringify(familySeed));return true;}catch(_){return false;}}
    function initPartner(saved){
      if(familySeed===null){try{familySeed=JSON.parse(localStorage.getItem(partnerKey));}catch(_){} }
      if(!validSeed(familySeed)){familySeed=validSeed(saved?.partnerSeed)?saved.partnerSeed:freshSeed();persistSeed();}
      let state=familySeed;const rand=()=>{state=(Math.imul(state,1664525)+1013904223)>>>0;return state/4294967296;},i=n=>Math.floor(rand()*n),pick=a=>a[i(a.length)];
      const gender=character.gender==='male'?'female':'male';partner=clampChar({...DEFAULT_CHARACTER,gender,skin:i(charCatalog.skinPalette.length),face_style:i(charCatalog.faceStyles.length),hair_style:pick(gender==='female'?[1,2,3,4,5,6,12,13,19]:[0,3,7,8,9,11,14,16]),hair_color:i(charCatalog.hairPalette.length),top_style:pick([0,1,2,3,4]),top_color:i(charCatalog.topPalette.length),bottom_style:pick([0,1,2]),bottom_color:i(charCatalog.bottomPalette.length),shoe_style:0,shoe_color:i(4),accessory:0});partnerLook=0;
    }
    el('set-partner').onclick=()=>{let next=freshSeed();if(next===familySeed)next=(next+1)>>>0;familySeed=next;const written=persistSeed();initPartner(null);
      if(lastSlot){lastSlot.partnerSeed=familySeed;try{localStorage.setItem(storyKey,JSON.stringify(lastSlot));}catch(_){}}
      el('partner-status').textContent=written?'Новая внешность сохранена':'Новая внешность — до закрытия вкладки';};
    const chairPos=[-2.90,12.20,31.37],chairYaw=-Math.PI/2;
    function drawDiningPartner(eye,elapsed){if(!partner||mode!=='home'||hallDone)return;const focus=hallT>=0?{eyes:point(3.9,1.60,7.85),head:point(3.9,1.60,7.85)}:supperT<0?{eyes:eye,head:eye}:dinnerFocus(false,supperT);const watch=focus.head,dx=watch[0]-chairPos[0],dz=watch[2]-chairPos[2];
      let a=Math.atan2(dx,dz)-chairYaw;a=Math.atan2(Math.sin(a),Math.cos(a));const target=Math.abs(a)>2.8?0:Math.max(-.84,Math.min(.84,a));partnerLook+=(target-partnerLook)*.15;
      const performanceCue=dinnerAct('partner',supperT,supperT<0?elapsed:supperT+1.78);
      const bones=evaluateCharacterBones(charJoints,partner.gender,chairPos,chairYaw,{elapsed,delivery:supperT>=0?linePerformance('partner',supperT,'supper'):null,acting:performanceCue,seatedDinner:true,worried:phase(supperT,19.53,21.83)*(1-.58*phase(supperT,31.3,34.4))*(1-phase(supperT,bellAt,bellAt+.7)),npcHeadLookYaw:partnerLook});
      // Keep the elbow planted and the closed hand in contact as the head turns.
      const j=charJoints[partner.gender],shoulder=mat4TransformPt(bones.upper_arm_r,[0,0,0]);
      const elbow=[chairPos[0]-.255,chairPos[1]+.822,chairPos[2]-.19],knuckles=mat4TransformPt(bones.head,[-.067,.053,.047]);
      const dir=normalize(vec3Sub(knuckles,elbow)),wrist=knuckles.map((v,i)=>v-dir[i]*.075);
      bones.upper_arm_r=armSegment(shoulder,elbow,-j.lower_arm_offset[1]);bones.lower_arm_r=armSegment(elbow,wrist,-j.hand_offset[1]);bones.hand_r=mat4Mul(armSegment(wrist,knuckles,.075),mat4Scale(1.04,.57,1.42));
      drawCharacterModel(partner,bones,supperT<0?elapsed:supperT+2.13,supperT>=0?supperMouth('partner',supperT):'',performanceCue.face,livingEyes(eyeAim(bones,focus.eyes),supperT<0?elapsed:supperT,.30,2.2));}
    // Two-link IK with a retained elbow plane. A hand that can't reach releases instead
    // of stretching an arm. The wrist rolls toward the surface just before contact.
    function contactArm(b,g,side,goal,mix,tip){if(mix<=.001)return;const j=charJoints[g],upper='upper_arm_'+side,lower='lower_arm_'+side,hand='hand_'+side,at=m=>[m[3],m[7],m[11]],A=at(b[upper]),E=at(b[lower]),W=at(b[hand]);
      const target=mix3(W,goal,mix),delta=vec3Sub(target,A),u=normalize(delta),l=-j.lower_arm_offset[1],s=-j.hand_offset[1],dist=Math.max(.08,Math.min(l+s-.002,Math.hypot(...delta))),along=(l*l-s*s+dist*dist)/(2*dist),bend=Math.sqrt(Math.max(0,l*l-along*along)),ev=vec3Sub(E,A),dot=ev.reduce((q,v,i)=>q+v*u[i],0);let pole=ev.map((v,i)=>v-u[i]*dot);if(Math.hypot(...pole)<.002)pole=vec3Cross(u,[0,0,1]);pole=normalize(pole);
      const elbow=A.map((v,i)=>v+along*u[i]+bend*pole[i]),wrist=A.map((v,i)=>v+dist*u[i]),rest=normalize(vec3Sub(mat4TransformPt(b[hand],[0,-.08,0]),W)),dir=normalize(mix3(rest,tip,mix));b[upper]=armSegment(A,elbow,l);b[lower]=armSegment(elbow,wrist,s);b[hand]=armSegment(wrist,wrist.map((v,i)=>v+dir[i]*.08),.08);}
    function armSegment(a,b,length){const y=normalize(vec3Sub(a,b)),x=normalize(vec3Cross(y,[1,0,0])),z=vec3Cross(x,y),k=Math.hypot(...vec3Sub(a,b))/length;
        return [x[0],y[0]*k,z[0],a[0],x[1],y[1]*k,z[1],a[1],x[2],y[2]*k,z[2],a[2],0,0,0,1];}
    // Ease into actions, hold a thought, and release. Windows are timed to the
    // actual edit rather than random idle gestures that cut across dialogue.
    // Authored fixations, in kitchen coordinates. The listener watches the speaker,
    // not the editing camera. At reload the same time gives the same performance.
    function dinnerFocus(hero,t){
      const seat=phase(t,4.46,6.72)*(1-phase(t,41.48,43.50));
      const other=hero?[7.02,1.22,10.37]:[4.86+.20*seat,1.65-.43*seat,10.01+.36*seat];
      const paper=[5.63,.82,10.96], glass=[6.56,.90,10.78],door=[3.91,1.56,7.9],window=[6.18,1.92,12.68];
      const cues=hero?[[0,.95,other],[1,.74,[4.87,.87,10.2]],[1.91,.95,other],[4.62,.94,[5.20,.70,10.39]],[5.65,.93,other],[15.57,.95,paper],[17.72,.94,other],[30.14,.70,window],[30.90,.92,other],[38.58,1,door],[39.74,.97,other],[41.52,1,door]]:
        [[0,.88,[4.86,.87,10.2]],[1.83,.98,other],[6.84,.83,glass],[8.32,.98,other],[15.05,.97,paper],[17.96,.90,other],[19.84,.91,glass],[23.72,.96,other],[32.22,.69,window],[33.22,.92,other],[38.66,1,door],[40.31,.96,other]];
      function target(time,slow){let target=other;for(const [at,strength,dest] of cues){if(time<at)break;target=mix3(target,mix3(other,dest,strength),phase(time,at,at+(slow?.57:.14)));}return point(...target);}
      return {eyes:target(t,false),head:target(t-.16,true)};
    }
    function focusTurn(pos,yaw,target){const dx=target[0]-pos[0],dz=target[2]-pos[2],a=Math.atan2(dx,dz)-yaw;return Math.atan2(Math.sin(a),Math.cos(a));}
    function eyeAim(bones,target){const m=bones.head,origin=mat4TransformPt(m,[0,.092,.085]),v=vec3Sub(target,origin);
      const x=m[0]*v[0]+m[4]*v[1]+m[8]*v[2],y=m[1]*v[0]+m[5]*v[1]+m[9]*v[2],z=m[2]*v[0]+m[6]*v[1]+m[10]*v[2];
      return [Math.tanh(Math.atan2(x,z)*2.5),-Math.tanh(Math.atan2(y,Math.hypot(x,z))*2.45)];}
    function dinnerAct(who,t,clock){
      const p=(a,b,c,d)=>phase(t,a,b)*(1-phase(t,c,d));const flinch=p(bellAt,bellAt+.15,bellAt+.28,bellAt+1.22),isHero=who==='hero';
      const ask=p(11.65,12.43,13.16,14.58),comfort=p(23.04,24.13,29.55,32.56),openPalm=p(25.41,26.72,27.68,29.43),doubt=p(14.80,16.42,20.75,22.13),hope=p(32.60,33.74,34.6,36.53),drawLook=p(15.3,15.83,18.68,20.73);
      const worry=isHero?.45*p(8.8,10.36,21,23.72):.18+.53*p(6.87,8.93,13.02,14.78)+.79*p(14.75,17.14,30.65,33.46)+.29*hope;
      const smile=isHero?comfort*.94:hope*.42;const surprise=isHero?ask*.60+flinch:flinch;
      const glance=(isHero?-.39:.73)*drawLook+.49*flinch+.13*Math.sin(clock*.88+(isHero?0:2))+.09*Math.tanh(5*Math.sin(clock*1.3));
      return {face:[worry,smile,surprise,glance],ask:isHero?ask:0,comfort:isHero?comfort:0,palm:isHero?openPalm:0,doubt:isHero?0:doubt,hope:isHero?0:hope,flinch,drawLook,breath:Math.sin(clock*1.52+(isHero?0:1.32)),settle:isHero?p(4.36,5.40,5.58,6.95)+p(41.14,41.82,42.07,43.14):0};
    }
    const supperEnd=44.6,bellAt=38.48,guestSeat=[-4.94,12.20,31.37];
    let supperT=-1,supperDone=false,spoken=-1,printed=0;
    const supperLines=[
      [8.70,11.46,'partner','Я так волнуюсь…'],[12.18,14.50,'hero','Насчёт чего?'],
      [15.24,21.95,'partner','Насчёт детей. Бабушка в последнее время ведёт себя…'],
      [23.55,25.44,'hero','Не беспокойся.'],[25.85,32.70,'hero','Всё нормально… Завтра вернутся целыми и здоровыми, даже лучше.'],
      [33.52,36.01,'partner','Надеюсь…'],[40.18,42.38,'hero','Я открою.']
    ];
    function nearGuestSeat(){return mode==='home'&&!supperDone&&supperT<0&&Math.hypot(player.pos[0]-guestSeat[0],player.pos[2]-guestSeat[2])<1.28;}
    function resetSupCaption(){el('inner-door').hidden=true;spoken=-1;printed=0;el('supper-caption').hidden=true;el('chair-prompt').hidden=true;}
    function startSupper(time=0){if(mode!=='home'||supperDone)return;keysDown.clear();dragging=false;if(document.pointerLockElement===canvas)document.exitPointerLock();getAudioCtx();ensureWeaponWavsLoaded();stopCall();resetSupCaption();supperT=time;storyUI.classList.add('cinema');el('story-save').hidden=true;el('story-skip').hidden=false;storyNote.textContent='ГЛАВА 00   /   ЗА СТОЛОМ';storeStory(true);}
    function endSupper(){supperDone=true;supperT=-1;resetSupCaption();finishFilm({pos:point(4.68,0,9.67),yaw:-.41,pitch:-.035});}
    function updateSupper(delta){const prev=supperT;supperT=Math.min(supperEnd,supperT+delta);const t=supperT;
      if(prev<bellAt&&t>=bellAt)playWeaponWavIfReady('flat_chime');
      const i=supperLines.findIndex(([a,b])=>a<=t&&t<b);const bellSub=t>=bellAt&&t<39.98;
      if(i<0){if(spoken!==-1)_stopNpcVoice();spoken=-1;printed=0;el('supper-caption').hidden=!bellSub;if(bellSub){el('supper-who').textContent='ЗА ДВЕРЬЮ';el('supper-words').textContent='Звонок в дверь.';}}
      else{const [at,,who,text]=supperLines[i];const wife=who==='partner'&&partner.gender==='female';const name=who==='hero'?'ВЫ':wife?'ЖЕНА':'МУЖ';if(i!==spoken){_stopNpcVoice();printed=0;spoken=i;}
        const n=Math.min(text.length,1+Math.floor((t-at)*23));el('supper-caption').hidden=false;el('supper-who').textContent=name;el('supper-words').textContent=text.slice(0,n);
        if(n>printed&&/[аеёиоуыэюя]/i.test(text.slice(printed,n))){if(who==='hero')_playNpcVoiceMono('hero_text',{gain:.92,cutoff:6300});else if(wife)_playNpcVoiceMono('spouse_toriel',{gain:.79,cutoff:2850});else _playNpcVoiceMono('npc_voice_'+((i+n)%8),{rate:.89,gain:.64,cutoff:2450});}printed=n;}
      black.style.opacity=Math.max(1-phase(t,0,.34),phase(t,44.16,44.57));if(t>=supperEnd)endSupper();}
    function supperMouth(who,t){if(spoken<0)return '';const [at,,actor,text]=supperLines[spoken];if(actor!==who)return '';
      const u=(t-at)*23,k=Math.floor(u);if(k<0||k>=text.length)return '';const c=text[k].toLowerCase();if(/[бмп\s.,…?!]/.test(c))return '';return /[аояэ]/.test(c)?'_talk2':/[еёиуыю]/.test(c)||u-k<.52?'_talk1':'';}
    // Shot/reverse-shot stays on the near side of the eyeline until the doorbell breaks it.
    function supperCam(t){let e,a,f;
      const move=(v,w,l,r)=>mix3(v,w,phase(t,l,r));
      if(t<2.22){e=move([4.02,1.20,11.05],[4.16,1.13,10.94],0,2.22);a=[4.865,.866,10.208];f=40;}
      else if(t<4.56){e=[6.35,1.50,9.86];a=[4.83,1.56,10.01];f=39;}
      else if(t<7.24){e=move([5.94,1.74,8.22],[5.93,1.43,8.43],4.56,7.24);a=move([5.90,1.19,10.34],[5.94,.99,10.37],4.56,7.24);f=55;}
      else if(t<12.03){e=move([5.19,1.37,9.83],[5.49,1.31,9.91],7.24,12.03);a=[7.01,1.20,10.34];f=38;}
      else if(t<15.25){e=move([6.79,1.37,9.60],[6.70,1.36,9.65],12.03,15.25);a=[5.12,1.17,10.37];f=41;}
      else if(t<17.55){e=move([5.86,1.36,10.97],[5.78,1.26,10.98],15.25,17.55);a=[5.63,.806,10.95];f=49;}
      else if(t<23.43){e=move([5.92,1.38,9.73],[6.19,1.30,9.87],17.55,23.43);a=move([7.0,1.16,10.34],[6.98,1.035,10.35],20.1,23.0);f=40;}
      else if(t<28.84){e=move([6.65,1.37,9.80],[6.38,1.33,9.90],23.43,28.84);a=[5.15,1.15,10.37];f=44;}
      else if(t<33.30){e=move([6.0,1.51,8.0],[6.07,1.43,8.20],28.84,33.30);a=[6.07,1.02,10.39];f=51;}
      else if(t<38.48){e=move([5.81,1.32,9.87],[5.94,1.30,9.90],33.30,38.48);a=[6.99,1.20,10.37];f=37;}
      else if(t<40.15){e=[5.41,1.52,9.92];a=move([3.96,1.65,7.89],[4.16,1.79,7.89],38.48,40.15);f=42;}
      else{e=move([6.35,1.40,9.79],[6.42,1.66,9.66],40.15,44.6);a=move([5.10,1.22,10.35],[4.81,1.58,9.90],41.39,44.50);f=45;}
      const eye=point(...e);return {eye,fwd:normalize(vec3Sub(point(...a),eye)),fov:f};}
    function drawDinnerHero(t){const seat=phase(t,4.46,6.72)*(1-phase(t,41.48,43.50)),slide=phase(t,3.95,5.77),stand=phase(t,42.62,44.4);
      const performanceCue=dinnerAct('hero',t,t);const stepShift=Math.sin(Math.PI*slide)*.032+Math.sin(Math.PI*stand)*.026;
      const pos=point(4.80+.26*slide-.27*stand+stepShift,0,9.98+.39*slide-.45*stand);const focus=dinnerFocus(true,t),yaw=Math.PI/2-.95*stand;const turn=focusTurn(pos,yaw,focus.head);const bones=evaluateCharacterBones(charJoints,character.gender,pos,yaw,{elapsed:t,npcHeadLookYaw:Math.max(-.65,Math.min(.65,turn))*.64,delivery:linePerformance('hero',t,'supper'),acting:performanceCue,heroSit:seat,heroShake:phase(t,23.43,23.92)*(1-phase(t,25.33,25.86)),doorLook:phase(t,bellAt,bellAt+.8),onGround:true});
      // Reach from the shoulder to the actual top rail, hold, then release before sitting.
      const grip=phase(t,0,.95)*(1-phase(t,3.91,4.53));if(grip>0){const j=charJoints[character.gender],joint=m=>mat4TransformPt(m,[0,0,0]);const shoulder=joint(bones.upper_arm_r),wrist=mix3(joint(bones.hand_r),point(4.863,.907,10.208),grip),elbow=mix3(joint(bones.lower_arm_r),point(4.86,1.11,10.18),grip);
        bones.upper_arm_r=armSegment(shoulder,elbow,-j.lower_arm_offset[1]);bones.lower_arm_r=armSegment(elbow,wrist,-j.hand_offset[1]);bones.hand_r=mat4Mul(armSegment(wrist,vec3Add(wrist,[.001,-.062,0]),.063),mat4Scale(1,.6,1.18));}
      drawCharacterModel(character,bones,t,supperMouth('hero',t),performanceCue.face,livingEyes(eyeAim(bones,focus.eyes),t,.36,.7));}
    // Second movement: the threshold, a player-paced conversation, then an ellipsis.
    const landingGpu=Object.fromEntries(Object.entries(scene.prologue.landingProps).map(([k,v])=>[k,uploadBatches(gl,v)]));
    const sam=clampChar({...DEFAULT_CHARACTER,gender:'male',skin:1,hair_style:charCatalog.hairStyles.findIndex(h=>h.id==='crew_cut'),hair_color:0,face_style:4,top_style:0,top_color:1,bottom_style:0,bottom_color:0,shoe_style:0,shoe_color:0,accessory:5,accessory_color:0});
    const hallTalkAt=49,hallExitAt=50,hallEnd=65.4,groundSpawn=[-3.35,.201,26.97];
    let hallT=-1,hallDone=false,hallPage=0,pageTime=0,hallSpoken='',hallLetters=0;
    const gateLines=[
      [2.35,4.82,'thought','Я узнаю этот шрам. Мы были в одной группировке.'],
      [5.25,7.23,'hero','Ты, э-э…?'],[7.78,9.20,'sam','Да, я.'],
      [12.25,15.84,'hero','Сэм… Какого хера ты тут делаешь?!'],[16.42,18.62,'sam','Мы можем поговорить?'],
      [21.32,24.82,'hero','Ещё шаг — и я вызову полицию.'],[25.46,30.18,'sam','Я не убивать тебя пришёл. Поговорить надо.'],
      [30.94,36.28,'hero','…Мне не о чем с тобой разговаривать. Последний раз предупреждаю…'],
      [39.00,41.02,'sam','Уверен?'],[42.45,44.75,'hero','Скоро вернусь…']
    ];
    const landingLines=[['hero','Что тебе от меня нужно?'],['sam','Ты жить хочешь или как?'],['hero','Я и так живу. Я же под программой защиты свидетелей.'],['sam','Нихрена. Уже нет. Спеклась твоя защита.'],['hero','Чего?!'],['sam','Деньги правят миром.'],['sam','Слушай. Сейчас сиди дома и не рыпайся, если жизнь дорога. Дальше расскажу, что делать.']];
    const hallLights=[{position:[-5.62,14.24,25.73],color:[.79,.85,1],intensity:.83,radius:3.7},{position:[-4.12,14.48,27.9],color:[1,.83,.57],intensity:1.94,radius:6.7},{position:[-6.16,13.87,28.72],color:[.80,.85,1],intensity:.94,radius:4.5},{position:[-4.2,2.68,27.35],color:[1,.86,.61],intensity:2,radius:7},{position:[-2.48,1.92,26.9],color:[.66,.83,1],intensity:.92,radius:3.9}];
    // Lobby exit. The latch, moving leaf, objective and saved position share one state.
    const EXIT_X=-1.61,EXIT_Z=27.68,closedStreet=[-1.68,.19,26.49,-1.51,2.37,27.69],openStreet=[-1.62,.19,27.62,-.42,2.37,27.74];
    const routeSolids=[scene.prologue.colliders.concat([closedStreet]),scene.prologue.colliders.concat([openStreet])];
    let streetStage=0,breachTime=2,rattleTime=2,jamText=0,streetImpact=false;
    // A nightmare is one continuous edit. All particle positions and camera cues depend on
    // scene time, so a save in the middle restores the same frame without replaying the bang.
    const flashAt=5.10,wakeAt=6.65,dreamEnd=16.65,bed=[-5.52,-27.80,23.67],motelSpawn=[-4.18,-27.798,24.65],oldBalcony=[-10,13.61,20.96];
    const motelLights=[{position:[-4.18,-26.77,22.73],color:[1,.64,.35],intensity:1.48,radius:4.5},{position:[-6.58,-26.04,25.06],color:[.48,.72,.90],intensity:1.21,radius:5.3}];
    let dreamT=-1,awake=false,dreamYaw=0,dreamSrc=null;
    const dreamParts=Array.from({length:99},(_,i)=>{const h=n=>{const a=Math.sin(71.6*i+n*9.13+.62)*43758.3;return a-Math.floor(a);};return [h(1),h(2),h(3),h(4),h(5),h(6)];});
    function cutSound(){if(dreamSrc){try{dreamSrc.stop();}catch{}dreamSrc=null;}}
    function dreamSound(key,volume,rate,offset=0){cutSound();const ac=getAudioCtx();if(!ac)return;let buf=wpnWavBufs[key];if(!buf&&key==='waking_breath')return;if(!buf){const boom=key==='window_report',dur=boom?1.24:1.17;buf=ac.createBuffer(1,Math.floor(ac.sampleRate*dur),ac.sampleRate);const out=buf.getChannelData(0);let low=0,mid=0;for(let i=0;i<out.length;i++){const t=i/ac.sampleRate,v=Math.random()*2-1;low+=.035*(v-low);mid+=.24*(v-mid);out[i]=boom?Math.tanh((4.4*low+.5*Math.sin(2*Math.PI*58*t))*Math.exp(-4.6*t)+v*.62*Math.exp(-26*t))*Math.min(1,t/.004):(.67*mid+.50*low)*Math.pow(Math.abs(Math.sin(Math.PI*t/dur*2)),1.2)*(1-.25*t/dur);}wpnWavBufs[key]=buf;}const src=ac.createBufferSource(),g=ac.createGain();src.buffer=buf;src.playbackRate.value=rate;g.gain.value=getSfxVol()*volume;src.connect(g).connect(ac.destination);src.onended=()=>{src.disconnect();g.disconnect();if(dreamSrc===src)dreamSrc=null;};src.start(0,Math.min(offset,buf.duration-.001));dreamSrc=src;}
    function startDream(time=0,yaw=player.yaw){el('inner-door').hidden=true;dreamT=time;dreamYaw=yaw;streetStage=5;keysDown.clear();dragging=false;if(document.pointerLockElement===canvas)document.exitPointerLock();player.vel=[0,0,0];closeStreetUI();black.style.opacity=0;storyUI.classList.add('cinema');el('story-skip').hidden=false;el('story-save').hidden=true;storyNote.textContent='';storeStory(true);}
    function finishDream(keepBreath=false){if(!keepBreath)cutSound();dreamT=-1;awake=true;el('motel-caption').hidden=true;closeStreetUI();finishFilm({pos:motelSpawn,yaw:-2.56,pitch:-.045});}
    function updateDream(dt){const old=dreamT;dreamT+=dt;const crossed=t=>old<t&&dreamT>=t;if(crossed(flashAt))dreamSound('window_report',.94,1);if(crossed(wakeAt))cutSound();
      const u=dreamT-wakeAt;if(u>=0&&u<23.7&&!dreamSrc&&wpnWavBufs.waking_breath)dreamSound('waking_breath',.94,1,u);
      black.style.opacity=phase(u,6.92,7.62)*(1-phase(u,7.80,8.69));
      const card=el('motel-caption');card.hidden=u<3.8||u>6.80;card.style.opacity=phase(u,3.8,4.46)*(1-phase(u,6.28,6.80));if(dreamT>=dreamEnd)finishDream(true);}
    function wakingBones(t){const edge=phase(t,8.35,12.74),up=phase(t,13.95,17.62),step=phase(t,18.1,20.32),yaw=1.50*phase(t,8.17,12.51)-1.23*phase(t,17.75,20.17);const root=[bed[0]+.655*edge+.46*up+.225*step,bed[1],bed[2]+.17*edge+.81*step];const bones=evaluateCharacterBones(charJoints,character.gender,root,yaw,{elapsed:t,onGround:true,waking:Math.max(0,t),bustSize:character.bust_size});
      const press=phase(t,9.3,10.35)*(1-phase(t,14.10,15.16));const brace=[-4.99,bed[1]+.59,23.48];contactArm(bones,character.gender,'l',brace,press,[-.73,-.29,.61]);
      const steady=phase(t,4.43,5.45)*(1-phase(t,7.56,8.92));contactArm(bones,character.gender,'r',mat4TransformPt(bones.torso,[-.10,.25,.105]),steady,[.62,.59,-.40]);return bones;}
    function nightmareCam(){const t=dreamT,eye=player.eyePos;
      if(t<2.38){const end=Math.atan2(oldBalcony[0]-eye[0],eye[2]-oldBalcony[2]),arc=Math.atan2(Math.sin(end-dreamYaw),Math.cos(end-dreamYaw)),q=phase(t,.12,1.76),yaw=dreamYaw+arc*q,pitch=lerp(player.pitch,Math.atan2(oldBalcony[1]-eye[1],Math.hypot(oldBalcony[0]-eye[0],oldBalcony[2]-eye[2])),q);return {eye,fwd:[Math.sin(yaw)*Math.cos(pitch),Math.sin(pitch),-Math.cos(yaw)*Math.cos(pitch)],fov:68-20*q};}
      if(t<wakeAt){const j=Math.max(0,t-flashAt),shake=j>0?.105*Math.exp(-4*j):0,e=mix3([-6.4,4.13,2.82],[-6.70,4.36,3.70],phase(t,2.38,flashAt));e[0]+=shake*Math.sin(j*58);e[1]+=shake*Math.sin(j*37);return {eye:e,fwd:normalize(vec3Sub(oldBalcony,e)),fov:26-3*phase(t,2.38,flashAt)+5*phase(j,0,.19)};}
      const u=t-wakeAt,bones=wakingBones(u),face=mat4TransformPt(bones.head,[0,.10,.07]);let e,v,f;
      if(u<3.75){const q=phase(u,0,.86);e=[face[0]+.54,face[1]+.56-.35*q,face[2]+.49+.27*q];v=face;f=36+4*q;}
      else if(u<7.72){const q=phase(u,3.75,7.60);e=mix3([-3.02,-26.55,25.96],[-3.19,-26.51,25.84],q);v=mix3([-5.47,-26.81,23.52],face,.23*q);f=54-2*q;}
      else{const q=phase(u,7.72,9.94),yaw=-2.56+.09*(1-q),pitch=-.045+.035*(1-q);e=[motelSpawn[0],motelSpawn[1]+player.cfg.stand_eye_height,motelSpawn[2]];v=[e[0]+Math.sin(yaw)*Math.cos(pitch),e[1]+Math.sin(pitch),e[2]-Math.cos(yaw)*Math.cos(pitch)];f=68;}return {eye:e,fwd:normalize(vec3Sub(v,e)),fov:f};}
    function drawDream(clock){const t=dreamT;if(t<0)return;if(t>=wakeAt){if(t>=wakeAt+7.72)return;const u=t-wakeAt,b=wakingBones(u),q=phase(u,.04,.74),settle=phase(u,3,9);drawCharacterModel(character,b,clock,u<3.7&&Math.sin(u*5.1)>.16?'_talk1':'',[.43+.34*q-.59*settle,.07*settle,.83*q-.77*settle,0],livingEyes([0,.04],u,1.48-.67*settle,.91));return;}
      const age=t-flashAt;if(age<0)return;
      function piece(mesh,pos,size,rot,tint,em){const m=mat4Mul(mat4Translate(...pos),mat4Mul(mat4RotZ(rot),mat4Mul(mat4RotY(rot*.71),mat4Scale(...size))));drawHallProp(mesh,m,tint,em);}
      for(let i=0;i<dreamParts.length;i++){const [a,b,c,d,e,f]=dreamParts[i];
        if(i<62){const smoke=i>=39,u=age-(smoke?.28+.34*c:i<19?1.09*c:.28*c);if(u<=0)continue;const k=(smoke?.48:.27)+(.55+.45*d)*phase(u,0,.32),heat=1-phase(u,.42,1.29),x=-12.20+4.4*a+(a-.5)*u*(smoke?1:1.7),y=13.36+.56*e+u*(smoke?2.7:1.1+1.6*f),z=20.91-(.4+1.8*b)*u;
          gl.uniform1f(su.softParticle,smoke?4:3);piece(landingGpu.plume,[x,y,z],[k*(1+.4*c),k*(.8+.77*f),k*.87],e+u*(f-.5),smoke?[.20+.08*f,.17,.17]:[1,.23+.55*d*heat,.025+.20*c*heat],smoke?.50:1.35+.95*heat);gl.uniform1f(su.softParticle,0);}
        else{const u=age-.05-.14*b;if(u<=0)continue;const k=.075+.17*e;piece(landingGpu.splinter,[-12.0+4*a+(a-.5)*6.4*u,13.34+.94*b+(1.5+4*c)*u-4.7*u*u,20.76-(2.1+4.7*f)*u],[k*(.5+d),k*.24,k*(.64+1.48*b)],c*4+u*(5+7*d),i%3?[1,.69,.43]:[.59,.79,.83],.65);}}
      gl.uniform3f(su.colorTint,1,1,1);}
    // Hinges are in world space. The six small collision spans follow the leaf, not the wall.
    const interiorDoors=[[-9.89,12.20,29.01,1.571,-1.54,'Коридор'],[-14.75,12.20,31.13,0,1.54,'Детская'],[-11.41,12.20,31.13,0,1.54,'Спальня'],[-2.11,-27.80,23.50,1.571,1.54,'Санузел']].map(([x,y,z,base,swing,name])=>({x,y,z,base,swing,name,p:0,target:0}));
    const leafAngle=(d,p)=>d.base+(d.y<0?0:d.swing*smooth01(p));
    const pocketShift=(d,p)=>d.y<0?1.27*smooth01(p):0;
    function leafPos(d,p,r){const a=leafAngle(d,p);return [d.x-r*Math.cos(a),d.z+pocketShift(d,p)+r*Math.sin(a)];}
    function nearbyDoor(){if(mode!=='home'||dreamT>=0||supperT>=0||hallT>=0)return null;const [px,py,pz]=player.pos;let best=null,score=2;
      for(const d of interiorDoors){if(Math.abs(d.y-py)>.5)continue;const [cx,cz]=leafPos(d,d.y<0?0:d.p,.63),dx=cx-px,dz=cz-pz,dist=Math.hypot(dx,dz);if(dist>(d.y<0?1.8:1.55)||(d.y>=0||dist>.95)&&dx*player.flatForward[0]+dz*player.flatForward[2]<.30*dist)continue;if(dist<score){best=d;score=dist;}}return best;}
    function toggleInterior(){const d=nearbyDoor();if(!d)return false;d.target=1-d.target;playWeaponWavIfReady(d.target?'door_open':'door_quiet');storeStory(true);return true;}
    function interiorPhysics(dt,base){let extra=[];for(const d of interiorDoors){if(Math.abs(d.y-player.pos[1])>.9)continue;const p=d.p+Math.sign(d.target-d.p)*Math.min(Math.abs(d.target-d.p),dt*1.58);let blocked=false;
        function spans(p){let a=leafPos(d,p,0),list=[];for(let i=1;i<=6;i++){const b=leafPos(d,p,1.14*i/6);list.push([Math.min(a[0],b[0])-.034,d.y,Math.min(a[1],b[1])-.034,Math.max(a[0],b[0])+.034,d.y+2.15,Math.max(a[1],b[1])+.034]);a=b;}return list;}
        let list=spans(p);if(p!==d.p){blocked=list.some(c=>aabbIntersects(player.getAABB(...player.pos),c));if(blocked)list=spans(d.p);else d.p=p;}extra.push(...list);}return extra.length?base.concat(extra):base;}
    function nearCase(){if(!awake||mode!=='home'||dreamT>=0)return false;const dx=-5.38-player.pos[0],dz=26.55-player.pos[2],l=Math.hypot(dx,dz);return l<2.53&&dx*player.flatForward[0]+dz*player.flatForward[2]>.55*l;}
    function showCase(on){el('case-panel').hidden=!on;keysDown.clear();if(on){if(document.pointerLockElement===canvas)document.exitPointerLock();dragging=false;el('inner-door').hidden=true;}else canvas.focus();}
    el('case-close').onclick=()=>showCase(false);
    function doorHint(){const d=nearbyDoor(),board=!d&&nearCase(),hint=el('inner-door');hint.hidden=!d&&!board;el('inner-key').textContent=board?'F':'E';if(d||board)el('inner-verb').textContent=board?'Рассмотреть доску':(d.target?'Закрыть':'Открыть')+' · '+d.name;}
    function drawInteriorLeaves(motel){for(const d of interiorDoors){if((d.y<0)!==motel)continue;drawHallProp(landingGpu.interiorLeaf,mat4Mul(mat4Translate(d.x,d.y,d.z+pocketShift(d,d.p)),mat4RotY(leafAngle(d,d.p))),d.y<0?[.69,.83,.80]:[1,1,1]);}}
    function nearStreetDoor(){const [x,y,z]=player.pos;return mode==='home'&&hallDone&&y<.65&&x> -4&&x<EXIT_X+.03&&Math.abs(z-27.04)<.85&&Math.hypot(x-EXIT_X,z-27.04)<1.92&&player.flatForward[0]>.26;}
    function useStreetDoor(code){if(!nearStreetDoor()||streetStage>=2||!(code==='KeyE'||code==='KeyV'))return false;
      if(code==='KeyE'){streetStage=1;rattleTime=0;jamText=5;playWeaponWavIfReady('door_quiet');}
      else if(streetStage===1){streetStage=2;breachTime=0;streetImpact=false;jamText=0;playKickWhoosh();}
      else return false;
      storeStory(true);return true;}
    function closeStreetUI(){for(const id of ['street-act','latch-box','route-guide','route-pin'])el(id).hidden=true;}
    function streetCam(){const eye=[...player.eyePos],fwd=player.forward;if(streetStage===2&&breachTime<.82){const t=breachTime,p=phase(t,0,.19)-phase(t,.19,.36),k=phase(t,.19,.28)-phase(t,.28,.70);eye[0]+=(.14*k-.055*p)*fwd[0];eye[1]-=.078*k;eye[2]+=(.14*k-.055*p)*fwd[2];return {eye,fwd:normalize([fwd[0],fwd[1]+.035*p-.054*k,fwd[2]]),fov:68+2*k};}return {eye,fwd,fov:68};}
    function drawStreetDoor(){let angle=streetStage<2?.016*Math.sin(rattleTime*37)*(1-phase(rattleTime,0,.52)):-1.58*phase(breachTime,.20,.51)-.14*Math.sin((breachTime-.51)*17)*phase(breachTime,.49,.54)*(1-phase(breachTime,.54,.94));
      drawHallProp(landingGpu.streetLeaf,mat4Mul(mat4Translate(EXIT_X,.20,EXIT_Z),mat4RotY(angle)));}
    function tickStreet(delta){if(!hallDone||awake){closeStreetUI();return;}rattleTime=Math.min(2,rattleTime+delta);breachTime=Math.min(2,breachTime+delta);jamText=Math.max(0,jamText-delta);
      if(streetStage===2&&!streetImpact&&breachTime>=.20){streetImpact=true;playWeaponWavIfReady('door_kick');}
      const [x,y,z]=player.pos;let before=streetStage;
      // Crossing the plane, not the kick itself, starts wayfinding. Stay on the footpath, then turn west.
      if(streetStage===2&&x>EXIT_X+.68&&y<.60)streetStage=3;
      if(streetStage===3&&z<=-.85&&x>=-13.2&&x<=.55&&y<.55)streetStage=4;
      if(streetStage===4&&z<=-.85&&y<.55&&player.flatForward[0]<-.62){startDream();return;}
      if(before!==streetStage)storeStory(true);
      const near=nearStreetDoor();el('street-act').hidden=!near||streetStage>=2;el('street-key').textContent=streetStage===1?'V':'E';el('street-verb').textContent=streetStage===1?'Выбить дверь':'Открыть дверь';el('latch-box').hidden=streetStage!==1||jamText<=0||!near;
      const outside=y<.65&&(x>EXIT_X+.12||z<21.1);const guide=el('route-guide');guide.hidden=!outside||streetStage<3||streetStage===5;const pin=el('route-pin');pin.hidden=true;if(guide.hidden)return;
      // First left out of the door is north. The second is west only once the opposite block is reached.
      const turn=streetStage===4,oldAlcove=!turn&&x< -3.78&&z>18.62,dx=(oldAlcove?-1.18:turn?-11.0:-1.10)-x,dz=(oldAlcove?19.64:turn?-6.55:-6.52)-z,dist=Math.hypot(dx,dz),angle=Math.atan2(dx,-dz)-player.yaw;
      const firstLeft=!turn&&z>25.65&&player.flatForward[0]>.55;
      el('route-arrow').style.transform=`rotate(${angle}rad)`;el('route-range').textContent=Math.ceil(dist)+' м';el('route-copy').textContent=oldAlcove?'К тротуару':turn?'У соседнего дома — налево':firstLeft?'Из подъезда — налево':'Прямо до соседнего дома';el('route-seq').textContent=turn?'02 / 02':'01 / 02';
      // Place the beacon at the actual waypoint. When it leaves the field of view the card stays readable.
      const eye=player.eyePos,v=[x+dx-eye[0],1.63-eye[1],z+dz-eye[2]],dot=(a,b)=>a[0]*b[0]+a[1]*b[1]+a[2]*b[2],depth=dot(v,player.forward),half=Math.tan(34*Math.PI/180),aspect=canvas.clientWidth/Math.max(1,canvas.clientHeight);
      if(depth>.6){const u=50+50*dot(v,player.right)/(depth*half*aspect),h=50-50*dot(v,player.rolledUp)/(depth*half);if(u>6&&u<94&&h>12&&h<83){pin.style.left=u+'%';pin.style.top=h+'%';pin.textContent=turn?'←':'◆';pin.hidden=false;}}}
    function atEntry(){return mode==='home'&&supperDone&&!hallDone&&hallT<0&&supperT<0&&player.pos[2]>28.67&&Math.hypot(player.pos[0]+6.12,player.pos[2]-28.78)<1.55;}
    function inLandingTalk(){return hallT>=hallTalkAt&&hallT<hallExitAt;}
    function hideLandingUI(){el('entry-prompt').hidden=el('hall-next').hidden=el('later-card').hidden=true;}
    function startHall(t=0){hallT=t;hallSpoken='';hallLetters=0;keysDown.clear();dragging=false;if(document.pointerLockElement===canvas)document.exitPointerLock();getAudioCtx();ensureWeaponWavsLoaded();stopCall();resetSupCaption();hideLandingUI();storyUI.classList.add('cinema');el('story-save').hidden=true;el('story-skip').hidden=false;storyNote.textContent='ГЛАВА 00   /   ПОРОГ';storeStory(true);}
    function finishHall(){hallT=-1;hallDone=true;hideLandingUI();resetSupCaption();finishFilm({pos:groundSpawn,yaw:Math.PI/2,pitch:-.035});}
    function nextHall(){if(!inLandingTalk())return;const pause=hallPage===6?2.45:0;if(pageTime<pause+.20)return;const len=landingLines[hallPage][1].length;if(pageTime<pause+len/24){pageTime=pause+len/24+.05;return;}if(hallPage===landingLines.length-1){hallT=hallExitAt;el('hall-next').hidden=true;}else hallPage++;pageTime=0;hallSpoken='';_stopNpcVoice();storeStory(true);}
    el('hall-next').onclick=nextHall;
    function hallCaption(key,who,text,age){const n=who==='thought'?text.length:Math.min(text.length,1+Math.floor(age*24));if(key!==hallSpoken){_stopNpcVoice();hallSpoken=key;hallLetters=0;}
      el('supper-caption').hidden=false;el('supper-who').textContent=who==='thought'?'ПРО СЕБЯ':who==='sam'?'СЭМ':'ВЫ';el('supper-words').textContent=text.slice(0,n);
      if(who!=='thought'&&n>hallLetters&&/[аеёиоуыюяэ]/i.test(text.slice(hallLetters,n)))_playNpcVoiceMono(who==='hero'?'hero_text':'npc_voice_'+((n+2)%8),{rate:who==='hero'?1:.79,gain:who==='hero'?.92:.82,cutoff:who==='hero'?6300:2390});hallLetters=n;
    }
    function updateHall(dt){const old=hallT;if(inLandingTalk())pageTime+=dt;else{hallT+=dt;if(old<hallTalkAt&&hallT>=hallTalkAt){hallT=hallTalkAt;pageTime=0;}}
      const t=hallT;if(old<.56&&t>=.56)playWeaponWavIfReady('door_open');if(old<47.31&&t>=47.31)playWeaponWavIfReady('door_close');
      if([18.96,19.50,20.27,20.61,45.19,45.80,46.48].some(c=>old<c&&t>=c))playFootstepSfx('crouch');
      if([50.85,51.47,52.15,52.81,53.32,53.86,54.37,54.94,55.46,56.05,56.58,57.13,57.73,58.35,59.02,59.57,60.09,60.65,61.21].some(c=>old<c&&t>=c))playFootstepSfx('walk',phase(t,50.8,61.21));
      const dialog=inLandingTalk(),pause=hallPage===6?2.45:0,line=gateLines.find(l=>l[0]<=t&&t<l[1]);el('hall-next').hidden=!dialog||pageTime<pause;el('hall-next').textContent='E · Далее';
      if(dialog&&pageTime>=pause){const [who,text]=landingLines[hallPage];hallCaption('d'+hallPage,who,text,pageTime-pause);}
      else if(line)hallCaption('c'+line[0],line[2],line[3],t-line[0]);else if(t>=55.78&&t<58.18)hallCaption('bye','sam','И никому не говори.',t-55.78);
      else{if(hallSpoken)_stopNpcVoice();hallSpoken='';el('supper-caption').hidden=true;}
      el('later-card').hidden=t<61.5;black.style.opacity=Math.max(1-phase(t,0,.28),phase(t,59.25,61.4));if(t>=hallEnd)finishHall();}
    function gatePose(t){const f=(a,b)=>phase(t,a,b),entry=f(18.56,19.96),shove=f(19.97,20.42),out=f(44.92,47.0),moveAside=f(43.8,45.35),depart=f(50.22,53.0),down=f(53.0,57.68),turn=f(57.62,59.02),down2=f(59.02,62.75);
      const A=[3.90+.94*f(45.98,47.0),0,8.55-.45*f(1,2.3)-.16*entry+.15*shove-1.71*out];const B=[3.91+.81*moveAside-.27*depart+2.4*turn,-1.50*down-1.50*down2,6.85+.87*entry-.96*shove-.91*moveAside-1.07*depart-3.35*down+3.35*down2];
      return {A,B,shove,out,down,turn,depart,gun:f(36.53,38.58)*(1-f(47.4,48.53)),reach:f(36.12,37.26)*(1-f(38.10,39.18)),push:f(19.67,20.07)*(1-f(20.37,20.99)),back:f(41.59,42.63)*(1-f(44.60,45.40))};}
    function drawHallProp(list,m,tint=[1,1,1],em=0){gl.uniformMatrix4fv(su.model,false,toGlMat4(m));gl.uniform3f(su.colorTint,...tint);for(const b of list){gl.bindTexture(gl.TEXTURE_2D,textures[b.texture]);gl.uniform1f(su.emissive,em||b.emissive||0);gl.bindVertexArray(b.vao);gl.drawArrays(gl.TRIANGLES,0,b.vertexCount);}gl.uniform1f(su.emissive,0);}
    function linePerformance(who,t,area){const dinner=area==='supper',dialog=area==='hall'&&inLandingTalk(),clock=dialog?pageTime:t;
      const row=dinner?supperLines.find(l=>t>=l[0]-.32&&t<l[1]):dialog?[hallPage===6?2.45:0,99,...landingLines[hallPage]]:area==='balcony'?[17.4,21.9,'partner','Ужин готов!']:gateLines.find(l=>t>=l[0]-.32&&t<l[1]);
      const age=row?clock-row[0]:-2,text=row?.[3]||'',speaker=row?.[2]===who,side=who==='hero'?1:-1;
      const env=(a,b,c,d)=>phase(age,a,b)*(1-phase(age,c,d)),end=text.length/(dinner?23:24),accent=text.search(/[,!?…]/),beat=accent>4?Math.min(2.7,accent/24):Math.min(.86,end*.53);
      return {side,seated:dinner,breath:Math.sin(clock*1.46+side*.93)*(.7+.3*Math.sin(clock*.52)),catch:speaker?env(-.32,-.04,.08,.29):env(.12,.35,.44,.88),offer:speaker?env(beat-.22,beat+.10,beat+.21,beat+.73):0,tail:speaker?env(beat,beat+.29,beat+.40,beat+.82):0,ack:speaker?.38*env(beat-.1,beat+.17,beat+.24,beat+.55):env(.31,.63,.72,1.13),release:env(end-.15,end+.20,end+.45,end+1.02)};}
    function performanceBeats(who,t){const hero=who==='hero',p=(x,a,b,c,d)=>phase(x,a,b)*(1-phase(x,c,d)),q=(a,b,c,d)=>p(t,a,b,c,d);
      if(inLandingTalk()){const age=pageTime-(hallPage===6?2.45:0),talk=landingLines[hallPage][0]===who,hold=p(age,.04,.5,1.2,2.12),after=p(age,1.25,1.65,2.10,2.89);return {ask:talk&&[0,1,4].includes(hallPage)?hold:0,stop:talk&&[3,6].includes(hallPage)?hold:0,chest:talk&&hallPage===2?hold:0,tight:hero?.65:.22,nod:talk?after:.55*hold};}
      return {ask:hero?q(4.9,5.6,6.3,7.3)+q(12,12.85,14.3,15.8):q(16.05,16.76,17.89,18.63),stop:hero?q(21,21.7,23.34,24.68)+q(32.3,33.36,34.8,36.10):q(26.39,27.04,27.8,28.7),chest:hero?q(30.6,31.31,31.85,32.63)+.7*q(42.1,42.76,43.36,44.39):q(25.02,25.64,26.23,26.93),tight:hero?q(36.53,37.13,41.36,42.38)+phase(t,51.3,53.1):0,nod:hero?q(5.2,5.49,5.60,6.0):q(7.64,7.96,8.14,8.67)};}
    function livingEyes(base,t,amp,seed){const stops=[[0,-.45,.09],[.69,.65,-.3],[1.11,.29,-.13],[1.93,-.91,.13],[2.5,-.25,.71],[3.68,.81,-.42],[4.32,-.51,-.08],[5.16,.36,.18],[5.81,-.45,.09]];
      const u=(Math.max(0,t)+seed)%6.73;let x=stops[0][1],y=stops[0][2];for(const [at,h,v] of stops){if(u<at)break;const f=phase(u,at,at+.092);x=lerp(x,h,f);y=lerp(y,v,f);}return [Math.tanh(base[0]+amp*x),Math.tanh(base[1]+amp*y)];}
    function drawThreshold(clock){drawStreetDoor();const t=hallT;const hinge=point(4.42,0,7.77);const open=t<0?0:phase(t,.52,2.12)*(1-phase(t,46.63,47.50));drawHallProp(landingGpu.entryLeaf,mat4Mul(mat4Translate(...hinge),mat4RotY(open*1.48)));if(t<0||t>61.6)return;
      const k=gatePose(t),a=point(...k.A),b=point(...k.B),walk=(t>=44.94&&t<47)||(t>=1&&t<2.30),samWalk=t>50.24||(t>18.55&&t<20.69)||(t>43.80&&t<45.35);
      const tracking=Math.atan2(b[0]-a[0],b[2]-a[2]),panic=phase(t,51.5,53.2),aYaw=tracking+panic*Math.atan2(Math.sin(-2.95-tracking),Math.cos(-2.95-tracking))+k.back*2.15;let sy=Math.atan2(a[0]-b[0],a[2]-b[2]);sy=lerp(sy,Math.PI,phase(t,50,51.5));sy-=Math.PI*phase(t,57.65,59.1);
      const line=inLandingTalk()?[hallPage===6?2.45:0,99,...landingLines[hallPage]]:gateLines.find(l=>t>=l[0]&&t<l[1])||(t>=55.78&&t<58.18?[55.78,58.18,'sam','И никому не говори.']:null);
      const mouth=name=>{if(!line||line[2]!==name)return '';const age=(inLandingTalk()?pageTime:t)-line[0],index=Math.floor(age*24),letter=line[3][index];if(!letter||/[мбп\s!?.,…]/i.test(letter))return '';return /[аяоэ]/i.test(letter)?'_talk2':'_talk1';};
      const ab=evaluateCharacterBones(charJoints,character.gender,a,aYaw,{elapsed:clock,onGround:true,horizSpeed:walk?.7:0,walkPhase:t*5.8,delivery:linePerformance('hero',t,'hall'),beats:performanceBeats('hero',t),entryPush:k.push,doorReach:phase(t,.1,.57)*(1-phase(t,1.35,1.9)),npcHeadLookYaw:k.back*.43});
      const sb=evaluateCharacterBones(charJoints,'male',b,sy,{elapsed:clock+.9,onGround:true,horizSpeed:samWalk?.94:0,walkPhase:t*6.1,delivery:linePerformance('sam',t,'hall'),beats:performanceBeats('sam',t),recoilBody:phase(t,19.96,20.23)*(1-phase(t,20.39,21.10)),samBeat:{glance:phase(t,34.90,35.23)*(1-phase(t,35.85,36.39)),follow:phase(t,35.04,35.61)*(1-phase(t,36.05,36.67)),shift:phase(t,35.78,36.67)*(1-phase(t,38.78,39.64)),settle:phase(t,38.20,38.66)*(1-phase(t,39.05,40.23))},pocketReach:k.reach,showProp:k.gun,npcHeadLookYaw:.64*phase(t,55.4,55.93)*(1-phase(t,57.1,57.72))});
      const handle=mat4TransformPt(mat4Mul(mat4Translate(...hinge),mat4RotY(open*1.48)),[-.98,1.04,.075]);contactArm(ab,character.gender,'r',handle,phase(t,.08,.49)*(1-phase(t,1.27,1.85)),[.48,-.48,-.73]);
      for(const side of ['l','r'])contactArm(ab,character.gender,side,mat4TransformPt(sb.torso,[side==='l'?-.14:.14,.28,.14]),k.push,[0,.25,-.97]);
      drawCharacterModel(character,ab,clock,mouth('hero'),[.49+.33*panic,0,.58*phase(t,2.1,3.3)*(1-phase(t,8,10))+.26*panic,0],livingEyes(mix3(eyeAim(ab,k.back>.4?chairPos:b.map((v,i)=>v+(i===1?1.65:0))),[0,.03],panic),inLandingTalk()?pageTime:t,.31+1.29*panic,.3));drawCharacterModel(sam,sb,clock+.91,mouth('sam'),[.20,0,.15,0],livingEyes(eyeAim(sb,a.map((v,i)=>v+(i===1?1.63:0))),inLandingTalk()?pageTime:t,.37,2.9));
      if(k.gun>.13){const m=mat4Mul(sb.hand_r,mat4Mul(mat4Translate(0,-.082,.022),mat4Mul(mat4RotX(Math.PI*.5+.08),mat4RotY(Math.PI))));for(const part of ['frame','slide','trigger','hammer','magazine'])drawWeaponBatches(weaponGpu.m1911[part],m);}}
    function hallCam(t){let e,v,f=45;const k=gatePose(t),A=k.A,B=k.B,head=(p,h=1.57)=>[p[0],p[1]+h,p[2]],mv=(a,b,l,r)=>mix3(a,b,phase(t,l,r));
      if(t<2.32){e=mv([3.48,1.57,9.24],[3.64,1.57,8.82],0,2.32);v=mv([3.44,1.26,7.77],head(B),.32,2.2);}
      else if(t<5.20||t>=7.73&&t<12.17||t>=16.36&&t<18.60||t>=25.36&&t<30.88){e=mv([3.62,1.65,8.54],[3.70,1.63,8.40],2.32,5.1);v=head(B);f=37;}
      else if(t<7.73||t>=12.17&&t<16.36||t>=21.27&&t<25.36||t>=30.88&&t<36.22){e=[4.19,1.65,6.92];v=head(A);f=42;}
      else if(t<21.27){e=[4.09,1.57,9.74];v=[3.90,1.28,7.64];f=53;}
      else if(t<39){e=mv([3.28,1.22,7.41],[3.19,1.24,7.56],36.2,38.5);v=mv(head(B,1.35),[B[0]+.02,1.02,B[2]+.25],36.2,38.5);f=44;}
      else if(t<41.56){e=[3.62,1.54,8.45];v=head(B);f=36;}
      else if(t<44.88){e=[3.39,1.52,7.19];v=mv(head(A),[6.94,1.22,10.35],42.9,44.38);f=51;}
      else if(t<49){e=mv([6.47,1.58,5.42],[6.21,1.57,5.59],44.88,49);v=[4.22,1.40,7.38];f=56;}
      else if(inLandingTalk()){const hero=landingLines[hallPage][0]==='hero';e=hero?[4.03,1.64,5.59]:[4.18,1.59,7.23];const dolly=.074*phase(pageTime,.16,2.1);v=head(hero?A:B);e=mix3(e,v,dolly);f=41-2*phase(pageTime,.16,2.1);}
      else if(t<53.0){e=[6.57,1.68,7.36];v=head(B,1.16);f=55;}
      else{const push=phase(t,53,59.3);e=[A[0]-.29+.05*push,1.64,A[2]-1.04+.18*push];v=head(A,character.gender==='female'?1.53:1.65);f=32-4*push;}
      const eye=point(...e);return {eye,fwd:normalize(vec3Sub(point(...v),eye)),fov:f};}
    function validSave(s){return s?.version===1 && s.chapter===0 && ['cinema','home'].includes(s.phase) && Number.isFinite(s.time) && s.character && ['male','female'].includes(s.character.gender) && Array.isArray(s.pos) && s.pos.length===3 && s.pos.every(Number.isFinite) && Number.isFinite(s.yaw)&&Number.isFinite(s.pitch);}
    try{const parsed=JSON.parse(localStorage.getItem(storyKey)); if(validSave(parsed))lastSlot=parsed;}catch(_){}
    function slotLabel(){el('story-continue').hidden=!lastSlot;el('story-continue').textContent='Продолжить · '+(lastSlot?.awake?'Мотель':lastSlot?.dreamT>=0?'Кошмар':lastSlot?.streetStage>=3?'Двор':lastSlot?.hallDone||lastSlot?.hallTime>=0?'Подъезд':lastSlot?.phase==='home'?'Квартира':'Пролог');el('story-new').textContent=lastSlot?'Пролог с начала':'Начать кампанию · Пролог';}
    for(const chapter of chapters){const tile=document.createElement('span');tile.textContent=String(chapter.id).padStart(2,'0');tile.title=chapter.available?'Пока горят окна':'Глава пока закрыта';el('chapter-grid').appendChild(tile);}
    slotLabel();
    function storeStory(force=false){if(!storyActive())return;
      const s={version:1,cut:3,partnerSeed:familySeed,leaves:interiorDoors.map(d=>[d.p,d.target]),streetRoute:2,streetStage,dreamT,dreamYaw,awake,hallTime:hallT,hallDone,hallPage,pageTime,supperTime:supperT,supperDone,chapter:0,phase:mode,time:storyTime,pos:mode==='home'?[...player.pos]:[...roomSpawn],yaw:player.yaw,pitch:player.pitch,character:{...character}};
      const raw=JSON.stringify(s);if(!force&&raw===storyStamp)return;
      try{localStorage.setItem(storyKey,raw);saveOK=true;}catch(_){saveOK=false;}
      lastSlot=s;storyStamp=raw;slotLabel();
      if(mode==='home'&&dreamT<0&&hallT<0&&supperT<0)storyNote.textContent=awake?'Мотель «Маяк» · Номер 07   /   WASD — идти':streetStage===5?'За углом · Двор   /   WASD — идти':streetStage>=3?'Двор   /   WASD — идти':hallDone?'Первый этаж · Как добраться до полиции, не встретив Сэма?   /   WASD — идти':(saveOK?'Сохранено на устройстве':'Память браузера недоступна · только до закрытия')+'  ·  WASD — идти   ·   мышь / стрелки — обзор';
      else if(!saveOK)storyNote.textContent='Сохранение доступно только до закрытия вкладки';
    }
    function stopCall(){_stopNpcVoice();lastBlip=-1;npcDlgEl.classList.remove('open','typing');}
    // One short recorded NPC sample per syllable; a comma or breath stays silent.
    function displayDinnerCall(text,t){
      const beats=[0,.105,.205,.31,.42,.54,.64, .99,1.14,1.28,1.39,1.50,1.61,1.73,1.84,1.96,2.08,2.19,2.30,2.42,2.52];
      const count=Math.min(text.length,beats.filter(v=>v<=t*(character.gender==='male'?1.85:1.4)).length);
      dlgTextSpanEl.textContent=text.slice(0,count);
      npcDlgEl.classList.toggle('typing',count<text.length);
      const vowels=/[аеёиоуыэюя]/i;
      if(count>lastBlip){const chars=text.slice(Math.max(0,lastBlip),count);lastBlip=count;
        if(vowels.test(chars)||(character.gender==='male'&&/[джгт]/i.test(chars))){const i=(count*5+(character.gender==='female'?3:1))%8;
          if(character.gender==='male') _playNpcVoiceMono('spouse_toriel',{gain:.79,cutoff:2450});
          else _playNpcVoiceMono('npc_voice_'+i,{rate:.89,gain:.63,cutoff:1640});}}
    }
    function finishFilm(saved){stopCall();mode='home';storyTime=duration;player.reset();let pos=saved?.pos||roomSpawn;
      const [x,y,z]=pos;const inFlat=x>=-12.82&&x<=-7.12&&z>=21.12&&z<=28.61,door=x>=-8.45&&x<=-7.22&&z>=28.45&&z<=28.90,kitchen=x>=-9.82&&x<=-1.55&&z>=28.68&&z<=33.67;
      if(awake){const wc=x>=-2.48&&x<=.29&&z>=23.08&&z<=26.43;if(y< -27.88||y> -26.30||!wc&&(x< -6.64||x> -2.16||z<22.20||z>26.58))pos=motelSpawn;}else if(hallDone){const pavement=streetStage>=2&&y>=.11&&y<.65&&((x>=-1.92&&x<=.4&&z>=18.63&&z<=31.16)||(x>=-13.6&&x<=.4&&z>=18.62&&z<=21.22)||(x>=-3.85&&x<=.43&&z>=-7.99&&z<18.64)||(x>=-13.20&&x<=.43&&z>=-7.99&&z<=-.54));if(!pavement&&(x< -6.94||x> -1.58||z<21.24||z>28.68||y<.16||y>13.60))pos=groundSpawn;}else if(!(inFlat||door||kitchen||(x>=-17.54&&x<=-9.58&&z>=28.78&&z<=33.72))||y<12.17||y>13.7)pos=roomSpawn;
      if(awake){pos=[pos[0],Math.max(-27.80,pos[1]),pos[2]];if(player.collidesAny(player.getAABB(...pos),scene.prologue.colliders))pos=pos[0]>-2.11?[-1.19,-27.8,24.18]:motelSpawn;}
      if(!awake&&!hallDone&&x<-9.95&&z>31.1&&player.collidesAny(player.getAABB(...pos),scene.prologue.colliders))pos=[x< -13.74?-15.43:-12.09,12.201,30.53];
      player.pos=[...pos];player.smoothFootY=pos[1];player.yaw=player.targetYaw=saved?.yaw??Math.PI;player.pitch=player.targetPitch=saved?.pitch??-.045;
      storyUI.classList.remove('cinema');el('story-chapter').hidden=true;black.style.opacity=0;el('story-skip').hidden=true;el('story-save').hidden=false;keysDown.clear();canvas.focus();storeStory(true);}
    function beginStory(saved=null){ getAudioCtx();ensureWeaponWavsLoaded();mp.disconnect(true);testNpc.close(true);closeSettings();
      interiorDoors.forEach((d,i)=>{const v=saved?.leaves?.[i];d.p=Array.isArray(v)&&Number.isFinite(v[0])?clamp01(v[0]):0;d.target=v?.[1]===1?1:0;});el('inner-door').hidden=true;cutSound();awake=saved?.awake===true;dreamT=-1;el('motel-caption').hidden=true;streetStage=saved?.hallDone?Math.min(5,Math.max(0,Math.trunc(saved.streetStage)||0)):0;if(saved?.streetRoute!==2&&streetStage>=3)streetStage=saved.pos[0]<EXIT_X+.68&&saved.pos[2]>26.4&&saved.pos[2]<27.7?2:3;breachTime=rattleTime=2;streetImpact=true;jamText=streetStage===1?5:0;closeStreetUI();hallT=-1;hallDone=saved?.hallDone===true;hallPage=Math.max(0,Math.min(6,Math.trunc(saved?.hallPage)||0));pageTime=Math.max(0,Number(saved?.pageTime)||0);hideLandingUI();supperT=-1;supperDone=!!saved?.supperDone;resetSupCaption();if(saved)character=clampChar({...DEFAULT_CHARACTER,...saved.character});
      initPartner(saved);mode='cinema';storyLife=createMenuLifeController(true); keysDown.clear();dragging=false;syncUiVisibility();storyUI.hidden=false;storyUI.classList.add('cinema');el('story-save').hidden=true;el('story-skip').hidden=false;
      storyTime=saved?Math.max(0,Math.min(duration,saved.cut===3?saved.time:(saved.cut===2?Math.min(saved.time,24.65)+Math.max(0,saved.time-24.65)*(duration-24.65)/(35-24.65):saved.time*duration/26))):0;autoAt=0;releaseMatrix=null;stopCall();storyLife.stageStreet(storyTime);black.style.opacity=1;storyNote.textContent='ДЕКАБРЬ 2007   /   ПОСЛЕ ОДИННАДЦАТИ';canvas.focus();
      if(saved?.phase==='home'){finishFilm(saved);if(!awake&&streetStage===5){startDream(Math.min(dreamEnd-.01,Math.max(0,Number(saved.dreamT)||0)),saved.dreamYaw??player.yaw);return;}if(!hallDone&&supperDone&&Number.isFinite(saved.hallTime)&&saved.hallTime>=0)startHall(Math.min(hallEnd-.01,saved.hallTime));else if(!supperDone&&Number.isFinite(saved.supperTime)&&saved.supperTime>=0)startSupper(Math.min(supperEnd-.01,saved.supperTime));return;}

      storeStory(true);
    }
    function newStory(){if(lastSlot && performance.now()>restartUntil){restartUntil=performance.now()+4500;el('story-new').textContent='Заменить сохранение? Нажать ещё раз';setTimeout(()=>{if(!storyActive())slotLabel()},4550);return;}restartUntil=0;beginStory();}
    function leaveStory(){showCase(false);storeStory(true);cutSound();dreamT=-1;el('motel-caption').hidden=true;hallT=-1;hideLandingUI();supperT=-1;resetSupCaption();stopCall();storyUI.hidden=true;storyLife=null;enterMenu(false);mainMenuEl.classList.add('menu-ready');slotLabel();}
    el('story-new').onclick=newStory;el('story-continue').onclick=()=>{if(lastSlot)beginStory(lastSlot)};el('story-exit').onclick=leaveStory;el('story-save').onclick=()=>storeStory(true);el('story-skip').onclick=()=>dreamT>=0?finishDream():hallT>=0?finishHall():supperT>=0?endSupper():finishFilm(null);
    document.addEventListener('visibilitychange',()=>{if(document.hidden){storeStory(true);keysDown.clear();}});window.addEventListener('pagehide',()=>storeStory(true));
    const point=(x,y,z)=>[-10+x,12.2+y,21+z], mix3=(a,b,u)=>a.map((v,i)=>lerp(v,b[i],u));
    const phase=(t,a,b)=>smooth01(clamp01((t-a)/(b-a)));
    function actor(t){return {pos:point(0,0,.54),yaw:Math.PI-.32*phase(t,23.9,24.65)};}
    // Shot / reverse shot: the street, the same light from above, the listener and the threshold.
    function movieCam(t){let eye,aim,fov;
      const move=(a,b,start,end)=>mix3(a,b,phase(t,start,end));
      if(t<4.6){ // Roofline dissolves out of black. Drift down to the laundry green.
        eye=move([8.1,10.8,18],[6.2,8.9,16.7],0,4.6);
        aim=move([-16,10,-9],[-15,3.35,-5.5],0,4.6);fov=57;
      }else if(t<7.8){ // Cut on the approaching headlights, below the pedestrian's eye line.
        eye=move([-1.17,.92,8.9],[-1.04,1.12,9.5],4.6,7.8);
        aim=move([1.64,.9,-5.4],[2.15,1.25,6.9],4.6,7.8);fov=58;
      }else if(t<10.6){ // A lit window high over the lane. Match to the reverse view inside it.
        const n=neighbours[0].pos;eye=move([n[0]+3.3,n[1]+2.18,n[2]+6.8],[n[0]+2.78,n[1]+2.05,n[2]+6.0],7.8,10.6);aim=[n[0]-.32,n[1]+1.13,n[2]];fov=43;
      }else if(t<14.1){ // Over the shoulder; keep the warm windows in the outlined panes.
        eye=move(point(-1.06,1.57,1.66),point(-.78,1.52,1.46),10.6,14.1);aim=point(.29,1.43,-1.05);fov=54;
      }else if(t<16.4){ // Insert: empties, worn sill, falling ash. The kitchen call follows the quiet beat.
        eye=move(point(-.75,.63,1.31),point(-.85,.58,1.23),14.1,16.4);aim=point(-1.57,.19,.42);fov=48;
      }else if(t<21.9){ // Reaction in profile. Hold the frame while the viewer reads.
        eye=move(point(1.37,1.50,.15),point(1.19,1.50,.22),16.4,21.9);aim=point(-.06,1.44,.56);fov=52;
      }else if(t<24.65){ // Cut on movement. Pan past the hand towards the ember's arc.
        const u=phase(t,23.13,24.35);eye=move(point(-.79,1.33,1.26),point(-.77,1.43,.99),21.9,24.65);
        aim=mix3(point(.16,1.15,.45),point(.42,1.12,-1.6),u);fov=48+4*u;
      }else if(t<28){ // Ellipsis: tea, remote and today's paper; the crossing is heard, never shown.
        eye=move(point(.89,1.18,4.89),point(1.07,1.09,4.73),24.65,28);aim=point(2.04,.59,3.95);fov=45;
      }else{const u=phase(t,28,duration),y=player.cfg.stand_eye_height;
        eye=move(point(.02,y-.025,3.05),point(0,y,3.12),28,duration);
        aim=mix3(point(-.32,1.29,7.13),[eye[0],eye[1]-.18,eye[2]+4],u);fov=62+6*u;
      }
      return {eye,fwd:normalize(vec3Sub(aim,eye)),fov};
    }
    function tickStory(delta){if(!el('case-panel').hidden)return;if(document.hidden)return; autoAt+=delta;
      if(mode==='cinema'){const prev=storyTime;storyTime=Math.min(duration,storyTime+delta);const t=storyTime;
        if([24.81,25.41,26.07,26.81].some(cue=>prev<cue && t>=cue))playFootstepSfx('crouch');
        black.style.opacity=Math.max(1-phase(t,.35,2.1),1-phase(Math.abs(t-24.65),0,.25),.65*(1-phase(Math.abs(t-28),0,.22)),.64*(1-phase(Math.abs(t-10.6),0,.34)));el('story-chapter').hidden=t<1.25||t>5.8;
        el('story-chapter').style.opacity=Math.min(phase(t,1.25,2.2),1-phase(t,4.75,5.8));
        const talk=t>=17.4&&t<21.9;npcDlgEl.classList.toggle('open',talk);
        if(talk){const text=character.gender==='female'?'Дорогая! Ужин готов!':'Дорогой! Ужин готов!';dlgSpeakerNameEl.textContent='Голос из кухни';dlgSpeakerRoleEl.textContent='· ЗА ДВЕРЬЮ';dlgCounterEl.textContent='01 / 01';dlgHintEl.textContent='';displayDinnerCall(text,t-17.4);}
        if(t>=duration)finishFilm(null);
      }else if(dreamT>=0){updateDream(delta);}else if(hallT>=0){updateHall(delta);}else if(supperT>=0){updateSupper(delta);}else{const dt=Math.min(.05,delta);player.targetYaw+=((keysDown.has('ArrowRight')?1:0)-(keysDown.has('ArrowLeft')?1:0))*dt*1.7;player.targetPitch=Math.max(-1.45,Math.min(1.45,player.targetPitch+((keysDown.has('ArrowUp')?1:0)-(keysDown.has('ArrowDown')?1:0))*dt));
        player.update(dt,{forward:.53*(+isActionDown('forward')-isActionDown('backward')),strafe:.53*(+isActionDown('right')-isActionDown('left')),crouch:isActionDown('crouch'),jump:false,sprint:false,slidePress:false,kick:false},interiorPhysics(dt,hallDone?routeSolids[+(streetStage>=2&&breachTime>=.4)]:scene.prologue.colliders));tickStreet(delta);el('chair-prompt').hidden=!nearGuestSeat();el('entry-prompt').hidden=!atEntry();doorHint();}
      if(autoAt>=2.5){autoAt=0;storeStory();}}

    if(lastSlot)el('btn-char-play').textContent='Сохранить и вернуться в меню';
    // First launch: open interactive 3D Character Creator! Returning players go to 3D Main Menu.
    banner.style.display = "none";
    updateQuickBarButtons();
    if (!hasSavedCharacter) {
      enterCreator(true);
    } else {
      enterMenu();
    }

    // Reusable light uniform buffers
    const lPosArr = new Float32Array(MAX_LIGHTS * 3);
    const lColArr = new Float32Array(MAX_LIGHTS * 3);
    const lIntArr = new Float32Array(MAX_LIGHTS);
    const lRadArr = new Float32Array(MAX_LIGHTS);
    const idMat4 = toGlMat4(mat4Identity());

    let lastTime = performance.now();
    let fpsSmooth = 60;
    let hudTimer = 0;

    function computeThirdPersonCamera() {
      const anchor = [player.pos[0], player.pos[1] + (player.crouching || player.sliding ? 1.05 : 1.46), player.pos[2]];
      const fwd = player.forward;
      const right = normalize(vec3Cross(fwd, [0, 1, 0]));
      // Slightly over-the-right-shoulder cinematic PSX 3rd-person offset
      const desired = [
        anchor[0] - fwd[0] * 2.55 + right[0] * 0.28,
        anchor[1] - fwd[1] * 2.55 + 0.22,
        anchor[2] - fwd[2] * 2.55 + right[2] * 0.28,
      ];
      const delta = vec3Sub(desired, anchor);
      const dist = Math.hypot(delta[0], delta[1], delta[2]);
      let camPos = desired;
      if (dist > 0.05) {
        const dir = [delta[0] / dist, delta[1] / dist, delta[2] / dist];
        let bestT = dist;
        for (const box of scene.colliders) {
          const hit = raycastAABB(anchor, dir, box, dist);
          if (hit !== null && hit < bestT) bestT = hit;
        }
        const safeT = Math.max(0.35, bestT - 0.18);
        camPos = [anchor[0] + dir[0] * safeT, anchor[1] + dir[1] * safeT, anchor[2] + dir[2] * safeT];
      }
      // Look forward along aim direction from anchor
      const lookTarget = [anchor[0] + fwd[0] * 8.0, anchor[1] + fwd[1] * 8.0, anchor[2] + fwd[2] * 8.0];
      const camFwd = normalize(vec3Sub(lookTarget, camPos));
      return { eye: camPos, fwd: camFwd, up: [0, 1, 0] };
    }

    function frame(now) {
      requestAnimationFrame(frame);
      try {
      const realDt = Math.min(.35,Math.max(0,(now-lastTime)/1000));
      const dt = Math.min(realDt,.05);
      if(storyActive())tickStory(realDt);
      lastTime = now;
      const elapsed = now / 1000;
      if (dt > 0.0001) fpsSmooth += (1.0 / dt - fpsSmooth) * 0.12;

      if (mode === "creator") {
        const k = 1.0 - Math.exp(-dt * 10.0);
        creatorYawOffset += (creatorTargetYawOffset - creatorYawOffset) * k;
        creatorZoom += (creatorZoomTarget - creatorZoom) * k;
      } else if (mode === "menu") {
        // Calm, subtle cursor-watching camera parallax in the 3D night courtyard menu
        const smooth = 1.0 - Math.exp(-dt * 3.8);
        menuCursorX += (menuTargetX - menuCursorX) * smooth;
        menuCursorY += (menuTargetY - menuCursorY) * smooth;
      } else if (mode === "playing") {
        // Optional smooth arrow-key look fallback (paused while typing in chat)
        const turnSpeed = 2.2 * dt;
        const maxP = 85 * Math.PI / 180;
        if (!chatOpen) {
          if (keysDown.has("ArrowLeft")) player.targetYaw -= turnSpeed;
          if (keysDown.has("ArrowRight")) player.targetYaw += turnSpeed;
          if (keysDown.has("ArrowUp")) player.targetPitch = Math.min(maxP, player.targetPitch + turnSpeed * 0.7);
          if (keysDown.has("ArrowDown")) player.targetPitch = Math.max(-maxP, player.targetPitch - turnSpeed * 0.7);
        }

        const inp = chatOpen ? {
          forward: 0, strafe: 0, jump: false, sprint: false, crouch: false, slidePress: false, kick: false,
        } : {
          forward: (isActionDown("forward") ? 1 : 0) - (isActionDown("backward") ? 1 : 0),
          strafe: (isActionDown("right") ? 1 : 0) - (isActionDown("left") ? 1 : 0),
          jump: isActionDown("jump") && !testNpc.active,
          sprint: isActionDown("sprint"),
          crouch: isActionDown("crouch"),
          slidePress: slideQueued,
          kick: kickQueued && !testNpc.active,
        };
        const didKickPress = kickQueued && !testNpc.active && !chatOpen;
        kickQueued = false;
        slideQueued = false;
        const prevOnGroundW = player.onGround;
        const prevFallSpdW = player.vel[1];
        const doorColNow = doorCtrl.getActiveCollider();
        const activeColliders = doorColNow ? scene.colliders.concat([doorColNow]) : scene.colliders;
        player.update(dt, inp, activeColliders);
        testNpc.update(dt, player.pos);

        // v14: Landing & Power-Slide 3D dust / spark VFX
        if (!prevOnGroundW && player.onGround && Math.abs(prevFallSpdW) > 1.8) {
          weapon.spawnLandingFx(player.pos, Math.abs(prevFallSpdW));
        }
        if (player.sliding && player.onGround) {
          weapon._slideFxAccum += dt;
          if (weapon._slideFxAccum >= 0.045) {
            weapon._slideFxAccum = 0.0;
            weapon.spawnSlideFx(player.pos, player.vel);
          }
        } else {
          weapon._slideFxAccum = 0.0;
        }

        // ---- Interactive 3D Tactical Door (E tap open/close, Hold E stealth peek, V kick breach!) ----
        const nearNpcPre = testNpc.canInteract(player.pos);
        const nearWpPre = weapon.nearestPickup(player.pos);
        const doorCanUse = !chatOpen && !testNpc.active && !nearNpcPre && !nearWpPre && doorCtrl.canInteract(player.pos);
        if (didKickPress) {
          if (doorCtrl.tryKick(player.eyePos, player.forward)) {
            weapon.spawnDoorBreachFx(doorCtrl.centerPos(), player.forward);
          }
        }
        doorCtrl.handleEInput(dt, isActionDown("pickup") && doorCanUse, doorCanUse);
        if(mp.connected&&(doorCtrl.sfxQueue.length||doorCtrl.quietActive&&Math.abs(mp.lastDoor-doorCtrl.targetAngle)>.025)){mp.lastDoor=doorCtrl.targetAngle;mp.sendJson({type:'door',target:mp.lastDoor,sound:doorCtrl.sfxQueue[0]||''});}
        doorCtrl.update(dt, player.pos);
        for (const devD of doorCtrl.popSfx()) {
          playWeaponEvent(devD);
          if (devD === "door_open" || devD === "door_close") {
            weapon.spawnDoorDustFx(doorCtrl.centerPos());
          }
        }

        // ---- Weapons, Bodycam Mag Check ([Hold R]), Suppressor ([P]), Firing & Dynamic Camera Recoil ----
        const horizSpdW = Math.hypot(player.vel[0], player.vel[2]);
        const sprintingW = inp.sprint && horizSpdW > 5.0 && !player.crouching && !player.sliding;
        const vm2wCur = mat4CameraToWorld(player.eyePos, player.forward, player.rolledUp);
        const rHeldNow = !chatOpen && !testNpc.active && isActionDown("reload");
        if (rHoldActive && rHeldNow) {
          rHoldTimer += dt;
          if (!rMagCheckFired && rHoldTimer >= 0.26) {
            rMagCheckFired = true;
            for (const ev of weapon.startMagCheck()) playWeaponEvent(ev);
          }
        } else if (rHoldActive && !rHeldNow) {
          if (!rMagCheckFired) reloadQueued = true;
          rHoldActive = false;
          rHoldTimer = 0.0;
          rMagCheckFired = false;
        }
        if (!chatOpen && pickupQueued) {
          pickupQueued = false;
          if (testNpc.canInteract(player.pos)) {
            testNpc.startOrAdvance();
          } else if(mp.connected&&nearWpPre){if(weapon.ownedWeapons().length<2)mp.sendJson({type:'take',weapon:nearWpPre.weaponId});
          } else if (nearWpPre && weapon.tryPickup(player.pos)) {
            updateQuickBarButtons();
            updateAmmoHud(true);
          }
        }
        if (!chatOpen && dropQueued) {
          dropQueued = false;
          if (weapon.tryDropWeapon(player.eyePos, player.forward, activeColliders)) {
            if(mp.connected){const dropped=weapon.worldPickups.pop();mp.sendJson({type:'drop',weapon:dropped.weaponId,ammo:dropped.magAmmo,reserve:dropped.reserveAmmo,spent:dropped.spentShells,locked:dropped.slideLocked});}
            updateQuickBarButtons();
            updateAmmoHud(true);
          }
        }
        if (!chatOpen && reloadQueued) {
          weapon.startReload();
          reloadQueued = false;
        }
        const autoFireNow = !chatOpen && lmbHeld && weapon.isFullAuto && !dragMoved;
        if (!chatOpen && (fireQueued || autoFireNow)) {
          const isAutoHold = !fireQueued;
          fireQueued = false;
          const firedOk = weapon.tryFire(player.eyePos, player.forward, vm2wCur, activeColliders, isAutoHold);
          if (firedOk && mp.connected) {
            mp.sendShootAndHitscan(player.eyePos, weapon.lastShotDir || player.forward);
          }
        } else if (chatOpen) {
          fireQueued = false;
        }
        const zoomSens = 1.0 - 0.66 * (weapon.scopeAmount || 0.0);
        weapon.update(
          dt,
          chatOpen ? 0.0 : frameMouseDx * zoomSens,
          chatOpen ? 0.0 : frameMouseDy * zoomSens,
          horizSpdW,
          player.onGround,
          sprintingW,
          !chatOpen && adsDown && !thirdPerson,
          player.crouching,
          player.sliding,
          player.kick.active,
          vm2wCur,
          activeColliders,
          player.eyePos,
          player.forward,
          !chatOpen && isActionDown("inspect"),
          player.vel[1],
          rHeldNow
        );
        // Update interactive door HUD prompt & Bodycam mag-check badge
        if (doorPromptEl) {
          const showDoorUi = !chatOpen && !testNpc.active && !nearNpcPre && !nearWpPre && (doorCtrl.nearPlayer || doorCtrl.quietActive);
          doorPromptEl.style.display = showDoorUi ? "block" : "none";
          if (showDoorUi) {
            const pct = Math.round(Math.max(0, Math.min(1, doorCtrl.angle / Math.max(doorCtrl.openAngle, 1e-4))) * 100);
            if (doorCtrl.quietActive) {
              doorPromptMainEl.textContent = `ТИХОЕ ПРИОТКРЫВАНИЕ ДВЕРИ  ${pct}%`;
              doorPromptMainEl.style.color = "#8af6ba";
              doorPromptSubEl.textContent = "ОТПУСТИ [E] ЧТОБЫ ЗАФИКСИРОВАТЬ УГОЛ";
              doorQuietBarWrapEl.style.display = "block";
              doorQuietBarEl.style.width = `${pct}%`;
            } else {
              const verb = (doorCtrl.angle < 0.62) ? "ОТКРЫТЬ" : "ЗАКРЫТЬ";
              doorPromptMainEl.textContent = `[E] ${verb} ДВЕРЬ   |   [ЗАЖАТЬ E] ТИХО`;
              doorPromptMainEl.style.color = "#f2e4a6";
              doorPromptSubEl.textContent = (pct < 65) ? "[V] ВЫБИТЬ НОГОЙ (БРИЧ)" : `ОТКРЫТА НА ${pct}% · [E] ЗАКРЫТЬ`;
              doorQuietBarWrapEl.style.display = "none";
            }
          }
        }
        if (magCheckBadgeEl) {
          const mcTxt = weapon.magCheckLabel || "";
          if (mcTxt && !testNpc.active) {
            magCheckBadgeEl.style.display = "block";
            magCheckBadgeEl.textContent = "● " + mcTxt;
            const isEmp = mcTxt.includes("ПУСТ");
            magCheckBadgeEl.style.color = isEmp ? "#ff8670" : "#a4f4ba";
            magCheckBadgeEl.style.borderColor = isEmp ? "rgba(240,118,98,0.88)" : "rgba(126,214,152,0.88)";
          } else {
            magCheckBadgeEl.style.display = "none";
          }
        }
        const [ckPitch, ckRoll] = weapon.consumeCamKick();
        if (Math.abs(ckPitch) > 1e-6 || Math.abs(ckRoll) > 1e-6) {
          const maxP2 = 85 * Math.PI / 180;
          player.targetPitch = Math.max(-maxP2, Math.min(maxP2, player.targetPitch + ckPitch));
          player.pitch = Math.max(-maxP2, Math.min(maxP2, player.pitch + ckPitch));
          player.targetYaw += ckRoll;
          player.yaw += ckRoll;
        }
        frameMouseDx = 0.0;
        frameMouseDy = 0.0;
        for (const ev of weapon.popSfxEvents()) playWeaponEvent(ev);
        updateAmmoHud(false);
        updatePickupPrompt();
        mp.update(dt, elapsed);
        renderChatFeed(false);
      }

      if(mode==='paused'&&mp.connected)mp.update(dt,elapsed);
      // Update HUD text ~5x/sec (only during gameplay/pause, never in 3D menu or creator)
      hudTimer += dt;
      const inStory=storyActive();
      const inCourtyard = inStory || (mode === "menu" || mode === "creator");
      if (!menuLife) {
        menuLife = createMenuLifeController();
        window.__menuLife = menuLife;
      }
      // Persistent living Winter Menu Street stays alive ABOVE the test room even during gameplay!
      // Passing traffic cars are COMPLETELY FROZEN while customizing in Character Creator (mode === "creator")!
      const life = inStory ? storyLife : menuLife;
      const menuHeroBones = life.update(
        (mode === "paused") ? 0.0 : dt,
        elapsed,
        character,
        mode !== "creator",
        mode !== "creator"
      );
      if ((mode === "menu" || (mode === "cinema" && storyTime<11)) && life.carPassEvent) {
        playWeaponEvent(life.carPassEvent === "near" ? "menu_car_pass_near" : "menu_car_pass");
      }
      if (!inCourtyard && showHud && hudTimer >= 0.12) {
        hudTimer = 0;
        const locked = document.pointerLockElement === canvas;
        hudMain.textContent = `RES: ${lowW}x${lowH}   FPS: ${Math.round(fpsSmooth)}   CAM[F5]: ${thirdPerson ? "3RD" : "1ST"}`;
        const kFlHud = formatKeyCode(userSettings.binds.flashlight);
        hudFlags.textContent =
          `DITHER[B]: ${preset.dither_enabled ? "ON" : "OFF"} (${Math.round(preset.color_levels)}lvl)  ` +
          `JITTER[J]: ${preset.vertex_jitter ? "ON" : "OFF"}  ` +
          `CRT[T]: ${preset.scanlines ? "ON" : "OFF"}  ` +
          `LIGHT[${kFlHud}]: ${player.flashlightOn ? "ON" : "OFF"}`;
        const badges = [];
        if (player.noclip) badges.push(">> NOCLIP ON (Space вверх / Ctrl вниз / Shift быстро · sudo noclip off) <<");
        else if (player.sliding) badges.push(">> SLIDING <<");
        else if (player.crouching) badges.push(">> CROUCHING <<");
        if (!weapon.hasWeapon && weapon.worldPickups.length > 0) {
          badges.push("8 ВИДОВ ОРУЖИЯ НА АРЕНЕ (ИЩИ БЛИКИ · [E] ПОДОБРАТЬ · [1..8/Q])");
        }
        if (mp.connected) {
          badges.push(`ОНЛАЙН: ${mp.roomName} (${mp.remotes.size + 1} игр.) | HP: ${mp.hp}`);
        }
        if (mode === "playing" && !locked && !chatOpen) badges.push("ОБЗОР: кликни по экрану для захвата мыши (или зажми ЛКМ / стрелки)");
        hudBadge.textContent = badges.join("   ");
      }

      // ---------------- PASS 1: 3D Scene (Creator / Main Menu Courtyard OR Arena + Persistent Menu Street ABOVE!) ----------------
      const MENU_ELEV_Y = Number(menuCfg.elevationY ?? 12.0);
      const mMenuElev = mat4Translate(0.0, MENU_ELEV_Y, 0.0);
      const glMenuElev = toGlMat4(mMenuElev);

      let eye, fwd, up, projMat;
      const aspect = lowW / Math.max(lowH, 1);

      if(inStory){const cam=mode==='cinema'?movieCam(storyTime):dreamT>=0?nightmareCam():hallT>=0?hallCam(hallT):supperT>=0?supperCam(supperT):streetCam();eye=cam.eye;fwd=cam.fwd;up=[0,1,0];projMat=mat4Perspective(cam.fov,aspect,.045,230);}
      else if (mode === "creator") {
        const z = creatorZoom;
        eye = [
          lerp(CREATOR_BODY_CAM[0], CREATOR_FACE_CAM[0], z),
          lerp(CREATOR_BODY_CAM[1], CREATOR_FACE_CAM[1], z),
          lerp(CREATOR_BODY_CAM[2], CREATOR_FACE_CAM[2], z),
        ];
        const tgt = [
          lerp(CREATOR_BODY_TGT[0], CREATOR_FACE_TGT[0], z),
          lerp(CREATOR_BODY_TGT[1], CREATOR_FACE_TGT[1], z),
          lerp(CREATOR_BODY_TGT[2], CREATOR_FACE_TGT[2], z),
        ];
        fwd = normalize(vec3Sub(tgt, eye));
        up = [0, 1, 0];
        projMat = mat4Perspective(lerp(50.0, 44.0, z), aspect, 0.1, menuCfg.farPlane ?? 230.0);
      } else if (mode === "menu") {
        const breatheX = Math.sin(elapsed * 0.42) * 0.032;
        const breatheY = Math.cos(elapsed * 0.58) * 0.022;
        eye = [
          menuCfg.baseEye[0] + menuCursorX * menuCfg.cursorPosXAmp + breatheX,
          menuCfg.baseEye[1] + menuCursorY * menuCfg.cursorPosYAmp + breatheY,
          menuCfg.baseEye[2] - menuCursorX * (menuCfg.cursorPosZAmp ?? 0.08),
        ];
        const yaw = menuCfg.baseYaw + menuCursorX * menuCfg.cursorYawAmp;
        const pitch = menuCfg.basePitch + menuCursorY * menuCfg.cursorPitchAmp;
        const roll = menuCfg.baseRoll - menuCursorX * (menuCfg.cursorRollAmp ?? 0.009);
        const cp = Math.cos(pitch);
        fwd = [Math.sin(yaw) * cp, Math.sin(pitch), -Math.cos(yaw) * cp];
        const br = normalize(vec3Cross(fwd, [0, 1, 0]));
        const bu = vec3Cross(br, fwd);
        const cr = Math.cos(roll), sr = Math.sin(roll);
        up = [bu[0]*cr + br[0]*sr, bu[1]*cr + br[1]*sr, bu[2]*cr + br[2]*sr];
        projMat = mat4Perspective(menuCfg.fovDeg, aspect, 0.1, menuCfg.farPlane ?? 230.0);
      } else if (thirdPerson) {
        const tp = computeThirdPersonCamera();
        eye = tp.eye;
        fwd = tp.fwd;
        up = tp.up;
        const farNow = (eye[1] > 9.2) ? (menuCfg.farPlane ?? 230.0) : Math.max(player.cfg.far_plane, 100.0);
        projMat = mat4Perspective(
          player.cfg.fov_deg + player.slideFovOffset * 0.6 + player.sprintFovOffset * 0.6,
          aspect,
          player.cfg.near_plane,
          farNow
        );
      } else {
        eye = player.eyePos;
        const bcP = weapon.bodycamCamPitch || 0.0;
        const bcY = weapon.bodycamCamYaw || 0.0;
        const kickJ = Math.sin((doorCtrl.kickShake || 0.0) * 22.0) * 0.026 * (doorCtrl.kickShake || 0.0);
        const effYaw = player.yaw + bcY;
        const effP = Math.max(-1.48, Math.min(1.48, player.pitch + bcP + kickJ * 0.55));
        const cpF = Math.cos(effP);
        fwd = [Math.sin(effYaw) * cpF, Math.sin(effP), -Math.cos(effYaw) * cpF];
        const effRoll = (player.slideRoll || 0.0) + (weapon.bodycamCamRoll || 0.0) + kickJ;
        const rRight = [Math.cos(effYaw), 0, Math.sin(effYaw)];
        const up0 = normalize(vec3Cross(rRight, fwd));
        up = [
          up0[0] * Math.cos(effRoll) + rRight[0] * Math.sin(effRoll),
          up0[1] * Math.cos(effRoll) + rRight[1] * Math.sin(effRoll),
          up0[2] * Math.cos(effRoll) + rRight[2] * Math.sin(effRoll),
        ];
        const farNow = (eye[1] > 9.2) ? (menuCfg.farPlane ?? 230.0) : Math.max(player.cfg.far_plane, 100.0);
        let curFov = player.cfg.fov_deg - 11.0 * weapon.adsT + player.kickFovOffset + player.slideFovOffset + player.sprintFovOffset;
        if (weapon.isSniper && weapon.scopeAmount > 0.005) {
          const ironFov = player.cfg.fov_deg - 12.0 * weapon.adsT;
          curFov = ironFov + (22.0 - ironFov) * weapon.scopeAmount;
        }
        projMat = mat4Perspective(curFov, aspect, player.cfg.near_plane, farNow);
      }
      const onUpperStreet = !inCourtyard && (eye[1] > 9.2);
      const useWinterAtmo = inCourtyard || onUpperStreet;

      gl.bindFramebuffer(gl.FRAMEBUFFER, fbo);
      gl.viewport(0, 0, lowW, lowH);
      gl.enable(gl.DEPTH_TEST);
      gl.enable(gl.CULL_FACE);
      gl.cullFace(gl.BACK);
      gl.frontFace(gl.CCW);
      gl.disable(gl.BLEND);

      const fc = inStory ? [.17,.205,.25] : useWinterAtmo ? menuCfg.fogColor : preset.fog_color;
      const fogStart = inStory ? 10 : useWinterAtmo ? menuCfg.fogStart : preset.fog_start;
      const fogEnd = inStory ? 58 : useWinterAtmo ? menuCfg.fogEnd : preset.fog_end;
      const ambBoost = inStory ? ((mode==='home'||storyTime>10.6)?1.4:1.18) : useWinterAtmo ? menuCfg.ambientBoost : preset.ambient_boost;
      const activeBatches = inStory ? prologueBatches : (inCourtyard ? menuBatches : worldBatches);
      const streetView=mode==='home'&&hallDone&&streetStage>=3&&eye[1]<3&&(eye[0]>EXIT_X+.10||eye[2]<21.1);
      const motelView=inStory&&(awake||dreamT>=wakeAt);const blast= dreamT>=flashAt&&dreamT<wakeAt;
      const wingLights=[{position:[-14.1,14.55,30.04],color:[1,.83,.60],intensity:1.61,radius:6},{position:[-16.14,14.24,32.66],color:[1,.85,.66],intensity:1.47,radius:4.4},{position:[-12.5,14.21,32.60],color:[.81,.87,1],intensity:1.51,radius:4.3}];
      const activeLights = inStory ? (motelView?motelLights.concat([{position:[-5.34,-25.89,25.82],color:[1,.83,.60],intensity:.82,radius:2.81},{position:[-.80,-25.62,25.65],color:[.74,.87,.91],intensity:1.68,radius:3.45}]):eye[0]<-9.76&&eye[2]>28.85&&eye[1]>12&&mode==='home'?wingLights.concat(kitchenLights):blast?[{position:[-10,13.81,20.65],color:[1,.45,.12],intensity:10,radius:18},...yardLights]:streetView?life.getDynamicLights(yardLights): (mode==='home'||storyTime>=10.6)?(hallT>=0||hallDone?hallLights.concat(kitchenLights):mode==='home'&&eye[2]>27.2?kitchenLights.concat(storyLights):storyLights).concat(life.getDynamicLights(yardLights)):life.getDynamicLights(yardLights))
        : (inCourtyard?life.getDynamicLights(menuLights):(onUpperStreet?life.getDynamicLights(menuLights,MENU_ELEV_Y):mp.connected?[...Array.from(mp.remotes.values()).filter(r=>r.flash).slice(0,3).map(r=>({position:[r.pos[0]+2*Math.sin(r.yaw),r.pos[1]+1.4,r.pos[2]-2*Math.cos(r.yaw)],color:[.88,.92,1],intensity:1.8,radius:5})),...scene.lights]:scene.lights));

      gl.clearColor(fc[0], fc[1], fc[2], 1.0);
      gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);

      gl.useProgram(sceneProg);
      life.view=[eye,fwd];const viewMat = mat4LookAt(eye, vec3Add(eye, fwd), up);

      gl.uniformMatrix4fv(su.model, false, idMat4);
      gl.uniformMatrix4fv(su.view, false, toGlMat4(viewMat));
      gl.uniformMatrix4fv(su.proj, false, toGlMat4(projMat));
      gl.uniform1f(su.jitterEn, preset.vertex_jitter ? 1.0 : 0.0);
      gl.uniform2f(su.jitterGrid, preset.jitter_grid[0]*(inStory?2:1), preset.jitter_grid[1]*(inStory?2:1));

      gl.uniform3f(su.camPos, eye[0], eye[1], eye[2]);
      gl.uniform3f(su.camDir, fwd[0], fwd[1], fwd[2]);
      gl.uniform1f(su.ambient, ambBoost);
      gl.uniform1f(su.flashOn, (!inCourtyard && player.flashlightOn) ? 1.0 : 0.0);

      gl.uniform3f(su.fogColor, fc[0], fc[1], fc[2]);
      gl.uniform1f(su.mistClock,inStory?1+elapsed:0);gl.uniform1f(su.fogStart, fogStart);
      gl.uniform1f(su.fogEnd, fogEnd);

      const hasMuzzleLight = !inCourtyard && weapon.flashLightTimer > 0.0;
      const hasCigLight = (mode === "menu" || (mode==='cinema'&&storyTime<23.2) || onUpperStreet) && menuLife && life.emberGlow > 0.05;
      const cigElev = onUpperStreet ? MENU_ELEV_Y : 0.0;
      const baseLimit = hasMuzzleLight ? (MAX_LIGHTS - 1) : MAX_LIGHTS;
      let nLights = Math.min(activeLights.length, baseLimit);
      for (let i = 0; i < nLights; i++) {
        const lt = activeLights[i];
        const b = i * 3;
        lPosArr[b] = lt.position[0]; lPosArr[b+1] = lt.position[1]; lPosArr[b+2] = lt.position[2];
        lColArr[b] = lt.color[0]; lColArr[b+1] = lt.color[1]; lColArr[b+2] = lt.color[2];
        lIntArr[i] = evalLightIntensity(lt, elapsed);
        lRadArr[i] = lt.radius;
      }
      if (hasMuzzleLight) {
        const b = nLights * 3;
        const mInt = Math.min(1.0, weapon.flashLightTimer / 0.085);
        lPosArr[b] = weapon.flashWorldPos[0];
        lPosArr[b+1] = weapon.flashWorldPos[1];
        lPosArr[b+2] = weapon.flashWorldPos[2];
        lColArr[b] = 1.0; lColArr[b+1] = 0.68; lColArr[b+2] = 0.28;
        lIntArr[nLights] = 3.2 * mInt;
        lRadArr[nLights] = 9.5;
        nLights += 1;
      } else if (hasCigLight) {
        const slot = nLights >= MAX_LIGHTS ? 4 : nLights;
        const b = slot * 3;
        lPosArr[b] = life.cigTipWorld[0];
        lPosArr[b+1] = life.cigTipWorld[1] + cigElev;
        lPosArr[b+2] = life.cigTipWorld[2];
        lColArr[b] = 1.0; lColArr[b+1] = 0.38; lColArr[b+2] = 0.10;
        lIntArr[slot] = life.emberGlow * 1.45;
        lRadArr[slot] = 2.6;
        if (nLights < MAX_LIGHTS) nLights += 1;
      }
      gl.uniform1i(su.numLights, nLights);
      gl.uniform3fv(su.lightPos, lPosArr);
      gl.uniform3fv(su.lightCol, lColArr);
      gl.uniform1fv(su.lightInt, lIntArr);
      gl.uniform1fv(su.lightRad, lRadArr);
      gl.uniform1i(su.tex, 0);
      gl.uniform3f(su.colorTint, 1.0, 1.0, 1.0);
      gl.uniform1f(su.alpha, 1.0);
      gl.uniform1f(su.softParticle, 0.0);
      gl.activeTexture(gl.TEXTURE0);

      for (const batch of activeBatches) {
        gl.bindTexture(gl.TEXTURE_2D, textures[batch.texture]);
        gl.uniform1f(su.emissive, batch.emissive);
        gl.bindVertexArray(batch.vao);
        gl.drawArrays(gl.TRIANGLES, 0, batch.vertexCount);
      }

      // Keep the entire Winter Menu Street + ALL menu models alive directly ABOVE the test room during gameplay!
      if (!inCourtyard) {
        // Draw articulated 3D Tactical Door in Sector B doorway!
        if (doorCtrl.enabled && doorBatches.length) {
          gl.uniformMatrix4fv(su.model, false, toGlMat4(doorCtrl.modelMatrix()));
          gl.uniform3f(su.colorTint, 1.0, 1.0, 1.0);
          for (const batch of doorBatches) {
            gl.bindTexture(gl.TEXTURE_2D, textures[batch.texture]);
            gl.uniform1f(su.emissive, batch.emissive);
            gl.bindVertexArray(batch.vao);
            gl.drawArrays(gl.TRIANGLES, 0, batch.vertexCount);
          }
          gl.uniformMatrix4fv(su.model, false, idMat4);
        }

        gl.uniformMatrix4fv(su.model, false, glMenuElev);
        for (const batch of menuBatches) {
          gl.bindTexture(gl.TEXTURE_2D, textures[batch.texture]);
          gl.uniform1f(su.emissive, batch.emissive);
          gl.bindVertexArray(batch.vao);
          gl.drawArrays(gl.TRIANGLES, 0, batch.vertexCount);
        }
        gl.uniformMatrix4fv(su.model, false, idMat4);

        // Draw menu hero + cigarette + sidewalk NPCs + traffic cars on the upper street (Y = +MENU_ELEV_Y)!
        if (life) {
          if (onUpperStreet) gl.uniform1f(su.ambient, 0.72);
          if (menuHeroBones) {
            const elevHeroBones = {};
            for (const [bk, bm] of Object.entries(menuHeroBones)) {
              elevHeroBones[bk] = mat4Mul(mMenuElev, bm);
            }
            drawCharacterModel(character, elevHeroBones, elapsed);
            if (cigBatches && elevHeroBones.hand_r) {
              gl.uniformMatrix4fv(su.model, false, toGlMat4(elevHeroBones.hand_r));
              gl.uniform3f(su.colorTint, 1.0, 1.0, 1.0);
              for (const b of cigBatches.body) {
                gl.bindTexture(gl.TEXTURE_2D, textures[b.texture]);
                gl.uniform1f(su.emissive, 0.0);
                gl.bindVertexArray(b.vao);
                gl.drawArrays(gl.TRIANGLES, 0, b.vertexCount);
              }
              const emberEm = 0.85 + 2.65 * life.emberGlow;
              for (const b of cigBatches.ember) {
                gl.bindTexture(gl.TEXTURE_2D, textures[b.texture]);
                gl.uniform1f(su.emissive, emberEm);
                gl.bindVertexArray(b.vao);
                gl.drawArrays(gl.TRIANGLES, 0, b.vertexCount);
              }
            }
          }
          for (let i = 0; i < life.npcs.length; i++) {
            const npc = life.npcs[i];
            const npcPosElev = [npc.pos[0], npc.pos[1] + MENU_ELEV_Y, npc.pos[2]];
            const nProp = npc.prop || "none";
            const npcBones = evaluateCharacterBones(
              charJoints,
              npc.appearance.gender,
              npcPosElev,
              npc.yaw,
              {
                elapsed: elapsed + npc.blinkOffset,
                horizSpeed: npc.speed,
                onGround: true,
                walkPhase: npc.walkPhase,
                bustSize: (typeof npc.appearance.bust_size === "number") ? npc.appearance.bust_size : 0.5,
                npcProp: nProp,
              }
            );
            drawCharacterModel(npc.appearance, npcBones, elapsed + npc.blinkOffset);
            if (nProp !== "none") drawNpcProp(nProp, npcBones, npc.propTint || [1, 1, 1]);
          }
          if (life.trafficCars.length) drawTrafficCars(life.trafficCars, MENU_ELEV_Y);
          gl.uniform1f(su.ambient, ambBoost);
          gl.uniform3f(su.colorTint, 1.0, 1.0, 1.0);
          gl.uniformMatrix4fv(su.model, false, idMat4);
        }
      }

      // ---------------- PASS 1A: Articulated 3D Character (Creator Showcase / Menu Protagonist+NPCs / 3rd-Person) ----------------
      let thirdPersonHandR = null;
      if (mode === "creator") {
        gl.uniform1f(su.ambient, 0.78);
        const cTopId = charCatalog.topStyles[character.top_style % charCatalog.topStyles.length].id;
        const cBustSz = (typeof character.bust_size === "number") ? character.bust_size : 0.5;
        const cBust = (character.gender === "female")
          ? creatorBustPhys.update(dt, elapsed, CREATOR_BASE_YAW + creatorYawOffset, { topId: cTopId, bustSize: cBustSz })
          : null;
        const bones = evaluateCharacterBones(
          charJoints,
          character.gender,
          CREATOR_STAGE_POS,
          CREATOR_BASE_YAW + creatorYawOffset,
          { elapsed, showcase: true, bustState: cBust, bustSize: cBustSz }
        );
        drawCharacterModel(character, bones, elapsed);
        gl.uniform1f(su.ambient, ambBoost);
        if (menuLife && life.trafficCars.length) drawTrafficCars(life.trafficCars);
      } else if ((mode === "menu" || inStory) && life) {
        gl.uniform1f(su.ambient, 0.72);
        if (menuHeroBones && (!inStory||(mode==='cinema'&&storyTime<24.65))) {
          if(inStory){const t=storyTime,scan=phase(t,11.35,11.56)*(1-phase(t,13.03,13.41)),call=phase(t,17.41,17.59)*(1-phase(t,21.52,22.08)),ash=phase(t,22.62,22.90);
            drawCharacterModel(character,menuHeroBones,t,'',[.08,0,.40*call,.01],[-.71*scan+.87*call,.26*scan+.74*ash]);
          }else drawCharacterModel(character,menuHeroBones,elapsed);
          // 3D cigarette held in hero's right hand + glowing orange ember tip during inhale drag!
          if (cigBatches && (!inStory||storyTime<24.75)) {
            let cigaretteM=menuHeroBones.hand_r;
            if(inStory && storyTime>=23.15){if(!releaseMatrix)releaseMatrix=cigaretteM.slice();const t=storyTime-23.15; cigaretteM=mat4Mul(mat4Translate(.32*t,2.05*t-3.6*t*t,-2.5*t),mat4Mul(releaseMatrix,mat4RotX(t*8)));}
            const handM = toGlMat4(cigaretteM);
            gl.uniformMatrix4fv(su.model, false, handM);
            gl.uniform3f(su.colorTint, 1.0, 1.0, 1.0);
            for (const b of cigBatches.body) {
              gl.bindTexture(gl.TEXTURE_2D, textures[b.texture]);
              gl.uniform1f(su.emissive, 0.0);
              gl.bindVertexArray(b.vao);
              gl.drawArrays(gl.TRIANGLES, 0, b.vertexCount);
            }
            const emberEm = 0.85 + 2.65 * life.emberGlow;
            for (const b of cigBatches.ember) {
              gl.bindTexture(gl.TEXTURE_2D, textures[b.texture]);
              gl.uniform1f(su.emissive, emberEm);
              gl.bindVertexArray(b.vao);
              gl.drawArrays(gl.TRIANGLES, 0, b.vertexCount);
            }
          }
        }
        // Occasional random winter sidewalk passersby (sometimes carrying shopping bags, books, coffee, briefcases, parcels!)
        for (let i = 0; i < life.npcs.length; i++) {
          const npc = life.npcs[i];
          const nProp = npc.prop || "none";
          const npcBones = evaluateCharacterBones(
            charJoints,
            npc.appearance.gender,
            npc.pos,
            npc.yaw,
            {
              elapsed: elapsed + npc.blinkOffset,
              horizSpeed: npc.speed,
              onGround: true,
              walkPhase: npc.walkPhase,
              bustSize: (typeof npc.appearance.bust_size === "number") ? npc.appearance.bust_size : 0.5,
              npcProp: nProp,
            }
          );
          drawCharacterModel(npc.appearance, npcBones, elapsed + npc.blinkOffset);
          if (nProp !== "none") drawNpcProp(nProp, npcBones, npc.propTint || [1, 1, 1]);
        }
        gl.uniform1f(su.ambient, ambBoost);
        if(inStory){drawInteriorLeaves(motelView);drawDream(elapsed);}if(inStory&&!motelView){drawTrafficCars(parkedCars);gl.uniform1f(su.ambient,1.65);drawBalconyLife(eye,fwd,elapsed);drawDiningPartner(eye,elapsed);drawThreshold(elapsed);if(supperT>=0)drawDinnerHero(supperT);gl.uniform1f(su.ambient,ambBoost);}
        if (life.trafficCars.length) drawTrafficCars(life.trafficCars);
        gl.uniform3f(su.colorTint, 1.0, 1.0, 1.0);
      } else if (!inCourtyard && thirdPerson) {
        const horizSpeed = Math.hypot(player.vel[0], player.vel[2]);
        const kickProg = player.kick.active ? (player.kick.timer / Math.max(player.kick.duration, 1e-4)) : 0.0;
        const relU = weapon.reloading ? (weapon.reloadTimer / Math.max(weapon.reloadDuration, 1e-4)) : 0.0;
        const pTopId = charCatalog.topStyles[character.top_style % charCatalog.topStyles.length].id;
        const pBustSz = (typeof character.bust_size === "number") ? character.bust_size : 0.5;
        const pBust = (character.gender === "female")
          ? playerBustPhys.update(dt, elapsed, Math.PI - player.yaw, {
              horizSpeed,
              velY: player.vel[1],
              onGround: player.onGround,
              crouching: player.crouching,
              sliding: player.sliding,
              kickPhase: kickProg,
              recoil: weapon.recoilPitch,
              walkPhase: player.walkPhase,
              topId: pTopId,
              bustSize: pBustSz,
            })
          : null;
        const bones = evaluateCharacterBones(
          charJoints,
          character.gender,
          player.pos,
          Math.PI - player.yaw,
          {
            elapsed,
            walkPhase: player.walkPhase,
            horizSpeed,
            onGround: player.onGround,
            crouching: player.crouching,
            sliding: player.sliding,
            kickProgress: kickProg,
            velY: player.vel[1],
            showcase: false,
            holdingPistol: weapon.hasWeapon && weapon.equipped,
            pistolRecoil: weapon.recoilPitch,
            pistolReloading: weapon.reloading,
            pistolReloadU: relU,
            pistolReloadType: weapon.reloadType,
            cameraPitch: player.pitch,
            bustState: pBust,
            bustSize: pBustSz,
          }
        );
        thirdPersonHandR = bones.hand_r;
        gl.uniform1f(su.ambient, preset.ambient_boost * 1.25);
        drawCharacterModel(character, bones, elapsed);
        gl.uniform1f(su.ambient, ambBoost);
      }

      // Interactive Test NPC ("Стас") rendered inside the test arena in both 1st- and 3rd-person!
      if (!inCourtyard && testNpc.enabled && testNpc.appearance) {
        const tnBones = evaluateCharacterBones(
          charJoints,
          testNpc.appearance.gender,
          testNpc.pos,
          testNpc.yaw,
          {
            elapsed: elapsed + 2.3,
            horizSpeed: 0.0,
            onGround: true,
            showcase: false,
            npcTalking: testNpc.talkBlend,
            npcHeadLookYaw: testNpc.headLookYaw,
          }
        );
        gl.uniform1f(su.ambient, preset.ambient_boost * 1.28);
        drawCharacterModel(testNpc.appearance, tnBones, elapsed + 2.3, testNpc.mouthSuffix);
        gl.uniform1f(su.ambient, ambBoost);
        gl.uniform3f(su.colorTint, 1.0, 1.0, 1.0);
      }

      // ---------------- PASS 1A2: World-Space Bullet FX, Grounded Pickups (M1911 & Glock-17), Floor Mags & 3rd-Person ----------------
      if (!inCourtyard) {
        if (weapon.worldPickups.length > 0) {
          const pickTint = [1.06, 1.04, 1.02];
          gl.uniform1f(su.ambient, preset.ambient_boost * 1.45);
          for (let wi = 0; wi < weapon.worldPickups.length; wi++) {
            const wp = weapon.worldPickups[wi];
            const wid = wp.weaponId || "m1911";
            const mdl = weaponGpu.models[wid] || weaponGpu.m1911;
            const wSpec = weapon.SPECS[wid] || weapon.SPECS.m1911;
            const wSc = ((wid === "remington11" || wid === "double_barrel" || wid === "ak47" || wid === "m4a1" || wid === "m110") ? 0.96 : ((wid === "mp7" || wid === "mp5s") ? 1.06 : (wid === "revolver" ? 1.24 : 1.28)));
            const baseM = mat4Mul(
              mat4Translate(wp.pos[0], wp.pos[1], wp.pos[2]),
              mat4Mul(
                rotYXZ(wp.rot[1], wp.rot[0], wp.rot[2]),
                mat4Scale(wSc, wSc, wSc)
              )
            );
            drawWeaponBatches(mdl.frame, baseM, null, pickTint);
            const wpSlide = wp.slideLocked ? mat4Mul(baseM, mat4Translate(0.0, 0.0, wSpec.slide_travel)) : baseM;
            drawWeaponBatches(mdl.slide, wpSlide, null, pickTint);
            drawWeaponBatches(mdl.trigger, baseM, null, pickTint);
            if (mdl.hammer && mdl.hammer.length) drawWeaponBatches(mdl.hammer, baseM, null, pickTint);
            if (mdl.suppressor && mdl.suppressor.length) drawWeaponBatches(mdl.suppressor, baseM, null, pickTint);
            if (mdl.magazine && mdl.magazine.length) drawWeaponBatches(mdl.magazine, baseM, null, pickTint);
            if (wp.grounded) {
              const pulse = 0.55 + 0.45 * Math.sin(elapsed * 3.4 + wi * 1.7);
              const glSc = 0.44 + 0.26 * pulse;
              const glintM = mat4Mul(
                mat4Translate(wp.pos[0], wp.pos[1] + 0.145 + 0.025 * Math.sin(elapsed * 2.6 + wi), wp.pos[2]),
                mat4Mul(mat4RotY(elapsed * 1.4), mat4Scale(glSc, glSc, glSc))
              );
              const glintMap = {
                remington11: [1.0, 0.46, 0.28],
                double_barrel: [1.0, 0.68, 0.22],
                revolver: [0.92, 0.82, 1.0],
                ak47: [1.0, 0.56, 0.18],
                m4a1: [0.48, 0.88, 1.0],
                mp7: [1.0, 0.28, 0.36],
                m110: [0.96, 0.84, 0.44],
                mp5s: [0.42, 0.98, 0.72],
                glock17: [0.72, 0.90, 1.0],
                m1911: [1.0, 0.92, 0.65],
              };
              const glintCol = glintMap[wid] || [1.0, 0.92, 0.65];
              drawWeaponBatches(weaponGpu.m1911.spark, glintM, null, glintCol);
            }
          }
          gl.uniform1f(su.ambient, ambBoost);
        }

        // Persistent 3D dropped magazines lying on floors, crates & tables!
        if (weapon.worldMags.length > 0) {
          gl.uniform1f(su.ambient, preset.ambient_boost * 1.32);
          for (let i = 0; i < weapon.worldMags.length; i++) {
            const wm = weapon.worldMags[i];
            const widM = wm.weaponId || "m1911";
            const mdl = weaponGpu.models[widM] || weaponGpu.m1911;
            const mc = (widM === "ak47")
              ? [0.0, -0.065, -0.195]
              : ((widM === "m4a1") ? weapon.M4_MAG_CENTER
              : ((widM === "m110") ? weapon.M110_MAG_CENTER
              : ((widM === "mp5s") ? weapon.MP5S_MAG_CENTER
              : ((widM === "mp7") ? weapon.MP7_MAG_CENTER : weapon.MAG_MESH_CENTER))));
            const magCenterFix = mat4Translate(-mc[0], -mc[1], -mc[2]);
            const wmM = mat4Mul(
              mat4Translate(wm.pos[0], wm.pos[1], wm.pos[2]),
              mat4Mul(
                rotYXZ(wm.rot[1], wm.rot[0], wm.rot[2]),
                mat4Mul(mat4Scale(1.22, 1.22, 1.22), magCenterFix)
              )
            );
            drawWeaponBatches(mdl.magazine, wmM);
            if (wm.hasBullet) {
              drawWeaponBatches(mdl.mag_round, wmM);
            }
          }
          gl.uniform1f(su.ambient, ambBoost);
        }

        for (const sh of weapon.shells) {
          let m = mat4Mul(
            mat4Translate(sh.pos[0], sh.pos[1], sh.pos[2]),
            rotYXZ(sh.rot[1], sh.rot[0], sh.rot[2])
          );
          let shBatch = weaponGpu.m1911.shell;
          if (sh.shellType === "rev_spent" && weaponGpu.m1911.rev_round_spent) {
            m = mat4Mul(m, mat4Scale(1.48, 1.48, 1.48));
            shBatch = weaponGpu.m1911.rev_round_spent;
          } else if (sh.shellType === "shotgun" && weaponGpu.m1911.shell_12g) {
            shBatch = weaponGpu.m1911.shell_12g;
          }
          drawWeaponBatches(shBatch, m);
        }
        for (const sp of weapon.sparks) {
          const sc = Math.max(0.25, sp.life / Math.max(sp.maxLife, 1e-4));
          const m = mat4Mul(
            mat4Translate(sp.pos[0], sp.pos[1], sp.pos[2]),
            mat4Mul(mat4RotZ(sp.roll), mat4Scale(sc, sc, sc))
          );
          drawWeaponBatches(weaponGpu.m1911.spark, m);
        }
        for (const bh of weapon.bulletHoles) {
          const age = bh.maxLife - bh.life;
          const fade = Math.min(1.0, bh.life / 1.2);
          const sc = 0.42 * (0.55 + 0.45 * fade);
          let tint = [0.10, 0.10, 0.11];
          if (age < 0.18) {
            const ht = age / 0.18;
            tint = [0.95 * (1.0 - ht) + 0.10 * ht, 0.46 * (1.0 - ht) + 0.10 * ht, 0.14 * (1.0 - ht) + 0.10 * ht];
          }
          const m = mat4Mul(
            mat4Translate(bh.pos[0], bh.pos[1], bh.pos[2]),
            mat4Mul(mat4RotZ(bh.roll), mat4Scale(sc, sc, sc))
          );
          drawWeaponBatches(weaponGpu.m1911.spark, m, null, tint);
        }
        // v14: Render 3D Bullet Tracers & Supersonic M110 Vapor Trails!
        const fxMdl = weaponGpu.models.mp5s || weaponGpu.m1911;
        if (weapon.tracers && weapon.tracers.length && fxMdl && fxMdl.tracer_beam) {
          for (const tr of weapon.tracers) {
            const [sx, sy, sz] = tr.start;
            const [ex, ey, ez] = tr.end;
            const dx = ex - sx, dy = ey - sy, dz = ez - sz;
            const dist = Math.hypot(dx, dy, dz);
            if (dist < 0.08) continue;
            const fade = Math.max(0.0, Math.min(1.0, tr.life / Math.max(tr.maxLife, 1e-4)));
            const fx = dx / dist, fy = dy / dist, fz = dz / dist;
            const upR = Math.abs(fy) < 0.96 ? [0, 1, 0] : [1, 0, 0];
            const rV = normalize(vec3Cross([fx, fy, fz], upR));
            const uV = vec3Cross(rV, [fx, fy, fz]);
            const th = tr.thickness * (0.35 + 0.65 * fade);
            const mTr = new Float32Array([
              rV[0] * th, uV[0] * th, -fx * dist, sx,
              rV[1] * th, uV[1] * th, -fy * dist, sy,
              rV[2] * th, uV[2] * th, -fz * dist, sz,
              0.0,        0.0,        0.0,        1.0,
            ]);
            const c = tr.color;
            drawWeaponBatches(fxMdl.tracer_beam, mTr, null, [c[0] * fade, c[1] * fade, c[2] * fade]);
          }
        }
        // v14: Render 3D Physics Particles (Gunsmoke, Brake Blast, Suppressor Steam, Breach Dust, Concrete Chips, Embers)!
        if (weapon.particles && weapon.particles.length && fxMdl && fxMdl.fx_cube) {
          for (const pt of weapon.particles) {
            const uP = 1.0 - Math.max(0.0, Math.min(1.0, pt.life / Math.max(pt.maxLife, 1e-4)));
            const fade = Math.max(0.0, 1.0 - uP);
            const sc = (pt.size0 + (pt.size1 - pt.size0) * uP) * (0.28 + 0.72 * fade);
            if (sc < 0.002) continue;
            const mPt = mat4Mul(
              mat4Translate(pt.pos[0], pt.pos[1], pt.pos[2]),
              mat4Mul(rotYXZ(pt.rot, pt.rot * 0.7, pt.rot * 0.4), mat4Scale(sc, sc, sc))
            );
            const c = pt.color;
            const kF = 0.45 + 0.55 * fade;
            const batches = (pt.emissive && weaponGpu.m1911.spark) ? weaponGpu.m1911.spark : fxMdl.fx_cube;
            drawWeaponBatches(batches, mPt, null, [c[0] * kF, c[1] * kF, c[2] * kF]);
          }
        }
        if (thirdPerson && thirdPersonHandR && weapon.hasWeapon && weapon.equipped) {
          const wSpec3 = weapon.activeSpec;
          const mdl = weaponGpu.models[weapon.activeWeapon] || weaponGpu.m1911;
          const scTp = weapon.isLongGun ? 0.96 : 1.12;
          const gripM = mat4Mul(
            thirdPersonHandR,
            mat4Mul(
              mat4Translate(0.0, -0.082, 0.022),
              mat4Mul(
                mat4RotX(Math.PI * 0.5 + 0.08),
                mat4Mul(mat4RotY(Math.PI), mat4Scale(scTp, scTp, scTp))
              )
            )
          );
          const pp3 = weapon._pumpPoseOffsets();
          const sb3 = weapon.pumpActive ? pp3.slideExtra : (weapon.slideLocked ? weapon.SLIDE_TRAVEL : (weapon.SLIDE_TRAVEL * Math.pow(weapon.slideKick, 0.75)));
          gl.uniform1f(su.ambient, preset.ambient_boost * 1.35);
          drawWeaponBatches(mdl.frame, gripM);
          drawWeaponBatches(mdl.slide, mat4Mul(gripM, mat4Translate(0, 0, sb3)));
          drawWeaponBatches(mdl.trigger, gripM);
          if (mdl.hammer && mdl.hammer.length) {
            drawWeaponBatches(mdl.hammer, gripM);
          }
          if (mdl.magazine && mdl.magazine.length) drawWeaponBatches(mdl.magazine, gripM);
          if (weapon.muzzleTimer > 0.0) {
            let ml = wSpec3.muzzle_local || weapon.MUZZLE_LOCAL;
                const fsc3 = wSpec3.flash_scale || 1.0;
            const mfM = mat4Mul(gripM, mat4Mul(mat4Translate(ml[0], ml[1], ml[2]), mat4Mul(mat4RotZ(weapon.muzzleRoll), mat4Scale(fsc3, fsc3, fsc3))));
            drawWeaponBatches(weaponGpu.m1911.muzzle_flash, mfM);
          }
          gl.uniform1f(su.ambient, ambBoost);
        }

        // ---- Multiplayer Remote Players (Full 18-Bone R16 + 3D Weapons + Floating Nameplates) ----
        if (mp.connected && mp.remotes.size > 0) {
          for (const rp of mp.remotes.values()) {
            const rBones = evaluateCharacterBones(
              charJoints,
              rp.appearance.gender,
              rp.pos,
              Math.PI - rp.yaw,
              {
                elapsed,
                walkPhase: rp.walkPhase,
                horizSpeed: rp.speed,
                onGround: rp.grounded,
                crouching: rp.crouch,
                sliding: rp.slide,
                kickProgress: rp.kickT,
                velY: rp.velY,
                showcase: false,
                holdingPistol: rp.equipped,
                pistolRecoil: rp.recoilPitch,
                onlineInspect:rp.ins,onlineADS:rp.ads,pistolReloading:rp.reload,pistolReloadU:rp.reloadU,pistolReloadType:rp.reloadType,
                cameraPitch: rp.pitch,
                bustState: rp.lastBust,
                bustSize: (typeof rp.appearance.bust_size === "number") ? rp.appearance.bust_size : 0.5,
              }
            );
            gl.uniform1f(su.ambient, preset.ambient_boost * 1.32);
            drawCharacterModel(rp.appearance, rBones, elapsed);
            if (rp.equipped && rBones.hand_r) {
              const rWid = weaponGpu.models[rp.weapon] ? rp.weapon : "m1911";
              const rMdl = weaponGpu.models[rWid] || weaponGpu.m1911;
              const rSpec = weapon.SPECS[rWid] || weapon.SPECS.m1911;
              const rLong = (rWid === "remington11" || rWid === "double_barrel" || rWid === "ak47" || rWid === "m4a1" || rWid === "mp7" || rWid === "m110" || rWid === "mp5s");
              const rSc = rLong ? 0.96 : 1.12;
              const rGripM = mat4Mul(
                rBones.hand_r,
                mat4Mul(
                  mat4Translate(0.0, -0.082, 0.022),
                  mat4Mul(
                    mat4RotX(Math.PI * 0.5 + 0.08),
                    mat4Mul(mat4RotY(Math.PI), mat4Scale(rSc, rSc, rSc))
                  )
                )
              );
              drawWeaponBatches(rMdl.frame, rGripM);
              drawWeaponBatches(rMdl.slide,mat4Mul(rGripM,mat4Translate(0,0,.045*rp.slideKick)));if(rWid==='mp5s'&&rp.suppressed)drawWeaponBatches(rMdl.suppressor,rGripM);
              drawWeaponBatches(rMdl.trigger, rGripM);
              if (rMdl.hammer && rMdl.hammer.length) drawWeaponBatches(rMdl.hammer, rGripM);
              if (rMdl.magazine && rMdl.magazine.length) drawWeaponBatches(rMdl.magazine, rGripM);
              if (rp.muzzleTimer > 0.0) {
                const ml = rSpec.muzzle_local || weapon.MUZZLE_LOCAL;
                const fscR = rSpec.flash_scale || 1.0;
                const mfM = mat4Mul(rGripM, mat4Mul(mat4Translate(ml[0], ml[1], ml[2]), mat4Scale(fscR, fscR, fscR)));
                drawWeaponBatches(weaponGpu.m1911.muzzle_flash, mfM);
              }
            }
            gl.uniform1f(su.ambient, ambBoost);
          }
        }
        mp.renderNameplates(viewMat, projMat);
        gl.uniform3f(su.colorTint, 1.0, 1.0, 1.0);
      }

      if(inStory){
        gl.enable(gl.BLEND);gl.blendFunc(gl.SRC_ALPHA,gl.ONE_MINUS_SRC_ALPHA);gl.depthMask(false);
        gl.uniformMatrix4fv(su.model,false,idMat4);gl.uniform3f(su.colorTint,1,1,1);gl.uniform1f(su.alpha,.13);
        for(const b of glassPanes){gl.uniform1f(su.emissive,b.emissive);gl.bindTexture(gl.TEXTURE_2D,textures[b.texture]);gl.bindVertexArray(b.vao);gl.drawArrays(gl.TRIANGLES,0,b.vertexCount);}
        gl.uniform1f(su.alpha,1);gl.depthMask(true);gl.disable(gl.BLEND);
      }
      // ---------------- PASS 1A3: Batched 3D Falling Winter Snow & Curling Cigarette Smoke! ----------------
      if (life&&!motelView) {
        if (inCourtyard) {
          life.renderParticles(eye, fwd, up, mode==='menu'||(mode==='cinema'&&storyTime<24.75), 0.0);
        } else {
          life.renderParticles(eye, fwd, up, true, MENU_ELEV_Y);
        }
      }

      // ---------------- PASS 1B: First-Person Depth-Cleared Viewmodels (Leg Kick + Colt M1911 & Articulated Player Hands!) ----------------
      if (!inCourtyard && !thirdPerson && (player.kick.active || (weapon.hasWeapon && weapon.equipT > 0.005))) {
        const vmToWorld = mat4CameraToWorld(eye, fwd, up);
        const slotCols = getCharacterSlotColors(charCatalog, character);
        gl.clear(gl.DEPTH_BUFFER_BIT);
        gl.uniform1f(su.ambient, preset.ambient_boost * 1.48);
        gl.uniform1f(su.flashOn, player.flashlightOn ? 0.30 : 0.0);

        if (player.kick.active) {
          const draws = player.kick.buildDrawCalls(legBatches, vmToWorld);
          for (const d of draws) {
            gl.uniformMatrix4fv(su.model, false, toGlMat4(d.model));
            for (const batch of d.batches) {
              if (batch.texture === "pants_dark") {
                gl.uniform3f(
                  su.colorTint,
                  Math.min(1.8, slotCols.bottom[0] * 1.9 + 0.25),
                  Math.min(1.8, slotCols.bottom[1] * 1.9 + 0.25),
                  Math.min(1.8, slotCols.bottom[2] * 1.9 + 0.25)
                );
              } else if (batch.texture === "boot_leather") {
                gl.uniform3f(
                  su.colorTint,
                  Math.min(1.8, slotCols.shoes[0] * 1.8 + 0.25),
                  Math.min(1.8, slotCols.shoes[1] * 1.8 + 0.25),
                  Math.min(1.8, slotCols.shoes[2] * 1.8 + 0.25)
                );
              } else {
                gl.uniform3f(su.colorTint, 1.0, 1.0, 1.0);
              }
              gl.bindTexture(gl.TEXTURE_2D, textures[batch.texture]);
              gl.uniform1f(su.emissive, batch.emissive);
              gl.bindVertexArray(batch.vao);
              gl.drawArrays(gl.TRIANGLES, 0, batch.vertexCount);
            }
          }
        }

        if (weapon.hasWeapon && weapon.equipT > 0.005 && (weapon.scopeAmount || 0.0) <= 0.72) {
          const kickProg = player.kick.active ? (player.kick.timer / Math.max(player.kick.duration, 1e-4)) : 0.0;
          const g = (character.gender === "female") ? "female" : "male";
          const mdl = weaponGpu.models[weapon.activeWeapon] || weaponGpu.m1911;
          const xf = weapon.computePartsViewMatrices(kickProg, player.sliding, g);
          const topId = charCatalog.topStyles[character.top_style % charCatalog.topStyles.length].id;
          const topStyleIdx = (typeof TOP_STYLE_INDEX[topId] === "number") ? TOP_STYLE_INDEX[topId] : 0;

          const vm = (localM) => mat4Mul(vmToWorld, localM);
          drawWeaponBatches(mdl.frame, vm(xf.gun), slotCols);
          drawWeaponBatches(mdl.slide, vm(xf.slide), slotCols);
          drawWeaponBatches(mdl.trigger, vm(xf.trigger), slotCols);
          if (mdl.hammer && mdl.hammer.length) drawWeaponBatches(mdl.hammer, vm(xf.hammer), slotCols);
          if (mdl.suppressor && mdl.suppressor.length && xf.suppressorVis) drawWeaponBatches(mdl.suppressor, vm(xf.suppressor), slotCols);

          if (xf.magVisible && mdl.magazine && mdl.magazine.length) {
            const mm = vm(xf.magazine);
            drawWeaponBatches(mdl.magazine, mm, slotCols);
            if (xf.magHasRound && mdl.mag_round && mdl.mag_round.length) drawWeaponBatches(mdl.mag_round, mm, slotCols);
          }
          if (xf.shell12g && weaponGpu.m1911.shell_12g) {
            drawWeaponBatches(weaponGpu.m1911.shell_12g, vm(xf.shell12g), slotCols);
          }
          if (xf.extraShells12g && xf.extraShells12g.length && weaponGpu.m1911.shell_12g) {
            for (const em of xf.extraShells12g) {
              drawWeaponBatches(weaponGpu.m1911.shell_12g, vm(em), slotCols);
            }
          }
          if (xf.revChamberRounds && xf.revChamberRounds.length && weaponGpu.m1911.rev_round_live && weaponGpu.m1911.rev_round_spent) {
            for (const rc of xf.revChamberRounds) {
              drawWeaponBatches(rc.live ? weaponGpu.m1911.rev_round_live : weaponGpu.m1911.rev_round_spent, vm(rc.m), slotCols);
            }
          }
          if (xf.revHandRound && weaponGpu.m1911.rev_round_live) {
            drawWeaponBatches(weaponGpu.m1911.rev_round_live, vm(xf.revHandRound), slotCols);
          }

          // Rigged 23-bone GLB 1st-person hands (low-poly_fps_hands.glb) + 6 sculpted 3D sleeves!
          if (weaponGpu.fpHandsGlb) {
            const pose23R = buildGlb23BonePose(xf.handR, xf.wristR, xf.elbowR, xf.poseR, false);
            const skinR = evalGlbHandSkinMatrices(weaponGpu.fpHandsGlb.right, pose23R);
            drawGlbSkinnedHand(weaponGpu.fpHandsGlb.right, vm(xf.handRGlb), skinR, topStyleIdx, slotCols);

            const pose23L = buildGlb23BonePose(xf.handL, xf.wristL, xf.elbowL, xf.poseL, true);
            const skinL = evalGlbHandSkinMatrices(weaponGpu.fpHandsGlb.left, pose23L);
            drawGlbSkinnedHand(weaponGpu.fpHandsGlb.left, vm(xf.handLGlb), skinL, topStyleIdx, slotCols);
          }

          if (xf.flash !== null) {
            drawWeaponBatches(weaponGpu.m1911.muzzle_flash, vm(xf.flash), slotCols);
          }
        }
        gl.uniform3f(su.colorTint, 1.0, 1.0, 1.0);
      }

      // ---------------- PASS 2: Fullscreen Buckshot Post-Process ----------------
      gl.bindFramebuffer(gl.FRAMEBUFFER, null);
      gl.viewport(0, 0, canvas.width, canvas.height);
      gl.disable(gl.DEPTH_TEST);
      gl.disable(gl.CULL_FACE);

      gl.useProgram(postProg);
      gl.activeTexture(gl.TEXTURE0);
      gl.bindTexture(gl.TEXTURE_2D, fboColor);

      gl.uniform1i(pu.sceneTex, 0);
      gl.uniform2f(pu.lowRes, lowW, lowH);
      gl.uniform2f(pu.screenRes, canvas.width, canvas.height);
      gl.uniform1f(pu.time, elapsed);
      gl.uniform1f(pu.ditherEn, preset.dither_enabled ? 1.0 : 0.0);
      gl.uniform1f(pu.ditherStr, preset.dither_strength);
      gl.uniform1f(pu.colorLvls, preset.color_levels);
      gl.uniform1f(pu.scanEn, preset.scanlines ? 1.0 : 0.0);
      gl.uniform1f(pu.scanStr, preset.scanline_strength);
      gl.uniform1f(pu.vigStr, preset.vignette_strength);
      gl.uniform1f(pu.grainStr, preset.grain_strength);
      gl.uniform1f(pu.contrast, preset.contrast);
      gl.uniform1f(pu.saturation, preset.saturation);
      gl.uniform3f(pu.tint, preset.tint[0], preset.tint[1], preset.tint[2]);
      const hideSniperHip = (weapon.hasWeapon && weapon.equipped && weapon.isSniper);
      gl.uniform1f(pu.crosshair, (mode === "playing" && !hideSniperHip && weapon.adsT < 0.72 && (weapon.scopeAmount || 0.0) < 0.45) ? 1.0 : 0.0);
      if (pu.crosshairSpread) gl.uniform1f(pu.crosshairSpread, weapon.crosshairSpreadPx || 0.0);
      if (pu.fadeBlack) gl.uniform1f(pu.fadeBlack, 0.0);
      if (pu.scopeAmount) gl.uniform1f(pu.scopeAmount, weapon.scopeAmount || 0.0);
      if (pu.scopeBlur) gl.uniform1f(pu.scopeBlur, weapon.scopeBlur || 0.0);
      if (pu.fxImpact) gl.uniform1f(pu.fxImpact, weapon.fxImpactPulse || 0.0);
      if (pu.fxSpeed) gl.uniform1f(pu.fxSpeed, weapon.fxSpeedLines || 0.0);

      gl.bindVertexArray(quadVao);
      gl.drawArrays(gl.TRIANGLES, 0, 6);
      gl.bindVertexArray(null);
      } catch (err) {
        console.error("[PSX frame error]", err);
      }
    }

    requestAnimationFrame(frame);
  }

  if (typeof module !== "undefined" && module.exports) {
    module.exports = { WeaponController };
    return;
  }
  boot().catch(err => {
    const banner = document.getElementById("loading-banner");
    banner.textContent = "Ошибка запуска: " + (err && err.message ? err.message : String(err));
    console.error(err);
  });
})();
