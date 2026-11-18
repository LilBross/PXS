#!/usr/bin/env python3
"""PSX 3D Sandbox — Lightweight Multiplayer WebSocket Server for Render.com.

Single-file aiohttp server serving:
  - GET  /health     -> Render.com health check
  - GET  /           -> Live server status page
  - GET  /api/rooms  -> List active rooms (name, players, maxPlayers, hasPassword)
  - POST /api/rooms  -> Create a new room (optional password)
  - WS   /ws         -> Realtime 20 Hz state sync, custom 3D skins, weapons, shots, PvP & U-chat

Deploy on Render.com (Free Web Service):
  Build Command: pip install -r requirements-server.txt
  Start Command: python multiplayer_server.py
"""
from __future__ import annotations

import asyncio
import math
from pathlib import Path
from html import escape
import hashlib
import json
import os
import re
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

from aiohttp import WSMsgType, web

MAX_PLAYERS_PER_ROOM = 16
MAX_CUSTOM_ROOMS = 40
TICK_HZ = 20.0
EMPTY_ROOM_TTL_SEC = 180.0
MAX_CHAT_HISTORY = 24
MAX_CHAT_LEN = 140
MAX_NICK_LEN = 20
MAX_ROOM_NAME_LEN = 28

GEOMETRY = json.loads(Path(__file__).with_name('arena_net.json').read_text(encoding='utf-8'))
SOLIDS = GEOMETRY['solids']
SPAWN_POINTS = GEOMETRY['spawns']

def num(v, default=0., lo=-1e5, hi=1e5):
    if not isinstance(v,(int,float)) or not math.isfinite(v):return default
    return max(lo,min(hi,v))

def vector(v):
    return isinstance(v,list) and len(v)==3 and all(isinstance(i,(int,float)) and math.isfinite(i) for i in v)

def ray(o,d,box):
    lo,hi=0.,90.
    for i in range(3):
        if abs(d[i])<1e-7:
            if not box[i]<=o[i]<=box[i+3]:return 91.
        else:
            a,b=sorted(((box[i]-o[i])/d[i],(box[i+3]-o[i])/d[i]));lo=max(a,lo);hi=min(b,hi)
            if hi<lo:return 91.
    return lo

WEAPON_LABELS = {
    "m1911": "M1911",
    "glock17": "Glock-17",
    "remington11": "Remington 11",
    "double_barrel": "Двустволка",
    "revolver": "Револьвер .357",
    "ak47": "AK-47", "m110":"M110", "mp7":"MP7", "mp5s":"MP5-S",
    "m4a1": "M4A1",
}


def _clean_text(s: Any, max_len: int) -> str:
    raw = str(s or "")
    raw = re.sub(r"[\x00-\x1f\x7f]", "", raw).strip()
    return raw[:max_len]


def _hash_pw(pw: str) -> str:
    if not pw:
        return ""
    return hashlib.sha256(pw.encode("utf-8")).hexdigest()


def _cors_headers() -> dict[str, str]:
    return {
        "Access-Control-Allow-Origin": "*",
        "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
        "Access-Control-Allow-Headers": "Content-Type",
    }


@dataclass
class PlayerSession:
    pid: str
    nick: str
    ws: web.WebSocketResponse
    room_id: str
    appearance: dict[str, Any] = field(default_factory=dict)
    x: float = 0.0
    y: float = 0.0
    z: float = 7.5
    yaw: float = 0.0
    pitch: float = 0.0
    speed: float = 0.0
    move_f: float = 0.0
    move_r: float = 0.0
    crouch: float = 0.0
    slide: bool = False
    grounded: bool = True
    kick_t: float = 0.0
    weapon: str = "m1911"
    equipped: bool = False
    ads: float = 0.0
    last_shot: float = -5.
    life: int = 0
    pose: dict = field(default_factory=dict)
    hp: int = 100
    last_seen: float = field(default_factory=time.monotonic)

    def to_public_dict(self) -> dict[str, Any]:
        return {
            "life": self.life, **self.pose,
            "id": self.pid,
            "nick": self.nick,
            "appearance": self.appearance,
            "x": round(self.x, 3),
            "y": round(self.y, 3),
            "z": round(self.z, 3),
            "yaw": round(self.yaw, 4),
            "pitch": round(self.pitch, 4),
            "speed": round(self.speed, 2),
            "moveF": round(self.move_f, 2),
            "moveR": round(self.move_r, 2),
            "crouch": round(self.crouch, 2),
            "slide": self.slide,
            "grounded": self.grounded,
            "kickT": round(self.kick_t, 3),
            "weapon": self.weapon,
            "equipped": self.equipped,
            "ads": round(self.ads, 2),
            "hp": self.hp,
        }

    def to_snapshot_dict(self) -> dict[str, Any]:
        return {
            "life": self.life, **self.pose,
            "id": self.pid,
            "x": round(self.x, 3),
            "y": round(self.y, 3),
            "z": round(self.z, 3),
            "yaw": round(self.yaw, 4),
            "pitch": round(self.pitch, 4),
            "speed": round(self.speed, 2),
            "moveF": round(self.move_f, 2),
            "moveR": round(self.move_r, 2),
            "crouch": round(self.crouch, 2),
            "slide": self.slide,
            "grounded": self.grounded,
            "kickT": round(self.kick_t, 3),
            "weapon": self.weapon,
            "equipped": self.equipped,
            "ads": round(self.ads, 2),
            "hp": self.hp,
        }


@dataclass
class GameRoom:
    room_id: str
    name: str
    password_hash: str = ""
    builtin: bool = False
    max_players: int = MAX_PLAYERS_PER_ROOM
    players: dict[str, PlayerSession] = field(default_factory=dict)
    chat_history: list[dict[str, Any]] = field(default_factory=list)
    created_at: float = field(default_factory=time.monotonic)
    last_empty_at: float = field(default_factory=time.monotonic)
    spawn_cursor: int = 0

    door_target: float = 0.
    items: dict = field(default_factory=lambda: dict(GEOMETRY['items']))
    owners: dict = field(default_factory=dict)

    def next_spawn(self) -> tuple[float, float, float, float]:
        for _ in range(len(SPAWN_POINTS)):
            sp = SPAWN_POINTS[self.spawn_cursor % len(SPAWN_POINTS)]; self.spawn_cursor += 1
            if all(math.hypot(sp[0]-p.x,sp[2]-p.z)>1.3 for p in self.players.values()):break
        return sp

    def to_summary(self) -> dict[str, Any]:
        return {
            "id": self.room_id,
            "name": self.name,
            "players": len(self.players),
            "maxPlayers": self.max_players,
            "hasPassword": bool(self.password_hash),
            "builtin": self.builtin,
        }

    def add_chat(self, msg: dict[str, Any]) -> None:
        self.chat_history.append(msg)
        if len(self.chat_history) > MAX_CHAT_HISTORY:
            del self.chat_history[:-MAX_CHAT_HISTORY]


class MultiplayerHub:
    def __init__(self) -> None:
        self.rooms: dict[str, GameRoom] = {}
        self._init_builtin_rooms()

    def _init_builtin_rooms(self) -> None:
        default_pw = os.environ.get("DEFAULT_ROOM_PASSWORD", "").strip()
        self.rooms["arena-1"] = GameRoom(
            room_id="arena-1",
            name="Арена #1 (Открытая)",
            password_hash="",
            builtin=True,
        )
        self.rooms["arena-2"] = GameRoom(
            room_id="arena-2",
            name="Арена #2 (PvP)",
            password_hash=_hash_pw(default_pw) if default_pw else "",
            builtin=True,
        )

    def list_rooms(self) -> list[dict[str, Any]]:
        self._cleanup_empty_rooms()
        ordered = sorted(
            self.rooms.values(),
            key=lambda r: (not r.builtin, -len(r.players), r.created_at),
        )
        return [r.to_summary() for r in ordered]

    def create_room(self, name: str, password: str = "") -> GameRoom:
        self._cleanup_empty_rooms()
        clean_name = _clean_text(name, MAX_ROOM_NAME_LEN) or f"Сервер #{len(self.rooms) + 1}"
        custom_count = sum(1 for r in self.rooms.values() if not r.builtin)
        if custom_count >= MAX_CUSTOM_ROOMS:
            # Drop oldest empty custom room if possible
            empties = [r for r in self.rooms.values() if not r.builtin and not r.players]
            if empties:
                empties.sort(key=lambda r: r.last_empty_at)
                del self.rooms[empties[0].room_id]
        rid = f"room-{uuid.uuid4().hex[:6]}"
        room = GameRoom(
            room_id=rid,
            name=clean_name,
            password_hash=_hash_pw(password.strip()) if password.strip() else "",
            builtin=False,
        )
        self.rooms[rid] = room
        return room

    def _cleanup_empty_rooms(self) -> None:
        now = time.monotonic()
        stale = [
            rid
            for rid, r in self.rooms.items()
            if (not r.builtin) and (not r.players) and (now - r.last_empty_at > EMPTY_ROOM_TTL_SEC)
        ]
        for rid in stale:
            del self.rooms[rid]

    async def broadcast(self, room: GameRoom, payload: dict[str, Any], exclude_pid: str | None = None) -> None:
        if not room.players:
            return
        raw = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        dead: list[str] = []
        for pid, pl in list(room.players.items()):
            if exclude_pid and pid == exclude_pid:
                continue
            if pl.ws.closed:
                dead.append(pid)
                continue
            try:
                await pl.ws.send_str(raw)
            except Exception:
                dead.append(pid)
        for pid in dead:
            await self.remove_player(room, pid)

    async def remove_player(self, room: GameRoom, pid: str) -> None:
        pl = room.players.pop(pid, None)
        if pl is None:
            return
        for w,owner in list(room.owners.items()):
            if owner==pid:
                del room.owners[w];room.items[w]=GEOMETRY['items'][w]
                await self.broadcast(room,{'type':'item','item':room.items[w],'weapon':w})
        if not room.players:
            room.door_target=0.;room.items=dict(GEOMETRY['items']);room.last_empty_at = time.monotonic()
        leave_chat = {
            "type": "chat",
            "id": "sys",
            "nick": "Сервер",
            "text": f"{pl.nick} вышел с сервера",
            "system": True,
            "ts": int(time.time()),
        }
        room.add_chat(leave_chat)
        await self.broadcast(room, {"type": "player_leave", "id": pid})
        await self.broadcast(room, leave_chat)

    async def snapshot_loop(self) -> None:
        dt = 1.0 / TICK_HZ
        while True:
            await asyncio.sleep(dt)
            self._cleanup_empty_rooms()
            for room in list(self.rooms.values()):
                if not room.players:
                    continue
                snap = {
                    "type": "snapshot",
                    "t": round(time.monotonic(), 3),
                    "players": [p.to_snapshot_dict() for p in room.players.values()],
                }
                await self.broadcast(room, snap)


def create_app() -> web.Application:
    hub = MultiplayerHub()
    app = web.Application()
    app["hub"] = hub

    async def respawn(room,pl):
        sp=room.next_spawn();pl.life+=1;pl.hp=100;pl.x,pl.y,pl.z,pl.yaw=sp;pl.speed=pl.pitch=pl.kick_t=0;pl.grounded=True;pl.slide=False;pl.crouch=0;pl.pose={}
        await pl.ws.send_json({'type':'respawn','spawn':sp,'hp':100,'life':pl.life})

    async def handle_options(_request: web.Request) -> web.Response:
        return web.Response(status=204, headers=_cors_headers())

    async def handle_health(_request: web.Request) -> web.Response:
        total_players = sum(len(r.players) for r in hub.rooms.values())
        return web.json_response(
            {"ok": True, "rooms": len(hub.rooms), "players": total_players, "version": "2.0"},
            headers=_cors_headers(),
        )

    async def handle_index(_request: web.Request) -> web.Response:
        rooms = hub.list_rooms()
        total = sum(r["players"] for r in rooms)
        rows = "".join(
            f"<li><b>{escape(r['name'])}</b> — {r['players']}/{r['maxPlayers']}"
            f"{' 🔒' if r['hasPassword'] else ''}</li>"
            for r in rooms
        )
        html = f"""<!doctype html>
<html lang="ru"><head><meta charset="utf-8"><title>PSX 3D Multiplayer Server</title>
<style>body{{background:#0d0f12;color:#dce6dc;font-family:monospace;padding:32px;max-width:640px;margin:auto;}}
h1{{color:#82d98e;}} code{{background:#1c2228;padding:2px 6px;border-radius:4px;color:#ffd27d;}}
ul{{line-height:1.8;}}</style></head><body>
<h1>PSX 3D Sandbox — Multiplayer Server</h1>
<p>Статус: <b style="color:#82d98e">ONLINE</b> · Игроков: <b>{total}</b> · Комнат: <b>{len(rooms)}</b></p>
<p>Вставьте URL этого сайта в поле <code>Сервер</code> в меню <b>МУЛЬТИПЛЕЕР</b> игры.</p>
<h3>Активные комнаты:</h3><ul>{rows}</ul>
</body></html>"""
        return web.Response(text=html, content_type="text/html", headers=_cors_headers())

    async def handle_list_rooms(_request: web.Request) -> web.Response:
        return web.json_response({"rooms": hub.list_rooms()}, headers=_cors_headers())

    async def handle_create_room(request: web.Request) -> web.Response:
        try:
            body = await request.json()
        except Exception:
            body = {}
        name = _clean_text(body.get("name", ""), MAX_ROOM_NAME_LEN)
        password = str(body.get("password", "") or "")[:48]
        room = hub.create_room(name=name, password=password)
        return web.json_response({"room": room.to_summary()}, status=201, headers=_cors_headers())

    async def handle_ws(request: web.Request) -> web.WebSocketResponse:
        ws = web.WebSocketResponse(heartbeat=25.0, max_msg_size=65536)
        await ws.prepare(request)

        player: PlayerSession | None = None
        room: GameRoom | None = None

        try:
            async for msg in ws:
                if msg.type == WSMsgType.TEXT:
                    try:
                        data = json.loads(msg.data)
                    except Exception:
                        continue
                    mtype = data.get("type")

                    if mtype == "ping":
                        await ws.send_json({"type": "pong", "t": data.get("t", 0)})
                        continue

                    if mtype == "join":
                        if player is not None and room is not None:
                            await hub.remove_player(room, player.pid)
                            player = None
                            room = None

                        nick = _clean_text(data.get("nick", ""), MAX_NICK_LEN) or f"Player_{uuid.uuid4().hex[:4]}"
                        req_room_id = _clean_text(data.get("room", "arena-1"), 40)
                        req_pw = str(data.get("password", "") or "").strip()[:48]
                        create_name = _clean_text(data.get("createName", ""), MAX_ROOM_NAME_LEN)

                        target_room = hub.rooms.get(req_room_id)
                        if target_room is None:
                            if create_name:
                                target_room = hub.create_room(name=create_name, password=req_pw)
                            else:
                                await ws.send_json({"type": "error", "message": "Сервер не найден. Обновите список."})
                                continue

                        if target_room.password_hash and _hash_pw(req_pw) != target_room.password_hash:
                            await ws.send_json({"type": "error", "message": "Неверный пароль!"})
                            continue

                        if len(target_room.players) >= target_room.max_players:
                            await ws.send_json({"type": "error", "message": "Сервер заполнен (16/16)"})
                            continue

                        pid = uuid.uuid4().hex[:8]
                        sx, sy, sz, syaw = target_room.next_spawn()
                        appearance = data.get("appearance")
                        if not isinstance(appearance, dict):
                            appearance = {}
                        player = PlayerSession(
                            pid=pid,
                            nick=nick,
                            ws=ws,
                            room_id=target_room.room_id,
                            appearance=appearance,
                            x=sx,
                            y=sy,
                            z=sz,
                            yaw=syaw,
                        )
                        room = target_room
                        room.players[pid] = player

                        await ws.send_json({
                            "type": "welcome",
                            "id": pid,
                            "spawn": [sx, sy, sz, syaw],
                            "room": room.to_summary(),
                            "players": [p.to_public_dict() for p in room.players.values() if p.pid != pid],
                            "life":0,"items":list(room.items.values()),"door":room.door_target,
                            "history": room.chat_history[-16:],
                        })

                        await hub.broadcast(
                            room,
                            {"type": "player_join", "player": player.to_public_dict()},
                            exclude_pid=pid,
                        )
                        join_chat = {
                            "type": "chat",
                            "id": "sys",
                            "nick": "Сервер",
                            "text": f"{player.nick} вошёл на сервер",
                            "system": True,
                            "ts": int(time.time()),
                        }
                        room.add_chat(join_chat)
                        await hub.broadcast(room, join_chat)
                        continue

                    if player is None or room is None:
                        continue

                    player.last_seen = time.monotonic()

                    if mtype == "state":
                        st=data.get('state',data)
                        if not isinstance(st,dict) or st.get('life',0)!=player.life:continue
                        coords=[st.get(k) for k in ('x','y','z')]
                        if not vector(coords) or coords[1]<-1 or max(abs(coords[0]),abs(coords[2]))>17 or coords[1]>10:
                            await respawn(room,player);continue
                        player.x,player.y,player.z=coords
                        player.yaw=num(st.get('yaw'));player.pitch=num(st.get('pitch'),0,-1.57,1.57)
                        player.speed=num(st.get('speed'),0,0,20);player.crouch=bool(st.get('crouch'));player.slide=bool(st.get('slide'));player.grounded=bool(st.get('grounded'));player.kick_t=num(st.get('kickT'),0,0,1);player.ads=num(st.get('ads'),0,0,1)
                        w=st.get('weapon');player.weapon=w if isinstance(w,str) and w in WEAPON_LABELS else player.weapon;player.equipped=bool(st.get('equipped'))
                        player.pose={k:num(st.get(k)) for k in ('walkPhase','velY','reloadU','inspect','magCheck','slideKick','ammo','reserve')};player.pose.update({k:bool(st.get(k)) for k in ('reload','flash','suppressed')});player.pose['reloadType']=str(st.get('reloadType','mag'))[:16]
                    elif mtype=='door':
                        room.door_target=num(data.get('target'),0,0,2.04)
                        await hub.broadcast(room,{'type':'door','target':room.door_target,'sound':data.get('sound') if data.get('sound') in ('door_open','door_close','door_quiet','door_kick') else ''},exclude_pid=player.pid)
                    elif mtype=='take':
                        w=data.get('weapon');it=room.items.get(w) if isinstance(w,str) else None
                        if not it or list(room.owners.values()).count(player.pid)>=2:continue
                        x,y,z=it['pos']
                        if math.hypot(player.x-x,player.z-z)>3.4 or abs(y-player.y-.95)>2.3:continue
                        room.owners[w]=player.pid;del room.items[w]
                        await hub.broadcast(room,{'type':'item','owner':player.pid,'taken':it,'weapon':w})
                    elif mtype=='drop':
                        w=data.get('weapon')
                        if not isinstance(w,str) or room.owners.get(w)!=player.pid:continue
                        x,z=player.x,player.z
                        # Land on the supporting surface, not inside the floor or behind a wall.
                        top=max((c[4] for c in SOLIDS if c[0]<=x<=c[3] and c[2]<=z<=c[5] and c[4]<=player.y+.19),default=0)
                        it={**GEOMETRY['items'][w],'pos':[x,top+.045,z],'magAmmo':int(num(data.get('ammo'),0,0,99)),'reserveAmmo':int(num(data.get('reserve'),0,0,999)),'spentShells':int(num(data.get('spent'),0,0,99)),'slideLocked':bool(data.get('locked')),'label':'На полу'}
                        room.items[w]=it;del room.owners[w];await hub.broadcast(room,{'type':'item','item':it,'weapon':w})
                    elif mtype=='sfx':
                        ev=data.get('event');
                        if isinstance(ev,str) and re.fullmatch(r'[a-z_0-9]{1,32}',ev):await hub.broadcast(room,{'type':'sfx','event':ev},exclude_pid=player.pid)

                    elif mtype == "appearance":
                        ap = data.get("appearance")
                        if isinstance(ap, dict):
                            player.appearance = ap
                            await hub.broadcast(
                                room,
                                {"type": "player_appearance", "id": player.pid, "appearance": ap},
                                exclude_pid=player.pid,
                            )

                    elif mtype == "chat":
                        text = _clean_text(data.get("text", ""), MAX_CHAT_LEN)
                        if not text:
                            continue
                        chat_msg = {
                            "type": "chat",
                            "id": player.pid,
                            "nick": player.nick,
                            "text": text,
                            "system": False,
                            "ts": int(time.time()),
                        }
                        room.add_chat(chat_msg)
                        await hub.broadcast(room, chat_msg)

                    elif mtype == "shoot":
                        w = str(data.get("weapon", player.weapon))
                        if w not in WEAPON_LABELS:
                            w = player.weapon
                        pos = data.get("pos", [player.x, player.y + 1.5, player.z])
                        direction = data.get("dir", [0.0, 0.0, -1.0])
                        if data.get('life',0)!=player.life or not vector(pos) or not vector(direction):continue
                        if math.dist(pos,[player.x,player.y+1.35,player.z])>2.8:continue
                        length=math.hypot(*direction)
                        if length<.001:continue
                        if room.owners.get(w)!=player.pid:continue
                        spec=GEOMETRY['shots'][w];now=time.monotonic()
                        if now-player.last_shot<spec[1]*.72:continue
                        player.last_shot=now;solids=list(SOLIDS);angle=room.door_target
                        ex=4.8+1.26*math.sin(angle);ez=10.52+1.26*math.cos(angle)
                        solids.append([4.66,0,10.45,4.94,2.40,11.85] if angle<.52 else [min(4.8,ex)-.09,0,min(10.52,ez)-.09,max(4.8,ex)+.09,2.36,max(10.52,ez)+.09])
                        packet=[];hits={};base={'m110':85,'revolver':52,'double_barrel':65,'remington11':48,'ak47':34,'m4a1':30,'mp7':22,'glock17':22,'mp5s':24}.get(w,28)
                        rays=data.get('rays');rays=rays[:spec[0]] if isinstance(rays,list) and rays else [direction]
                        for rv in rays:
                            if not vector(rv) or math.hypot(*rv)<.001:continue
                            d=[v/math.hypot(*rv) for v in rv];t,box=min(((ray(pos,d,c),c) for c in solids),key=lambda pair:pair[0]);stop=min(85.,t);target=None
                            for q in room.players.values():
                                if q is player:continue
                                tt=ray(pos,d,[q.x-.36,q.y,q.z-.36,q.x+.36,q.y+(1.08 if q.crouch or q.slide else 1.75),q.z+.36])
                                if tt<stop:stop=tt;target=q
                            end=[pos[i]+stop*d[i] for i in range(3)];n=[0.,0.,0.];face=min(range(6),key=lambda j:abs(end[j%3]-box[j]));n[face%3]=1 if face>=3 else -1
                            packet.append({'end':end,'impact':target is None and stop<85.,'normal':n});
                            if target:hits[target.pid]=hits.get(target.pid,0)+max(1,round(base/spec[0]))
                        if not packet:continue
                        await hub.broadcast(room,{'type':'shoot','id':player.pid,'pos':pos,'dir':direction,'end':packet[0]['end'],'rays':packet,'weapon':w,'suppressed':bool(data.get('suppressed'))},exclude_pid=player.pid)
                        for tid,dmg in hits.items():
                            target=room.players.get(tid);w_name=WEAPON_LABELS[w]
                            if not target:continue
                            target.hp=max(0,target.hp-dmg)
                            await player.ws.send_json({'type':'hit_confirm','target':target.pid})
                            if not target.ws.closed:
                                await target.ws.send_json({
                                    "type": "damaged",
                                    "hp": target.hp,
                                    "damage": dmg,
                                    "by": player.nick,
                                    "weapon": w,
                                })
                            if target.hp <= 0:
                                await respawn(room,target)
                                kill_chat = {
                                    "type": "chat",
                                    "id": "sys",
                                    "nick": "Бой",
                                    "text": f"{player.nick} -> {target.nick} ({w_name})",
                                    "system": True,
                                    "ts": int(time.time()),
                                }
                                room.add_chat(kill_chat)
                                await hub.broadcast(room, kill_chat)


                elif msg.type in (WSMsgType.ERROR, WSMsgType.CLOSE):
                    break
        finally:
            if player is not None and room is not None:
                await hub.remove_player(room, player.pid)
        return ws

    app.router.add_route("OPTIONS", "/{tail:.*}", handle_options)
    from pathlib import Path
    site = Path(__file__).resolve().parent / 'site'
    async def game_page(request):
        return web.FileResponse(site / {'/':'index.html','/game.js':'game.js','/caseboard.png':'caseboard.png'}[request.path])
    async def scene(request):
        return web.Response(body=(site/'scene.json.gz').read_bytes(),content_type='application/json',headers={'Content-Encoding':'gzip'})
    for path in ('/','/game.js','/caseboard.png'):app.router.add_get(path,game_page)
    app.router.add_get('/api/scene',scene)
    app.router.add_static('/assets/sfx',site/'audio',show_index=False)
    app.router.add_get('/status', handle_index)
    app.router.add_get("/health", handle_health)
    app.router.add_get("/api/rooms", handle_list_rooms)
    app.router.add_post("/api/rooms", handle_create_room)
    app.router.add_get("/ws", handle_ws)

    async def on_startup(application: web.Application) -> None:
        application["snap_task"] = asyncio.create_task(hub.snapshot_loop())

    async def on_cleanup(application: web.Application) -> None:
        task = application.get("snap_task")
        if task is not None:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

    app.on_startup.append(on_startup)
    app.on_cleanup.append(on_cleanup)
    return app


def main() -> None:
    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", "8765"))
    print(f"[PSX Multiplayer Server] Listening on http://{host}:{port} (WS: /ws, Rooms: /api/rooms)")
    web.run_app(create_app(), host=host, port=port, print=None)


if __name__ == "__main__":
    main()
