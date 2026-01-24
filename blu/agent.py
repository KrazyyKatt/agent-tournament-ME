from collections import deque
import heapq
from config import *

WALL = ASCII_TILES["wall"]
UNK  = ASCII_TILES["unknown"]
EMP  = ASCII_TILES["empty"]

DIRS = {
    "right": (1, 0),
    "left": (-1, 0),
    "down": (0, 1),
    "up": (0, -1),
}

class Agent:

    def __init__(self, color, index):
        self.color = color
        self.index = index

        self.enemy_flag_tile = ASCII_TILES["red_flag"] if color == "blue" else ASCII_TILES["blue_flag"]
        self.home_flag_tile  = ASCII_TILES["blue_flag"] if color == "blue" else ASCII_TILES["red_flag"]

        # blue ide desno, red ide lijevo :contentReference[oaicite:1]{index=1}
        self.enemy_bias_dx = 1 if color == "blue" else -1

        # movement feedback
        self.prev_pos = None
        self.pending_target = None

        # anti-loop
        self.recent = deque(maxlen=6)
        self.stuck_count = 0

    # ---------- Shared knowledge ----------
    def _sk_init(self, sk):
        sk.setdefault("map", {})              # (x,y)->tile
        sk.setdefault("enemy_flag_pos", None)
        sk.setdefault("home_flag_pos", None)

    def _update_shared_map(self, visible_world, position, sk):
        cx, cy = position
        r = AGENT_VISION_RANGE
        m = sk["map"]

        for dy in range(-r, r + 1):
            row = visible_world[dy + r]
            for dx in range(-r, r + 1):
                tile = row[dx + r]
                if tile == UNK:
                    continue
                wx, wy = cx + dx, cy + dy

                # treat dynamic entities as empty (navigation)
                if tile in (
                    ASCII_TILES["blue_agent"], ASCII_TILES["red_agent"],
                    ASCII_TILES["blue_agent_f"], ASCII_TILES["red_agent_f"],
                    ASCII_TILES["bullet"]
                ):
                    tile = EMP

                m[(wx, wy)] = tile

                if tile == self.enemy_flag_tile:
                    sk["enemy_flag_pos"] = (wx, wy)
                elif tile == self.home_flag_tile:
                    sk["home_flag_pos"] = (wx, wy)

    # ---------- Learning from bumps ----------
    def _apply_bump_learning(self, position, sk):
        if self.prev_pos is None:
            return
        if position == self.prev_pos and self.pending_target is not None:
            sk["map"][self.pending_target] = WALL
            self.stuck_count += 1
        else:
            self.stuck_count = 0

        self.pending_target = None

    # ---------- Pathfinding (Dijkstra with unknown penalty) ----------
    def _tile_cost(self, sk_map, x, y):
        # out of bounds are walls :contentReference[oaicite:2]{index=2}
        if x <= 0 or y <= 0 or x >= WIDTH - 1 or y >= HEIGHT - 1:
            return None  # impassable

        t = sk_map.get((x, y), None)
        if t == WALL:
            return None
        if t is None:
            return 4  # UNKNOWN penalty (optimistic but cautious)
        return 1      # known empty/flags

    def _dijkstra_next_step(self, start, goal, sk_map):
        if goal is None or goal == start:
            return None

        pq = [(0, start)]
        dist = {start: 0}
        prev = {start: None}

        while pq:
            d, (x, y) = heapq.heappop(pq)
            if (x, y) == goal:
                break
            if d != dist.get((x, y), 10**9):
                continue

            for dx, dy in DIRS.values():
                nx, ny = x + dx, y + dy
                c = self._tile_cost(sk_map, nx, ny)
                if c is None:
                    continue
                nd = d + c
                if nd < dist.get((nx, ny), 10**9):
                    dist[(nx, ny)] = nd
                    prev[(nx, ny)] = (x, y)
                    heapq.heappush(pq, (nd, (nx, ny)))

        if goal not in prev:
            return None

        # reconstruct: find first step after start
        cur = goal
        while prev[cur] != start:
            cur = prev[cur]
            if cur is None:
                return None
        return cur

    # ---------- Exploration target ----------
    def _pick_explore_goal(self, position, sk):
        sk_map = sk["map"]
        px, py = position

        def is_frontier(x, y):
            if sk_map.get((x, y), None) in (None, WALL):
                return False
            for dx, dy in DIRS.values():
                if sk_map.get((x + dx, y + dy), None) is None:
                    return True
            return False

        best = None
        best_score = None

        # 1) frontiers from known map
        for (x, y), t in sk_map.items():
            if t == WALL:
                continue
            if not is_frontier(x, y):
                continue
            dist = abs(x - px) + abs(y - py)
            forward = (x - px) * self.enemy_bias_dx  # prefer forward
            score = dist - 0.7 * forward

            if (x, y) in self.recent:
                score += 2.5

            if best_score is None or score < best_score:
                best_score = score
                best = (x, y)

        if best is not None:
            return best

        # 2) if no frontier known yet: push forward deterministically
        # aim a point forward + small vertical variation by agent index to spread
        target_x = px + 6 * self.enemy_bias_dx
        target_y = py + ((self.index % 3) - 1) * 2
        target_x = max(1, min(WIDTH - 2, target_x))
        target_y = max(1, min(HEIGHT - 2, target_y))
        return (target_x, target_y)

    # ---------- Move decision ----------
    def _step_to_dir(self, position, nxt):
        x, y = position
        nx, ny = nxt
        if nx > x: return "right"
        if nx < x: return "left"
        if ny > y: return "down"
        if ny < y: return "up"
        return None

    def _set_pending(self, position, direction):
        dx, dy = DIRS[direction]
        self.pending_target = (position[0] + dx, position[1] + dy)

    def _escape_move(self, position, sk):
        sk_map = sk["map"]
        px, py = position

        # deterministic order biased forward, then vertical, then back
        ordered = []
        forward_dir = "right" if self.enemy_bias_dx == 1 else "left"
        back_dir = "left" if forward_dir == "right" else "right"
        ordered.append(forward_dir)
        ordered += ["up", "down", back_dir]

        for d in ordered:
            dx, dy = DIRS[d]
            nx, ny = px + dx, py + dy
            if self._tile_cost(sk_map, nx, ny) is None:
                continue
            if (nx, ny) in self.recent:
                continue
            return d

        # if everything repeats, just take any passable
        for d in ordered:
            dx, dy = DIRS[d]
            nx, ny = px + dx, py + dy
            if self._tile_cost(sk_map, nx, ny) is not None:
                return d

        return None

    def update(self, visible_world, position, can_shoot, holding_flag, shared_knowledge, hp, ammo):
        self._sk_init(shared_knowledge)

        # bump learning uses last tick outcome
        self._apply_bump_learning(position, shared_knowledge)

        # update map with new vision
        self._update_shared_map(visible_world, position, shared_knowledge)

        sk_map = shared_knowledge["map"]
        enemy_flag_pos = shared_knowledge["enemy_flag_pos"]
        home_flag_pos  = shared_knowledge["home_flag_pos"]

        self.recent.append(position)
        self.prev_pos = position

        # Choose goal
        if holding_flag:
            goal = home_flag_pos
        else:
            goal = enemy_flag_pos if enemy_flag_pos is not None else self._pick_explore_goal(position, shared_knowledge)

        nxt = self._dijkstra_next_step(position, goal, sk_map)

        # If pathfinding fails or we are stuck, escape
        if nxt is None or self.stuck_count >= 2:
            d = self._escape_move(position, shared_knowledge)
            if d is None:
                return "", ""
            self._set_pending(position, d)
            return "move", d

        d = self._step_to_dir(position, nxt)
        if d is None:
            return "", ""

        self._set_pending(position, d)
        return "move", d

    def terminate(self, reason):
        pass
