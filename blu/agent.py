# Simple reliable team: one seeker (index 0) uses global A* to fetch enemy flag and return,
# two supporters (index != 0) shoot and chase enemies. No reservation system.
#
# Usage: copy to your team folder and run:
#   python main.py my_team other_team --headless --ascii
#
from collections import deque
import heapq
import random
from config import *

DIR_DELTAS = {
    "left": (-1, 0),
    "right": (1, 0),
    "up": (0, -1),
    "down": (0, 1)
}

def manhattan(a, b):
    return abs(a[0]-b[0]) + abs(a[1]-b[1])

class Agent:
    def __init__(self, color, index):
        self.color = color
        self.index = index
        self._rnd = random.Random(1000 + index)

        if color == "blue":
            self.enemy_agent_char = "r"
            self.enemy_flag_char = ASCII_TILES["red_flag"]
            self.friendly_flag_char = ASCII_TILES["blue_flag"]
            self.attack_direction = "right"
            self.home_direction = "left"
        else:
            self.enemy_agent_char = "b"
            self.enemy_flag_char = ASCII_TILES["blue_flag"]
            self.friendly_flag_char = ASCII_TILES["red_flag"]
            self.attack_direction = "left"
            self.home_direction = "right"

    # ---------------- Map utilities ----------------
    def _ensure_map(self, shared_knowledge):
        if "map" not in shared_knowledge:
            shared_knowledge["map"] = {}

    def _update_map_from_visible(self, visible_world, position, shared_knowledge):
        """Write observed visible_world tiles into shared_knowledge['map'] using absolute coords."""
        self._ensure_map(shared_knowledge)
        gm = shared_knowledge["map"]
        size = AGENT_VISION_RANGE * 2 + 1
        center = AGENT_VISION_RANGE
        for vy in range(size):
            for vx in range(size):
                ch = visible_world[vy][vx]
                abs_x = position[0] + (vx - center)
                abs_y = position[1] + (vy - center)
                gm[(abs_x, abs_y)] = ch
                if ch == self.enemy_flag_char:
                    shared_knowledge["enemy_flag"] = (abs_x, abs_y)
                if ch == self.friendly_flag_char:
                    shared_knowledge["friendly_flag"] = (abs_x, abs_y)

    # ---------------- A* on observed map ----------------
    def _astar(self, start, goal, shared_knowledge, max_nodes=20000):
        """A* using known map; unknown tiles treated as traversable."""
        if start == goal:
            return [start]
        self._ensure_map(shared_knowledge)
        gm = shared_knowledge["map"]
        open_heap = []
        heapq.heappush(open_heap, (manhattan(start, goal), 0, start))
        came_from = {start: None}
        gscore = {start: 0}
        visited = 0
        while open_heap:
            _, g, current = heapq.heappop(open_heap)
            visited += 1
            if visited > max_nodes:
                break
            if current == goal:
                # reconstruct
                path = []
                cur = current
                while cur is not None:
                    path.append(cur)
                    cur = came_from[cur]
                path.reverse()
                return path
            cx, cy = current
            for dx, dy in DIR_DELTAS.values():
                nb = (cx + dx, cy + dy)
                # skip known wall
                if gm.get(nb) == ASCII_TILES["wall"]:
                    continue
                tentative_g = g + 1
                if nb not in gscore or tentative_g < gscore[nb]:
                    gscore[nb] = tentative_g
                    f = tentative_g + manhattan(nb, goal)
                    heapq.heappush(open_heap, (f, tentative_g, nb))
                    came_from[nb] = current
        return []

    def _coords_to_dir(self, cur, nxt):
        dx = nxt[0] - cur[0]
        dy = nxt[1] - cur[1]
        for d, (ddx, ddy) in DIR_DELTAS.items():
            if (ddx, ddy) == (dx, dy):
                return d
        return None

    # Bresenham-like check for clear line in visible window (used for shooting)
    def _visible_has_clear_line(self, visible_world, sx, sy, tx, ty):
        x1, y1, x2, y2 = sx, sy, tx, ty
        dx = abs(x2 - x1); dy = abs(y2 - y1)
        sx_step = 1 if x1 < x2 else -1
        sy_step = 1 if y1 < y2 else -1
        err = dx - dy
        txc, tyc = x1, y1
        while True:
            if visible_world[tyc][txc] == ASCII_TILES["wall"]:
                return False
            if txc == x2 and tyc == y2:
                return True
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                txc += sx_step
            if e2 < dx:
                err += dx
                tyc += sy_step

    # ---------------- Supporter behavior ----------------
    def _supporter(self, visible_world, position, can_shoot, holding_flag, shared_knowledge, hp, ammo):
        size = AGENT_VISION_RANGE * 2 + 1
        center = AGENT_VISION_RANGE

        # publish pos so teammates can use it if needed
        shared_knowledge[f"agent_{self.index}_pos"] = position

        # 1) shooting priority: aligned & clear LOS
        enemies = []
        for y in range(size):
            for x in range(size):
                ch = visible_world[y][x]
                if ch.lower() == self.enemy_agent_char:
                    enemies.append((x, y))
        if can_shoot and ammo > 0 and enemies:
            sx, sy = center, center
            for (ex, ey) in enemies:
                if ex == sx or ey == sy:
                    if self._visible_has_clear_line(visible_world, sx, sy, ex, ey):
                        if ex == sx:
                            return "shoot", "up" if ey < sy else "down"
                        else:
                            return "shoot", "left" if ex < sx else "right"

        # 2) chase nearest visible enemy using A* on known map (robust)
        if enemies:
            target = min(enemies, key=lambda e: abs(e[0]-center) + abs(e[1]-center))
            tgt_abs = (position[0] + (target[0]-center), position[1] + (target[1]-center))
            path = self._astar(position, tgt_abs, shared_knowledge, max_nodes=2000)
            if path and len(path) >= 2:
                d = self._coords_to_dir(position, path[1])
                return "move", d
            # fallback greedy toward target
            dx = tgt_abs[0] - position[0]
            dy = tgt_abs[1] - position[1]
            prefer = []
            if abs(dx) >= abs(dy):
                prefer.append("right" if dx > 0 else "left")
                if dy != 0:
                    prefer.append("down" if dy > 0 else "up")
            else:
                prefer.append("down" if dy > 0 else "up")
                if dx != 0:
                    prefer.append("right" if dx > 0 else "left")
            # avoid immediate wall
            for d in prefer:
                ddx, ddy = DIR_DELTAS[d]
                tx, ty = center + ddx, center + ddy
                if 0 <= tx < size and 0 <= ty < size and visible_world[ty][tx] != ASCII_TILES["wall"]:
                    return "move", d

        # 3) go to remembered enemy_flag (help seeker) if known
        if "enemy_flag" in shared_knowledge and shared_knowledge["enemy_flag"]:
            tgt = shared_knowledge["enemy_flag"]
            path = self._astar(position, tgt, shared_knowledge, max_nodes=3000)
            if path and len(path) >= 2:
                return "move", self._coords_to_dir(position, path[1])

        # 4) default exploration toward attack side
        size_vis = AGENT_VISION_RANGE*2+1
        center = AGENT_VISION_RANGE
        prefer = [self.attack_direction, "up", "down", "left" if self.attack_direction == "right" else "right"]
        for d in prefer:
            ddx, ddy = DIR_DELTAS[d]
            tx, ty = center + ddx, center + ddy
            if 0 <= tx < size_vis and 0 <= ty < size_vis and visible_world[ty][tx] != ASCII_TILES["wall"]:
                return "move", d

        return "", None

    # ---------------- Seeker behavior ----------------
    def _seeker(self, visible_world, position, can_shoot, holding_flag, shared_knowledge, hp, ammo):
        """Seeker plans A* to enemy_flag (absolute) and, after pickup, A* to friendly_flag."""
        # publish pos
        shared_knowledge[f"agent_{self.index}_pos"] = position

        # compute goals
        goal = None
        goal_type = None
        if holding_flag:
            # return to friendly flag spawn (if known)
            if "friendly_flag" in shared_knowledge:
                goal = shared_knowledge["friendly_flag"]
                goal_type = "friendly"
        else:
            if "enemy_flag" in shared_knowledge:
                goal = shared_knowledge["enemy_flag"]
                goal_type = "enemy"

        # If we have a goal, compute A* from current position to it each tick (robust to dynamic changes)
        if goal:
            path = self._astar(position, goal, shared_knowledge, max_nodes=20000)
            if path and len(path) >= 2:
                next_coord = path[1]
                d = self._coords_to_dir(position, next_coord)
                # move toward next step
                return "move", d
            else:
                # A* failed (maybe unknown obstacles). Fallback: greedy move toward goal that avoids immediate walls.
                dx = goal[0] - position[0]
                dy = goal[1] - position[1]
                prefer = []
                if abs(dx) >= abs(dy):
                    prefer.append("right" if dx > 0 else "left")
                    if dy != 0:
                        prefer.append("down" if dy > 0 else "up")
                else:
                    prefer.append("down" if dy > 0 else "up")
                    if dx != 0:
                        prefer.append("right" if dx > 0 else "left")
                size_vis = AGENT_VISION_RANGE*2+1
                center = AGENT_VISION_RANGE
                for d in prefer:
                    ddx, ddy = DIR_DELTAS[d]
                    tx, ty = center + ddx, center + ddy
                    if 0 <= tx < size_vis and 0 <= ty < size_vis and visible_world[ty][tx] != ASCII_TILES["wall"]:
                        return "move", d
                # if blocked locally, try any free neighbor
                for d, (ddx, ddy) in DIR_DELTAS.items():
                    tx, ty = center + ddx, center + ddy
                    if 0 <= tx < size_vis and 0 <= ty < size_vis and visible_world[ty][tx] != ASCII_TILES["wall"]:
                        return "move", d
                return "", None

        # If no goal (haven't seen enemy flag yet), behave opportunistically: shoot/chase visible enemies or explore
        # shooting priority:
        size = AGENT_VISION_RANGE*2+1
        center = AGENT_VISION_RANGE
        enemies = []
        for y in range(size):
            for x in range(size):
                ch = visible_world[y][x]
                if ch.lower() == self.enemy_agent_char:
                    enemies.append((x, y))
        if can_shoot and ammo > 0 and enemies:
            sx, sy = center, center
            for (ex, ey) in enemies:
                if ex == sx or ey == sy:
                    if self._visible_has_clear_line(visible_world, sx, sy, ex, ey):
                        if ex == sx:
                            return "shoot", "up" if ey < sy else "down"
                        else:
                            return "shoot", "left" if ex < sx else "right"
        # chase nearest visible enemy (A* or greedy)
        if enemies:
            target = min(enemies, key=lambda e: abs(e[0]-center) + abs(e[1]-center))
            tgt_abs = (position[0] + (target[0]-center), position[1] + (target[1]-center))
            path = self._astar(position, tgt_abs, shared_knowledge, max_nodes=3000)
            if path and len(path) >= 2:
                return "move", self._coords_to_dir(position, path[1])
            # greedy fallback
            dx = tgt_abs[0] - position[0]
            dy = tgt_abs[1] - position[1]
            prefer = []
            if abs(dx) >= abs(dy):
                prefer.append("right" if dx > 0 else "left")
                if dy != 0:
                    prefer.append("down" if dy > 0 else "up")
            else:
                prefer.append("down" if dy > 0 else "up")
                if dx != 0:
                    prefer.append("right" if dx > 0 else "left")
            size_vis = AGENT_VISION_RANGE*2+1
            center = AGENT_VISION_RANGE
            for d in prefer:
                ddx, ddy = DIR_DELTAS[d]
                tx, ty = center + ddx, center + ddy
                if 0 <= tx < size_vis and 0 <= ty < size_vis and visible_world[ty][tx] != ASCII_TILES["wall"]:
                    return "move", d

        # exploratory movement: move toward attack side (avoid immediate wall)
        size_vis = AGENT_VISION_RANGE*2+1
        center = AGENT_VISION_RANGE
        prefer = [self.attack_direction, "up", "down", "left" if self.attack_direction == "right" else "right"]
        for d in prefer:
            ddx, ddy = DIR_DELTAS[d]
            tx, ty = center + ddx, center + ddy
            if 0 <= tx < size_vis and 0 <= ty < size_vis and visible_world[ty][tx] != ASCII_TILES["wall"]:
                return "move", d
        return "", None

    # ---------------- Main API ----------------
    def update(self, visible_world, position, can_shoot, holding_flag, shared_knowledge, hp, ammo):
        """
        visible_world: 2D list; agent centered at (AGENT_VISION_RANGE, AGENT_VISION_RANGE)
        position: absolute (x,y)
        holding_flag: truthy if agent holds enemy flag (AgentEngine passes Flag object or None)
        shared_knowledge: dict persisted across team agents for this match
        """
        # update global map and any seen flag positions
        self._update_map_from_visible(visible_world, position, shared_knowledge)

        # basic survival: if very low hp or out of ammo, retreat to friendly flag if known
        if hp <= 1 or ammo == 0:
            if "friendly_flag" in shared_knowledge and shared_knowledge["friendly_flag"]:
                refuge = shared_knowledge["friendly_flag"]
                path = self._astar(position, refuge, shared_knowledge, max_nodes=5000)
                if path and len(path) >= 2:
                    return "move", self._coords_to_dir(position, path[1])
            # fallback local retreat
            size_vis = AGENT_VISION_RANGE*2+1
            center = AGENT_VISION_RANGE
            for d in [self.home_direction, "up", "down", "left", "right"]:
                ddx, ddy = DIR_DELTAS[d]
                tx, ty = center + ddx, center + ddy
                if 0 <= tx < size_vis and 0 <= ty < size_vis and visible_world[ty][tx] != ASCII_TILES["wall"]:
                    return "move", d
            return "", None

        # dispatch role
        if self.index == 0:
            return self._seeker(visible_world, position, can_shoot, holding_flag, shared_knowledge, hp, ammo)
        else:
            return self._supporter(visible_world, position, can_shoot, holding_flag, shared_knowledge, hp, ammo)

    def terminate(self, reason):
        return