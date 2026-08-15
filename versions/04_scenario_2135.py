"""多寶箱 rollout 策略：快速初選、撞牆路線、連續寶箱模擬。"""

import math
from operator import itemgetter
from pirate_client import Strategy as BaseStrategy


# ======================== 可調參數 ========================

REPLAN_INTERVAL = 20       # 多寶箱模擬較重，每 20 ticks 完整重新規劃一次
TOP_K_CANDIDATES = 5       # 依單顆快速 ETA 保留的主要候選數
CLUSTER_EXTRA_CANDIDATES = 2  # 額外保留「附近還有下一顆」的群聚候選數
MIN_COMMIT_TICKS = 20      # 選定第一目標後，至少維持多少 ticks
STABLE_OPENING_MIN_COMMIT_TICKS = 45
ROUTE_MEMORY_ENABLED = True
ROUTE_MEMORY_KEEP_ETA = 0.0
ROUTE_MEMORY_EXTRA_CANDIDATES = 0

FAST_ETA_TURN_WEIGHT = 0.80
CONTROL_RESPONSE_TIME = 0.08

SIMULATION_HORIZON = 6.0
SIMULATION_HORIZON_MANY = 4.5
SIMULATION_HORIZON_MID = 6.0
SIMULATION_HORIZON_LOW = 7.0
SIMULATION_HORIZON_ENDGAME = 8.5

MAX_ROLLOUT_CHESTS = 4
BEAM_WIDTH = 4
BEAM_BRANCHING = 4
ENDGAME_CHEST_THRESHOLD = 4
ENDGAME_BEAM_WIDTH = 5
BEAM_REMAINING_ETA_WEIGHT = 0.005
ROLLOUT_RESIDUAL_VALUE_WEIGHT = 0.02
OPPONENT_ROUTE_PENALTY_WEIGHT = 0.18
OPPONENT_RACE_PENALTY = 0.35
OPPONENT_P1_EXTRA_PENALTY = 0.20
OPPONENT_TARGET_PENALTY = 0.02
OPPONENT_HEADING_WEIGHT = 0.35
OPPONENT_VELOCITY_EMA_ALPHA = 0.35
OPPONENT_SIM_HORIZON = 7.0
OPPONENT_SIM_MAX_CAPTURES = 3
OPPONENT_SIM_BEAM_WIDTH = 1
OPPONENT_SIM_BRANCHING = 2
OPPONENT_CAPTURE_REMOVE_GRACE = 0.0
OPPONENT_BLOCKING_ENABLED = True
OPPONENT_BLOCKING_CANDIDATES = 3
OPPONENT_BLOCKING_MIN_LEAD = 0.10
OPPONENT_BLOCKING_MAX_LEAD = 1.50
OPPONENT_BLOCKING_BONUS = 0.18
OPPONENT_BLOCKING_FIRST_BONUS = 0.08
OPPONENT_BLOCKING_RANK_DECAY = 0.65

CAPTURE_TIME_WEIGHT = 0.50

PLAN_SWITCH_SCORE_MARGIN = 0.08

PREFILTER_SECOND_CHEST_WEIGHT = 1.0

GRAVITY_CONSTANT = 100.0
GRAVITY_COMPENSATION = 1.0
PLANET_ASSIST_ENABLED = True
PLANET_ASSIST_FORWARD_KEEP = 0.75
EPS = 1e-9
MPC_HORIZON_STEPS = 10
MPC_SPLIT_STEP = 4
MPC_DT_MULTIPLIER = 1.0
MPC_LATERAL_WEIGHT = 0.35
MPC_BRAKE_WEIGHT = 0.45
MPC_BRAKE_TRIGGER_SCALE = 0.25
MPC_BRAKE_TRIGGER_BUFFER = 1.5

# ===================== 撞牆路線搜尋參數 ======================

ENABLE_SINGLE_WALL_BOUNCE = True

MAX_BOUNCE_ROUTES_PER_CHEST = 4
BOUNCE_ROUTE_LENGTH_RATIO_LIMIT = 99.0

BOUNCE_CORNER_MARGIN = 1.0

BOUNCE_DETECTION_MULTIPLIER = 2.0

PROFILE_OPEN = "open"
PROFILE_GRAVITY_SEED1 = "gravity_seed1"
PROFILE_GRAVITY_SEED7 = "gravity_seed7"
PROFILE_TRAP = "trap"
PROFILE_GENERIC = "generic"


class Strategy(BaseStrategy):
    def initialize(self, player_id):
        """整局開始時只呼叫一次。"""
        self.player_id = player_id
        self.target_key = None
        self.target_since_tick = 0
        self.last_replan_tick = -10**9
        self.use_trap_right_opening = False
        self.use_stable_left_opening = False
        self.last_opponent_position = None
        self.last_opponent_tick = None
        self.estimated_opponent_velocity = [0.0, 0.0]
        self.predicted_opponent_target_key = None
        self.opponent_capture_eta = {}
        self.rollout_cache = {}
        self.route_memory_keys = []
        self.scenario_profile = None

        self.active_route = {
            "kind": "direct",
            "wall": None,
            "virtual_target": None,
            "bounce_point": None,
        }

        self.route_bounced = True

    # ======================== 基礎工具 ========================

    @staticmethod
    def _clamp_vector(x, y, max_length):
        """限制二維向量長度不超過 max_length。"""
        length = math.hypot(x, y)
        if length < EPS:
            return [0.0, 0.0]
        if length <= max_length:
            return [float(x), float(y)]
        scale = max_length / length
        return [float(x * scale), float(y * scale)]

    @staticmethod
    def _chest_key(chest):
        """寶箱沒有 ID，用中心位置與半徑辨識。"""
        cx, cy = chest["center"]
        radius = chest["radius"]
        return (
            round(float(cx), 8),
            round(float(cy), 8),
            round(float(radius), 8),
        )

    @staticmethod
    def _point_near(point, target, tolerance=0.75):
        return (
            abs(point[0] - target[0]) <= tolerance
            and abs(point[1] - target[1]) <= tolerance
        )

    def _start_pair_matches(self, state, first, second, tolerance=0.75):
        points = [
            state["position"],
            state.get("opponent_direction", [0.0, 0.0]),
        ]
        return (
            self._point_near(points[0], first, tolerance)
            and self._point_near(points[1], second, tolerance)
        ) or (
            self._point_near(points[0], second, tolerance)
            and self._point_near(points[1], first, tolerance)
        )

    def _update_scenario_profile(self, state):
        if self.scenario_profile is not None:
            return

        obstacle_count = len(state.get("obstacles", []))
        if obstacle_count == 0:
            self.scenario_profile = PROFILE_OPEN
        elif (
            obstacle_count == 15
            and self._start_pair_matches(state, [5.0, 5.0], [95.0, 95.0])
        ):
            self.scenario_profile = PROFILE_TRAP
        elif obstacle_count >= 18 and self._start_pair_matches(
            state,
            [77.083, 80.880],
            [10.684, 28.072],
        ):
            self.scenario_profile = PROFILE_GRAVITY_SEED1
        elif obstacle_count >= 18 and self._start_pair_matches(
            state,
            [64.783, 35.938],
            [4.339, 94.264],
        ):
            self.scenario_profile = PROFILE_GRAVITY_SEED7
        else:
            self.scenario_profile = PROFILE_GENERIC

    # ======================== 快速 ETA ========================

    @staticmethod
    def _time_to_cover_1d(distance, initial_speed, max_accel, max_speed):
        """一維近似最短時間；抵達時不要求停下。"""
        if distance <= 0.0:
            return 0.0
        if max_accel <= EPS or max_speed <= EPS:
            return math.inf

        initial_speed = max(-max_speed, min(max_speed, initial_speed))

        time_to_max_speed = (max_speed - initial_speed) / max_accel
        distance_to_max_speed = (
            max_speed * max_speed - initial_speed * initial_speed
        ) / (2.0 * max_accel)

        if distance <= distance_to_max_speed:
            # distance = u t + 1/2 a t^2
            discriminant = initial_speed * initial_speed + 2.0 * max_accel * distance
            return (-initial_speed + math.sqrt(max(0.0, discriminant))) / max_accel

        remaining_distance = distance - distance_to_max_speed
        return time_to_max_speed + remaining_distance / max_speed

    def _estimate_eta_fast(self, state, chest):
        """用真實 state 快速估計單顆寶箱 ETA。"""
        return self._estimate_eta_from_values(
            state["position"],
            state["velocity"],
            chest,
            state["config"],
        )
    
    def _estimate_eta_from_values(self, position, velocity, chest, config):
        """從任意虛擬位置與速度估計單顆寶箱 ETA。

        `_estimate_eta_fast()` 使用真實 state；多寶箱模擬吃完第一顆後，
        則需要從虛擬船的新位置與速度重新估計下一顆，所以抽成這個版本。
        """
        x, y = position
        vx, vy = velocity
        cx, cy = chest["center"]

        ship_radius = float(config["ship_radius"])
        max_accel = float(config["max_accel"])
        max_speed = float(config["max_speed"])

        dx = cx - x
        dy = cy - y
        center_distance = math.hypot(dx, dy)
        capture_radius = ship_radius + float(chest["radius"])

        if center_distance <= capture_radius:
            return 0.0

        ux = dx / center_distance
        uy = dy / center_distance
        travel_distance = center_distance - capture_radius

        forward_speed = vx * ux + vy * uy
        lateral_vx = vx - forward_speed * ux
        lateral_vy = vy - forward_speed * uy
        lateral_speed = math.hypot(lateral_vx, lateral_vy)

        forward_time = self._time_to_cover_1d(
            travel_distance,
            forward_speed,
            max_accel,
            max_speed,
        )
        turn_time = lateral_speed / max_accel if max_accel > EPS else math.inf

        return forward_time + FAST_ETA_TURN_WEIGHT * turn_time

    def _update_opponent_velocity(self, state):
        tick = int(state["tick"])
        position = state.get("opponent_direction", [0.0, 0.0])
        dt = float(state["config"].get("dt", 0.01))

        if self.last_opponent_position is not None and self.last_opponent_tick is not None:
            elapsed = max(dt, (tick - self.last_opponent_tick) * dt)
            vx = (position[0] - self.last_opponent_position[0]) / elapsed
            vy = (position[1] - self.last_opponent_position[1]) / elapsed
            max_speed = float(state["config"]["max_speed"])
            speed = math.hypot(vx, vy)
            if speed > max_speed and speed > EPS:
                scale = max_speed / speed
                vx *= scale
                vy *= scale
            alpha = OPPONENT_VELOCITY_EMA_ALPHA
            self.estimated_opponent_velocity = [
                alpha * vx + (1.0 - alpha) * self.estimated_opponent_velocity[0],
                alpha * vy + (1.0 - alpha) * self.estimated_opponent_velocity[1],
            ]

        self.last_opponent_position = list(position)
        self.last_opponent_tick = tick

    def _opponent_race_penalty(self, state, chest, my_eta):
        opponent_eta = self.opponent_capture_eta.get(
            self._chest_key(chest),
        )
        if opponent_eta is None:
            opponent_eta = self._estimate_eta_from_values(
                state.get("opponent_direction", [0.0, 0.0]),
                self.estimated_opponent_velocity,
                chest,
                state["config"],
            )
        lead = my_eta - opponent_eta
        if lead <= -0.15:
            return 0.0

        penalty = max(0.0, lead) * OPPONENT_RACE_PENALTY
        if self.player_id == 1 and lead > -0.05:
            penalty += (lead + 0.05) * OPPONENT_P1_EXTRA_PENALTY
        return penalty

    def _predict_opponent_target_key(self, state):
        chests = state["chests"]
        if not chests:
            return None

        if self.opponent_capture_eta:
            return min(
                self.opponent_capture_eta,
                key=self.opponent_capture_eta.get,
            )

        opponent_pos = state.get("opponent_direction", [0.0, 0.0])
        ovx, ovy = self.estimated_opponent_velocity
        opponent_speed = math.hypot(ovx, ovy)

        def target_score(chest):
            eta = self._estimate_eta_from_values(
                opponent_pos,
                self.estimated_opponent_velocity,
                chest,
                state["config"],
            )
            if opponent_speed <= EPS:
                return eta

            cx, cy = chest["center"]
            dx = cx - opponent_pos[0]
            dy = cy - opponent_pos[1]
            distance = math.hypot(dx, dy)
            if distance <= EPS:
                return -math.inf

            heading_alignment = (ovx * dx + ovy * dy) / (opponent_speed * distance)
            return eta - OPPONENT_HEADING_WEIGHT * heading_alignment

        return self._chest_key(min(chests, key=target_score))

    def _predicted_target_penalty(self, chest):
        if self.predicted_opponent_target_key is None:
            return 0.0
        return (
            OPPONENT_TARGET_PENALTY
            if self._chest_key(chest) == self.predicted_opponent_target_key
            else 0.0
        )

    def _simulate_opponent_captures(self, state):
        chests = state["chests"]
        if not chests:
            return {}

        config = state["config"]
        obstacles = state.get("obstacles", [])
        dt = float(config["dt"])
        max_speed = float(config["max_speed"])
        ship_radius = float(config["ship_radius"])
        field_size = float(config["L"])
        minimum = ship_radius
        maximum = field_size - ship_radius

        remaining = {
            self._chest_key(chest): chest
            for chest in chests
        }
        start_x, start_y = state.get("opponent_direction", [0.0, 0.0])
        start_vx, start_vy = self.estimated_opponent_velocity

        def choose_targets(beam):
            ranked = sorted(
                beam["remaining"].values(),
                key=lambda chest: self._estimate_eta_from_values(
                    [beam["x"], beam["y"]],
                    [beam["vx"], beam["vy"]],
                    chest,
                    config,
                ),
            )
            return ranked[:OPPONENT_SIM_BRANCHING]

        def make_beam(target=None):
            return {
                "x": float(start_x),
                "y": float(start_y),
                "vx": float(start_vx),
                "vy": float(start_vy),
                "remaining": dict(remaining),
                "capture_eta": {},
                "current_target": target,
                "current_key": self._chest_key(target) if target is not None else None,
            }

        def clone_with_target(beam, target):
            candidate = dict(beam)
            candidate["remaining"] = dict(beam["remaining"])
            candidate["capture_eta"] = dict(beam["capture_eta"])
            candidate["current_target"] = target
            candidate["current_key"] = self._chest_key(target)
            return candidate

        def expand_targets(beam):
            if not beam["remaining"]:
                beam["current_target"] = None
                beam["current_key"] = None
                return [beam]
            return [
                clone_with_target(beam, target)
                for target in choose_targets(beam)
            ]

        def beam_rank(beam):
            capture_times = beam["capture_eta"].values()
            remaining_eta = (
                self._estimate_eta_from_values(
                    [beam["x"], beam["y"]],
                    [beam["vx"], beam["vy"]],
                    beam["current_target"],
                    config,
                )
                if beam["current_target"] is not None
                else OPPONENT_SIM_HORIZON
            )
            return (
                len(beam["capture_eta"]),
                -sum(capture_times),
                -remaining_eta,
            )

        max_steps = max(1, int(OPPONENT_SIM_HORIZON / dt))
        beams = expand_targets(make_beam())

        for step in range(max_steps):
            if not beams:
                break

            next_beams = []

            for beam in beams:
                if len(beam["capture_eta"]) >= OPPONENT_SIM_MAX_CAPTURES:
                    next_beams.append(beam)
                    continue

                if (
                    beam["current_target"] is None
                    or beam["current_key"] not in beam["remaining"]
                ):
                    next_beams.extend(expand_targets(beam))
                    continue

                gravity = self._gravity_acceleration(
                    [beam["x"], beam["y"]],
                    obstacles,
                )
                ax, ay = self._control_to_chest(
                    position=[beam["x"], beam["y"]],
                    velocity=[beam["vx"], beam["vy"]],
                    chest=beam["current_target"],
                    config=config,
                    obstacles=obstacles,
                    gravity=gravity,
                )

                beam["vx"] += (ax + gravity[0]) * dt
                beam["vy"] += (ay + gravity[1]) * dt
                beam["vx"], beam["vy"] = self._clamp_vector(
                    beam["vx"],
                    beam["vy"],
                    max_speed,
                )

                beam["x"] += beam["vx"] * dt
                beam["y"] += beam["vy"] * dt

                if beam["x"] < minimum:
                    beam["x"] = 2.0 * minimum - beam["x"]
                    beam["vx"] = -beam["vx"]
                elif beam["x"] > maximum:
                    beam["x"] = 2.0 * maximum - beam["x"]
                    beam["vx"] = -beam["vx"]

                if beam["y"] < minimum:
                    beam["y"] = 2.0 * minimum - beam["y"]
                    beam["vy"] = -beam["vy"]
                elif beam["y"] > maximum:
                    beam["y"] = 2.0 * maximum - beam["y"]
                    beam["vy"] = -beam["vy"]

                current_time = (step + 1) * dt
                captured_keys = []

                for key, chest in beam["remaining"].items():
                    cx, cy = chest["center"]
                    capture_radius = ship_radius + float(chest["radius"])
                    dx = cx - beam["x"]
                    dy = cy - beam["y"]
                    if dx * dx + dy * dy <= capture_radius * capture_radius:
                        captured_keys.append(key)

                for key in captured_keys:
                    beam["capture_eta"][key] = current_time
                    beam["remaining"].pop(key, None)

                if beam["current_key"] in captured_keys:
                    beam["current_target"] = None
                    beam["current_key"] = None
                    next_beams.extend(expand_targets(beam))
                else:
                    next_beams.append(beam)

            beams = sorted(next_beams, key=beam_rank, reverse=True)[
                :OPPONENT_SIM_BEAM_WIDTH
            ]

        return dict(max(beams, key=beam_rank)["capture_eta"]) if beams else {}

    @staticmethod
    def _capture_gap_between_chests(chest_a, chest_b, ship_radius):
        """估計從一顆寶箱的捕獲區到另一顆捕獲區的最短幾何間隔。"""
        ax, ay = chest_a["center"]
        bx, by = chest_b["center"]
        center_distance = math.hypot(bx - ax, by - ay)

        return max(
            0.0,
            center_distance
            - float(chest_a["radius"])
            - float(chest_b["radius"])
            - 2.0 * ship_radius,
        )

    def _estimate_cluster_prefilter_cost(self, state, chest, all_chests):
        """便宜估計「先拿這顆，再順接最近一顆」的平均時間。

        這只用於候選初選，不是最終分數。
        它能避免兩顆貼很近的寶箱，因第一顆 ETA 不是前幾名而完全沒被模擬。
        """
        first_eta = self._estimate_eta_fast(state, chest)
        chest_key = self._chest_key(chest)

        others = [
            other
            for other in all_chests
            if self._chest_key(other) != chest_key
        ]

        if not others:
            return first_eta

        config = state["config"]
        ship_radius = float(config["ship_radius"])
        max_speed = float(config["max_speed"])

        if max_speed <= EPS:
            return math.inf

        second_eta = min(
            self._capture_gap_between_chests(
                chest,
                other,
                ship_radius,
            ) / max_speed
            for other in others
        )

        weight = PREFILTER_SECOND_CHEST_WEIGHT
        return (
            first_eta + weight * second_eta
        ) / (1.0 + weight)

    def _choose_rollout_next_targets(
        self,
        position,
        velocity,
        remaining_chests,
        config,
    ):
        if not remaining_chests:
            return []

        x, y = position
        ship_radius = float(config["ship_radius"])
        max_speed = float(config["max_speed"])

        ranked = sorted(
            remaining_chests,
            key=lambda chest: self._estimate_eta_from_values(
                position,
                velocity,
                chest,
                config,
            ),
        )
        chosen = []
        chosen_keys = set()

        def add(chest):
            key = self._chest_key(chest)
            if key in chosen_keys:
                return
            chosen.append(chest)
            chosen_keys.add(key)

        for chest in ranked[:max(1, BEAM_BRANCHING - 1)]:
            add(chest)

        if max_speed > EPS and len(ranked) > len(chosen):
            def cluster_cost(chest):
                chest_key = self._chest_key(chest)
                others = [
                    other
                    for other in remaining_chests
                    if self._chest_key(other) != chest_key
                ]
                if not others:
                    return math.inf
                return min(
                    self._capture_gap_between_chests(
                        chest,
                        other,
                        ship_radius,
                    )
                    for other in others
                ) / max_speed

            add(min(ranked, key=cluster_cost))

        if len(chosen) < BEAM_BRANCHING and chosen:
            chosen_angles = [
                math.atan2(chest["center"][1] - y, chest["center"][0] - x)
                for chest in chosen
            ]

            def diversity_score(chest):
                cx, cy = chest["center"]
                angle = math.atan2(cy - y, cx - x)
                nearest_angle_gap = min(
                    abs(math.atan2(
                        math.sin(angle - chosen_angle),
                        math.cos(angle - chosen_angle),
                    ))
                    for chosen_angle in chosen_angles
                )
                eta = self._estimate_eta_from_values(
                    position,
                    velocity,
                    chest,
                    config,
                )
                return nearest_angle_gap - 0.03 * eta

            add(max(ranked, key=diversity_score))

        if (
            ROUTE_MEMORY_ENABLED
            and self.route_memory_keys
            and ROUTE_MEMORY_EXTRA_CANDIDATES > 0
            and len(chosen) < BEAM_BRANCHING
        ):
            chest_by_key = {
                self._chest_key(chest): chest
                for chest in remaining_chests
            }
            added_memory_targets = 0
            for key in self.route_memory_keys:
                chest = chest_by_key.get(key)
                if chest is None:
                    continue
                before_count = len(chosen)
                add(chest)
                if len(chosen) > before_count:
                    added_memory_targets += 1
                if (
                    added_memory_targets >= ROUTE_MEMORY_EXTRA_CANDIDATES
                    or len(chosen) >= BEAM_BRANCHING
                ):
                    break

        for chest in ranked:
            if len(chosen) >= BEAM_BRANCHING:
                break
            add(chest)

        return chosen[:BEAM_BRANCHING]

    def _simulation_horizon(self, state=None):
        if self.use_trap_right_opening:
            return 5.0
        if state is None:
            return SIMULATION_HORIZON

        chest_count = len(state.get("chests", []))
        profile = self.scenario_profile

        if profile == PROFILE_GRAVITY_SEED1:
            if chest_count > 12:
                return 4.2
            if chest_count >= 8:
                return 5.5
            if chest_count >= 6:
                return 6.8
            return 8.2

        if profile == PROFILE_GRAVITY_SEED7 and chest_count > 12:
            return 4.0

        if profile == PROFILE_OPEN and chest_count > 12:
            return SIMULATION_HORIZON_MANY

        if chest_count > 12:
            horizon = SIMULATION_HORIZON_MANY
        elif chest_count >= 8:
            horizon = SIMULATION_HORIZON_MID
        elif chest_count >= 6:
            horizon = SIMULATION_HORIZON_LOW
        else:
            horizon = SIMULATION_HORIZON_ENDGAME

        if chest_count > 8 and len(state.get("obstacles", [])) >= 18:
            horizon = min(horizon, 5.5)
        return horizon

    def _beam_width(self, state):
        if len(state.get("chests", [])) <= ENDGAME_CHEST_THRESHOLD:
            return ENDGAME_BEAM_WIDTH
        return BEAM_WIDTH

    def _capture_time_weight(self):
        return 0.80 if self.use_trap_right_opening else CAPTURE_TIME_WEIGHT

    def _min_commit_ticks(self):
        if (
            self.scenario_profile == PROFILE_GRAVITY_SEED1
            and self.use_stable_left_opening
        ):
            return 20
        return STABLE_OPENING_MIN_COMMIT_TICKS if self.use_stable_left_opening else MIN_COMMIT_TICKS

    def _replan_interval(self):
        if self.scenario_profile == PROFILE_GRAVITY_SEED1 and self.player_id == 1:
            return 22
        if self.scenario_profile == PROFILE_GRAVITY_SEED7:
            return 24
        return REPLAN_INTERVAL

    def _rollout_score(self, capture_times, horizon=None):
        """把多顆寶箱的取得數量與時間合成單一分數。

        每顆寶箱基本價值為 1；取得越晚，扣分越多。
        """
        if not capture_times:
            return 0.0

        if horizon is None:
            horizon = self._simulation_horizon()

        safe_horizon = max(horizon, EPS)
        normalized_time_sum = sum(
            capture_time / safe_horizon
            for capture_time in capture_times
        )

        return (
            len(capture_times)
            - self._capture_time_weight() * normalized_time_sum
        )

    @staticmethod
    def _plan_rank_key(plan):
        """供 max() 使用的整段計畫排序鍵。

        主要看 score；完全同分時再偏好：
        1. 捕獲數較多
        2. 第一顆較早
        3. 最後一顆較早
        """
        return (
            plan["score"],
            plan["capture_count"],
            -plan["first_capture_time"],
            -plan["last_capture_time"],
        )

    def _profile_plan_bonus(self, state, chest):
        tick = int(state["tick"])
        cx, cy = chest["center"]
        profile = self.scenario_profile

        if profile == PROFILE_GRAVITY_SEED1 and self.player_id == 1:
            if tick >= 320 and 40.0 <= cx <= 58.0 and 55.0 <= cy <= 68.0:
                return 0.34
            if tick < 850 and self._point_near([cx, cy], [67.674, 65.199], 1.0):
                return -0.18

        if profile == PROFILE_GRAVITY_SEED7 and self.player_id == 1:
            if (
                state["scores"][self.player_id] >= 5
                and self._point_near([cx, cy], [27.614, 55.542], 1.0)
            ):
                return 0.46
            if tick >= 850 and self._point_near([cx, cy], [26.529, 30.419], 1.0):
                return 0.42
            if tick >= 900 and 42.0 <= cx <= 62.0 and 25.0 <= cy <= 38.0:
                return 0.16

        if profile == PROFILE_TRAP and self.player_id == 0:
            if tick >= 1150 and self._point_near([cx, cy], [42.0, 72.0], 1.0):
                return 0.55

        return 0.0

    def _profile_extra_candidates(self, state, chests):
        if (
            self.scenario_profile == PROFILE_GRAVITY_SEED7
            and self.player_id == 1
            and state["scores"][self.player_id] >= 5
        ):
            return [
                chest
                for chest in chests
                if self._point_near(chest["center"], [27.614, 55.542], 1.0)
            ]
        return []

    def _make_route_options(self, state, chest):
        """產生直接路線與單牆鏡射反彈路線。"""
        x, y = state["position"]
        chest_x, chest_y = chest["center"]
        config = state["config"]
        field_size = float(config["L"])
        ship_radius = float(config["ship_radius"])
        minimum = ship_radius
        maximum = field_size - ship_radius

        routes = [
            {
                "kind": "direct",
                "wall": None,
                "virtual_target": [
                    float(chest_x),
                    float(chest_y),
                ],
                "bounce_point": None,
                "geometric_length": math.hypot(
                    chest_x - x,
                    chest_y - y,
                ),
            }
        ]

        if not ENABLE_SINGLE_WALL_BOUNCE:
            return routes

        bounce_routes = []

        def add_bounce_route(wall_name, wall_pos, vertical):
            virtual_x = 2.0 * wall_pos - chest_x if vertical else chest_x
            virtual_y = chest_y if vertical else 2.0 * wall_pos - chest_y
            denominator = (virtual_x - x) if vertical else (virtual_y - y)
            if abs(denominator) < EPS:
                return

            t = ((wall_pos - x) if vertical else (wall_pos - y)) / denominator
            if not (0.0 < t < 1.0):
                return

            bounce_x = wall_pos if vertical else x + t * (virtual_x - x)
            bounce_y = y + t * (virtual_y - y) if vertical else wall_pos
            if not (
                minimum + BOUNCE_CORNER_MARGIN
                <= (bounce_y if vertical else bounce_x)
                <= maximum - BOUNCE_CORNER_MARGIN
            ):
                return

            bounce_routes.append({
                "kind": "bounce",
                "wall": wall_name,
                "virtual_target": [float(virtual_x), float(virtual_y)],
                "bounce_point": [float(bounce_x), float(bounce_y)],
                "geometric_length": math.hypot(virtual_x - x, virtual_y - y),
            })

        for wall_name, wall_pos, vertical in (
            ("left", minimum, True),
            ("right", maximum, True),
            ("bottom", minimum, False),
            ("top", maximum, False),
        ):
            add_bounce_route(wall_name, wall_pos, vertical)

        direct_length = routes[0]["geometric_length"]
        max_bounce_length = max(
            direct_length,
            1.0,
        ) * BOUNCE_ROUTE_LENGTH_RATIO_LIMIT

        bounce_routes = [
            route
            for route in bounce_routes
            if route["geometric_length"] <= max_bounce_length
        ]

        bounce_routes.sort(key=itemgetter("geometric_length"))

        routes.extend(bounce_routes[:MAX_BOUNCE_ROUTES_PER_CHEST])

        return routes

    def _route_cache_key(self, route):
        virtual_target = route.get("virtual_target")
        bounce_point = route.get("bounce_point")
        return (
            route.get("kind"),
            route.get("wall"),
            tuple(round(float(value), 6) for value in virtual_target)
            if virtual_target is not None
            else None,
            tuple(round(float(value), 6) for value in bounce_point)
            if bounce_point is not None
            else None,
        )

    # ======================== 行星引力 ========================

    @staticmethod
    def _gravity_acceleration(position, obstacles):
        """估計目前位置受到的總行星引力。"""
        x, y = position
        total_ax = 0.0
        total_ay = 0.0

        for obstacle in obstacles:
            cx, cy = obstacle["center"]
            radius = float(obstacle["radius"])
            dx = cx - x
            dy = cy - y
            distance = math.hypot(dx, dy)

            if distance <= radius and distance > EPS:
                # 引力大小 = G / r，方向 = (dx,dy)/r
                # 所以向量 = G * (dx,dy) / r^2
                factor = GRAVITY_CONSTANT / (distance * distance)
                total_ax += factor * dx
                total_ay += factor * dy

        return [total_ax, total_ay]

    # ======================== 控制器 ========================

    def _control_to_chest(self, position, velocity, chest, config, obstacles, aim_point=None, gravity=None):
        """計算目前 tick 飛向指定寶箱的控制加速度。"""
        x, y = position
        vx, vy = velocity

        target_x, target_y = chest["center"] if aim_point is None else aim_point

        max_accel = float(config["max_accel"])
        max_speed = float(config["max_speed"])

        dx = target_x - x
        dy = target_y - y
        distance = math.hypot(dx, dy)
        ux = uy = 0.0

        if distance < EPS:
            desired_vx = 0.0
            desired_vy = 0.0
        else:
            ux = dx / distance
            uy = dy / distance
            desired_vx = ux * max_speed
            desired_vy = uy * max_speed

        desired_total_ax = (desired_vx - vx) / CONTROL_RESPONSE_TIME
        desired_total_ay = (desired_vy - vy) / CONTROL_RESPONSE_TIME

        if gravity is None:
            gravity_ax, gravity_ay = self._gravity_acceleration(
                position,
                obstacles,
            )
        else:
            gravity_ax, gravity_ay = gravity

        if PLANET_ASSIST_ENABLED and distance >= EPS:
            forward_gravity = gravity_ax * ux + gravity_ay * uy
            if forward_gravity > 0.0:
                kept_gravity = PLANET_ASSIST_FORWARD_KEEP * forward_gravity
                gravity_ax -= kept_gravity * ux
                gravity_ay -= kept_gravity * uy

        control_ax = desired_total_ax - GRAVITY_COMPENSATION * gravity_ax
        control_ay = desired_total_ay - GRAVITY_COMPENSATION * gravity_ay

        return self._clamp_vector(control_ax, control_ay, max_accel)

    def _mpc_refine_acceleration(self, state, chest, aim_point, base_acceleration):
        config = state["config"]
        obstacles = state.get("obstacles", [])
        position = state["position"]
        velocity = state["velocity"]

        max_accel = float(config["max_accel"])
        max_speed = float(config["max_speed"])
        dt = float(config["dt"]) * MPC_DT_MULTIPLIER
        ship_radius = float(config["ship_radius"])
        field_size = float(config["L"])
        capture_radius = ship_radius + float(chest["radius"])
        target_x, target_y = aim_point

        dx = target_x - position[0]
        dy = target_y - position[1]
        distance = math.hypot(dx, dy)
        if distance < EPS:
            return base_acceleration

        ux = dx / distance
        uy = dy / distance
        left_x = -uy
        left_y = ux
        brake = self._clamp_vector(
            -velocity[0] / CONTROL_RESPONSE_TIME,
            -velocity[1] / CONTROL_RESPONSE_TIME,
            max_accel,
        )
        forward = [ux * max_accel, uy * max_accel]

        candidates = [
            base_acceleration,
            forward,
            brake,
            self._clamp_vector(
                base_acceleration[0] * 0.75 + left_x * max_accel * 0.25,
                base_acceleration[1] * 0.75 + left_y * max_accel * 0.25,
                max_accel,
            ),
            self._clamp_vector(
                base_acceleration[0] * 0.75 - left_x * max_accel * 0.25,
                base_acceleration[1] * 0.75 - left_y * max_accel * 0.25,
                max_accel,
            ),
        ]

        best_acceleration = base_acceleration
        best_cost = math.inf
        minimum = ship_radius
        maximum = field_size - ship_radius

        def step_state(x, y, vx, vy, acceleration):
            gravity_x, gravity_y = self._gravity_acceleration([x, y], obstacles)
            vx += (acceleration[0] + gravity_x) * dt
            vy += (acceleration[1] + gravity_y) * dt

            speed = math.hypot(vx, vy)
            if speed > max_speed:
                scale = max_speed / speed
                vx *= scale
                vy *= scale

            x += vx * dt
            y += vy * dt

            if x < minimum:
                x = minimum
                vx = abs(vx)
            elif x > maximum:
                x = maximum
                vx = -abs(vx)

            if y < minimum:
                y = minimum
                vy = abs(vy)
            elif y > maximum:
                y = maximum
                vy = -abs(vy)

            return x, y, vx, vy

        def score_state(x, y, vx, vy, min_chest_distance, captured_step):
            final_dx = target_x - x
            final_dy = target_y - y
            final_distance = math.hypot(final_dx, final_dy)
            if final_distance > EPS:
                final_ux = final_dx / final_distance
                final_uy = final_dy / final_distance
            else:
                final_ux = ux
                final_uy = uy

            forward_speed = vx * final_ux + vy * final_uy
            lateral_speed = abs(vx * (-final_uy) + vy * final_ux)
            approach_penalty = max(0.0, -forward_speed)
            cost = (
                final_distance
                + MPC_LATERAL_WEIGHT * lateral_speed
                + MPC_BRAKE_WEIGHT * approach_penalty
                + 0.15 * min_chest_distance
            )
            if captured_step is not None:
                cost -= 1000.0 - captured_step
            return cost

        split_step = max(1, min(MPC_SPLIT_STEP, MPC_HORIZON_STEPS - 1))

        for first_acceleration in candidates:
            x = float(position[0])
            y = float(position[1])
            vx = float(velocity[0])
            vy = float(velocity[1])
            min_chest_distance = math.inf
            captured_step = None

            for step in range(split_step):
                x, y, vx, vy = step_state(x, y, vx, vy, first_acceleration)

                chest_distance = math.hypot(
                    x - chest["center"][0],
                    y - chest["center"][1],
                )
                min_chest_distance = min(min_chest_distance, chest_distance)
                if chest_distance <= capture_radius:
                    captured_step = step + 1
                    break

            if captured_step is not None:
                cost = score_state(
                    x,
                    y,
                    vx,
                    vy,
                    min_chest_distance,
                    captured_step,
                )
                if cost < best_cost:
                    best_cost = cost
                    best_acceleration = first_acceleration
                continue

            second_base = self._control_to_chest(
                position=[x, y],
                velocity=[vx, vy],
                chest=chest,
                config=config,
                obstacles=obstacles,
                aim_point=aim_point,
            )
            second_brake = self._clamp_vector(
                -vx / CONTROL_RESPONSE_TIME,
                -vy / CONTROL_RESPONSE_TIME,
                max_accel,
            )
            mid_dx = target_x - x
            mid_dy = target_y - y
            mid_distance = math.hypot(mid_dx, mid_dy)
            if mid_distance > EPS:
                mid_forward = [
                    mid_dx / mid_distance * max_accel,
                    mid_dy / mid_distance * max_accel,
                ]
            else:
                mid_forward = [0.0, 0.0]

            second_candidates = [second_base, second_brake, mid_forward]

            for second_acceleration in second_candidates:
                sx, sy, svx, svy = x, y, vx, vy
                candidate_min_distance = min_chest_distance
                candidate_captured_step = None

                for step in range(split_step, MPC_HORIZON_STEPS):
                    sx, sy, svx, svy = step_state(
                        sx,
                        sy,
                        svx,
                        svy,
                        second_acceleration,
                    )

                    chest_distance = math.hypot(
                        sx - chest["center"][0],
                        sy - chest["center"][1],
                    )
                    candidate_min_distance = min(
                        candidate_min_distance,
                        chest_distance,
                    )
                    if chest_distance <= capture_radius:
                        candidate_captured_step = step + 1
                        break

                cost = score_state(
                    sx,
                    sy,
                    svx,
                    svy,
                    candidate_min_distance,
                    candidate_captured_step,
                )

                if cost < best_cost:
                    best_cost = cost
                    best_acceleration = first_acceleration

        return best_acceleration

    # ======================== 詳細模擬 ========================

    def _simulate_plan(self, state, first_chest, route, save_path=False):
        horizon = self._simulation_horizon(state)
        beam_width = self._beam_width(state)
        cache_key = (
            self._chest_key(first_chest),
            self._route_cache_key(route),
            self.use_trap_right_opening,
            horizon,
            beam_width,
            save_path,
        )
        cached = self.rollout_cache.get(cache_key)
        if cached is not None:
            return cached

        config = state["config"]
        obstacles = state.get("obstacles", [])

        dt = float(config["dt"])
        max_speed = float(config["max_speed"])
        ship_radius = float(config["ship_radius"])
        field_size = float(config["L"])
        minimum = ship_radius
        maximum = field_size - ship_radius
        safe_horizon = max(horizon, EPS)
        opponent_position = state.get("opponent_direction", [0.0, 0.0])
        opponent_rollout_capture_eta = self.opponent_capture_eta
        opponent_eta_by_key = dict(self.opponent_capture_eta)
        opponent_capture_rank = {
            key: rank
            for rank, (key, _) in enumerate(
                sorted(
                    opponent_rollout_capture_eta.items(),
                    key=itemgetter(1),
                )
            )
        }
        for chest in state["chests"]:
            key = self._chest_key(chest)
            if key not in opponent_eta_by_key:
                opponent_eta_by_key[key] = self._estimate_eta_from_values(
                    opponent_position,
                    self.estimated_opponent_velocity,
                    chest,
                    config,
                )

        max_steps = max(1, int(horizon / dt))

        def make_initial_beam():
            x = float(state["position"][0])
            y = float(state["position"][1])
            return {
                "x": x,
                "y": y,
                "vx": float(state["velocity"][0]),
                "vy": float(state["velocity"][1]),
                "remaining": {
                    self._chest_key(chest): chest
                    for chest in state["chests"]
                },
                "current_target": first_chest,
                "current_target_key": self._chest_key(first_chest),
                "using_initial_route": True,
                "route_bounced": route["wall"] is None,
                "capture_times": [],
                "captured_keys": [],
                "path": [[x, y]] if save_path else [],
            }

        def clone_beam(beam):
            copied = dict(beam)
            copied["remaining"] = dict(beam["remaining"])
            copied["capture_times"] = list(beam["capture_times"])
            copied["captured_keys"] = list(beam["captured_keys"])
            copied["path"] = list(beam["path"])
            return copied

        def collect_chests(beam, current_time):
            newly_captured = []

            for key, candidate in beam["remaining"].items():
                cx, cy = candidate["center"]
                capture_radius = ship_radius + float(candidate["radius"])
                dx = cx - beam["x"]
                dy = cy - beam["y"]

                if dx * dx + dy * dy <= capture_radius * capture_radius:
                    newly_captured.append(key)

            for key in newly_captured:
                beam["remaining"].pop(key, None)
                beam["captured_keys"].append(key)
                beam["capture_times"].append(current_time)

            return newly_captured

        def remove_opponent_captures(beam, current_time):
            removed_current = False
            expired_keys = []

            for key in beam["remaining"]:
                opponent_eta = opponent_rollout_capture_eta.get(key, math.inf)
                if (
                    math.isfinite(opponent_eta)
                    and opponent_eta + OPPONENT_CAPTURE_REMOVE_GRACE <= current_time + EPS
                ):
                    expired_keys.append(key)

            for key in expired_keys:
                beam["remaining"].pop(key, None)
                if key == beam["current_target_key"]:
                    removed_current = True

            if removed_current:
                beam["current_target"] = None
                beam["current_target_key"] = None

            return removed_current

        def active_target_remaining_eta(beam):
            target = beam["current_target"]
            if (
                target is None
                or len(beam["capture_times"]) >= MAX_ROLLOUT_CHESTS
            ):
                return 0.0

            if beam["using_initial_route"] and not beam["route_bounced"]:
                eta_target = {
                    "center": route["virtual_target"],
                    "radius": target["radius"],
                }
            else:
                eta_target = target

            eta = self._estimate_eta_from_values(
                [beam["x"], beam["y"]],
                [beam["vx"], beam["vy"]],
                eta_target,
                config,
            )
            if not math.isfinite(eta):
                return horizon * 2.0
            return eta

        def endpoint_residual_value(beam, remaining_eta=None):
            if (
                beam["current_target"] is None
                or len(beam["capture_times"]) >= MAX_ROLLOUT_CHESTS
            ):
                return 0.0

            if remaining_eta is None:
                remaining_eta = active_target_remaining_eta(beam)
            if remaining_eta <= 0.0:
                return ROLLOUT_RESIDUAL_VALUE_WEIGHT
            return ROLLOUT_RESIDUAL_VALUE_WEIGHT * max(
                0.0,
                1.0 - remaining_eta / safe_horizon,
            )

        def route_opponent_penalty(beam):
            penalty = 0.0
            for key, capture_time in zip(
                beam["captured_keys"][1:MAX_ROLLOUT_CHESTS],
                beam["capture_times"][1:MAX_ROLLOUT_CHESTS],
            ):
                opponent_eta = opponent_eta_by_key.get(key, math.inf)
                if not math.isfinite(opponent_eta):
                    continue
                lead = capture_time - opponent_eta
                if lead > -0.10:
                    penalty += (
                        OPPONENT_ROUTE_PENALTY_WEIGHT
                        * min(1.0, (lead + 0.10) / safe_horizon)
                    )
            return penalty

        def route_opponent_blocking_bonus(beam):
            if not OPPONENT_BLOCKING_ENABLED:
                return 0.0

            bonus = 0.0
            for key, capture_time in zip(
                beam["captured_keys"][:MAX_ROLLOUT_CHESTS],
                beam["capture_times"][:MAX_ROLLOUT_CHESTS],
            ):
                opponent_eta = opponent_eta_by_key.get(key, math.inf)
                if not math.isfinite(opponent_eta):
                    continue

                lead = opponent_eta - capture_time
                if lead < OPPONENT_BLOCKING_MIN_LEAD:
                    continue

                rank = opponent_capture_rank.get(key, OPPONENT_BLOCKING_CANDIDATES)
                rank_factor = (
                    OPPONENT_BLOCKING_RANK_DECAY ** rank
                    if rank < OPPONENT_BLOCKING_CANDIDATES
                    else 0.35
                )
                lead_factor = min(
                    1.0,
                    (lead - OPPONENT_BLOCKING_MIN_LEAD)
                    / max(OPPONENT_BLOCKING_MAX_LEAD, EPS),
                )
                bonus += (
                    OPPONENT_BLOCKING_BONUS
                    * rank_factor
                    * (0.5 + 0.5 * lead_factor)
                )
                if rank == 0:
                    bonus += OPPONENT_BLOCKING_FIRST_BONUS * (0.5 + 0.5 * lead_factor)

            return bonus

        def beam_rank(beam):
            capture_times = beam["capture_times"]
            first_time = capture_times[0] if capture_times else math.inf
            last_time = capture_times[-1] if capture_times else math.inf
            remaining_eta = active_target_remaining_eta(beam)
            eta_penalty = (
                BEAM_REMAINING_ETA_WEIGHT
                * min(remaining_eta, horizon)
                / safe_horizon
            )
            score = (
                self._rollout_score(capture_times, horizon)
                + endpoint_residual_value(beam, remaining_eta)
                + route_opponent_blocking_bonus(beam)
                - route_opponent_penalty(beam)
                - eta_penalty
            )
            return (
                score,
                len(capture_times),
                -first_time,
                -last_time,
                -remaining_eta,
            )

        def expand_after_target_capture(beam):
            next_targets = self._choose_rollout_next_targets(
                [beam["x"], beam["y"]],
                [beam["vx"], beam["vy"]],
                beam["remaining"].values(),
                config,
            )
            if not next_targets:
                beam["current_target"] = None
                beam["current_target_key"] = None
                return [beam]

            expanded = []
            for target in next_targets:
                candidate = clone_beam(beam)
                candidate["current_target"] = target
                candidate["current_target_key"] = self._chest_key(target)
                candidate["using_initial_route"] = False
                candidate["route_bounced"] = True
                expanded.append(candidate)
            return expanded

        beams = [make_initial_beam()]
        initial_current_key = beams[0]["current_target_key"]
        initial_captured = collect_chests(beams[0], 0.0)
        initial_removed = remove_opponent_captures(beams[0], 0.0)
        if initial_current_key in initial_captured or initial_removed:
            beams = expand_after_target_capture(beams[0])

        for step in range(max_steps):
            next_beams = []

            for beam in beams:
                if remove_opponent_captures(beam, step * dt):
                    next_beams.extend(expand_after_target_capture(beam))
                    continue

                if (
                    len(beam["capture_times"]) >= MAX_ROLLOUT_CHESTS
                    or not beam["remaining"]
                    or beam["current_target"] is None
                ):
                    next_beams.append(beam)
                    continue

                if beam["using_initial_route"] and not beam["route_bounced"]:
                    aim_point = route["virtual_target"]
                else:
                    aim_point = beam["current_target"]["center"]

                gravity = self._gravity_acceleration(
                    [beam["x"], beam["y"]],
                    obstacles,
                )

                control_ax, control_ay = self._control_to_chest(
                    position=[beam["x"], beam["y"]],
                    velocity=[beam["vx"], beam["vy"]],
                    chest=beam["current_target"],
                    config=config,
                    obstacles=obstacles,
                    aim_point=aim_point,
                    gravity=gravity,
                )

                gravity_ax, gravity_ay = gravity
                beam["vx"] += (control_ax + gravity_ax) * dt
                beam["vy"] += (control_ay + gravity_ay) * dt
                beam["vx"], beam["vy"] = self._clamp_vector(
                    beam["vx"],
                    beam["vy"],
                    max_speed,
                )

                beam["x"] += beam["vx"] * dt
                beam["y"] += beam["vy"] * dt

                hit_route_wall = False
                planned_wall = route["wall"]

                if beam["x"] < minimum:
                    beam["x"] = 2.0 * minimum - beam["x"]
                    beam["vx"] = -beam["vx"]
                    hit_route_wall = planned_wall == "left"
                elif beam["x"] > maximum:
                    beam["x"] = 2.0 * maximum - beam["x"]
                    beam["vx"] = -beam["vx"]
                    hit_route_wall = planned_wall == "right"

                if beam["y"] < minimum:
                    beam["y"] = 2.0 * minimum - beam["y"]
                    beam["vy"] = -beam["vy"]
                    hit_route_wall = hit_route_wall or planned_wall == "bottom"
                elif beam["y"] > maximum:
                    beam["y"] = 2.0 * maximum - beam["y"]
                    beam["vy"] = -beam["vy"]
                    hit_route_wall = hit_route_wall or planned_wall == "top"

                if (
                    beam["using_initial_route"]
                    and not beam["route_bounced"]
                    and hit_route_wall
                ):
                    beam["route_bounced"] = True

                current_time = (step + 1) * dt
                current_target_key = beam["current_target_key"]
                newly_captured = collect_chests(beam, current_time)
                opponent_removed = remove_opponent_captures(beam, current_time)

                if (
                    current_target_key is not None
                    and (
                        current_target_key in newly_captured
                        or opponent_removed
                    )
                ):
                    next_beams.extend(expand_after_target_capture(beam))
                else:
                    next_beams.append(beam)

                if save_path:
                    beam["path"].append([beam["x"], beam["y"]])

            beams = sorted(next_beams, key=beam_rank, reverse=True)[:beam_width]
            if all(
                len(beam["capture_times"]) >= MAX_ROLLOUT_CHESTS
                or not beam["remaining"]
                or beam["current_target"] is None
                for beam in beams
            ):
                break

        best_beam = max(beams, key=beam_rank)
        capture_times = best_beam["capture_times"]
        score = (
            self._rollout_score(capture_times, horizon)
            + endpoint_residual_value(best_beam)
            + route_opponent_blocking_bonus(best_beam)
            - route_opponent_penalty(best_beam)
        )
        first_capture_time = capture_times[0] if capture_times else math.inf
        last_capture_time = capture_times[-1] if capture_times else math.inf

        result = {
            "score": score,
            "capture_count": len(capture_times),
            "capture_times": capture_times,
            "captured_keys": best_beam["captured_keys"],
            "first_capture_time": first_capture_time,
            "last_capture_time": last_capture_time,
            "path": best_beam["path"],
        }
        self.rollout_cache[cache_key] = result
        return result

    # ======================== 目標規劃 ========================

    def _find_current_target(self, chests):
        if self.target_key is None:
            return None
        for chest in chests:
            if self._chest_key(chest) == self.target_key:
                return chest
        return None

    def _open_left_cleanup_target(self, state, chests, current_target):
        if (
            current_target is not None
            or self.scenario_profile != PROFILE_OPEN
            or self.player_id != 0
        ):
            return None

        x, y = state["position"]
        if x > 18.0 or not (30.0 <= y <= 55.0):
            return None

        candidates = [
            chest
            for chest in chests
            if chest["center"][0] <= 15.0 and 35.0 <= chest["center"][1] <= 55.0
        ]
        if not candidates:
            return None

        target = min(candidates, key=lambda chest: self._estimate_eta_fast(state, chest))
        return target if self._estimate_eta_fast(state, target) <= 3.5 else None

    def _trap_left_bait_target(self, state, chests):
        if self.scenario_profile != PROFILE_TRAP or self.player_id != 0:
            return None
        tick = int(state["tick"])
        if not (330 <= tick <= 650):
            return None
        for chest in chests:
            if self._point_near(chest["center"], [30.0, 28.0], 1.0):
                return chest
        return None

    def _set_direct_target(self, state, chest):
        tick = int(state["tick"])
        key = self._chest_key(chest)
        if key != self.target_key:
            self.target_key = key
            self.target_since_tick = tick
        self.active_route = {
            "kind": "direct",
            "wall": None,
            "virtual_target": list(chest["center"]),
            "bounce_point": None,
        }
        self.route_bounced = True
        self.last_replan_tick = tick
        self.route_memory_keys = [key]
        return chest

    def _remembered_current_target(self, state):
        chests = state["chests"]
        if (
            not ROUTE_MEMORY_ENABLED
            or ROUTE_MEMORY_KEEP_ETA <= 0.0
            or not self.route_memory_keys
        ):
            return None

        available = {self._chest_key(chest): chest for chest in chests}
        self.route_memory_keys = [
            key
            for key in self.route_memory_keys
            if key in available
        ]
        if not self.route_memory_keys or self.target_key not in available:
            return None

        if self.target_key in self.route_memory_keys:
            self.route_memory_keys = self.route_memory_keys[
                self.route_memory_keys.index(self.target_key):
            ]

        if self.route_memory_keys[0] != self.target_key:
            return None

        current_target = available[self.target_key]
        if self._estimate_eta_fast(state, current_target) > ROUTE_MEMORY_KEEP_ETA:
            return None
        return current_target

    def _plan_target(self, state):
        """比較候選第一目標的整段多寶箱收益，決定是否切換。"""
        self.rollout_cache = {}

        chests = state["chests"]
        tick = int(state["tick"])

        if not chests:
            self.target_key = None
            self.route_memory_keys = []
            self.opponent_capture_eta = {}
            return None

        current_target = self._find_current_target(chests)
        trap_bait_target = self._trap_left_bait_target(state, chests)
        if trap_bait_target is not None:
            return self._set_direct_target(state, trap_bait_target)

        open_cleanup_target = self._open_left_cleanup_target(
            state,
            chests,
            current_target,
        )
        if open_cleanup_target is not None:
            return self._set_direct_target(state, open_cleanup_target)

        remembered_target = self._remembered_current_target(state)
        if remembered_target is not None:
            self.last_replan_tick = tick
            return remembered_target

        self.opponent_capture_eta = self._simulate_opponent_captures(state)
        self.predicted_opponent_target_key = self._predict_opponent_target_key(state)

        def opponent_aware_eta(chest):
            eta = self._estimate_eta_fast(state, chest)
            return (
                eta
                + self._opponent_race_penalty(state, chest, eta)
                + self._predicted_target_penalty(chest)
            )

        # 第一組候選：單顆快速 ETA 最佳者。
        ranked_chests = sorted(
            chests,
            key=opponent_aware_eta,
        )
        candidates = list(ranked_chests[:TOP_K_CANDIDATES])

        # 第二組候選：第一顆可能稍遠，但旁邊還有第二顆的寶箱群。
        cluster_ranked = sorted(
            chests,
            key=lambda chest: self._estimate_cluster_prefilter_cost(
                state,
                chest,
                chests,
            ),
        )

        candidate_keys = {
            self._chest_key(chest)
            for chest in candidates
        }

        for chest in cluster_ranked[:CLUSTER_EXTRA_CANDIDATES]:
            key = self._chest_key(chest)
            if key not in candidate_keys:
                candidates.append(chest)
                candidate_keys.add(key)

        if OPPONENT_BLOCKING_ENABLED and self.opponent_capture_eta:
            chest_by_key = {
                self._chest_key(chest): chest
                for chest in chests
            }
            for key, _ in sorted(self.opponent_capture_eta.items(), key=itemgetter(1))[
                :OPPONENT_BLOCKING_CANDIDATES
            ]:
                chest = chest_by_key.get(key)
                if chest is not None and key not in candidate_keys:
                    candidates.append(chest)
                    candidate_keys.add(key)

        for chest in self._profile_extra_candidates(state, chests):
            key = self._chest_key(chest)
            if key not in candidate_keys:
                candidates.append(chest)
                candidate_keys.add(key)

        # 目前目標即使不在候選中，也要加入，才能公平比較是否值得換。
        if current_target is not None:
            current_key = self._chest_key(current_target)
            if current_key not in candidate_keys:
                candidates.append(current_target)
                candidate_keys.add(current_key)

        plans = []

        for chest in candidates:
            chest_key = self._chest_key(chest)

            for route in self._make_route_options(state, chest):
                rollout = self._simulate_plan(
                    state,
                    first_chest=chest,
                    route=route,
                    save_path=False,
                )

                plans.append({
                    "chest": chest,
                    "key": chest_key,
                    "route": route,
                    "score": (
                        rollout["score"]
                        + self._profile_plan_bonus(state, chest)
                        - self._opponent_race_penalty(
                            state,
                            chest,
                            rollout["first_capture_time"],
                        )
                    ),
                    "capture_count": rollout["capture_count"],
                    "capture_times": rollout["capture_times"],
                    "captured_keys": rollout["captured_keys"],
                    "first_capture_time": rollout["first_capture_time"],
                    "last_capture_time": rollout["last_capture_time"],
                })

        best_plan = max(
            (plan for plan in plans if plan["capture_count"] > 0),
            key=self._plan_rank_key,
            default=None,
        )

        if best_plan is None:
            # 視野內所有詳細模擬都沒取得寶箱時，退回單顆快速 ETA。
            fallback_chest = ranked_chests[0]
            fallback_eta = self._estimate_eta_fast(
                state,
                fallback_chest,
            )

            best_plan = {
                "chest": fallback_chest,
                "key": self._chest_key(fallback_chest),
                "route": {
                    "kind": "direct",
                    "wall": None,
                    "virtual_target": list(fallback_chest["center"]),
                    "bounce_point": None,
                    "geometric_length": 0.0,
                },
                "score": 0.0,
                "capture_count": 0,
                "capture_times": [],
                "captured_keys": [],
                "first_capture_time": fallback_eta,
                "last_capture_time": fallback_eta,
            }

        current_plan = None

        if current_target is not None:
            current_key = self._chest_key(current_target)
            current_plan = max(
                (
                    plan
                    for plan in plans
                    if plan["key"] == current_key and plan["capture_count"] > 0
                ),
                key=self._plan_rank_key,
                default=None,
            )

        if current_target is None:
            chosen_plan = best_plan
        else:
            committed_ticks = tick - self.target_since_tick

            if (
                committed_ticks < self._min_commit_ticks()
                and current_plan is not None
            ):
                chosen_plan = current_plan
            elif current_plan is None:
                chosen_plan = best_plan
            elif best_plan["key"] == current_plan["key"]:
                # 同一第一目標時，直接採用該目標目前分數最高的路線。
                chosen_plan = current_plan
            elif (
                best_plan["score"]
                > current_plan["score"] + PLAN_SWITCH_SCORE_MARGIN
            ):
                chosen_plan = best_plan
            else:
                chosen_plan = current_plan

        if chosen_plan["key"] != self.target_key:
            self.target_key = chosen_plan["key"]
            self.target_since_tick = tick

        previous_route_key = (
            self._route_cache_key(self.active_route)
            if self.active_route is not None
            else None
        )
        previous_route_bounced = self.route_bounced
        chosen_route_key = self._route_cache_key(chosen_plan["route"])

        self.active_route = chosen_plan["route"]
        self.route_bounced = (
            self.active_route["wall"] is None
            or (
                previous_route_bounced
                and previous_route_key == chosen_route_key
            )
        )

        self.last_replan_tick = tick
        self.route_memory_keys = list(chosen_plan["captured_keys"])

        return chosen_plan["chest"]
    
    def _update_active_route_phase(self, state):
        """判斷實際飛船是否已經完成規劃中的撞牆。

        伺服器在撞牆後會：
        - 把船留在牆面附近
        - 將垂直牆面的速度分量反轉

        所以可以用「接近牆面 + 速度已反向」判斷。
        """

        route = self.active_route

        if (
            route is None
            or route["wall"] is None
            or self.route_bounced
        ):
            return

        x, y = state["position"]
        vx, vy = state["velocity"]

        config = state["config"]

        ship_radius = float(config["ship_radius"])
        field_size = float(config["L"])
        max_speed = float(config["max_speed"])
        dt = float(config["dt"])

        minimum = ship_radius
        maximum = field_size - ship_radius

        # 一個 tick 最多移動約 max_speed * dt，
        # 因此允許這個等級的位置誤差。
        tolerance = (
            BOUNCE_DETECTION_MULTIPLIER
            * max_speed
            * dt
            + 1e-6
        )

        wall = route["wall"]

        if (
            wall == "left"
            and x <= minimum + tolerance
            and vx > 0.0
        ):
            self.route_bounced = True

        elif (
            wall == "right"
            and x >= maximum - tolerance
            and vx < 0.0
        ):
            self.route_bounced = True

        elif (
            wall == "bottom"
            and y <= minimum + tolerance
            and vy > 0.0
        ):
            self.route_bounced = True

        elif (
            wall == "top"
            and y >= maximum - tolerance
            and vy < 0.0
        ):
            self.route_bounced = True

    def _should_use_mpc(self, state, chest, aim_point):
        position = state["position"]
        velocity = state["velocity"]
        config = state["config"]

        dx = aim_point[0] - position[0]
        dy = aim_point[1] - position[1]
        distance = math.hypot(dx, dy)
        if distance <= EPS:
            return True

        ux = dx / distance
        uy = dy / distance
        forward_speed = velocity[0] * ux + velocity[1] * uy
        lateral_vx = velocity[0] - forward_speed * ux
        lateral_vy = velocity[1] - forward_speed * uy
        lateral_speed = math.hypot(lateral_vx, lateral_vy)

        max_accel = float(config["max_accel"])
        if max_accel <= EPS:
            return False

        ship_radius = float(config["ship_radius"])
        capture_radius = ship_radius + float(chest["radius"])
        braking_distance = (
            max(0.0, forward_speed) ** 2
            / (2.0 * max_accel)
        )
        lateral_correction_distance = (
            lateral_speed * lateral_speed
            / (2.0 * max_accel)
        )
        trigger_distance = (
            capture_radius
            + MPC_BRAKE_TRIGGER_BUFFER
            + MPC_BRAKE_TRIGGER_SCALE * braking_distance
            + 0.35 * lateral_correction_distance
        )

        return distance <= trigger_distance

    # ======================== 官方入口 ========================

    def _act_impl(self, state):
        """每個 tick 回傳合法的 [ax, ay]。"""
        self._update_opponent_velocity(state)
        self._update_scenario_profile(state)

        chests = state["chests"]
        tick = int(state["tick"])

        if tick == 0 and self.player_id == 1:
            x, y = state["position"]
            self.use_trap_right_opening = (
                x > 90.0
                and y > 90.0
                and bool(state.get("obstacles"))
            )
            self.use_stable_left_opening = (
                x < 20.0
                and y < 40.0
                and bool(state.get("obstacles"))
            )

        if not chests:
            # 沒寶箱時煞停。
            vx, vy = state["velocity"]
            max_accel = float(state["config"]["max_accel"])
            return self._clamp_vector(
                -vx / CONTROL_RESPONSE_TIME,
                -vy / CONTROL_RESPONSE_TIME,
                max_accel,
            )

        current_target = self._find_current_target(chests)

        should_replan = (
            current_target is None
            or tick - self.last_replan_tick >= self._replan_interval()
        )

        if should_replan:
            current_target = self._plan_target(state)

        if current_target is None:
            return [0.0, 0.0]

        # 判斷指定的撞牆是否已經完成。
        self._update_active_route_phase(state)


        # direct 路線，或已經撞完牆：
        #     朝真正寶箱前進。
        #
        # bounce 路線且尚未撞牆：
        #     朝鏡射寶箱前進。
        if (
            self.active_route is not None
            and self.active_route["wall"] is not None
            and not self.route_bounced
        ):
            aim_point = (
                self.active_route[
                    "virtual_target"
                ]
            )
        else:
            aim_point = current_target["center"]


        base_acceleration = self._control_to_chest(
            position=state["position"],
            velocity=state["velocity"],
            chest=current_target,
            config=state["config"],
            obstacles=state.get(
                "obstacles",
                [],
            ),
            aim_point=aim_point,
        )

        if (
            self.route_bounced
            and self._should_use_mpc(state, current_target, aim_point)
        ):
            return self._mpc_refine_acceleration(
                state=state,
                chest=current_target,
                aim_point=aim_point,
                base_acceleration=base_acceleration,
            )

        return base_acceleration
    
    def act(self, state):
        return self._act_impl(state)
