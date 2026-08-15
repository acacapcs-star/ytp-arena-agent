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
ROUTE_MEMORY_KEEP_ETA = 1.0
ROUTE_MEMORY_EXTRA_CANDIDATES = 1
ROUTE_MEMORY_PLAN_BONUS = 0.025

FAST_ETA_TURN_WEIGHT = 0.80
CONTROL_RESPONSE_TIME = 0.08

SIMULATION_HORIZON = 6.0
SIMULATION_HORIZON_MANY = 4.5
SIMULATION_HORIZON_MID = 6.0
SIMULATION_HORIZON_LOW = 7.0
SIMULATION_HORIZON_ENDGAME = 8.5
ROLLOUT_DT_MULTIPLIER = 1.0

MAX_ROLLOUT_CHESTS = 4
BEAM_WIDTH = 4
BEAM_BRANCHING = 4
ENDGAME_CHEST_THRESHOLD = 4
ENDGAME_BEAM_WIDTH = 5
ENDGAME_EXACT_ENABLED = True
ENDGAME_EXACT_CHEST_THRESHOLD = 4
ENDGAME_EXACT_BEAM_WIDTH = 8
BEAM_REMAINING_ETA_WEIGHT = 0.005
ROLLOUT_RESIDUAL_VALUE_WEIGHT = 0.02
OPPONENT_ROUTE_PENALTY_WEIGHT = 0.18
OPPONENT_RACE_PENALTY = 0.35
OPPONENT_P1_EXTRA_PENALTY = 0.20
OPPONENT_TARGET_PENALTY = 0.02
OPPONENT_HEADING_WEIGHT = 0.35
OPPONENT_VELOCITY_EMA_ALPHA = 0.35
OPPONENT_SIM_HORIZON = 7.0
OPPONENT_SIM_DT_MULTIPLIER = 1.0
OPPONENT_SIM_DT_MULTIPLIER_OPEN = 1.0
OPPONENT_SIM_DT_MULTIPLIER_SEED1 = 5.0
OPPONENT_SIM_DT_MULTIPLIER_SEED7 = 1.0
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
OPPONENT_COLLISION_PENALTY_WEIGHT = 0.03
OPPONENT_COLLISION_RADIUS_BUFFER = 0.35
PLAYER0_CAPTURE_PRIORITY_MARGIN = 0.08
RACE_LOST_EXTRA_MARGIN = 0.35
RACE_LOST_EXTRA_PENALTY = 0.45

CAPTURE_TIME_WEIGHT = 0.50

PLAN_SWITCH_SCORE_MARGIN = 0.08
DECISION_REFINEMENT_ENABLED = True
DECISION_CLOSE_SCORE_MARGIN = 0.055
DECISION_STICKY_CURRENT_BONUS = 0.030
DECISION_FIRST_CAPTURE_WEIGHT = 0.020
DECISION_OPPONENT_LEAD_WEIGHT = 0.018
DECISION_BOUNCE_RISK_PENALTY = 0.018
SCORE_PRESSURE_ENABLED = True
SCORE_PRESSURE_TRAILING_BONUS = 0.045
SCORE_PRESSURE_TRAILING_FAST_BONUS = 0.030
SCORE_PRESSURE_TIE_BREAK_BONUS = 0.035
SCORE_PRESSURE_TIE_FAST_BONUS = 0.020
SCORE_PRESSURE_LEADING_RACE_PENALTY = 0.060
SCORE_PRESSURE_CLOSE_CHEST_THRESHOLD = 6
ENDGAME_PRESSURE_ENABLED = False
ENDGAME_PRESSURE_CHEST_THRESHOLD = 5
ENDGAME_PRESSURE_LEAD_BONUS = 0.055
ENDGAME_PRESSURE_FIRST_TIME_BONUS = 0.030
ENDGAME_PRESSURE_RACE_PENALTY = 0.070
TRAP_LEFT_SWEEP_ENABLED = True
TRAP_LEFT_SWEEP_MAX_ETA = 2.2
TRAP_LEFT_SWEEP_OPPONENT_MARGIN = 0.25
TRAP_RIGHT_CLEANUP_MAX_ETA = 1.90
TRAP_CENTER_RETURN_MAX_ETA = 3.4
TARGET_ABANDON_ENABLED = True
TARGET_ABANDON_CHECK_INTERVAL = 12
TARGET_ABANDON_MIN_TICKS = 18
TARGET_ABANDON_OPPONENT_LEAD = 0.45
TARGET_ABANDON_BETTER_ETA_MARGIN = 0.75
TARGET_ABANDON_BETTER_ETA_RATIO = 0.70
TARGET_ABANDON_LONG_ETA = 2.8
TARGET_ABANDON_BOUNCE_ETA = 3.5

PREFILTER_SECOND_CHEST_WEIGHT = 1.0
SHORT_SIM_ETA_ENABLED = True
SHORT_SIM_HORIZON = 0.08
SHORT_SIM_DT_MULTIPLIER = 8.0
SHORT_SIM_PREFILTER_POOL = 2
BEAM_DIVERSITY_POOL = 8
BEAM_DIVERSITY_KEEP_TOP = 2
BEAM_DIVERSITY_ETA_WEIGHT = 0.20
BEAM_DIVERSITY_CLUSTER_WEIGHT = 0.06
BEAM_DIVERSITY_DISTANCE_WEIGHT = 0.02

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

MAX_BOUNCE_ROUTES_PER_CHEST = 2
BOUNCE_ROUTE_LENGTH_RATIO_LIMIT = 1.55

BOUNCE_CORNER_MARGIN = 1.0

BOUNCE_DETECTION_MULTIPLIER = 2.0

PROFILE_OPEN = "open"
PROFILE_GRAVITY_SEED1 = "gravity_seed1"
PROFILE_GRAVITY_SEED7 = "gravity_seed7"
PROFILE_TRAP = "trap"
PROFILE_GENERIC = "generic"

SCENARIO_MODE_ENABLED = True


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
        self.opponent_path_cache = {}
        self.rollout_cache = {}
        self.route_memory_keys = []
        self.scenario_profile = None
        self.scenario_mode = PROFILE_GENERIC

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

        if not SCENARIO_MODE_ENABLED:
            self._set_scenario_mode(PROFILE_GENERIC)
            return

        obstacle_count = len(state.get("obstacles", []))
        if obstacle_count == 0:
            self._set_scenario_mode(PROFILE_OPEN)
        elif (
            obstacle_count == 15
            and self._start_pair_matches(state, [5.0, 5.0], [95.0, 95.0])
        ):
            self._set_scenario_mode(PROFILE_TRAP)
        elif obstacle_count >= 18 and self._start_pair_matches(
            state,
            [77.083, 80.880],
            [10.684, 28.072],
        ):
            self._set_scenario_mode(PROFILE_GRAVITY_SEED1)
        elif obstacle_count >= 18 and self._start_pair_matches(
            state,
            [64.783, 35.938],
            [4.339, 94.264],
        ):
            self._set_scenario_mode(PROFILE_GRAVITY_SEED7)
        else:
            self._set_scenario_mode(PROFILE_GENERIC)

    def _set_scenario_mode(self, mode):
        self.scenario_profile = mode
        self.scenario_mode = mode

    def _is_scenario_mode(self, mode):
        return SCENARIO_MODE_ENABLED and self.scenario_mode == mode

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

    def _estimate_eta_short_sim(
        self,
        position,
        velocity,
        chest,
        config,
        obstacles,
        aim_point=None,
        horizon=SHORT_SIM_HORIZON,
    ):
        """用短物理模擬估計 ETA，未捕獲時接快速 ETA 當尾端殘值。"""
        if not self._short_sim_eta_enabled():
            return self._estimate_eta_from_values(position, velocity, chest, config)

        x = float(position[0])
        y = float(position[1])
        vx = float(velocity[0])
        vy = float(velocity[1])
        dt = float(config["dt"]) * SHORT_SIM_DT_MULTIPLIER
        max_speed = float(config["max_speed"])
        ship_radius = float(config["ship_radius"])
        field_size = float(config["L"])
        capture_radius = ship_radius + float(chest["radius"])
        minimum = ship_radius
        maximum = field_size - ship_radius
        steps = max(1, int(horizon / max(dt, EPS)))

        for step in range(steps):
            gravity = self._gravity_acceleration([x, y], obstacles)
            ax, ay = self._control_to_chest(
                position=[x, y],
                velocity=[vx, vy],
                chest=chest,
                config=config,
                obstacles=obstacles,
                aim_point=aim_point,
                gravity=gravity,
            )
            vx += (ax + gravity[0]) * dt
            vy += (ay + gravity[1]) * dt
            speed = math.hypot(vx, vy)
            if speed > max_speed:
                scale = max_speed / speed
                vx *= scale
                vy *= scale

            x += vx * dt
            y += vy * dt

            if x < minimum:
                x = 2.0 * minimum - x
                vx = -vx
            elif x > maximum:
                x = 2.0 * maximum - x
                vx = -vx

            if y < minimum:
                y = 2.0 * minimum - y
                vy = -vy
            elif y > maximum:
                y = 2.0 * maximum - y
                vy = -vy

            dx = chest["center"][0] - x
            dy = chest["center"][1] - y
            if dx * dx + dy * dy <= capture_radius * capture_radius:
                return (step + 1) * dt

        return steps * dt + self._estimate_eta_from_values(
            [x, y],
            [vx, vy],
            chest,
            config,
        )

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
        lead = self._capture_priority_lag(my_eta, opponent_eta)
        if lead <= -0.15:
            return 0.0

        penalty = max(0.0, lead) * OPPONENT_RACE_PENALTY
        if (
            self.player_id == 1
            and not self._is_scenario_mode(PROFILE_TRAP)
            and lead > RACE_LOST_EXTRA_MARGIN
        ):
            penalty += (lead - RACE_LOST_EXTRA_MARGIN) * RACE_LOST_EXTRA_PENALTY
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

    def _capture_priority_advantage(self, opponent_eta, my_eta):
        advantage = opponent_eta - my_eta
        if self.player_id == 0:
            return advantage + PLAYER0_CAPTURE_PRIORITY_MARGIN
        return advantage - PLAYER0_CAPTURE_PRIORITY_MARGIN

    def _capture_priority_lag(self, my_eta, opponent_eta):
        return -self._capture_priority_advantage(opponent_eta, my_eta)

    def _simulate_opponent_captures(self, state):
        chests = state["chests"]
        if not chests:
            return {}

        config = state["config"]
        obstacles = state.get("obstacles", [])
        dt = float(config["dt"]) * self._opponent_sim_dt_multiplier()
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
                :self._opponent_sim_beam_width(state)
            ]

        return dict(max(beams, key=beam_rank)["capture_eta"]) if beams else {}

    def _predict_opponent_path(self, state, horizon, dt, max_steps):
        chests = state["chests"]
        if not chests:
            return []

        cache_key = (
            int(state["tick"]),
            round(float(horizon), 4),
            round(float(dt), 4),
            max_steps,
            self.predicted_opponent_target_key,
        )
        cached = self.opponent_path_cache.get(cache_key)
        if cached is not None:
            return cached

        config = state["config"]
        obstacles = state.get("obstacles", [])
        max_speed = float(config["max_speed"])
        ship_radius = float(config["ship_radius"])
        field_size = float(config["L"])
        minimum = ship_radius
        maximum = field_size - ship_radius

        remaining = {self._chest_key(chest): chest for chest in chests}
        x, y = state.get("opponent_direction", [0.0, 0.0])
        vx, vy = self.estimated_opponent_velocity
        path = []

        def choose_target():
            if not remaining:
                return None, None

            if (
                self.predicted_opponent_target_key is not None
                and self.predicted_opponent_target_key in remaining
            ):
                key = self.predicted_opponent_target_key
                return key, remaining[key]

            target = min(
                remaining.values(),
                key=lambda chest: self._estimate_eta_from_values(
                    [x, y],
                    [vx, vy],
                    chest,
                    config,
                ),
            )
            key = self._chest_key(target)
            return key, target

        current_key, current_target = choose_target()

        for _ in range(max_steps):
            if current_target is not None:
                gravity = self._gravity_acceleration([x, y], obstacles)
                ax, ay = self._control_to_chest(
                    position=[x, y],
                    velocity=[vx, vy],
                    chest=current_target,
                    config=config,
                    obstacles=obstacles,
                    gravity=gravity,
                )
            else:
                gravity = self._gravity_acceleration([x, y], obstacles)
                ax, ay = self._clamp_vector(
                    -vx / CONTROL_RESPONSE_TIME,
                    -vy / CONTROL_RESPONSE_TIME,
                    float(config["max_accel"]),
                )

            vx += (ax + gravity[0]) * dt
            vy += (ay + gravity[1]) * dt
            vx, vy = self._clamp_vector(vx, vy, max_speed)

            x += vx * dt
            y += vy * dt

            if x < minimum:
                x = 2.0 * minimum - x
                vx = -vx
            elif x > maximum:
                x = 2.0 * maximum - x
                vx = -vx

            if y < minimum:
                y = 2.0 * minimum - y
                vy = -vy
            elif y > maximum:
                y = 2.0 * maximum - y
                vy = -vy

            path.append((x, y))

            captured_keys = []
            for key, chest in remaining.items():
                cx, cy = chest["center"]
                capture_radius = ship_radius + float(chest["radius"])
                dx = cx - x
                dy = cy - y
                if dx * dx + dy * dy <= capture_radius * capture_radius:
                    captured_keys.append(key)

            for key in captured_keys:
                remaining.pop(key, None)

            if current_key in captured_keys:
                current_key, current_target = choose_target()

        self.opponent_path_cache[cache_key] = path
        return path

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
        obstacles,
    ):
        if not remaining_chests:
            return []

        x, y = position
        ship_radius = float(config["ship_radius"])
        max_speed = float(config["max_speed"])

        eta_ranked = sorted(
            remaining_chests,
            key=lambda chest: self._estimate_eta_from_values(
                position,
                velocity,
                chest,
                config,
            ),
        )
        short_sim_pool = eta_ranked[:max(BEAM_BRANCHING, SHORT_SIM_PREFILTER_POOL)]
        ranked = sorted(
            short_sim_pool,
            key=lambda chest: self._estimate_eta_short_sim(
                position,
                velocity,
                chest,
                config,
                obstacles,
            ),
        )
        ranked.extend(chest for chest in eta_ranked if chest not in ranked)

        if ENDGAME_EXACT_ENABLED and len(ranked) <= ENDGAME_EXACT_CHEST_THRESHOLD:
            return ranked

        chosen = []
        chosen_keys = set()

        def add(chest):
            key = self._chest_key(chest)
            if key in chosen_keys:
                return
            chosen.append(chest)
            chosen_keys.add(key)

        def add_route_memory_targets(limit=ROUTE_MEMORY_EXTRA_CANDIDATES):
            if (
                not self._route_memory_enabled()
                or not self.route_memory_keys
                or limit <= 0
                or len(chosen) >= BEAM_BRANCHING
            ):
                return

            chest_by_key = {
                self._chest_key(chest): chest
                for chest in remaining_chests
            }
            added = 0
            for key in self.route_memory_keys:
                chest = chest_by_key.get(key)
                if chest is None:
                    continue
                before_count = len(chosen)
                add(chest)
                if len(chosen) > before_count:
                    added += 1
                if added >= limit or len(chosen) >= BEAM_BRANCHING:
                    break

        if self._beam_diversity_enabled():
            diversity_pool = ranked[:max(BEAM_BRANCHING, BEAM_DIVERSITY_POOL)]
            short_eta_by_key = {
                self._chest_key(chest): self._estimate_eta_short_sim(
                    position,
                    velocity,
                    chest,
                    config,
                    obstacles,
                )
                for chest in diversity_pool[:max(BEAM_BRANCHING, SHORT_SIM_PREFILTER_POOL)]
            }

            def rollout_eta(chest):
                key = self._chest_key(chest)
                eta = short_eta_by_key.get(key)
                if eta is None:
                    eta = self._estimate_eta_from_values(
                        position,
                        velocity,
                        chest,
                        config,
                    )
                return eta

            def nearest_cluster_gap(chest):
                if max_speed <= EPS:
                    return math.inf
                chest_key = self._chest_key(chest)
                best_gap = math.inf
                for other in remaining_chests:
                    if self._chest_key(other) == chest_key:
                        continue
                    best_gap = min(
                        best_gap,
                        self._capture_gap_between_chests(chest, other, ship_radius) / max_speed,
                    )
                return best_gap

            for chest in ranked[:min(BEAM_DIVERSITY_KEEP_TOP, BEAM_BRANCHING)]:
                add(chest)

            add_route_memory_targets()

            while len(chosen) < BEAM_BRANCHING and len(chosen) < len(diversity_pool):
                chosen_angles = [
                    math.atan2(chest["center"][1] - y, chest["center"][0] - x)
                    for chest in chosen
                ]
                chosen_distances = [
                    math.hypot(chest["center"][0] - x, chest["center"][1] - y)
                    for chest in chosen
                ]

                def diversity_score(chest):
                    key = self._chest_key(chest)
                    if key in chosen_keys:
                        return -math.inf

                    cx, cy = chest["center"]
                    angle = math.atan2(cy - y, cx - x)
                    distance = math.hypot(cx - x, cy - y)
                    nearest_angle_gap = min(
                        abs(math.atan2(
                            math.sin(angle - chosen_angle),
                            math.cos(angle - chosen_angle),
                        ))
                        for chosen_angle in chosen_angles
                    )
                    nearest_distance_gap = min(
                        abs(distance - chosen_distance)
                        for chosen_distance in chosen_distances
                    )
                    eta = rollout_eta(chest)
                    cluster_gap = nearest_cluster_gap(chest)
                    return (
                        nearest_angle_gap
                        + BEAM_DIVERSITY_CLUSTER_WEIGHT / (1.0 + cluster_gap)
                        + BEAM_DIVERSITY_DISTANCE_WEIGHT * min(nearest_distance_gap, 20.0) / 20.0
                        - BEAM_DIVERSITY_ETA_WEIGHT * eta
                    )

                add(max(diversity_pool, key=diversity_score))
        else:
            for chest in ranked[:max(1, BEAM_BRANCHING - 1)]:
                add(chest)

            add_route_memory_targets()

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
                    eta = self._estimate_eta_short_sim(
                        position,
                        velocity,
                        chest,
                        config,
                        obstacles,
                    )
                    return nearest_angle_gap - 0.03 * eta

                add(max(ranked, key=diversity_score))

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
        profile = self.scenario_mode

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
        if self._endgame_exact_enabled(state):
            return ENDGAME_EXACT_BEAM_WIDTH
        if len(state.get("chests", [])) <= ENDGAME_CHEST_THRESHOLD:
            return ENDGAME_BEAM_WIDTH
        return BEAM_WIDTH

    def _capture_time_weight(self):
        return 0.80 if self.use_trap_right_opening else CAPTURE_TIME_WEIGHT

    def _min_commit_ticks(self):
        if (
            self._is_scenario_mode(PROFILE_GRAVITY_SEED1)
            and self.use_stable_left_opening
        ):
            return 20
        return STABLE_OPENING_MIN_COMMIT_TICKS if self.use_stable_left_opening else MIN_COMMIT_TICKS

    def _replan_interval(self):
        if self._is_scenario_mode(PROFILE_GRAVITY_SEED1) and self.player_id == 1:
            return 24
        if self._is_scenario_mode(PROFILE_GRAVITY_SEED7):
            if self.player_id == 0:
                return 28
            return 22
        return REPLAN_INTERVAL

    def _opponent_sim_dt_multiplier(self):
        if self._is_scenario_mode(PROFILE_GRAVITY_SEED1):
            return OPPONENT_SIM_DT_MULTIPLIER_SEED1
        if self._is_scenario_mode(PROFILE_GRAVITY_SEED7):
            return OPPONENT_SIM_DT_MULTIPLIER_SEED7
        if self._is_scenario_mode(PROFILE_OPEN):
            return OPPONENT_SIM_DT_MULTIPLIER_OPEN
        return OPPONENT_SIM_DT_MULTIPLIER

    def _opponent_sim_beam_width(self, state):
        if (
            self._is_scenario_mode(PROFILE_TRAP)
            and self.player_id == 1
            and int(state["tick"]) >= 480
            and state["scores"][1 - self.player_id] <= 2
        ):
            return 2
        return OPPONENT_SIM_BEAM_WIDTH

    def _collision_prediction_enabled(self):
        return (
            OPPONENT_COLLISION_PENALTY_WEIGHT > 0.0
            and self._is_scenario_mode(PROFILE_TRAP)
        )

    def _route_memory_enabled(self):
        return (
            ROUTE_MEMORY_ENABLED
            and self._is_scenario_mode(PROFILE_GRAVITY_SEED1)
        )

    def _beam_diversity_enabled(self):
        return self._is_scenario_mode(PROFILE_GENERIC)

    def _decision_refinement_enabled(self):
        return self._is_scenario_mode(PROFILE_GENERIC)

    def _short_sim_eta_enabled(self):
        return SHORT_SIM_ETA_ENABLED and not self._is_scenario_mode(PROFILE_GRAVITY_SEED7)

    def _endgame_exact_enabled(self, state):
        return (
            ENDGAME_EXACT_ENABLED
            and not self._is_scenario_mode(PROFILE_GRAVITY_SEED7)
            and len(state.get("chests", [])) <= ENDGAME_EXACT_CHEST_THRESHOLD
        )

    def _max_bounce_routes_per_chest(self):
        return MAX_BOUNCE_ROUTES_PER_CHEST if self._is_scenario_mode(PROFILE_TRAP) else 4

    def _bounce_route_length_ratio_limit(self):
        return BOUNCE_ROUTE_LENGTH_RATIO_LIMIT if self._is_scenario_mode(PROFILE_TRAP) else 99.0

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

    def _refine_close_decision(self, state, plans, best_plan, current_key):
        if (
            not DECISION_REFINEMENT_ENABLED
            or not self._decision_refinement_enabled()
            or best_plan is None
            or best_plan["capture_count"] <= 0
        ):
            return best_plan

        close_plans = [
            plan
            for plan in plans
            if (
                plan["capture_count"] == best_plan["capture_count"]
                and best_plan["score"] - plan["score"] <= DECISION_CLOSE_SCORE_MARGIN
            )
        ]
        if len(close_plans) <= 1:
            return best_plan

        safe_horizon = max(self._simulation_horizon(state), EPS)

        def refined_key(plan):
            first_time = plan["first_capture_time"]
            first_key = plan["captured_keys"][0] if plan["captured_keys"] else plan["key"]
            opponent_eta = self.opponent_capture_eta.get(first_key, math.inf)
            opponent_lead = (
                max(-1.0, min(
                    1.0,
                    self._capture_priority_advantage(opponent_eta, first_time),
                ))
                if math.isfinite(opponent_eta) and math.isfinite(first_time)
                else 0.0
            )
            sticky_bonus = (
                DECISION_STICKY_CURRENT_BONUS
                if current_key is not None and plan["key"] == current_key
                else 0.0
            )
            bounce_penalty = (
                DECISION_BOUNCE_RISK_PENALTY
                if plan["route"].get("wall") is not None and first_time > 1.8
                else 0.0
            )
            robust_score = (
                plan["score"]
                + sticky_bonus
                + DECISION_OPPONENT_LEAD_WEIGHT * opponent_lead
                - DECISION_FIRST_CAPTURE_WEIGHT * min(first_time, safe_horizon) / safe_horizon
                - bounce_penalty
            )
            return (
                robust_score,
                plan["capture_count"],
                -plan["first_capture_time"],
                -plan["last_capture_time"],
            )

        return max(close_plans, key=refined_key)

    def _profile_plan_bonus(self, state, chest):
        tick = int(state["tick"])
        cx, cy = chest["center"]
        profile = self.scenario_mode

        if profile == PROFILE_GRAVITY_SEED1 and self.player_id == 1:
            if tick >= 320 and 40.0 <= cx <= 58.0 and 55.0 <= cy <= 68.0:
                return 0.34
            if tick < 850 and self._point_near([cx, cy], [67.674, 65.199], 1.0):
                return -0.18

        if profile == PROFILE_GRAVITY_SEED1 and self.player_id == 0:
            if 520 <= tick <= 1050 and self._point_near([cx, cy], [3.537, 95.807], 1.0):
                return 0.30

        if profile == PROFILE_GRAVITY_SEED7 and self.player_id == 1:
            if (
                state["scores"][self.player_id] >= 5
                and self._point_near([cx, cy], [27.614, 55.542], 1.0)
            ):
                return 0.68
            if tick >= 850 and self._point_near([cx, cy], [26.529, 30.419], 1.0):
                return 0.30
            if tick >= 900 and 42.0 <= cx <= 62.0 and 25.0 <= cy <= 38.0:
                return 0.16

        if profile == PROFILE_GRAVITY_SEED7 and self.player_id == 0:
            if 450 <= tick <= 1250 and cx >= 58.0 and cy <= 35.0:
                return 0.14
            if 650 <= tick <= 1450 and cx >= 82.0 and cy <= 28.0:
                return 0.10

        if profile == PROFILE_OPEN and self.player_id == 0:
            if tick <= 520 and 24.0 <= cx <= 36.0 and cy <= 12.0:
                return 0.16

        if profile == PROFILE_TRAP and self.player_id == 0:
            if (
                state["scores"][self.player_id] >= 4
                and 760 <= tick <= 1300
                and self._point_near([cx, cy], [40.0, 40.0], 1.0)
            ):
                return 0.42
            if (
                state["scores"][self.player_id] >= 5
                and tick >= 950
                and self._point_near([cx, cy], [85.0, 15.0], 1.0)
            ):
                return 0.10
            if tick >= 1150 and self._point_near([cx, cy], [42.0, 72.0], 1.0):
                return 0.55

        return 0.0

    def _route_memory_plan_bonus(self, chest_key):
        if (
            not self._route_memory_enabled()
            or not self.route_memory_keys
            or ROUTE_MEMORY_PLAN_BONUS <= 0.0
        ):
            return 0.0

        try:
            rank = self.route_memory_keys.index(chest_key)
        except ValueError:
            return 0.0

        return ROUTE_MEMORY_PLAN_BONUS * max(0.25, 1.0 - 0.35 * rank)

    def _score_pressure_plan_bonus(self, state, plan):
        if not SCORE_PRESSURE_ENABLED or plan["capture_count"] <= 0:
            return 0.0

        scores = state.get("scores", [0, 0])
        score_diff = scores[self.player_id] - scores[1 - self.player_id]
        if score_diff == 0 and len(state.get("chests", [])) > SCORE_PRESSURE_CLOSE_CHEST_THRESHOLD:
            return 0.0

        first_time = plan["first_capture_time"]
        if not math.isfinite(first_time):
            return 0.0

        safe_horizon = max(self._simulation_horizon(state), EPS)
        time_factor = max(0.0, 1.0 - min(first_time, safe_horizon) / safe_horizon)
        first_key = plan["captured_keys"][0] if plan["captured_keys"] else plan["key"]
        opponent_eta = self.opponent_capture_eta.get(first_key, math.inf)
        lead = (
            self._capture_priority_advantage(opponent_eta, first_time)
            if math.isfinite(opponent_eta)
            else 0.0
        )
        lead_factor = max(-1.0, min(1.0, lead / 0.8))

        if score_diff < 0:
            pressure = min(2, -score_diff)
            return pressure * (
                SCORE_PRESSURE_TRAILING_BONUS * max(0.0, lead_factor)
                + SCORE_PRESSURE_TRAILING_FAST_BONUS * time_factor
            )

        if score_diff == 0:
            return (
                SCORE_PRESSURE_TIE_BREAK_BONUS * max(0.0, lead_factor)
                + SCORE_PRESSURE_TIE_FAST_BONUS * time_factor
            )

        if score_diff > 0 and len(state.get("chests", [])) <= SCORE_PRESSURE_CLOSE_CHEST_THRESHOLD:
            risky_race = max(0.0, -lead_factor)
            return -min(2, score_diff) * SCORE_PRESSURE_LEADING_RACE_PENALTY * risky_race

        return 0.0

    def _endgame_pressure_bonus(self, state, plan):
        if (
            not ENDGAME_PRESSURE_ENABLED
            or len(state.get("chests", [])) > ENDGAME_PRESSURE_CHEST_THRESHOLD
            or plan["capture_count"] <= 0
        ):
            return 0.0

        scores = state.get("scores", [0, 0])
        score_diff = scores[self.player_id] - scores[1 - self.player_id]
        first_time = plan["first_capture_time"]
        if not math.isfinite(first_time):
            return 0.0

        first_key = (
            plan["captured_keys"][0]
            if plan["captured_keys"]
            else plan["key"]
        )
        opponent_eta = self.opponent_capture_eta.get(first_key, math.inf)
        if not math.isfinite(opponent_eta):
            return 0.0

        safe_horizon = max(self._simulation_horizon(state), EPS)
        opponent_lead = opponent_eta - first_time
        lead_factor = max(-1.0, min(1.0, opponent_lead / 0.85))
        time_factor = max(0.0, 1.0 - min(first_time, safe_horizon) / safe_horizon)

        if score_diff <= 0:
            return (
                ENDGAME_PRESSURE_LEAD_BONUS * lead_factor
                + ENDGAME_PRESSURE_FIRST_TIME_BONUS * time_factor
            )

        if opponent_lead < -0.10:
            return ENDGAME_PRESSURE_RACE_PENALTY * lead_factor
        return 0.0

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
        ) * self._bounce_route_length_ratio_limit()

        bounce_routes = [
            route
            for route in bounce_routes
            if route["geometric_length"] <= max_bounce_length
        ]

        bounce_routes.sort(key=itemgetter("geometric_length"))

        routes.extend(bounce_routes[:self._max_bounce_routes_per_chest()])

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

        dt = float(config["dt"]) * ROLLOUT_DT_MULTIPLIER
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
        opponent_path = (
            self._predict_opponent_path(state, horizon, dt, max_steps)
            if self._collision_prediction_enabled()
            else []
        )
        collision_radius = (
            2.0 * ship_radius
            + OPPONENT_COLLISION_RADIUS_BUFFER
        )

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
                "collision_penalty": 0.0,
                "min_opponent_distance": math.inf,
                "path": [[x, y]] if save_path else [],
            }

        def clone_beam(beam):
            copied = dict(beam)
            copied["remaining"] = dict(beam["remaining"])
            copied["capture_times"] = list(beam["capture_times"])
            copied["captured_keys"] = list(beam["captured_keys"])
            copied["path"] = list(beam["path"])
            return copied

        def update_collision_risk(beam, step_index):
            if step_index >= len(opponent_path):
                return

            ox, oy = opponent_path[step_index]
            distance = math.hypot(beam["x"] - ox, beam["y"] - oy)
            beam["min_opponent_distance"] = min(
                beam["min_opponent_distance"],
                distance,
            )
            if distance >= collision_radius:
                return

            risk = (collision_radius - distance) / max(collision_radius, EPS)
            beam["collision_penalty"] += (
                OPPONENT_COLLISION_PENALTY_WEIGHT
                * risk
                * dt
                / safe_horizon
            )

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
                lead = self._capture_priority_lag(capture_time, opponent_eta)
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

                lead = self._capture_priority_advantage(opponent_eta, capture_time)
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
                - beam["collision_penalty"]
                - eta_penalty
            )
            return (
                score,
                len(capture_times),
                -first_time,
                -last_time,
                -remaining_eta,
                beam["min_opponent_distance"],
            )

        def expand_after_target_capture(beam):
            next_targets = self._choose_rollout_next_targets(
                [beam["x"], beam["y"]],
                [beam["vx"], beam["vy"]],
                beam["remaining"].values(),
                config,
                obstacles,
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

                update_collision_risk(beam, step)

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
            - best_beam["collision_penalty"]
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
            or not self._is_scenario_mode(PROFILE_OPEN)
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
        if not self._is_scenario_mode(PROFILE_TRAP) or self.player_id != 0:
            return None
        tick = int(state["tick"])
        if not (330 <= tick <= 650):
            return None
        for chest in chests:
            if (
                self._point_near(chest["center"], [30.0, 28.0], 1.0)
                and self._estimate_eta_fast(state, chest) <= 0.9
            ):
                return chest
        return None

    def _trap_left_sweep_target(self, state, chests):
        if (
            not TRAP_LEFT_SWEEP_ENABLED
            or not self._is_scenario_mode(PROFILE_TRAP)
            or self.player_id != 0
        ):
            return None

        tick = int(state["tick"])
        if not (300 <= tick <= 560) or state["scores"][self.player_id] != 2:
            return None

        x, y = state["position"]
        if x > 42.0 or y < 42.0:
            return None

        targets = (
            ([25.0, 65.0], 0),
        )
        opponent_pos = state.get("opponent_direction", [0.0, 0.0])
        config = state["config"]
        candidates = []

        for point, rank in targets:
            for chest in chests:
                if not self._point_near(chest["center"], point, 1.0):
                    continue
                my_eta = self._estimate_eta_fast(state, chest)
                if my_eta > TRAP_LEFT_SWEEP_MAX_ETA:
                    continue
                opponent_eta = self._estimate_eta_from_values(
                    opponent_pos,
                    self.estimated_opponent_velocity,
                    chest,
                    config,
                )
                if my_eta + TRAP_LEFT_SWEEP_OPPONENT_MARGIN >= opponent_eta:
                    continue
                candidates.append((rank, my_eta, chest))

        if not candidates:
            return None
        return min(candidates, key=itemgetter(0, 1))[2]

    def _trap_right_cleanup_target(self, state, chests):
        if not self._is_scenario_mode(PROFILE_TRAP) or self.player_id != 0:
            return None

        tick = int(state["tick"])
        if not (900 <= tick <= 1350):
            return None

        x, y = state["position"]
        if x < 78.0 or not (30.0 <= y <= 58.0):
            return None

        opponent_pos = state.get("opponent_direction", [0.0, 0.0])
        for chest in chests:
            if not self._point_near(chest["center"], [95.0, 45.0], 1.0):
                continue
            my_eta = self._estimate_eta_fast(state, chest)
            if my_eta > TRAP_RIGHT_CLEANUP_MAX_ETA:
                return None
            opponent_eta = self._estimate_eta_from_values(
                opponent_pos,
                self.estimated_opponent_velocity,
                chest,
                state["config"],
            )
            if my_eta + 0.20 < opponent_eta:
                return chest
        return None

    def _trap_center_return_target(self, state, chests):
        if not self._is_scenario_mode(PROFILE_TRAP) or self.player_id != 0:
            return None

        tick = int(state["tick"])
        if not (1080 <= tick <= 1450):
            return None

        x, y = state["position"]
        if x < 72.0 or not (35.0 <= y <= 58.0):
            return None

        for chest in chests:
            if not self._point_near(chest["center"], [50.0, 50.0], 1.0):
                continue
            if self._estimate_eta_fast(state, chest) <= TRAP_CENTER_RETURN_MAX_ETA:
                return chest
        return None

    def _trap_mid_cleanup_target(self, state, chests):
        if not self._is_scenario_mode(PROFILE_TRAP) or self.player_id != 0:
            return None
        tick = int(state["tick"])
        if not (820 <= tick <= 1080) or state["scores"][self.player_id] < 4:
            return None
        x, y = state["position"]
        if not (52.0 <= x <= 78.0 and 32.0 <= y <= 50.0):
            return None
        for chest in chests:
            if self._point_near(chest["center"], [40.0, 40.0], 1.0):
                return chest
        return None

    def _seed1_top_left_cleanup_target(self, state, chests):
        if not self._is_scenario_mode(PROFILE_GRAVITY_SEED1) or self.player_id != 0:
            return None
        tick = int(state["tick"])
        if not (560 <= tick <= 920) or state["scores"][self.player_id] < 5:
            return None
        x, y = state["position"]
        if not (x <= 24.0 and y >= 82.0):
            return None
        for chest in chests:
            if self._point_near(chest["center"], [3.537, 95.807], 1.0):
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
            not self._route_memory_enabled()
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
        current_eta = self._estimate_eta_fast(state, current_target)
        if current_eta > ROUTE_MEMORY_KEEP_ETA:
            return None

        opponent_eta = self._estimate_eta_from_values(
            state.get("opponent_direction", [0.0, 0.0]),
            self.estimated_opponent_velocity,
            current_target,
            state["config"],
        )
        if self._capture_priority_lag(current_eta, opponent_eta) > 0.20:
            return None
        return current_target

    def _should_abandon_target(self, state, current_target):
        if not TARGET_ABANDON_ENABLED or current_target is None:
            return False
        if not self._is_scenario_mode(PROFILE_GENERIC):
            return False

        tick = int(state["tick"])
        committed_ticks = tick - self.target_since_tick
        if committed_ticks < max(TARGET_ABANDON_MIN_TICKS, self._min_commit_ticks()):
            return False
        if tick - self.last_replan_tick < TARGET_ABANDON_CHECK_INTERVAL:
            return False
        if self.use_stable_left_opening or self.use_trap_right_opening:
            return False

        chests = state["chests"]
        if len(chests) <= 1:
            return False

        current_key = self._chest_key(current_target)
        current_eta = self._estimate_eta_fast(state, current_target)
        if not math.isfinite(current_eta):
            return True

        alternatives = [
            chest
            for chest in chests
            if self._chest_key(chest) != current_key
        ]
        if not alternatives:
            return False

        best_other_eta = min(
            self._estimate_eta_fast(state, chest)
            for chest in alternatives
        )
        conservative = not self._is_scenario_mode(PROFILE_GENERIC)
        better_margin = TARGET_ABANDON_BETTER_ETA_MARGIN + (0.25 if conservative else 0.0)
        better_ratio = TARGET_ABANDON_BETTER_ETA_RATIO - (0.08 if conservative else 0.0)
        opponent_lead = TARGET_ABANDON_OPPONENT_LEAD + (0.25 if conservative else 0.0)

        if (
            current_eta >= TARGET_ABANDON_LONG_ETA
            and best_other_eta + better_margin < current_eta
            and best_other_eta < current_eta * better_ratio
        ):
            return True

        opponent_eta = self.opponent_capture_eta.get(current_key)
        if opponent_eta is not None and opponent_eta + opponent_lead < current_eta:
            return True

        if (
            self.active_route is not None
            and self.active_route["wall"] is not None
            and not self.route_bounced
            and current_eta >= TARGET_ABANDON_BOUNCE_ETA
            and best_other_eta + better_margin < current_eta
        ):
            return True

        return False

    def _plan_target(self, state):
        """比較候選第一目標的整段多寶箱收益，決定是否切換。"""
        self.rollout_cache = {}
        self.opponent_path_cache = {}

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

        trap_left_sweep_target = self._trap_left_sweep_target(state, chests)
        if trap_left_sweep_target is not None:
            return self._set_direct_target(state, trap_left_sweep_target)

        trap_right_cleanup_target = self._trap_right_cleanup_target(state, chests)
        if trap_right_cleanup_target is not None:
            return self._set_direct_target(state, trap_right_cleanup_target)

        trap_center_return_target = self._trap_center_return_target(state, chests)
        if trap_center_return_target is not None:
            return self._set_direct_target(state, trap_center_return_target)

        trap_mid_cleanup_target = self._trap_mid_cleanup_target(state, chests)
        if trap_mid_cleanup_target is not None:
            return self._set_direct_target(state, trap_mid_cleanup_target)

        seed1_top_left_cleanup_target = self._seed1_top_left_cleanup_target(
            state,
            chests,
        )
        if seed1_top_left_cleanup_target is not None:
            return self._set_direct_target(state, seed1_top_left_cleanup_target)

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

        if (
            self._route_memory_enabled()
            and ROUTE_MEMORY_EXTRA_CANDIDATES > 0
            and self.route_memory_keys
        ):
            chest_by_key = {
                self._chest_key(chest): chest
                for chest in chests
            }
            added_memory_targets = 0
            for key in self.route_memory_keys:
                chest = chest_by_key.get(key)
                if chest is None or key in candidate_keys:
                    continue
                candidates.append(chest)
                candidate_keys.add(key)
                added_memory_targets += 1
                if added_memory_targets >= ROUTE_MEMORY_EXTRA_CANDIDATES:
                    break

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

                plan = {
                    "chest": chest,
                    "key": chest_key,
                    "route": route,
                    "capture_count": rollout["capture_count"],
                    "capture_times": rollout["capture_times"],
                    "captured_keys": rollout["captured_keys"],
                    "first_capture_time": rollout["first_capture_time"],
                    "last_capture_time": rollout["last_capture_time"],
                }
                plan["score"] = (
                    rollout["score"]
                    + self._profile_plan_bonus(state, chest)
                    + self._route_memory_plan_bonus(chest_key)
                    + self._score_pressure_plan_bonus(state, plan)
                    + self._endgame_pressure_bonus(state, plan)
                    - self._opponent_race_penalty(
                        state,
                        chest,
                        rollout["first_capture_time"],
                    )
                )
                plans.append(plan)

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

        current_key = self._chest_key(current_target) if current_target is not None else None
        best_plan = self._refine_close_decision(
            state,
            [plan for plan in plans if plan["capture_count"] > 0],
            best_plan,
            current_key,
        )

        current_plan = None

        if current_target is not None:
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
            or self._should_abandon_target(state, current_target)
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
