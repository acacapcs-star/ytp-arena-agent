"""多寶箱前瞻策略：快速初選 + 撞牆路線 + 多寶箱滾動模擬。

流程：
1. 用目前位置、速度與朝向，快速估計所有寶箱的 ETA。
2. 額外把「第一顆稍遠，但旁邊還有第二顆」的寶箱群加入候選。
3. 對候選第一目標模擬直接路線與四面牆反彈路線。
4. 模擬不會在吃到第一顆後立即停止，而會保留當時的位置與速度，
   繼續追下一顆寶箱。
5. 每個虛擬 tick 都檢查是否碰到任何寶箱，因此順路擦到的寶箱也會計分。
6. 依「取得寶箱數量 + 捕獲時間」評估整段路線，而不是只比較第一顆 ETA。
7. 真實 Agent 仍只執行目前 tick 的控制，並定期重新規劃。

目前仍是近似解：
- 第一段會比較直接與單次撞牆；後續寶箱暫時使用快速 ETA 貪心並直接前往。
- 不會搜尋所有可能的加速度序列。
- 已近似處理牆壁與行星引力。
- 尚未處理敵船碰撞與敵人搶先拿走寶箱。
"""

import math
import sys
import time
from pirate_client import Strategy as BaseStrategy


# ======================== 可調參數 ========================

REPLAN_INTERVAL = 20       # 多寶箱模擬較重，每 20 ticks 完整重新規劃一次
TOP_K_CANDIDATES = 5       # 依單顆快速 ETA 保留的主要候選數
CLUSTER_EXTRA_CANDIDATES = 2  # 額外保留「附近還有下一顆」的群聚候選數
MIN_COMMIT_TICKS = 20      # 選定第一目標後，至少維持多少 ticks

FAST_ETA_TURN_WEIGHT = 0.80
CONTROL_RESPONSE_TIME = 0.08

# 多寶箱模擬最多向未來看幾秒。
# 原本只模擬到第一顆時可設較長；現在每條路線會一路模擬到視野結束，
SIMULATION_HORIZON = 6.0

# 一條模擬路線最多計入幾顆寶箱；達到後便提早停止。
MAX_ROLLOUT_CHESTS = 4

# 路線分數：
#     score = 捕獲數 - CAPTURE_TIME_WEIGHT * sum(捕獲時間 / 視野)
#
# 數量仍是主體，但越早拿到越好。
# 0.5 代表兩顆非常晚的寶箱，不會無條件大勝一顆立即取得的寶箱。
CAPTURE_TIME_WEIGHT = 0.50

# 新第一目標的整段模擬分數，至少高出目前計畫多少才換。
# 這是絕對分數差；一顆寶箱的基本價值約為 1。
PLAN_SWITCH_SCORE_MARGIN = 0.08

# 快速候選初選時，會估計「第一顆 + 最近第二顆」的平均時間。
# 1.0 表示第一顆與第二顆同等參與群聚排序。
PREFILTER_SECOND_CHEST_WEIGHT = 1.0

GRAVITY_CONSTANT = 100.0
GRAVITY_COMPENSATION = 1.0
SAVE_DEBUG_PATH = False
DEBUG_PATH_SAMPLE_INTERVAL = 5
EPS = 1e-9

# 開發期間設為 True，會測量每個 tick 的實際運算時間。
# 正式提交前改成 False，避免額外輸出與少量測量成本。
PROFILE_TIMING = True

# 單一 tick 超過幾秒時顯示警告。
SLOW_TICK_WARNING = 0.20

# 每多少 ticks 印一次統計。
TIMING_REPORT_INTERVAL = 500

# ===================== 撞牆路線搜尋參數 ======================

# 是否讓規劃器主動搜尋一次撞牆反彈的路線。
ENABLE_SINGLE_WALL_BOUNCE = True


# 一顆寶箱最多保留幾條撞牆候選。
#
# 理論上有左、右、下、上共 4 條，
# 依計算量設定上限
MAX_BOUNCE_ROUTES_PER_CHEST = 4


# 避免規劃出太靠近角落的撞牆點。
#
# 靠近角落時可能同時撞兩面牆，
# 模擬結果會比較不穩定。
BOUNCE_CORNER_MARGIN = 1.0


# 用目前狀態判斷「實際船已經完成撞牆」時，
# 允許距離牆面有多少誤差。
BOUNCE_DETECTION_MULTIPLIER = 2.0


class Strategy(BaseStrategy):
    def initialize(self, player_id):
        """整局開始時只呼叫一次。"""
        self.player_id = player_id
        self.target_key = None
        self.target_since_tick = 0
        self.last_replan_tick = -10**9
        # last_plan_eta 現在表示整段計畫中「第一顆實際取得的時間」。
        self.last_plan_eta = math.inf
        self.last_plan_score = -math.inf
        self.last_plan_capture_times = []
        self.last_plan_path = []
        self.last_acceleration = [0.0, 0.0]

        # 目前採用的路線。
        #
        # direct:
        #     直接前往寶箱。
        #
        # bounce:
        #     先朝鏡射目標飛，撞指定牆後再前往真正寶箱。
        self.active_route = {
            "kind": "direct",
            "wall": None,
            "virtual_target": None,
            "bounce_point": None,
        }

        # direct 路線不需要撞牆，因此視為已完成撞牆階段。
        self.route_bounced = True

        # ==================== 效能測量資料 ====================

        # act() 總共被呼叫幾次。
        self.timing_call_count = 0

        # 所有 act() 累積使用的現實時間。
        self.timing_total_seconds = 0.0

        # 單一 act() 最久使用的時間。
        self.timing_max_seconds = 0.0

        # 最慢的 tick 編號。
        self.timing_max_tick = -1

        # 完整重新規劃的次數。
        self.timing_replan_count = 0

        # 所有 _plan_target() 的累積時間。
        self.timing_replan_total_seconds = 0.0

        # 單次 _plan_target() 最久時間。
        self.timing_replan_max_seconds = 0.0

    # ======================== 基礎工具 ========================

    @staticmethod
    def _clamp(value, lower, upper):
        return max(lower, min(upper, value))

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
    def _capture_distance(position, ship_radius, chest):
        """船還要前進多遠，才會與寶箱的圓相碰。"""
        x, y = position
        cx, cy = chest["center"]
        center_distance = math.hypot(cx - x, cy - y)
        return max(0.0, center_distance - ship_radius - chest["radius"])

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

        others = [
            other
            for other in all_chests
            if self._chest_key(other) != self._chest_key(chest)
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

    def _choose_rollout_next_target(
        self,
        position,
        velocity,
        remaining_chests,
        config,
    ):
        """在虛擬模擬中，吃到目前目標後選下一顆。

        第一版只在「成功取得目前目標」時重選，不會每個虛擬 tick 改目標。
        這可避免虛擬路線不停抖動，也能控制計算量。
        """
        if not remaining_chests:
            return None

        return min(
            remaining_chests,
            key=lambda chest: self._estimate_eta_from_values(
                position,
                velocity,
                chest,
                config,
            ),
        )

    @staticmethod
    def _rollout_score(capture_times):
        """把多顆寶箱的取得數量與時間合成單一分數。

        每顆寶箱基本價值為 1；取得越晚，扣分越多。
        """
        if not capture_times:
            return 0.0

        normalized_time_sum = sum(
            capture_time / SIMULATION_HORIZON
            for capture_time in capture_times
        )

        return (
            len(capture_times)
            - CAPTURE_TIME_WEIGHT * normalized_time_sum
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

    def _make_route_options(self, state, chest):
        """產生抵達指定寶箱的候選路線。

        回傳的候選至少包含：
            1. 直接路線

        如果 ENABLE_SINGLE_WALL_BOUNCE=True，
        還會嘗試：
            2. 左牆反彈
            3. 右牆反彈
            4. 下牆反彈
            5. 上牆反彈

        撞牆路線使用「鏡射法」：

        若想在牆上反彈後抵達真正寶箱，
        可以把寶箱對牆鏡射，先朝鏡射位置飛。

        例如右牆位置是 x = wall_x：

            真寶箱 x = chest_x

            鏡射寶箱 x：
                virtual_x = 2 * wall_x - chest_x

        從船到鏡射寶箱的直線與牆的交點，
        就是理想撞牆點。
        """

        x, y = state["position"]

        chest_x, chest_y = chest["center"]

        config = state["config"]

        field_size = float(config["L"])
        ship_radius = float(config["ship_radius"])

        # 船中心真正能到達的牆面位置。
        #
        # 因為船本身有半徑，
        # 船中心不可能到 x = 0 或 x = L。
        minimum = ship_radius
        maximum = field_size - ship_radius

        # 一定保留直接路線。
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

        # --------------------------------------------------------
        # 嘗試垂直牆：左牆或右牆
        # --------------------------------------------------------

        def try_vertical_wall(wall_name, wall_x):
            # 將寶箱中心對這面牆鏡射。
            virtual_x = 2.0 * wall_x - chest_x
            virtual_y = chest_y

            denominator = virtual_x - x

            if abs(denominator) < EPS:
                return

            # 從目前位置到鏡射目標的參數式：
            #
            # position(t)
            # = start + t * (virtual_target - start)
            #
            # 求它何時碰到 x = wall_x。
            t = (wall_x - x) / denominator

            # 交點必須位於船與鏡射目標之間。
            if not (0.0 < t < 1.0):
                return

            bounce_y = (
                y + t * (virtual_y - y)
            )

            # 撞擊點不能太靠近上下角落。
            if not (
                minimum + BOUNCE_CORNER_MARGIN
                <= bounce_y
                <= maximum - BOUNCE_CORNER_MARGIN
            ):
                return

            bounce_routes.append(
                {
                    "kind": "bounce",
                    "wall": wall_name,
                    "virtual_target": [
                        float(virtual_x),
                        float(virtual_y),
                    ],
                    "bounce_point": [
                        float(wall_x),
                        float(bounce_y),
                    ],

                    # 船到鏡射目標的直線長度，
                    # 就近似等於反彈路線總長。
                    "geometric_length": math.hypot(
                        virtual_x - x,
                        virtual_y - y,
                    ),
                }
            )

        # --------------------------------------------------------
        # 嘗試水平牆：下牆或上牆
        # --------------------------------------------------------

        def try_horizontal_wall(wall_name, wall_y):
            virtual_x = chest_x
            virtual_y = 2.0 * wall_y - chest_y

            denominator = virtual_y - y

            if abs(denominator) < EPS:
                return

            t = (wall_y - y) / denominator

            if not (0.0 < t < 1.0):
                return

            bounce_x = (
                x + t * (virtual_x - x)
            )

            # 撞擊點不能太靠近左右角落。
            if not (
                minimum + BOUNCE_CORNER_MARGIN
                <= bounce_x
                <= maximum - BOUNCE_CORNER_MARGIN
            ):
                return

            bounce_routes.append(
                {
                    "kind": "bounce",
                    "wall": wall_name,
                    "virtual_target": [
                        float(virtual_x),
                        float(virtual_y),
                    ],
                    "bounce_point": [
                        float(bounce_x),
                        float(wall_y),
                    ],
                    "geometric_length": math.hypot(
                        virtual_x - x,
                        virtual_y - y,
                    ),
                }
            )

        try_vertical_wall(
            wall_name="left",
            wall_x=minimum,
        )

        try_vertical_wall(
            wall_name="right",
            wall_x=maximum,
        )

        try_horizontal_wall(
            wall_name="bottom",
            wall_y=minimum,
        )

        try_horizontal_wall(
            wall_name="top",
            wall_y=maximum,
        )

        # 只保留幾何上較短的撞牆候選，
        # 避免模擬量膨脹太多。
        bounce_routes.sort(
            key=lambda route:
                route["geometric_length"]
        )

        routes.extend(
            bounce_routes[
                :MAX_BOUNCE_ROUTES_PER_CHEST
            ]
        )

        return routes

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

        # direct 路線：
        #     aim_point 為 None，直接朝真正寶箱。
        #
        # bounce 路線：
        #     撞牆前，aim_point 是鏡射寶箱位置。
        #     撞牆後，aim_point 會改回真正寶箱位置。
        if aim_point is None:
            target_x, target_y = chest["center"]
        else:
            target_x, target_y = aim_point

        max_accel = float(config["max_accel"])
        max_speed = float(config["max_speed"])

        dx = target_x - x
        dy = target_y - y
        distance = math.hypot(dx, dy)

        if distance < EPS:
            desired_vx = 0.0
            desired_vy = 0.0
        else:
            ux = dx / distance
            uy = dy / distance
            desired_vx = ux * max_speed
            desired_vy = uy * max_speed

        # 希望在 CONTROL_RESPONSE_TIME 秒內修正到期望速度。
        desired_total_ax = (desired_vx - vx) / CONTROL_RESPONSE_TIME
        desired_total_ay = (desired_vy - vy) / CONTROL_RESPONSE_TIME

        if gravity is None:
            gravity_ax, gravity_ay = self._gravity_acceleration(
                position,
                obstacles,
            )
        else:
            gravity_ax, gravity_ay = gravity

        # 先嘗試抵消行星引力，再交給加速度上限裁切。
        control_ax = desired_total_ax - GRAVITY_COMPENSATION * gravity_ax
        control_ay = desired_total_ay - GRAVITY_COMPENSATION * gravity_ay

        return self._clamp_vector(control_ax, control_ay, max_accel)

    # ======================== 詳細模擬 ========================

    def _simulate_plan(self, state, first_chest, route, save_path=False):
        """模擬一整段多寶箱計畫。

        第一目標使用傳入的 direct / bounce 路線。
        吃到第一目標後，保留虛擬船當下的位置與速度，從剩餘寶箱中
        用快速 ETA 選下一顆，並直接前往；如此持續到：

        - 模擬時間到達 SIMULATION_HORIZON，或
        - 已取得 MAX_ROLLOUT_CHESTS 顆，或
        - 場上已沒有剩餘寶箱。

        每個虛擬 tick 都檢查所有剩餘寶箱，因此順路碰到的非目標寶箱
        也會被記錄，且同一 tick 可以取得多顆。
        """
        x = float(state["position"][0])
        y = float(state["position"][1])
        vx = float(state["velocity"][0])
        vy = float(state["velocity"][1])

        config = state["config"]
        obstacles = state.get("obstacles", [])

        dt = float(config["dt"])
        max_speed = float(config["max_speed"])
        ship_radius = float(config["ship_radius"])
        field_size = float(config["L"])

        max_steps = max(1, int(SIMULATION_HORIZON / dt))
        path = [[x, y]] if save_path else []
        first_acceleration = [0.0, 0.0]

        remaining = {
            self._chest_key(chest): chest
            for chest in state["chests"]
        }

        current_target = first_chest
        current_target_key = self._chest_key(first_chest)

        # 只有第一段會執行規劃出的撞牆路線；後續段先採直接路線。
        using_initial_route = True
        route_bounced = route["wall"] is None

        capture_times = []
        captured_keys = []

        def collect_chests(current_time):
            """收集虛擬船目前位置碰到的所有剩餘寶箱。"""
            newly_captured = []

            for key, candidate in list(remaining.items()):
                cx, cy = candidate["center"]
                capture_radius = ship_radius + float(candidate["radius"])
                dx = cx - x
                dy = cy - y

                # 每個虛擬 tick 都會檢查許多寶箱；用平方距離避免大量 sqrt。
                if dx * dx + dy * dy <= capture_radius * capture_radius:
                    newly_captured.append(key)

            for key in newly_captured:
                remaining.pop(key, None)
                captured_keys.append(key)
                capture_times.append(current_time)

            return newly_captured

        # 理論上 state 中的寶箱通常不會在 t=0 已被捕獲，仍做完整防護。
        initially_captured = collect_chests(0.0)

        if current_target_key in initially_captured:
            current_target = self._choose_rollout_next_target(
                [x, y],
                [vx, vy],
                list(remaining.values()),
                config,
            )
            current_target_key = (
                self._chest_key(current_target)
                if current_target is not None
                else None
            )
            using_initial_route = False
            route_bounced = True

        for step in range(max_steps):
            if (
                len(capture_times) >= MAX_ROLLOUT_CHESTS
                or not remaining
                or current_target is None
            ):
                break

            # 第一段尚未撞到指定牆時，朝鏡射寶箱飛。
            # 撞牆完成，或已進入第二顆之後，就朝目前真正目標飛。
            if using_initial_route and not route_bounced:
                aim_point = route["virtual_target"]
            else:
                aim_point = current_target["center"]

            gravity = self._gravity_acceleration(
                [x, y],
                obstacles,
            )

            control_ax, control_ay = self._control_to_chest(
                position=[x, y],
                velocity=[vx, vy],
                chest=current_target,
                config=config,
                obstacles=obstacles,
                aim_point=aim_point,
                gravity=gravity,
            )

            if step == 0:
                first_acceleration = [control_ax, control_ay]

            gravity_ax, gravity_ay = gravity

            # 近似伺服器：先更新速度，再更新位置。
            vx += (control_ax + gravity_ax) * dt
            vy += (control_ay + gravity_ay) * dt
            vx, vy = self._clamp_vector(vx, vy, max_speed)

            x += vx * dt
            y += vy * dt

            # 近似牆壁反彈。
            minimum = ship_radius
            maximum = field_size - ship_radius
            hit_walls = set()

            if x < minimum:
                x = 2.0 * minimum - x
                vx = -vx
                hit_walls.add("left")
            elif x > maximum:
                x = 2.0 * maximum - x
                vx = -vx
                hit_walls.add("right")

            if y < minimum:
                y = 2.0 * minimum - y
                vy = -vy
                hit_walls.add("bottom")
            elif y > maximum:
                y = 2.0 * maximum - y
                vy = -vy
                hit_walls.add("top")

            if (
                using_initial_route
                and not route_bounced
                and route["wall"] in hit_walls
            ):
                route_bounced = True

            current_time = (step + 1) * dt
            newly_captured = collect_chests(current_time)

            # 只有目前正式目標被取得時，才重新決定下一顆。
            # 若只是順路吃到其他寶箱，仍維持原本方向。
            if (
                current_target_key is not None
                and current_target_key in newly_captured
            ):
                current_target = self._choose_rollout_next_target(
                    [x, y],
                    [vx, vy],
                    list(remaining.values()),
                    config,
                )
                current_target_key = (
                    self._chest_key(current_target)
                    if current_target is not None
                    else None
                )

                # 第一顆正式目標完成後，不再沿用原本的鏡射牆路線。
                using_initial_route = False
                route_bounced = True

            if (
                save_path
                and step % DEBUG_PATH_SAMPLE_INTERVAL == 0
            ):
                path.append([x, y])

        score = self._rollout_score(capture_times)
        first_capture_time = (
            capture_times[0]
            if capture_times
            else math.inf
        )
        last_capture_time = (
            capture_times[-1]
            if capture_times
            else math.inf
        )

        return {
            "score": score,
            "capture_count": len(capture_times),
            "capture_times": capture_times,
            "captured_keys": captured_keys,
            "first_capture_time": first_capture_time,
            "last_capture_time": last_capture_time,
            "path": path,
            "first_acceleration": first_acceleration,
        }

    # ======================== 目標規劃 ========================

    def _find_current_target(self, chests):
        if self.target_key is None:
            return None
        for chest in chests:
            if self._chest_key(chest) == self.target_key:
                return chest
        return None

    def _plan_target(self, state):
        """比較候選第一目標的整段多寶箱收益，決定是否切換。"""
        chests = state["chests"]
        tick = int(state["tick"])

        if not chests:
            self.target_key = None
            self.last_plan_eta = math.inf
            self.last_plan_score = -math.inf
            self.last_plan_capture_times = []
            self.last_plan_path = []
            return None

        current_target = self._find_current_target(chests)

        # 第一組候選：單顆快速 ETA 最佳者。
        ranked_chests = sorted(
            chests,
            key=lambda chest: self._estimate_eta_fast(state, chest),
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

        # 目前目標即使不在候選中，也要加入，才能公平比較是否值得換。
        if current_target is not None:
            current_key = self._chest_key(current_target)
            if current_key not in candidate_keys:
                candidates.append(current_target)
                candidate_keys.add(current_key)

        plans = []

        for chest in candidates:
            route_options = self._make_route_options(
                state,
                chest,
            )

            for route in route_options:
                rollout = self._simulate_plan(
                    state,
                    first_chest=chest,
                    route=route,
                    save_path=False,
                )

                plans.append({
                    "chest": chest,
                    "key": self._chest_key(chest),
                    "route": route,
                    "score": rollout["score"],
                    "capture_count": rollout["capture_count"],
                    "capture_times": rollout["capture_times"],
                    "captured_keys": rollout["captured_keys"],
                    "first_capture_time": rollout["first_capture_time"],
                    "last_capture_time": rollout["last_capture_time"],
                    "first_acceleration": rollout["first_acceleration"],
                })

        successful_plans = [
            plan
            for plan in plans
            if plan["capture_count"] > 0
        ]

        if successful_plans:
            best_plan = max(
                successful_plans,
                key=self._plan_rank_key,
            )
        else:
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
                "first_acceleration": self._control_to_chest(
                    position=state["position"],
                    velocity=state["velocity"],
                    chest=fallback_chest,
                    config=state["config"],
                    obstacles=state.get("obstacles", []),
                ),
            }

        current_plan = None

        if current_target is not None:
            current_key = self._chest_key(current_target)
            current_target_plans = [
                plan
                for plan in plans
                if (
                    plan["key"] == current_key
                    and plan["capture_count"] > 0
                )
            ]

            if current_target_plans:
                current_plan = max(
                    current_target_plans,
                    key=self._plan_rank_key,
                )

        if current_target is None:
            chosen_plan = best_plan
        else:
            committed_ticks = tick - self.target_since_tick

            if (
                committed_ticks < MIN_COMMIT_TICKS
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

        self.active_route = chosen_plan["route"]
        self.route_bounced = self.active_route["wall"] is None

        self.last_plan_eta = chosen_plan["first_capture_time"]
        self.last_plan_score = chosen_plan["score"]
        self.last_plan_capture_times = list(
            chosen_plan["capture_times"]
        )
        self.last_replan_tick = tick

        if SAVE_DEBUG_PATH:
            debug_plan = self._simulate_plan(
                state,
                first_chest=chosen_plan["chest"],
                route=chosen_plan["route"],
                save_path=True,
            )
            self.last_plan_path = debug_plan["path"]
        else:
            self.last_plan_path = []

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

        ship_radius = float(
            config["ship_radius"]
        )
        field_size = float(
            config["L"]
        )
        max_speed = float(
            config["max_speed"]
        )
        dt = float(
            config["dt"]
        )

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

    # ======================== 官方入口 ========================

    def _act_impl(self, state):
        """每個 tick 回傳合法的 [ax, ay]。"""
        chests = state["chests"]
        tick = int(state["tick"])

        if not chests:
            # 沒寶箱時煞停。
            vx, vy = state["velocity"]
            max_accel = float(state["config"]["max_accel"])
            acceleration = self._clamp_vector(
                -vx / CONTROL_RESPONSE_TIME,
                -vy / CONTROL_RESPONSE_TIME,
                max_accel,
            )
            self.last_acceleration = acceleration
            return acceleration

        current_target = self._find_current_target(chests)

        should_replan = (
            current_target is None
            or tick - self.last_replan_tick >= REPLAN_INTERVAL
        )

        if should_replan:
            # 完整重新規劃是最耗時間的部分，
            # 因此另外測量它花了多久。
            replan_start = time.perf_counter()

            current_target = self._plan_target(state)

            replan_elapsed = (
                time.perf_counter() - replan_start
            )

            if PROFILE_TIMING:
                self.timing_replan_count += 1
                self.timing_replan_total_seconds += replan_elapsed
                self.timing_replan_max_seconds = max(
                    self.timing_replan_max_seconds,
                    replan_elapsed,
                )

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


        acceleration = self._control_to_chest(
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

        self.last_acceleration = acceleration
        return acceleration
    
    def act(self, state):
        """官方真正呼叫的入口，同時計算本 tick 的執行時間。"""

        # 關閉效能測量時，直接執行原本策略。
        if not PROFILE_TIMING:
            return self._act_impl(state)

        tick = int(state["tick"])

        # 記錄 act() 開始的現實時間。
        start_time = time.perf_counter()

        try:
            # 執行真正策略。
            return self._act_impl(state)

        finally:
            # 即使 _act_impl() 中途 return，
            # finally 仍然一定會執行。
            elapsed = (
                time.perf_counter() - start_time
            )

            self.timing_call_count += 1
            self.timing_total_seconds += elapsed

            # 更新單一 tick 最慢紀錄。
            if elapsed > self.timing_max_seconds:
                self.timing_max_seconds = elapsed
                self.timing_max_tick = tick

            # 某個 tick 明顯太慢時立即警告。
            #
            # 使用 stderr，而不是 stdout，
            # 避免干擾 Agent 與遊戲伺服器的通訊格式。
            if elapsed >= SLOW_TICK_WARNING:
                print(
                    "[TIMING WARNING]"
                    f" tick={tick}"
                    f" act={elapsed * 1000:.3f} ms",
                    file=sys.stderr,
                    flush=True,
                )

            # 每隔一段時間報告整體統計。
            if (
                tick > 0
                and tick % TIMING_REPORT_INTERVAL == 0
            ):
                average_seconds = (
                    self.timing_total_seconds
                    / self.timing_call_count
                )

                max_ticks = int(
                    state["config"]["max_ticks"]
                )

                # 若後面平均速度不變，
                # 粗略估計跑滿 max_ticks 會花多少計算時間。
                projected_total_seconds = (
                    average_seconds * max_ticks
                )

                if self.timing_replan_count > 0:
                    average_replan_seconds = (
                        self.timing_replan_total_seconds
                        / self.timing_replan_count
                    )
                else:
                    average_replan_seconds = 0.0

                print(
                    "[TIMING]"
                    f" tick={tick}"
                    f" avg_act={average_seconds * 1000:.3f} ms"
                    f" max_act={self.timing_max_seconds * 1000:.3f} ms"
                    f" max_tick={self.timing_max_tick}"
                    f" avg_replan={average_replan_seconds * 1000:.3f} ms"
                    f" max_replan={self.timing_replan_max_seconds * 1000:.3f} ms"
                    f" used={self.timing_total_seconds:.3f} s"
                    f" projected={projected_total_seconds:.3f} s",
                    file=sys.stderr,
                    flush=True,
                )