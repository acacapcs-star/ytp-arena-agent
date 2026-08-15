"""strategy.py 的單元測試。

只測不需要對戰環境的純函式與靜態方法——這些是整個 Agent 的計算基礎，
一旦算錯，上層的 beam search 與 MPC 都會建立在錯誤的數字上。

執行：
    python -m pytest tests/ -v
或不裝 pytest：
    python tests/test_strategy.py
"""

import math
import os
import sys
import types
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# strategy.py 會 import 主辦提供的 pirate_client，本 repo 未收錄。
# 測試只針對純函式，因此注入一個最小替身讓 import 通過。
if "pirate_client" not in sys.modules:
    stub = types.ModuleType("pirate_client")

    class _BaseStrategy:
        def initialize(self, player_id):
            pass

        def act(self, state):
            return [0.0, 0.0]

    stub.Strategy = _BaseStrategy
    sys.modules["pirate_client"] = stub

from strategy import Strategy  # noqa: E402


class TestTimeToCover(unittest.TestCase):
    """_time_to_cover_1d：一維最短時間估計，抵達時不要求停下。"""

    f = staticmethod(Strategy._time_to_cover_1d)

    def test_zero_distance_takes_no_time(self):
        self.assertEqual(self.f(0.0, 0.0, 10.0, 20.0), 0.0)

    def test_negative_distance_takes_no_time(self):
        self.assertEqual(self.f(-5.0, 0.0, 10.0, 20.0), 0.0)

    def test_no_acceleration_never_arrives(self):
        self.assertEqual(self.f(10.0, 0.0, 0.0, 20.0), math.inf)

    def test_no_max_speed_never_arrives(self):
        self.assertEqual(self.f(10.0, 0.0, 10.0, 0.0), math.inf)

    def test_pure_acceleration_matches_closed_form(self):
        """未達最高速時，distance = ½at² → t = √(2d/a)。"""
        # a=10, 從靜止出發，d=5 → t=1.0
        self.assertAlmostEqual(self.f(5.0, 0.0, 10.0, 100.0), 1.0, places=6)

    def test_initial_speed_shortens_time(self):
        """有初速應該比靜止更快到。"""
        still = self.f(50.0, 0.0, 10.0, 20.0)
        moving = self.f(50.0, 10.0, 10.0, 20.0)
        self.assertLess(moving, still)

    def test_cruise_phase_when_distance_is_long(self):
        """距離夠長時會先加速到頂速再等速前進。"""
        a, v_max = 10.0, 20.0
        d_to_max = v_max * v_max / (2.0 * a)      # 20.0
        t_to_max = v_max / a                       # 2.0
        distance = d_to_max + 100.0
        expected = t_to_max + 100.0 / v_max        # 2.0 + 5.0
        self.assertAlmostEqual(self.f(distance, 0.0, a, v_max), expected, places=6)

    def test_initial_speed_is_clamped_to_max(self):
        """初速超過上限時應被夾住，不能因此算出不合理的短時間。"""
        clamped = self.f(50.0, 20.0, 10.0, 20.0)
        absurd = self.f(50.0, 1e6, 10.0, 20.0)
        self.assertAlmostEqual(clamped, absurd, places=6)

    def test_monotonic_in_distance(self):
        """距離越遠，時間必須越長。"""
        times = [self.f(d, 5.0, 10.0, 20.0) for d in (10.0, 30.0, 60.0, 120.0)]
        self.assertEqual(times, sorted(times))


class TestClampVector(unittest.TestCase):
    """_clamp_vector：把加速度限制在 max_accel 內，方向必須保持不變。"""

    f = staticmethod(Strategy._clamp_vector)

    def test_short_vector_is_unchanged(self):
        self.assertEqual(self.f(3.0, 4.0, 10.0), [3.0, 4.0])

    def test_exact_length_is_unchanged(self):
        self.assertEqual(self.f(3.0, 4.0, 5.0), [3.0, 4.0])

    def test_long_vector_is_scaled_to_limit(self):
        x, y = self.f(30.0, 40.0, 5.0)
        self.assertAlmostEqual(math.hypot(x, y), 5.0, places=9)

    def test_direction_is_preserved_when_scaling(self):
        """縮放後方向角必須完全一致——方向錯了船就往別處加速。"""
        ox, oy = 30.0, 40.0
        x, y = self.f(ox, oy, 5.0)
        self.assertAlmostEqual(math.atan2(y, x), math.atan2(oy, ox), places=9)

    def test_zero_vector_returns_zero(self):
        self.assertEqual(self.f(0.0, 0.0, 10.0), [0.0, 0.0])

    def test_near_zero_vector_returns_zero(self):
        """極小向量正規化會炸開，必須回傳零而不是放大成任意方向。"""
        self.assertEqual(self.f(1e-15, 1e-15, 10.0), [0.0, 0.0])

    def test_negative_components_are_handled(self):
        x, y = self.f(-30.0, -40.0, 5.0)
        self.assertLess(x, 0.0)
        self.assertLess(y, 0.0)
        self.assertAlmostEqual(math.hypot(x, y), 5.0, places=9)


class TestCaptureGap(unittest.TestCase):
    """_capture_gap_between_chests：兩顆寶箱捕獲區之間的實際空隙。"""

    f = staticmethod(Strategy._capture_gap_between_chests)

    @staticmethod
    def chest(x, y, r):
        return {"center": [x, y], "radius": r}

    def test_far_apart_gap_is_positive(self):
        a = self.chest(0.0, 0.0, 1.0)
        b = self.chest(20.0, 0.0, 1.0)
        # 20 - 1 - 1 - 2×0.5 = 17
        self.assertAlmostEqual(self.f(a, b, 0.5), 17.0, places=9)

    def test_overlapping_capture_zones_clamp_to_zero(self):
        """捕獲區重疊時應為 0，不能回傳負數。"""
        a = self.chest(0.0, 0.0, 2.0)
        b = self.chest(3.0, 0.0, 2.0)
        self.assertEqual(self.f(a, b, 0.5), 0.0)

    def test_identical_position_is_zero(self):
        a = self.chest(5.0, 5.0, 1.0)
        b = self.chest(5.0, 5.0, 1.0)
        self.assertEqual(self.f(a, b, 0.5), 0.0)

    def test_larger_ship_shrinks_the_gap(self):
        """船越大，實際要跨越的距離越短。"""
        a = self.chest(0.0, 0.0, 1.0)
        b = self.chest(20.0, 0.0, 1.0)
        self.assertLess(self.f(a, b, 2.0), self.f(a, b, 0.5))

    def test_is_symmetric(self):
        a = self.chest(0.0, 0.0, 1.5)
        b = self.chest(12.0, 9.0, 0.8)
        self.assertAlmostEqual(self.f(a, b, 0.5), self.f(b, a, 0.5), places=9)


class TestChestKey(unittest.TestCase):
    """_chest_key：寶箱沒有 ID，用座標與半徑當識別鍵。"""

    f = staticmethod(Strategy._chest_key)

    def test_same_chest_yields_same_key(self):
        a = {"center": [1.0, 2.0], "radius": 0.5}
        b = {"center": [1.0, 2.0], "radius": 0.5}
        self.assertEqual(self.f(a), self.f(b))

    def test_different_position_yields_different_key(self):
        a = {"center": [1.0, 2.0], "radius": 0.5}
        b = {"center": [1.0, 2.1], "radius": 0.5}
        self.assertNotEqual(self.f(a), self.f(b))

    def test_different_radius_yields_different_key(self):
        a = {"center": [1.0, 2.0], "radius": 0.5}
        b = {"center": [1.0, 2.0], "radius": 0.6}
        self.assertNotEqual(self.f(a), self.f(b))

    def test_key_is_hashable(self):
        """鍵會被放進 dict 與 set，必須可雜湊。"""
        key = self.f({"center": [1.0, 2.0], "radius": 0.5})
        self.assertIsInstance(hash(key), int)

    def test_floating_noise_below_rounding_collapses(self):
        """浮點雜訊不該讓同一顆寶箱被當成兩顆。"""
        a = {"center": [1.0, 2.0], "radius": 0.5}
        b = {"center": [1.0 + 1e-12, 2.0], "radius": 0.5}
        self.assertEqual(self.f(a), self.f(b))


class TestPointNear(unittest.TestCase):
    """_point_near：場景辨識用來比對出生點，容差 0.75。"""

    f = staticmethod(Strategy._point_near)

    def test_exact_match(self):
        self.assertTrue(self.f([5.0, 5.0], [5.0, 5.0]))

    def test_within_tolerance(self):
        self.assertTrue(self.f([5.5, 4.5], [5.0, 5.0]))

    def test_on_tolerance_boundary(self):
        self.assertTrue(self.f([5.75, 5.0], [5.0, 5.0]))

    def test_outside_tolerance(self):
        self.assertFalse(self.f([6.0, 5.0], [5.0, 5.0]))

    def test_both_axes_must_match(self):
        """只有一軸接近不算——否則會誤判成別張地圖。"""
        self.assertFalse(self.f([5.0, 50.0], [5.0, 5.0]))

    def test_custom_tolerance(self):
        self.assertTrue(self.f([7.0, 5.0], [5.0, 5.0], tolerance=3.0))
        self.assertFalse(self.f([7.0, 5.0], [5.0, 5.0], tolerance=1.0))


if __name__ == "__main__":
    unittest.main(verbosity=2)
