"""驗證重建環境的物理是否正確。

第一步做完之前不要開始寫 RL —— 環境錯了後面全部白做。
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import math
import env


def check_wall_bounce():
    """無重力下反彈應完全彈性，速度大小守恆。"""
    sc = {
        "config": dict(env.DEFAULT_CONFIG),
        "players": [[5, 50], [95, 50]],
        "chests": [{"center": [50, 50], "radius": 1.2}],
        "obstacles": [],
    }
    e = env.ArenaEnv(sc)
    e.reset()
    e.vel[0] = [-20.0, 0.0]

    speeds = []
    for _ in range(50):
        e.step([0, 0], [0, 0])
        speeds.append(math.hypot(*e.vel[0]))

    lo = env.DEFAULT_CONFIG["ship_radius"]
    hi = 100 - lo
    in_bounds = lo - 1e-9 <= e.pos[0][0] <= hi + 1e-9

    ok = abs(max(speeds) - 20.0) < 1e-9 and abs(min(speeds) - 20.0) < 1e-9
    print(f"[{'OK' if ok and in_bounds else '!!'}] 牆壁反彈  "
          f"速度 {min(speeds):.6f}~{max(speeds):.6f}（應恆為 20）  界內={in_bounds}")


def check_gravity_field():
    """重力是 1/r 力場，不是平方反比。"""
    sc = {
        "config": dict(env.DEFAULT_CONFIG),
        "players": [[56, 50], [95, 95]],
        "chests": [{"center": [10, 10], "radius": 1.2}],
        "obstacles": [{"center": [50, 50], "radius": 10}],
    }
    e = env.ArenaEnv(sc)

    print("\n重力大小 vs 半徑（應等於 G/r = 100/r）")
    for r in [1, 2, 5, 8, 10, 12]:
        gx, gy = e.gravity_at(50 + r, 50)
        mag = math.hypot(gx, gy)
        expect = 100.0 / r if r <= 10 else 0.0
        flag = "OK" if abs(mag - expect) < 1e-9 else "!!"
        print(f"  [{flag}] r={r:2d}  實測={mag:7.3f}  預期={expect:7.3f}"
              f"{'   ← 井外不受力' if r > 10 else ''}")


def check_escape():
    """靜止時逃不出，帶著速度就能逃。"""
    sc = {
        "config": dict(env.DEFAULT_CONFIG),
        "players": [[56, 50], [95, 95]],
        "chests": [{"center": [10, 10], "radius": 1.2}],
        "obstacles": [{"center": [50, 50], "radius": 10}],
    }
    e0 = env.ArenaEnv(sc)
    print(f"\n推力逃逸半徑 = {e0.thrust_escape_radius():.2f}"
          f"（等於井半徑，代表井內推力永遠輸給重力）")
    print(f"動能逃逸半徑 = {e0.momentum_escape_radius():.3f}（不計推力做功）")

    print("\n從半徑 r 起步能不能逃出井：")
    for v0 in [0.0, 20.0]:
        results = []
        for start in [1.0, 2.0, 5.0, 8.0]:
            e = env.ArenaEnv(sc)
            e.reset()
            e.pos[0] = [50 + start, 50.0]
            e.vel[0] = [v0, 0.0]
            escaped = False
            for _ in range(600):
                e.step([10.0, 0.0], [0.0, 0.0])
                r = math.hypot(e.pos[0][0] - 50, e.pos[0][1] - 50)
                if r > 10.0:
                    escaped = True
                    break
            results.append(f"r0={start:.0f}:{'逃出' if escaped else '困住'}")
        print(f"  初速 {v0:4.0f} 向外 + 全力推 →  " + "  ".join(results))


def check_symmetry():
    """對稱地圖、同一支策略對打，結果應該接近平分。"""
    import strategy
    sc = env.load_scenario("scenarios/open.csv")
    r = env.run_match(sc, strategy.Strategy(0), strategy.Strategy(1))
    print(f"\n對稱檢查（open 圖，strategy.py 自己打自己）")
    print(f"  比分 {r['scores']}  ticks={r['ticks']}  剩餘寶箱={r['chests_left']}")
    print(f"  首次捕獲 tick: {r['first_capture']}")


if __name__ == "__main__":
    check_wall_bounce()
    check_gravity_field()
    check_escape()
    check_symmetry()
