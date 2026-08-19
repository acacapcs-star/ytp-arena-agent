"""分區塊訓練（curriculum）

一次丟完整任務給 PPO 學不起來 —— 實測 120 輪之後得分還停在 2.4，
跟隨機策略的 2.83 差不多，熵只從 2.83 掉到 2.52。

原因有兩個：
  1. 單局太長：1000 個決策步才結束一局，245k 步只玩了 245 局
  2. 獎勵太稀疏：1000 步裡只有 20 次 +1，密度 2%

分區塊的作法是把難度拆開，每個區塊只加一個新挑戰，
權重一路帶到下一塊。每塊的單局也短得多，局數自然變多。

  區塊 1  導航      小地圖、少寶箱、沒有井、對手不動
  區塊 2  避開重力井  加入一個井
  區塊 3  多井       加到四個井
  區塊 4  對抗       換成 greedy 對手、完整地圖

驗收方式：每塊結束後在「該塊自己的環境」評估，
而不是只看最終任務 —— 否則看不出是哪一塊沒學會。
"""

import json
import os
import random
import statistics
import time

import torch

import env
import ppo
from rl_env import ArenaRLEnv


def make_scenario(n_chests, wells, seed, L=100.0, spread=(10, 90)):
    """程式產生 scenario，不寫檔。"""
    rng = random.Random(seed)
    chests = []
    for _ in range(n_chests):
        chests.append({
            "center": [rng.uniform(*spread), rng.uniform(*spread)],
            "radius": 1.2,
        })
    return {
        "config": dict(env.DEFAULT_CONFIG, L=L),
        "players": [[5.0, 5.0], [L - 5.0, L - 5.0]],
        "chests": chests,
        "obstacles": [{"center": [w[0], w[1]], "radius": w[2]} for w in wells],
    }


# 地圖大小是最重要的難度旋鈕。實測 greedy 在 6 顆寶箱、150 步下：
#   L=100 → 0.88   L=50 → 2.38   L=30 → 3.38   L=20 → 5.88
# 大地圖的獎勵密度太低（一局不到一次 +1），PPO 沒有東西可學。
# 所以區塊 1 從 L=20 開始，逐塊放大。

STAGES = [
    {
        "name": "1-小場地",
        "L": 20, "spread": (4, 16),
        "chests": 6, "wells": [],
        "opponent": "idle", "max_steps": 150, "iterations": 80,
    },
    {
        "name": "2-放大場地",
        "L": 40, "spread": (6, 34),
        "chests": 8, "wells": [],
        "opponent": "idle", "max_steps": 200, "iterations": 80,
    },
    {
        "name": "3-加入重力井",
        "L": 60, "spread": (8, 52),
        "chests": 10, "wells": [(30, 30, 8)],
        "opponent": "idle", "max_steps": 250, "iterations": 100,
    },
    {
        "name": "4-完整地圖",
        "L": 100, "spread": (10, 90),
        "chests": 20,
        "wells": [(30, 30, 10), (70, 70, 10), (30, 70, 10), (70, 30, 10), (50, 50, 12)],
        "opponent": "greedy", "max_steps": 400, "iterations": 120,
    },
]


def make_opponent(kind):
    return env.IdleAgent() if kind == "idle" else env.GreedyAgent()


def build_envs(stage, n_envs, seed_base):
    return [
        ArenaRLEnv(
            make_scenario(stage["chests"], stage["wells"], seed_base + i,
                          L=stage["L"], spread=stage["spread"]),
            opponent=make_opponent(stage["opponent"]),
            max_steps=stage["max_steps"],
        )
        for i in range(n_envs)
    ]


def evaluate(model, stage, n_episodes=12, seed_base=9000):
    """在這個區塊自己的環境上評估。回傳得分、掉井次數、報酬。"""
    scores, dangers, returns = [], [], []
    action_table = None

    for k in range(n_episodes):
        e = ArenaRLEnv(
            make_scenario(stage["chests"], stage["wells"], seed_base + k,
                          L=stage["L"], spread=stage["spread"]),
            opponent=make_opponent(stage["opponent"]),
            max_steps=stage["max_steps"],
            shaping=False,        # 評估看真實獎勵，不含 shaping
        )
        action_table = e.actions
        obs = e.reset(swap_spawns=(k % 2 == 1))
        total, done, info = 0.0, False, {}
        while not done:
            with torch.no_grad():
                logits, _ = model(torch.tensor(obs, dtype=torch.float32).unsqueeze(0))
                a = int(logits.argmax(dim=-1).item())
            obs, r, done, info = e.step(a)
            total += r
        scores.append(info["scores"][0])
        dangers.append(info["danger_entries"])
        returns.append(total)

    return {
        "score": statistics.mean(scores),
        "score_sd": statistics.pstdev(scores),
        "danger": statistics.mean(dangers),
        "return": statistics.mean(returns),
        "n_chests": stage["chests"],
    }


def baseline(stage, policy, n_episodes=12, seed_base=9000):
    """同一批評估環境上的 random / greedy 對照。"""
    scores, dangers = [], []
    for k in range(n_episodes):
        rng = random.Random(k)
        e = ArenaRLEnv(
            make_scenario(stage["chests"], stage["wells"], seed_base + k,
                          L=stage["L"], spread=stage["spread"]),
            opponent=make_opponent(stage["opponent"]),
            max_steps=stage["max_steps"],
            shaping=False,
        )
        e.reset(swap_spawns=(k % 2 == 1))
        g = env.GreedyAgent()
        g.initialize(0)
        done, info = False, {}
        while not done:
            if policy == "random":
                a = rng.randrange(e.n_actions)
            else:
                ax, ay = g.act(e.core.state_for(0))
                a = min(range(e.n_actions),
                        key=lambda i: (e.actions[i][0] - ax) ** 2
                                      + (e.actions[i][1] - ay) ** 2)
            _, _, done, info = e.step(a)
        scores.append(info["scores"][0])
        dangers.append(info["danger_entries"])
    return {"score": statistics.mean(scores),
            "score_sd": statistics.pstdev(scores),
            "danger": statistics.mean(dangers)}


def run_curriculum(seed=0, n_envs=4, out_prefix="curriculum"):
    random.seed(seed)
    torch.manual_seed(seed)

    probe = build_envs(STAGES[0], 1, 0)[0]
    model = ppo.ActorCritic(probe.obs_dim, probe.n_actions)
    opt = torch.optim.Adam([
        {"params": list(model.shared.parameters()) + list(model.actor.parameters()),
         "lr": ppo.LR_ACTOR},
        {"params": model.critic.parameters(), "lr": ppo.LR_CRITIC},
    ])

    report = []
    t0 = time.time()

    for si, stage in enumerate(STAGES):
        envs = build_envs(stage, n_envs, seed_base=seed * 1000 + si * 100)
        states = [e.reset(swap_spawns=(i % 2 == 1)) for i, e in enumerate(envs)]

        print(f"\n=== 區塊 {stage['name']}  L={stage['L']}  "
              f"寶箱 {stage['chests']}  井 {len(stage['wells'])}  "
              f"對手 {stage['opponent']}  單局上限 {stage['max_steps']} 步 ===")

        for it in range(stage["iterations"]):
            buf, last_value, ep_returns, ep_scores = ppo.collect_rollout(
                model, envs, states)
            stats = ppo.ppo_update(model, opt, buf, last_value)

            if (it + 1) % 20 == 0 or it == stage["iterations"] - 1:
                mr = statistics.mean(ep_returns) if ep_returns else float("nan")
                ms = statistics.mean(s[0] for s in ep_scores) if ep_scores else float("nan")
                print(f"  [{it+1:3d}] 報酬 {mr:6.2f}  得分 {ms:5.2f}  "
                      f"局數 {len(ep_returns):3d}  熵 {stats['ent']:.3f}  "
                      f"{time.time()-t0:5.0f}s")

        ev = evaluate(model, stage)
        rb = baseline(stage, "random")
        gb = baseline(stage, "greedy")
        print(f"  → 評估  PPO {ev['score']:5.2f}±{ev['score_sd']:.2f} "
              f"(掉井 {ev['danger']:.1f})   "
              f"random {rb['score']:5.2f}±{rb['score_sd']:.2f} "
              f"(掉井 {rb['danger']:.1f})   "
              f"greedy {gb['score']:5.2f}±{gb['score_sd']:.2f} "
              f"(掉井 {gb['danger']:.1f})   / 共 {stage['chests']} 顆")

        report.append({"stage": stage["name"], "ppo": ev,
                       "random": rb, "greedy": gb})
        torch.save(model.state_dict(), f"{out_prefix}_s{seed}_stage{si+1}.pt")

    json.dump(report, open(f"{out_prefix}_s{seed}_report.json", "w"),
              ensure_ascii=False, indent=2)
    return model, report


if __name__ == "__main__":
    run_curriculum(seed=0)
