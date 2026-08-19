"""Actor-Critic (PPO) —— YTP Arena

蘇教授提的 Actor-Critic 就是這個。核心只有一條式子：

    A_t = G_t - V(s_t)          「實際拿到的，比 Critic 預期的好多少」

Critic 不決定做什麼，它只提供基準線。
沒有基準線的話，在一個每步都能拿高分的局面裡，
所有動作都會被鼓勵 —— 包括不好的那些。

PPO 在這之上再加一個截斷，限制單次更新的幅度：

    ratio = exp(logp_new - logp_old)
    loss  = -min(ratio * A, clip(ratio, 1-eps, 1+eps) * A)

用 PyTorch 而不是手刻，因為拉密那份 (rummikub-ai-agent/rl/) 已經
證明過一次手刻可行（梯度檢查吻合到 1e-11）。同一件事不用證明兩次，
這次的重點是「規則式 vs 學出來的」那個比較。
"""

import math
import random
import statistics
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

import env
from rl_env import ArenaRLEnv

# ---------------- 超參數 ----------------

GAMMA = 0.99          # 配合 FRAME_SKIP=10，有效視野約 100 個決策步 = 1000 ticks
GAE_LAMBDA = 0.95
LR_ACTOR = 3e-4
LR_CRITIC = 1e-3
CLIP_EPS = 0.2
EPOCHS = 4
BATCH_SIZE = 256
ROLLOUT_STEPS = 2048
ENTROPY_COEF = 0.01   # 鼓勵探索。太低會太早收斂到一個爛策略
MAX_GRAD_NORM = 0.5

HIDDEN = 128


class ActorCritic(nn.Module):
    """兩個 head 共用前面的特徵層。

            狀態 s
              |
          [共用 MLP]
           /       \\
      [Actor]   [Critic]
      pi(a|s)     V(s)
    """

    def __init__(self, obs_dim, n_actions, hidden=HIDDEN):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
        )
        self.actor = nn.Linear(hidden, n_actions)
        self.critic = nn.Linear(hidden, 1)

        # 正交初始化。actor 的最後一層 gain 設小，
        # 讓一開始的策略接近均勻分布 —— 探索比較充分。
        for m in self.shared:
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=math.sqrt(2))
                nn.init.zeros_(m.bias)
        nn.init.orthogonal_(self.actor.weight, gain=0.01)
        nn.init.zeros_(self.actor.bias)
        nn.init.orthogonal_(self.critic.weight, gain=1.0)
        nn.init.zeros_(self.critic.bias)

    def forward(self, x):
        h = self.shared(x)
        return self.actor(h), self.critic(h).squeeze(-1)

    def act(self, obs):
        """取樣一個動作，回傳 (動作, log機率, 狀態價值)。"""
        with torch.no_grad():
            logits, value = self.forward(obs)
            dist = torch.distributions.Categorical(logits=logits)
            action = dist.sample()
            return action, dist.log_prob(action), value

    def evaluate(self, obs, actions):
        logits, value = self.forward(obs)
        dist = torch.distributions.Categorical(logits=logits)
        return dist.log_prob(actions), dist.entropy(), value


def compute_gae(rewards, values, dones, last_value, gamma=GAMMA, lam=GAE_LAMBDA):
    """GAE：advantage 的低變異數估計。

    lam=1 退化成蒙地卡羅（無偏但高變異），
    lam=0 退化成 TD(0)（低變異但有偏）。0.95 是折衷。
    """
    advantages = [0.0] * len(rewards)
    gae = 0.0
    next_value = last_value
    for t in reversed(range(len(rewards))):
        mask = 0.0 if dones[t] else 1.0
        delta = rewards[t] + gamma * next_value * mask - values[t]
        gae = delta + gamma * lam * mask * gae
        advantages[t] = gae
        next_value = values[t]
    returns = [a + v for a, v in zip(advantages, values)]
    return advantages, returns


def collect_rollout(model, envs, states, n_steps=ROLLOUT_STEPS):
    """跑 n_steps 收集資料。envs 是多個環境輪流跑，增加資料多樣性。"""
    buf = {k: [] for k in
           ("obs", "actions", "logps", "values", "rewards", "dones")}
    episode_returns = []
    episode_scores = []

    ep_return = [0.0] * len(envs)

    for _ in range(n_steps // len(envs)):
        for i, e in enumerate(envs):
            obs_t = torch.tensor(states[i], dtype=torch.float32).unsqueeze(0)
            action, logp, value = model.act(obs_t)
            a = int(action.item())

            next_obs, reward, done, info = e.step(a)

            buf["obs"].append(states[i])
            buf["actions"].append(a)
            buf["logps"].append(float(logp.item()))
            buf["values"].append(float(value.item()))
            buf["rewards"].append(reward)
            buf["dones"].append(done)

            ep_return[i] += reward

            if done:
                episode_returns.append(ep_return[i])
                episode_scores.append(info["scores"])
                ep_return[i] = 0.0
                states[i] = e.reset(swap_spawns=random.random() < 0.5)
            else:
                states[i] = next_obs

    with torch.no_grad():
        last = torch.tensor(states[0], dtype=torch.float32).unsqueeze(0)
        _, last_value = model(last)
        last_value = float(last_value.item())

    return buf, last_value, episode_returns, episode_scores


def ppo_update(model, opt, buf, last_value):
    adv, ret = compute_gae(buf["rewards"], buf["values"], buf["dones"], last_value)

    obs = torch.tensor(buf["obs"], dtype=torch.float32)
    actions = torch.tensor(buf["actions"], dtype=torch.long)
    old_logps = torch.tensor(buf["logps"], dtype=torch.float32)
    advantages = torch.tensor(adv, dtype=torch.float32)
    returns = torch.tensor(ret, dtype=torch.float32)

    # advantage 正規化。不做的話，獎勵尺度一變學習率就要重調。
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    n = len(actions)
    idx = list(range(n))
    stats = {"pg": [], "v": [], "ent": [], "kl": []}

    for _ in range(EPOCHS):
        random.shuffle(idx)
        for start in range(0, n, BATCH_SIZE):
            b = idx[start:start + BATCH_SIZE]
            if len(b) < 2:
                continue
            bi = torch.tensor(b, dtype=torch.long)

            logps, entropy, values = model.evaluate(obs[bi], actions[bi])

            ratio = torch.exp(logps - old_logps[bi])
            a = advantages[bi]

            # 這兩行就是 PPO 的全部
            surr1 = ratio * a
            surr2 = torch.clamp(ratio, 1 - CLIP_EPS, 1 + CLIP_EPS) * a
            pg_loss = -torch.min(surr1, surr2).mean()

            v_loss = F.mse_loss(values, returns[bi])
            ent = entropy.mean()

            loss = pg_loss + 0.5 * v_loss - ENTROPY_COEF * ent

            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
            opt.step()

            stats["pg"].append(float(pg_loss.item()))
            stats["v"].append(float(v_loss.item()))
            stats["ent"].append(float(ent.item()))
            with torch.no_grad():
                stats["kl"].append(float((old_logps[bi] - logps).mean().item()))

    return {k: statistics.mean(v) if v else 0.0 for k, v in stats.items()}


class PolicyAgent:
    """把訓練好的模型包成跟 strategy.py 一樣的介面，方便對打。"""

    def __init__(self, model, action_table, deterministic=True):
        self.model = model
        self.actions = action_table
        self.deterministic = deterministic
        self.rl_env = None      # 要外部塞一個 ArenaRLEnv 來做狀態編碼

    def initialize(self, player_id):
        self.player_id = player_id

    def act_from_obs(self, obs):
        with torch.no_grad():
            logits, _ = self.model(torch.tensor(obs, dtype=torch.float32).unsqueeze(0))
            if self.deterministic:
                a = int(logits.argmax(dim=-1).item())
            else:
                a = int(torch.distributions.Categorical(logits=logits).sample().item())
        return self.actions[a]


def train(scenario_names=("gravity",), iterations=60, n_envs=4, seed=0, verbose=True):
    random.seed(seed)
    torch.manual_seed(seed)

    scenarios = [env.load_scenario(f"scenarios/{n}.csv") for n in scenario_names]
    envs = [ArenaRLEnv(scenarios[i % len(scenarios)], opponent=env.GreedyAgent())
            for i in range(n_envs)]
    states = [e.reset(swap_spawns=(i % 2 == 1)) for i, e in enumerate(envs)]

    model = ActorCritic(envs[0].obs_dim, envs[0].n_actions)
    opt = torch.optim.Adam([
        {"params": list(model.shared.parameters()) + list(model.actor.parameters()),
         "lr": LR_ACTOR},
        {"params": model.critic.parameters(), "lr": LR_CRITIC},
    ])

    history = []
    t0 = time.time()

    for it in range(iterations):
        buf, last_value, ep_returns, ep_scores = collect_rollout(model, envs, states)
        stats = ppo_update(model, opt, buf, last_value)

        mean_r = statistics.mean(ep_returns) if ep_returns else float("nan")
        mean_score = statistics.mean(s[0] for s in ep_scores) if ep_scores else float("nan")
        history.append({"iter": it, "return": mean_r, "score": mean_score, **stats})

        if verbose:
            print(f"[{it:3d}] 報酬 {mean_r:7.2f}  得分 {mean_score:5.2f}  "
                  f"局數 {len(ep_returns):2d}  pg {stats['pg']:+.4f}  "
                  f"V {stats['v']:7.3f}  熵 {stats['ent']:.3f}  "
                  f"KL {stats['kl']:+.4f}  {time.time()-t0:5.0f}s")

    return model, history


if __name__ == "__main__":
    model, hist = train(iterations=8, n_envs=4)
    torch.save(model.state_dict(), "ppo_smoke.pt")
    print("\n煙霧測試完成。要正式訓練請把 iterations 調大。")
