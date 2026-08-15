# Rummikub AI Agent 設計實作：從規則驗證到策略優化

[🇹🇼 中文](#中文) ｜ [🇬🇧 English](#english)

---

<a name="中文"></a>
# 🇹🇼 中文

## 專案簡介

本專案是 2026 資訊之芽 C++ 班大作業二的延伸實作，主題為拉密（Rummikub）遊戲系統與 AI Agent 設計。專案目標不只是讓程式能夠正確判斷牌組是否合法，也需要設計一支能夠自動分析局勢、出牌、利用桌面牌重組，並嘗試打敗 baseline agent 的 AI。

在這個專案中，我完成了遊戲規則驗證、Human Agent 檔案讀取，以及 AI Agent 的策略設計。AI Agent 的核心目標是：在一般對戰中保留彈性與爆發力，在殘局中降低手牌失分，並且盡量避免提交非法盤面。

這個專案讓我第一次完整體驗到「遊戲規則 → 程式建模 → AI 策略 → 壓力測試 → 策略修正」的開發流程。

---

## 使用技術

```text
Language: C++
Build System: CMake
Environment: Docker
Version Control: Git / GitHub
Core Concepts: OOP, pointer identity, greedy strategy, board reconstruction, game AI
```

---

## 專案結構

```
sprout2026-hw2/
├── CMakeLists.txt
├── Dockerfile
├── README.md
├── server.py
├── src/
│   ├── main.cpp
│   ├── tile.h / tile.cpp
│   ├── validator.h / validator.cpp     ← 本次修改（遊戲引擎）
│   ├── board.h / board.cpp
│   ├── player.h / player.cpp
│   ├── game_manager.h / game_manager.cpp
│   ├── ai_agent_0.h / ai_agent_0.cpp   ← 本次修改（AI agent）
│   ├── ai_agent_baseline0.h
│   ├── ai_agent_baseline1.h
│   └── human_agent.h / human_agent.cpp ← 本次修改（human agent）
├── prebuilt/
├── grader/
└── visualizer/
    ├── index.html
    ├── app.js
    └── styles.css
```

---

## 核心功能實作

### 遊戲引擎（`validator.cpp`）

實作了 `isValidRun` 與 `isValidGroup` 兩個核心驗證函式：

- **Run（順組）**：3 張以上、同色、數字連續，Joker 可代入缺號位置
- **Group（群組）**：3～4 張、同數字、顏色互異，Joker 可補缺色

出牌的唯一入口是 `Board::applyProposedSets()`，它會依序檢查七件事——牌源合法性、有無重複、原桌面牌是否全部保留、未破冰前不可重組、每個 set 是否合法、破冰分數是否 ≥30——只要有一項不通過，整批提案會被打回，桌面與手牌完全不會被改動。

### 檔案讀取（`human_agent.cpp`）

完成 `waitForActionFile()` 輪詢等待 `action.json`，以及 `applyActionFile()` 解析並提交人類玩家的操作。

---

## AI Agent 策略設計

`AIAgent_0` 的決策邏輯依優先序分成五層：

| 階段 | 核心機制 | 對應程式碼 |
|---|---|---|
| 破冰 | 兩層貪心：先只用純手牌找候選，湊不到 30 分才把 Joker 納入；候選依「Joker 用量少 → 分數高」排序 | `tryInitialMeld()` |
| 反僵局 | 牌堆剩餘 ≤20 張時放棄囤牌、全力進攻 | `playTurn()` |
| 囤牌觀察 | 手牌出現「重疊牌型」（同數字 ≥2 色、且鄰近數字也在手上）時，冷卻 4 回合保留戰術彈性 | `internalCheckOverlap()` |
| 反擊 | 監控對手連續抽牌次數，達門檻（實測後由 3 調整為 8）視為對手卡關，觸發全域重組 | `enemy_draw_count` |
| 殘局 | 牌堆抽完後解除所有限制，迴圈呼叫重組直到打不出牌為止 | `playTurn()` 殘局分支 |

### 核心創新：「大風吹」全域重組

這是本次實作中投入最多心力的部分（對應 `tryExtendBoard()`）：

1. 把桌面與手牌全部攤開，依顏色重新分堆，掃描拼出最長合法 Run，缺口用 Joker 動態填補
2. 桌面上原本就合法的 Group 整組跳過、不拆進顏色堆，避免拆了拼不回去
3. Run 長度 ≥6 張時，模擬每個切點能讓手牌接上最多張牌，取最佳切點切成兩段
4. **安全回滾鎖**：重組完成後檢查原桌面每張牌是否都完整保留在合法組合中、且總張數確實增加，只要有一項不符合，整批 rollback，絕不提交不合法盤面

> 開發過程中原本規劃用遞迴窮舉（`solveRummikub`）來找真正的最佳解，但考量到需要疊加安全回滾鎖來確保正確性，改用「迭代＋貪心規則」整套寫進 `tryExtendBoard`。`solveRummikub` 保留了函式介面，但目前未實際使用。

---

## 開發過程中的挑戰

1. **Visualizer 讀檔格式不符**：把整場對局紀錄（陣列）直接覆蓋 `state.json`（預期是單一物件），導致 `state.players` 是 `undefined`、畫面報錯。拆成 `state.json`（當前局面）＋`state_history.json`（整場歷史）兩個檔案，對應 Replay Mode 的用途後解決。
2. **反擊門檻太敏感**：`enemy_draw_count` 門檻原本設 ≥3，實測發現對手只要連續抽牌 3 次就幾乎每輪都觸發大風吹，反而干擾正常節奏，調高到 ≥8 後才穩定，只在對手真正卡關時出手。
3. **本地一度測不到真正的 baseline**：`CMakeLists.txt` 裡負責串接 `prebuilt/ai_agent_baseline0.cpp.o` / `ai_agent_baseline1.cpp.o` 的區塊被誤註解掉，導致連結失敗；為了先能編譯，`main.cpp` 一度暫時把 `b0`/`b1` token 都導向自己的 `AIAgent_0` 頂替。後來把 `CMakeLists.txt` 那段還原、並在 Docker（`--platform=linux/amd64`，因為 prebuilt 物件檔是 x86_64 格式，跟 Apple Silicon 原生環境不相容）環境下重新編譯，才真正連結上助教提供的 baseline，跑出下方的真實對戰數據。

---

## 測試結果

修正建置設定並在 Docker（linux/amd64）環境重新編譯、確認連結到真正的 baseline 物件檔後，進行了兩輪測試：

**大樣本驗證（各 500 場）**

| 對手 | 勝場 | 勝率 |
|---|---|---|
| baseline0 | 500 / 500 | 100% |
| baseline1 | 320 / 500 | 64% |
| **合計** | **820 / 1000** | **82%** |

對 baseline1 的勝率沒有到 100%，符合預期——baseline1 本身會出手牌中的完整牌組，比完全不出牌的 baseline0 更有威脅性；但千場等級的大樣本下仍維持 64% 的整體優勢，排除了小樣本測試只是運氣好的疑慮。

**逐場明細（各 5 場，取自大樣本測試前的驗證階段）**

**10 / 10 全勝**（vs baseline0 五戰全勝、vs baseline1 五戰全勝）

| 對局 | AI_0_1 分數 | 對手分數 | 結果 |
|---|---|---|---|
| vs baseline0 #1 | 67 | 394 | 勝 (+327) |
| vs baseline0 #2 | 78 | 408 | 勝 (+330) |
| vs baseline0 #3 | 71 | 465 | 勝 (+394) |
| vs baseline0 #4 | 139 | 450 | 勝 (+311) |
| vs baseline0 #5 | 137 | 441 | 勝 (+304) |
| vs baseline1 #1 | 89 | 211 | 勝 (+122) |
| vs baseline1 #2 | 119 | 154 | 勝 (+35) |
| vs baseline1 #3 | 139 | 216 | 勝 (+77) |
| vs baseline1 #4 | 33 | 92 | 勝 (+59) |
| vs baseline1 #5 | 71 | 137 | 勝 (+66) |

分數為手牌點數總和，越低越好。對 baseline0 的平均勝場分差達 +333，對 baseline1 平均勝場分差 +72——分差較小的原因是 baseline1 本身會出手牌中的完整牌組，比完全不出牌的 baseline0 更有威脅性，但兩者最終都沒能撐過 `AIAgent_0` 的大風吹重組與殘局壓分能力。

> **驗證過程說明**：先前版本的測試數據是「本地鏡像測試」（因建置設定問題，`b0`/`b1` 一度都指向自己的 `AIAgent_0`），已在修正 `CMakeLists.txt` 並於 Docker 環境重新編譯後，替換成上方這組對戰真實 baseline 的數據。

---

## 心得

最一開始的版本邏輯很單純：只要能破冰就無腦出牌，藉此把手牌壓到最低。但實際測試後發現，這樣的策略太依賴「找到當下最大化的組合」，牌堆快見底時還在猶豫要不要湊更好的組合，反而常常演變成雙方一直抽牌的僵局。

為了打破這個情況，我把判斷依據從「手牌怎麼打分數最高」改成「牌堆還剩多少張」：當牌堆數量低於門檻時，就不再追求最大化，先求有牌出。這個調整讓 agent 終於能穩定推進牌局，而不是卡在互相抽牌。

但光靠這個還是太看運氣。於是我開始把桌面上已經打出去的牌也一起納入考慮，設計出「一條龍」式的最長 Run 掃描：不只看手牌能不能湊出組合，也會嘗試把手牌接到桌面現有 Run 的頭尾，甚至把整個桌面拆開重拼成更長的連續數字，這也是後來「大風吹」全域重組的雛形。

在對戰 baseline1 的過程中，我發現一味進攻其實不夠聰明——對手的行為本身就是資訊。我加入了「觀察對手連續抽牌次數」的機制：如果對手連續好幾輪都在抽牌，代表牌堆裡已經沒有他能用的組合，這時候正是發動大風吹重組、搶進度的最佳時機（門檻經過實測從 3 調整到 8，因為 3 太敏感、幾乎每回合都觸發）。

實作大風吹的過程中，我原本規劃寫一個遞迴窮舉版本（`solveRummikub`），理論上能找到真正的最佳解。但後來考量到需要加上「安全回滾鎖」——重組完一定要驗證原本桌面上的牌是否還完整保留、張數是否真的增加，否則整批打回原狀——這種需要層層驗證的邏輯，用迭代加貪心規則反而更好掌控每一步的正確性，所以最後把整套邏輯改寫進 `tryExtendBoard`，`solveRummikub` 保留了介面但沒有真的實作。

> **補充說明**：這份心得原本還包含兩個構想——「依手牌 Joker 張數（0／1／2 張）切換三種判斷分支」，以及「刻意出錯留數字空格、藉此引誘對手洩漏手牌」的戰術。誠實記錄一下：這兩個機制在最終版的 `ai_agent_0.cpp` 裡**沒有**實際寫成程式碼。目前的 Joker 處理是在 `tryInitialMeld()` 用「候選組合依 Joker 用量少、分數高排序」的統一貪心邏輯，而不是依手牌 Joker 數切換的三分支；囤牌偵測則是 `internalCheckOverlap()`，判斷的是手牌本身的重疊牌型（同數字 ≥2 色、且鄰近數字也在手上），跟主動誘敵完全無關。留下這段是想保留當時的思考方向，即使最後沒有真的做出來。

---

## 結論

這次做出的是一支混合戰術型 `AIAgent_0`：

- **破冰階段**：兩層貪心排序，優先省下 Joker
- **中局**：依牌堆張數與手牌重疊牌型，在囤牌與進攻間切換
- **反擊時機**：抓到對手連續抽牌，立刻用大風吹反擊
- **殘局**：解除所有限制，迴圈出牌直到打不出為止

學到最深的一課是「先求穩定合法，再談最佳化」——安全回滾鎖多次擋下了不合法的重組提案。未來想把 `solveRummikub` 補成真正的遞迴／動態規劃搜尋，取代現在的貪心啟發式。

---

## AI 協助揭露（Discussion & AI Policy）

依課程規範揭露本次開發過程中使用 AI（Claude）協助的部分：

**Claude 協助的範圍：**
- Debug：協助定位 visualizer 讀取 `state.json` 時的格式不符問題（陣列 vs 單一物件），以及 `CMakeLists.txt` 中 baseline 連結區塊被誤註解、跨架構（arm64/x86_64）編譯導致的連結失敗問題
- 內容校對：將這份 README／心得的文字敘述與實際送交的 `ai_agent_0.cpp` 逐項對照，指出敘述與程式碼不一致之處
- 文件整理：協助整理與潤飾簡報、口說逐字稿與本 README 的文字排版

**未經 Claude 協助、完全由本人設計與撰寫的部分：**
- `validator.cpp` 的 `isValidRun` / `isValidGroup` 驗證邏輯
- `ai_agent_0.cpp` 的所有策略與演算法設計，包括兩層貪心破冰、大風吹全域重組、長龍切斷優化、安全回滾鎖、重疊牌型偵測、對手抽牌監控反擊機制
- `human_agent.cpp` 的檔案讀取實作

本人對繳交的每一行程式碼負責，並理解其運作原理，如講師需要進行 Code Review，可隨時安排。

---

## 編譯與執行

```bash
# 編譯
rm -rf build && cmake -S . -B build && cmake --build build -j

# 執行（預設 baseline0 vs AI_0）
cd build && ./bin/rummikub && cd ..

# 視覺化（另開一個終端機）
python3 server.py
# 瀏覽器開啟 http://127.0.0.1:8080/visualizer/
```

---

## 致謝

感謝資訊之芽講師群提供的 `sprout2026-hw2` 起始框架與課程教學。
感謝 Claude 協助 debug 與內容校對（詳見上方 AI 協助揭露）。

<br><br>

---
---

<a name="english"></a>
# 🇬🇧 English

## Project Overview

This project is an extended implementation of Assignment 2 for the 2026 Sprout (資訊之芽) C++ track, centered on building the game engine and AI agent for **Rummikub**. The goal goes beyond simply validating whether a tile set is legal — it also requires designing an agent that can analyze the board, play tiles, reorganize existing sets on the table, and attempt to beat the course's baseline agents.

In this project I implemented the rule-validation engine, the human-agent file-reading interface, and the strategy design for the AI agent. The AI agent's core objectives are: stay flexible and aggressive during normal play, minimize leftover hand score in the endgame, and avoid submitting illegal board states as much as possible.

This project was my first full hands-on experience with the development cycle of **rules → modeling → AI strategy → stress testing → strategy revision**.

---

## Tech Stack

```text
Language: C++
Build System: CMake
Environment: Docker
Version Control: Git / GitHub
Core Concepts: OOP, pointer identity, greedy strategy, board reconstruction, game AI
```

---

## Project Structure

```
sprout2026-hw2/
├── CMakeLists.txt
├── Dockerfile
├── README.md
├── server.py
├── src/
│   ├── main.cpp
│   ├── tile.h / tile.cpp
│   ├── validator.h / validator.cpp     ← Modified (game engine)
│   ├── board.h / board.cpp
│   ├── player.h / player.cpp
│   ├── game_manager.h / game_manager.cpp
│   ├── ai_agent_0.h / ai_agent_0.cpp   ← Modified (AI agent)
│   ├── ai_agent_baseline0.h
│   ├── ai_agent_baseline1.h
│   └── human_agent.h / human_agent.cpp ← Modified (human agent)
├── prebuilt/
├── grader/
└── visualizer/
    ├── index.html
    ├── app.js
    └── styles.css
```

---

## Core Feature Implementation

### Game Engine (`validator.cpp`)

Implemented the two core validation functions, `isValidRun` and `isValidGroup`:

- **Run**: 3+ tiles, same color, consecutive numbers; a Joker can substitute for the missing number
- **Group**: 3–4 tiles, same number, all different colors; a Joker can fill a missing color

The single entry point for playing tiles is `Board::applyProposedSets()`. It checks seven conditions in sequence — legality of tile sources, no duplicates, all original board tiles preserved, no rearranging before the initial meld, every set must be a legal Run or Group, and the initial-meld score must be ≥30. If any check fails, the entire proposal is rejected and neither the board nor the hand is modified.

### File I/O (`human_agent.cpp`)

Implemented `waitForActionFile()`, which polls until `action.json` can be opened, and `applyActionFile()`, which parses and submits the human player's action.

---

## AI Agent Strategy Design

`AIAgent_0`'s decision logic is organized into five priority tiers:

| Stage | Core Mechanism | Corresponding Code |
|---|---|---|
| Initial meld | Two-tier greedy search: first tries candidates using only plain tiles; only brings Jokers into consideration if 30 points can't be reached otherwise. Candidates are sorted by "fewer Jokers used → higher score" | `tryInitialMeld()` |
| Anti-stalemate | Abandons hoarding and plays aggressively once the draw pile has ≤20 tiles left | `playTurn()` |
| Hoarding detection | Enters a 4-turn cooldown to preserve tactical flexibility when the hand shows an "overlapping pattern" (same number in ≥2 colors, with an adjacent number also in hand) | `internalCheckOverlap()` |
| Counter-attack | Monitors the opponent's consecutive draw count; once it crosses a threshold (tuned from 3 to 8 through testing), the opponent is considered stuck and a global reorganization is triggered | `enemy_draw_count` |
| Endgame | Once the draw pile is empty, all restrictions are lifted and reorganization is called in a loop until no more tiles can be played | `playTurn()` endgame branch |

### Core Innovation: "Windstorm" Global Reorganization

This is the part of the implementation I invested the most effort in (corresponds to `tryExtendBoard()`):

1. Lay out the entire board and hand, re-bucket tiles by color, and scan for the longest legal Run per color, dynamically filling gaps with Jokers
2. Any Group already legal on the board is skipped entirely — it's not broken apart into the color buckets, to avoid ending up unable to reassemble it
3. When a Run reaches ≥6 tiles, simulate every possible cut point and choose the one that lets the most hand tiles attach to either resulting segment
4. **Safety rollback lock**: after reorganizing, verify that every original board tile is still present in a legal set and that the total tile count has genuinely increased. If either check fails, the entire batch is rolled back — an illegal board state is never submitted

> I originally planned to implement `solveRummikub()` as a full recursive brute-force search to find the true optimal solution. However, since a safety rollback lock also needed to be layered on top to guarantee correctness, I switched to an "iterative + greedy rule" approach implemented entirely inside `tryExtendBoard`. `solveRummikub()` keeps its function signature but is currently unused.

---

## Challenges During Development

1. **Visualizer file-format mismatch**: The full match history (an array) was directly overwriting `state.json` (which expects a single object), causing `state.players` to be `undefined` and the UI to error out. Fixed by splitting it into `state.json` (current state) and `state_history.json` (full history), matching the intended use for Replay Mode.
2. **Counter-attack threshold too sensitive**: `enemy_draw_count`'s threshold was originally ≥3; testing showed that just 3 consecutive opponent draws triggered a windstorm almost every round, disrupting normal pacing. Raising it to ≥8 stabilized the behavior so it only fires when the opponent is genuinely stuck.
3. **Couldn't link the real baselines locally, for a while**: The block in `CMakeLists.txt` responsible for linking `prebuilt/ai_agent_baseline0.cpp.o` / `ai_agent_baseline1.cpp.o` had been accidentally commented out, breaking the link step. As a temporary workaround to keep things compiling, `main.cpp` briefly redirected both `b0`/`b1` tokens to `AIAgent_0` itself. After restoring the `CMakeLists.txt` block and rebuilding inside Docker (`--platform=linux/amd64`, since the prebuilt object files are x86_64 and incompatible with native Apple Silicon), the agent was genuinely linked against the instructor-provided baselines, producing the real match data shown below.

---

## Test Results

After fixing the build configuration and rebuilding inside Docker (linux/amd64) to confirm a real link against the baseline object files, two rounds of testing were run:

**Large-sample validation (500 games each)**

| Opponent | Wins | Win Rate |
|---|---|---|
| baseline0 | 500 / 500 | 100% |
| baseline1 | 320 / 500 | 64% |
| **Total** | **820 / 1000** | **82%** |

The win rate against baseline1 falling short of 100% is expected — baseline1 plays complete sets from its hand and is more threatening than baseline0, which never plays at all. Even so, a 64% edge held up at the thousand-game scale, ruling out the possibility that a smaller sample was just a lucky streak.

**Game-by-game detail (5 games each, from the validation phase before the large-sample run)**

**10 / 10 wins** (5/5 vs baseline0, 5/5 vs baseline1)

| Match | AI_0_1 Score | Opponent Score | Result |
|---|---|---|---|
| vs baseline0 #1 | 67 | 394 | Win (+327) |
| vs baseline0 #2 | 78 | 408 | Win (+330) |
| vs baseline0 #3 | 71 | 465 | Win (+394) |
| vs baseline0 #4 | 139 | 450 | Win (+311) |
| vs baseline0 #5 | 137 | 441 | Win (+304) |
| vs baseline1 #1 | 89 | 211 | Win (+122) |
| vs baseline1 #2 | 119 | 154 | Win (+35) |
| vs baseline1 #3 | 139 | 216 | Win (+77) |
| vs baseline1 #4 | 33 | 92 | Win (+59) |
| vs baseline1 #5 | 71 | 137 | Win (+66) |

Scores are the sum of remaining hand tile values — lower is better. The average winning margin against baseline0 was +333, versus +72 against baseline1. The smaller margin against baseline1 makes sense: it actually plays complete sets from its hand, making it a bigger threat than baseline0, which never plays at all — but neither could withstand `AIAgent_0`'s windstorm reorganization and endgame scoring pressure.

> **Note on the verification process**: an earlier version of this data was a "local mirror test" (due to the build-configuration issue, `b0`/`b1` briefly pointed to `AIAgent_0` itself). After fixing `CMakeLists.txt` and rebuilding inside Docker, that data was replaced with the real baseline match results shown above.

---

## Reflection

The very first version of the logic was simple: play tiles the moment an initial meld is possible, to minimize the hand as quickly as possible. In practice, though, this strategy relied too heavily on "finding the best possible combination right now" — near the end of the draw pile it would still hesitate over whether to hold out for a better combination, which often turned into both sides just drawing tiles back and forth in a stalemate.

To break that pattern, I changed the decision criterion from "which play scores the most" to "how many tiles are left in the draw pile": once the pile drops below a threshold, the agent stops optimizing and just plays whatever it can. That change let the agent finally make steady progress instead of getting stuck in a mutual-drawing loop.

But that alone still left too much to luck. So I started factoring in the tiles already played on the board, designing a "longest-run" scan: not just checking whether the hand alone can form a set, but also trying to attach hand tiles to either end of an existing board Run, or even tearing the whole board apart and reassembling it into longer consecutive runs — this became the prototype for what later became the "windstorm" global reorganization.

While playing against baseline1, I realized that attacking non-stop isn't actually smart — the opponent's behavior is itself information. I added a mechanism to watch the opponent's consecutive draw count: if they keep drawing turn after turn, it means the draw pile no longer has anything useful for them, which is exactly the moment to trigger a windstorm reorganization and seize the advantage (the threshold was tuned from 3 to 8 through testing, since 3 was too sensitive and fired almost every round).

While implementing the windstorm, I originally planned to write a full recursive brute-force version (`solveRummikub`) that could, in theory, find the true optimal solution. But once I factored in the need for a "safety rollback lock" — verifying after every reorganization that the original board tiles are all still intact and that the tile count genuinely increased, otherwise rolling the whole thing back — this kind of layered verification logic turned out to be easier to get right with iteration plus greedy rules. So I ended up rewriting the entire thing into `tryExtendBoard`, and `solveRummikub` kept its interface but was never actually implemented.

> **Addendum**: This reflection originally also included two other ideas — "switching between three decision branches based on how many Jokers are in hand (0/1/2)," and a tactic of "deliberately leaving a numeric gap to bait the opponent into revealing their hand." Being honest about it here: neither of these ended up actually implemented in the final `ai_agent_0.cpp`. The current Joker handling in `tryInitialMeld()` uses a single unified greedy rule — sort candidates by "fewer Jokers used, higher score" — rather than three branches keyed on Joker count; and the hoarding detector is `internalCheckOverlap()`, which judges overlapping patterns within the hand itself (same number in ≥2 colors, with an adjacent number also present), with no connection at all to actively baiting an opponent. I'm keeping this note to preserve that line of thinking at the time, even though it never actually got built.

---

## Conclusion

What came out of this project is a hybrid, tactics-driven `AIAgent_0`:

- **Initial meld**: two-tier greedy sorting, prioritizing saving Jokers
- **Midgame**: switches between hoarding and attacking based on draw-pile size and hand-overlap patterns
- **Counter-attack timing**: the moment the opponent's consecutive draws are detected, immediately counters with a windstorm
- **Endgame**: lifts all restrictions and loops on playing tiles until nothing more can be played

The deepest lesson learned was "establish stability and legality first, optimize later" — the safety rollback lock caught a number of illegal reorganization proposals along the way. Looking ahead, I'd like to fill in `solveRummikub` as a genuine recursive/dynamic-programming search, replacing the current greedy heuristic.

---

## AI Assistance Disclosure (Discussion & AI Policy)

As required by the course policy, here is a disclosure of how AI (Claude) was used during this project:

**What Claude helped with:**
- Debugging: helped pinpoint the visualizer's `state.json` format mismatch (array vs. single object), as well as the linking failure caused by a commented-out baseline-linking block in `CMakeLists.txt` and a cross-architecture (arm64/x86_64) compilation issue
- Content review: cross-checked the wording in this README/reflection against the actual submitted `ai_agent_0.cpp`, line by line, flagging places where the narrative didn't match the code
- Document formatting: helped organize and polish the wording and layout of the presentation, the spoken script, and this README

**What was designed and written entirely by me, without Claude's help:**
- The `isValidRun` / `isValidGroup` validation logic in `validator.cpp`
- Every strategy and algorithm in `ai_agent_0.cpp`, including the two-tier greedy initial meld, the windstorm global reorganization, the long-run cut-point optimization, the safety rollback lock, the overlap-pattern hoarding detector, and the opponent-draw-count counter-attack mechanism
- The file-reading implementation in `human_agent.cpp`

I take full responsibility for every line of code submitted and understand how it works. I'm available for a code review with the instructor at any time if needed.

---

## Build & Run

```bash
# Build
rm -rf build && cmake -S . -B build && cmake --build build -j

# Run (default: baseline0 vs AI_0)
cd build && ./bin/rummikub && cd ..

# Visualizer (open a separate terminal)
python3 server.py
# Then open http://127.0.0.1:8080/visualizer/ in your browser
```

---

## Acknowledgments

Thanks to the Sprout (資訊之芽) instructor team for providing the `sprout2026-hw2` starter framework and course instruction.
Thanks to Claude for assistance with debugging and content review (see the AI Assistance Disclosure above for details).
