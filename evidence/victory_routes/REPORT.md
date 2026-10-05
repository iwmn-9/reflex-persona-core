# 人格に沿った有限勝利ルート

32 headless four-player resource games; 654 choices replayed; shared core/runtime unchanged.

|profile|route layer|games|wins|switches|route decision counts|p50 ms|
|---|---|---:|---:|---:|---|---:|
|growth|False|4|4|0|{}|3.87|
|growth|True|4|4|1|{'power': 69, 'science': 2}|5.91|
|steady|False|4|4|0|{}|5.25|
|steady|True|4|4|5|{'culture': 41, 'power': 3}|7.35|
|care|False|4|4|0|{}|4.33|
|care|True|4|4|4|{'power': 44, 'culture': 28}|6.91|
|ego|False|4|1|0|{}|3.83|
|ego|True|4|2|0|{'power': 117}|6.12|

## 得たものと費用

ゲームが有限の勝利候補、公開資源による進めやすさ、本人への効果を供給し、共通Policyが選ぶ。特性と勝利名をコアに直結しない。科学は自己決定、文化は伝統、勢力は権力という結びつきはこの自作ゲームの仮定。
ルート記憶は単発行動の記憶と分離。わずかなスコア差では変更せず、利用不能・大きな利得差なら変更する。性格自体は更新しない。
研究/建築で消費済みの蓄積を完成済み能力として保持する専用代理値を接続した。ルールだけで評価が自然発生する能力は得ていない。比較はルート層と専用評価の組み合わせであり、どちらの単独効果かは断定できない。

## 同じ盤面における人格・機会の選択

|scenario|profile|route|
|---|---|---|
|initial_world|growth|power|
|initial_world|steady|culture|
|initial_world|care|power|
|initial_world|ego|power|
|science_capacity|growth|science|
|science_capacity|steady|science|
|science_capacity|care|science|
|science_capacity|ego|power|
|culture_capacity|growth|culture|
|culture_capacity|steady|culture|
|culture_capacity|care|culture|
|culture_capacity|ego|power|
|military_capacity|growth|power|
|military_capacity|steady|power|
|military_capacity|care|power|
|military_capacity|ego|power|
|military_closed|growth|culture|
|military_closed|steady|culture|
|military_closed|care|culture|
|military_closed|ego|culture|

## 残る制限

- route opportunities/effects/progress are game-authored proxies, not universal automatic win valuation
- no opponents/future event simulation in opportunity estimate
- development probe; no validated Big Five-to-victory psychological mapping
- route preference and route-specific value changed together, so attribution is limited
- hard rejection/known failure and hysteresis handle route change; calibrated route advantage prediction pending

- timings include possible concurrent local experiment load; not isolated throughput measurements
