# 公開履歴によるハゲタカ・有限予算オークション

576 games / 7776 judgments replayed. Shared core/runtime unchanged. No LLM or GPU.

|game|opponents|reading|games|win credit|mean margin|changed actions|median p50 ms|
|---|---|---|---:|---:|---:|---:|---:|
|hagetaka|pattern|uniform|32|3.00|-13.12|0|11.10|
|hagetaka|pattern|learned|32|12.00|-1.31|153|23.30|
|hagetaka|pattern|gated|32|5.25|-9.03|36|18.96|
|hagetaka|pressure|uniform|32|14.00|-0.31|0|10.75|
|hagetaka|pressure|learned|32|24.00|6.56|153|22.31|
|hagetaka|pressure|gated|32|17.50|2.72|35|19.49|
|hagetaka|responsive|uniform|32|12.50|-2.78|0|11.41|
|hagetaka|responsive|learned|32|15.00|2.41|142|23.86|
|hagetaka|responsive|gated|32|13.00|0.12|36|20.52|
|auction|pattern|uniform|32|2.00|-8.29|0|11.75|
|auction|pattern|learned|32|7.00|-7.12|131|24.32|
|auction|pattern|gated|32|3.50|-6.93|27|23.33|
|auction|pressure|uniform|32|0.50|-8.80|0|11.27|
|auction|pressure|learned|32|4.00|-6.46|134|23.57|
|auction|pressure|gated|32|3.00|-7.40|32|22.55|
|auction|responsive|uniform|32|6.50|-7.34|0|10.92|
|auction|responsive|learned|32|9.00|-6.04|128|23.23|
|auction|responsive|gated|32|10.50|-5.39|32|22.58|

uniform: no reading. learned: always score with observed predictions. gated: only change for material persona benefit; preserve currently favorable one-round forecast.

## 得たものと費用

ゲーム名に依存しない人格Policyへ、公開情報だけの入札分布を接続。相手は固定傾向、得点で不利になると変更、過去の公開入札に応答する3条件。選択前の同一盤面で全員が決め、公開後だけ観測する。
読みを使う場合は通常評価に加え、同じ候補を再評価する。各モデルは全合法候補×32標本（最大480解決）、両モデル最大960。1ゲーム先の勝率、他者の心理の真実、ゲーム横断の自動評価は得ていない。
競り資金の代理価格は max(0.25, 残商品価値合計 / 人数 / 残資金)、上限2。終局資金価値を下限に、将来の商品獲得に使える資金を評価する。価値の等分を仮定した未校正の機会費用で、次の山順や実相手方策を入力しない。
性格を変えず、行動の予測を更新する。学習・忘却は既存の有限仮説と減衰窓。正しい学習を保証せず、誤読も結果に残す。

## 解釈上の制限

- development scenarios, not held-out intelligence evidence
- current-round MC and hand/budget commitment proxy, not full-game win EV
- finite public behavior templates; imperfect change detection; confidence is heuristic
- 8-outcome compression loses some distribution/correlation detail
- public perfect bid ledger; unresolved final Hagetaka pot remains unawarded
- authored common-value auction, not a named board game or private-value equilibrium
- same seeded decks; public feedback diverges, so paired results are not same-state causal effects
- auction future capital price assumes equal shares of remaining common item value; not calibrated
- reported campaign wall times can include other local jobs; not isolated throughput benchmarks

## 資金評価の修正前後（同じ開発条件）

|mode|games per version|before win credit|after win credit|before mean margin|after mean margin|
|---|---:|---:|---:|---:|---:|
|uniform|96|1.00|9.00|-10.34|-8.14|
|learned|96|4.00|20.00|-9.32|-6.54|
|gated|96|1.50|17.00|-9.57|-6.57|

再現: `run(game="auction", ..., capital_pricing=False)` は終局資金価値0.25と取得量による旧達成評価。既定Trueは将来資金価値と純獲得による達成評価。2つを一緒に変えた結果で、独立の寄与は未分離。修正前576局はローカル保存、修正後576局が上表の対象。
ハゲタカでは読み常用の優勝持分が大きく、利益ゲートは有益な変更も抑える。競りでは今回の開発条件で資金評価が改善したが、全方式とも平均点差は負。人間並みの競り判断に到達したとは扱わない。

Hagetaka rules: https://www.amigo-spiele.de/kartenspiele/hols-der-geier_1943_1210
Zero-carry convention: https://mobius-games.co.jp/25th/rule.html
