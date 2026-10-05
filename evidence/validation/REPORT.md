# オークション・複数資源/勝利条件・戦闘・ギャンブル共通検証

480 games; seeds [50, 51]; 12044 real rule transitions replayed.

baseline は共通接続済み既存評価。pressure は有界な欲求不足の蓄積を追加。戦闘のみ forecast で目標進展の欲求効果と3tick静止脅威代理値も比較する。既定の他実験を置換しない。

|種別|方式|人格|勝利持分/局数|平均目的点|最大停滞|
|---|---|---|---|---|---|
|auction|baseline|growth|2.5/6|18.625|6|
|auction|baseline|steady|0/6|4.750|12|
|auction|baseline|care|4/6|17.583|5|
|auction|baseline|ego|0/6|10.583|8|
|auction|pressure|growth|2.5/6|18.625|6|
|auction|pressure|steady|0/6|6.958|12|
|auction|pressure|care|4/6|17.583|5|
|auction|pressure|ego|0/6|10.583|8|
|hagetaka|baseline|growth|4/6|16.333|13|
|hagetaka|baseline|steady|4/6|20.500|11|
|hagetaka|baseline|care|3/6|15.667|11|
|hagetaka|baseline|ego|4/6|16.333|13|
|hagetaka|pressure|growth|4/6|16.333|13|
|hagetaka|pressure|steady|4/6|20.500|11|
|hagetaka|pressure|care|3/6|15.667|11|
|hagetaka|pressure|ego|4/6|16.333|13|
|resources|baseline|growth|5/6|0.972|27|
|resources|baseline|steady|6/6|1.000|8|
|resources|baseline|care|6/6|1.000|16|
|resources|baseline|ego|3/6|0.778|25|
|resources|pressure|growth|5/6|0.972|27|
|resources|pressure|steady|6/6|1.000|10|
|resources|pressure|care|6/6|1.000|11|
|resources|pressure|ego|4/6|0.889|23|
|combat|baseline|growth|17/24|0.944|32|
|combat|baseline|steady|0/24|0.377|39|
|combat|baseline|care|7/24|0.779|10|
|combat|baseline|ego|12/24|0.824|7|
|combat|pressure|growth|12/24|0.868|29|
|combat|pressure|steady|0/24|0.390|39|
|combat|pressure|care|9/24|0.779|11|
|combat|pressure|ego|12/24|0.824|6|
|combat|forecast|growth|7/24|0.722|38|
|combat|forecast|steady|0/24|0.422|39|
|combat|forecast|care|4/24|0.606|38|
|combat|forecast|ego|13/24|0.881|34|
|gambling|baseline|growth|3/6|16.000|3|
|gambling|baseline|steady|1/6|14.667|2|
|gambling|baseline|care|3/6|16.000|3|
|gambling|baseline|ego|3/6|16.000|3|
|gambling|pressure|growth|3/6|16.000|3|
|gambling|pressure|steady|1/6|14.667|2|
|gambling|pressure|care|3/6|16.000|3|
|gambling|pressure|ego|3/6|16.000|3|

同一条件対の勝利持分（改善/悪化/同じ）：
- auction/pressure: 0/0/24。目的点の改善/悪化/同じ: 3/0/21。
- hagetaka/pressure: 0/0/24。目的点の改善/悪化/同じ: 0/0/24。
- resources/pressure: 1/0/23。目的点の改善/悪化/同じ: 2/0/22。
- combat/pressure: 4/7/85。目的点の改善/悪化/同じ: 4/5/87。
- combat/forecast: 6/18/72。目的点の改善/悪化/同じ: 21/29/46。
- gambling/pressure: 0/0/24。目的点の改善/悪化/同じ: 0/0/24。

人格間の勝利率の最大差（勝利を均一化する補正なし）：
- auction/baseline: 66.7%、最小率0.0%、全人格の勝利持分6.5。
- auction/pressure: 66.7%、最小率0.0%、全人格の勝利持分6.5。
- hagetaka/baseline: 16.7%、最小率50.0%、全人格の勝利持分15。
- hagetaka/pressure: 16.7%、最小率50.0%、全人格の勝利持分15。
- resources/baseline: 50.0%、最小率50.0%、全人格の勝利持分20。
- resources/pressure: 33.3%、最小率66.7%、全人格の勝利持分21。
- combat/baseline: 70.8%、最小率0.0%、全人格の勝利持分36。
- combat/pressure: 50.0%、最小率0.0%、全人格の勝利持分33。
- combat/forecast: 54.2%、最小率0.0%、全人格の勝利持分24。
- gambling/baseline: 33.3%、最小率16.7%、全人格の勝利持分10。
- gambling/pressure: 33.3%、最小率16.7%、全人格の勝利持分10。

得たもの: 同じ人格コア・欲求不足・判断/ルート接続を4系統+ハゲタカへ適用し、現実の勝敗/破産/資源不足を比較できる再実行基盤。性格別ハンデ、非公開の当手、実乱数の先読みは追加していない。

削ったもの: この試験では未来込みの効果を即時結果として経験学習する接続を使わない。現在の効果ベクトルが未定義な箇所は明示的欠測とし、実際の目的進展だけ別契約で欲求へ返す。

増えた費用: 個体ごとの固定5欲求の有界記憶と進展ラベル、JSONコピー/検査。将来評価は戦闘側の代理値であり、一般的な賢さの獲得ではない。

合否: 改善・悪化は evaluation.json の対条件と人格別結果で判定する。強い人格を弱めて差を縮める補正は不採用。慎重型の停滞が残る場合は未解決とする。

Colab/GPU/Drive・LLM教師・モデル訓練は使用していない。全てローカルCPU。

実測による採否：
- auction/pressure: 総勝利持分6.5→6.5、最小率0.0%→0.0%。最弱人格の勝利率改善は確認できない。
- hagetaka/pressure: 総勝利持分15→15、最小率50.0%→50.0%。最弱人格の勝利率改善は確認できない。
- resources/pressure: 総勝利持分20→21、最小率50.0%→66.7%。小標本の改善の兆候（一般改善の証明ではない）。
- combat/pressure: 総勝利持分36→33、最小率0.0%→0.0%。最弱人格の勝利率改善は確認できない。
  改善より悪化条件が多く、品質改善案として棄却。強い人格を弱めて差を縮める修正は採用しない。forecastは目標欲求とH3代理値の複数変更を含み、H3単独の効果ではない。
- combat/forecast: 総勝利持分36→24、最小率0.0%→0.0%。最弱人格の勝利率改善は確認できない。
  改善より悪化条件が多く、品質改善案として棄却。強い人格を弱めて差を縮める修正は採用しない。forecastは目標欲求とH3代理値の複数変更を含み、H3単独の効果ではない。
- gambling/pressure: 総勝利持分10→10、最小率16.7%→16.7%。最弱人格の勝利率改善は確認できない。
- ギャンブル48局: 負の期待利益の賭け0、破産0。不利条件16局の待機/資金/不足はevaluation.jsonで個別確認。初期資金12と有限16機会の自作ルール内の結果。

採用範囲: 再検証基盤と任意のNeedPressure接続。従来のゲーム接続・Policy・数値Populationの既定は維持。戦闘の不足/将来代理値を既定へ置換しない。

残る急所: 最強主義の厳密優先と目先の安全代理値が結びつくと、欲求不足だけでは慎重型を動かせない。実際の相手の動き・チームの競合・勝利期限を含む評価を別条件で検証する必要がある。主義を強制解除したり、防御を不可能扱いして押し出す修正はしない。汎用知性/遅延評価・経験学習の統合は未解決。
