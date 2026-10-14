# 残資源と将来の選択肢

`reflex.opportunity_graph` は、既知の有限な公開ルールを一度列挙し、現在の予測で終局まで評価し直す任意部品。残資源へ別の価格を付けるのではなく、その資源が残す合法な選択肢と結果を評価する。性格係数、主義、通常の反射経路は変更しない。実証の正本は [opportunity_graph](../evidence/opportunity_graph/REPORT.md)。

## 解決する問題

相手の行動予測が正確でも、一手先の利益だけを見て資源を使えば後で必要な手を打てなくなる。反射方策を終局まで試走する場合も、未来の本人が同じ近視眼的な手を打つ見積りになりうる。深さの追加だけでは、この継続の弱さを解消できない。

そこで公開ルールの状態・合法手・観測結果をDAGとして編成する。同じ状態へ合流する履歴を共有し、終局から逆向きに結果分布を計算する。各状態で将来の本人が何を選ぶかを決めてから、現在の手を採点する。ゲーム名、特定の資源、特定の勝ち手への補正はコアに持たない。

これは有限状態の動的計画法の利用であり、新しい理論的最適化手法の発明ではない。[Bertsekasの講義](https://ocw.mit.edu/courses/6-231-dynamic-programming-and-stochastic-control-fall-2015/5839a0924a3a8aac2413fff653a1a8d4_MIT6_231F15_Lec9.pdf)が扱う継続方策・終端評価・逐次的一貫性が背景になる。ただし、期待費用の最適化に関する保証を、人格の非線形採点や未知の相手に移してはいない。

## 入力と責務

`OpportunityGraph(actions, events, terminal, assess, record, model_id)` はキーワード引数で作る。

- `actions(state)`：全合法候補の一意なID。候補を恣意的に省くと、その先の機会も失う。
- `events(state, action)`：公開結果の記号、次の公開状態、当該一手の実現した効果。`PublicEvent` と同じ属性を持つ。条件付き効果の `p` は1。
- `terminal(state)` と `assess(state)`：実際の決着と、同じ目標IDに対する `Goal`。ルール・目的・残り時間はゲーム側が定める。
- `record(state)`：状態を共有するための十分な表現。ルール、予測の特徴、将来の選択に効く情報を保持する。自動的に安全な抽象化を発見する機能ではない。
- `model_id`：ルール・目的・観測表現の変更時に替える識別子。

`compile(initial)` は確率や実世界の隠れた傾向を読まず、全公開分岐を列挙する。有限の時計を含めても循環が残る場合は拒否する。ノード・辺・深さの上限では全体を辞退し、途中まで評価した候補を完全な案として返さない。

`solve(law, law_id=..., choose=..., chooser_id=..., initial=...)` は現在局面から到達可能な部分を再評価する。`law(state, action, symbols)` は、その時点で本人が持つ予測の固定スナップショット。状態に応じて確率は変わるが、この一回のsolve中に新しい学習は行わない。正規化した全記号の確率を要求する。

`choose(states, alternatives)` には、各手の当該一手の効果と、その先で選ばれた継続の終局目的分布を渡す。未指定は目的期待値の最大化で、人格NPCの継続モデルではなく比較役。本人を使う時は `persona_choices` へ十分な将来コンテキストと同じ `Policy` を渡す。

`OpportunitySolution.evaluate(state)` は `EndpointEstimate` を返し、既存の `ClosedLoop.frontier` へ接続できる。モデル内の終局分布であり、実世界の確定結果や予測の校正済み証拠ではない。グラフの再構築後や、今回解いていない状態の参照は拒否する。

## 人格と省略

`persona_choices` は既存の一手 `goal_forecast` / `deliberation.select` と同じ採点をバッチで呼ぶ。最優先の主義階層を守り、その中の目的許容幅で本人の選択を行う。危険な結果を平均効果へ潰さず、終局目的の分布を残す。主義のための損までゼロにする仕組みではない。

この許容幅は、その局面・同じ将来継続・同じ主義階層内の比較。全行程で目的だけを最大化する最善方策との差を、累積で同じ幅へ抑える保証ではない。

**将来の人格は将来の選択に入る。ただし、遠い将来までの欲求・主義・スタイルの効果の累積はここでは追わない。** 各状態で当該一手の心理的な効果と終局目的を組み合わせる。この基準に沿う本人が次の状態でも選び直すことを計算する。任意の心理履歴を持つ将来人格を自動で再現する保証はない。

**読みの中での将来学習は省略する。** 実際に一手進んだ公開結果は `EventLearner` に返し、次の実判断で新しい予測を使って解き直す。既存の学習込み `ClosedLoop` と異なり、「情報を得たことで将来の知識が変わる価値」をこのsolveの中では数えない。この交換で履歴の爆発を抑えて、残資源から終局までの選択を列挙した。

## 今回のゲーム接続

`tools.opportunity_models.OpenOpportunities` は既存の4つの `OpenGame` 診断ルールへ接続する。予算・得点・体力・時刻・残る賞品・直前の自分の手などは保持する。将来ルールと予測へ影響しない表示用の直前結果を省き、ゲーム番号は必要な偶奇だけ残す。これはこの4ルールの十分性を検査した変換であり、他ゲームへ無条件に使うものではない。

診断では欲求不足は固定、切替費用0、親しみ係数は一定。将来のmodeを既存Policyで解決して共有できる。追加のpressure/progress/social/predictorや効果学習記憶を持つownerは拒否する。それらの状態を含む別の十分な接続が必要。本人・モデルが違う見積りの混用も拒否する。

実行は毎手 `DecisionLoop` から一手の `ClosedLoop` を通し、その先に今回の分布を接続する。人格継続での根の選択と実際の選択が違えば、実験を失敗扱いにする。固定予測での2手・全公開分岐では、実際に毎手解き直した本人の期待目的と予測を照合する。実観測で予測が変われば、その後の計画変更は自然に起きる。

## 実行

```sh
python -m unittest discover -s tests -p test_opportunity_graph.py -v
python tools/run_opportunity.py --root FRESH_PILOT --stage pilot --workers 4
python tools/run_opportunity.py --root FRESH_DEV --stage development --workers 4
python tools/run_opportunity.py --root FRESH_LIBRARY --stage development --baseline-only --workers 4
python tools/analyze_opportunity.py --roots FRESH_DEV FRESH_LIBRARY --output analysis.json
python tools/probe_opportunity.py --root FRESH_PROBE --inputs FRESH_DEV --workers 4
python tools/probe_opportunity_cards.py --root FRESH_CARDS --workers 2
```

各実験はソースを複製・固定し、条件を実行前に保存する。`holdout` は別の公開順序・多い雑音・反応/途中変化/復帰/交互作用を使う。対照も同じseedと公開世界の生成法を使う。診断の再生は公開遷移・学習・目的値を照合するもので、全判断の再探索ではない。別のprobeが全人格campaignの最終ゲーム初手を再探索する。

`probe_opportunity_cards` は既存Goofspielの公開入札・最高同額なら賞品破棄のルールに同じコアを接続する。13ラウンドの合法な公開履歴から残り3ラウンドを作り、3〜5人・2賞品順・2seed・4人格・2目的を列挙する。相手は独立一様という宣言モデルで、未知の相手の学習試験ではない。残り2ラウンドの既存全列挙関数と目的優先値を比較し、本人の実DecisionLoopで根の選択も照合する。ここでの目的優先は全手を目的だけで選ぶ上限比較で、主試験の「未来だけ目的優先」と区別する。元のゲーム側の心理効果・温存の代理点は変更しておらず、その意味の妥当性をコアが自動で保証するものではない。

有限モデルを全列挙できる範囲の重い読みであり、実時間・多数NPCへ常時使う既定ではない。大規模な状態数、無期限のゲーム、未知ルール、動く心理状態との完全接続、未知環境での人間水準は未解決。上限で辞退することと、難しい局面で十分に賢く動けることは分ける。
