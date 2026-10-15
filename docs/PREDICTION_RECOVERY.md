# 予測の組が両方外れるとき

`EventReader` は、既存の `EventLearner` を通常の `DecisionLoop` とモデル内の本人へ接続する任意の予測器。人格・欲求の評価やゲームのルールは変更しない。実験と採否の正本は [検証報告](../evidence/prediction_recovery/REPORT.md)。

従来の `CategoricalReader` は直近4観測の分布と有限仮説を比較する。両方が誤る場合、どちらを採用するかという検査だけでは修正できない。新しい接続では、仮説に加え、公開された特徴の組合せと直近8・32・128観測から作る分布を、結果を見る前の予測成績で混合する。以前悪かった予測器にも少量の重みを残し、変化後の復帰を可能にする。ルール内で起きうる応答なら、仮説の混合では表せない傾向も学習できる。

これは [Tracking the Best Expert](https://mwarmuth.bitbucket.io/pubs/J39.pdf) の切替追従の考え方を使った既存実装の接続。論文の保証を、このNPCの勝率や人格の合理性へ拡張しない。今回 [Bayesian Online Changepoint Detection](https://arxiv.org/abs/0710.3742) も検討したが、変化点の確定診断を追加せずに既存の複数期間の競争で回復できる範囲を先に測る。原因の発見や正しい予測の保証ではない。

## 接続契約

```python
from reflex.event_reader import EventReader

reader = EventReader(
    scope, hypothesis_names, models_for, outcome_for,
    legal_for=legal_for, model_id="my-observation-and-rule-contract-v1",
    known_conditionals=True,
    factors=("public-signal",),
    features_for=lambda context: (context["facts"]["public-signal"],),
)
# DecisionLoop(..., predictor=reader)
# 別の先読み器も reader.predict(public_context) から同じ現在の予測を取得できる。
```

- `legal_for` はルール上可能で観測可能な応答の、ソートした文字列tupleを返す。「ありそうな応答」だけに絞らない。異なる合法集合の経験は同一の選好証拠にしない。
- `models_for` は各仮説の確率分布を返す。各分布は現在の合法集合全体を覆う。`outcome_for(context, action_id, response)` は既知の条件付き効果を返す。
- この接続は全候補が同じ応答法則を共有する場合に限る。自分の手によって相手の反応が変わるなら、その条件を表す別のモデルが必要。隠れた行動や観測できない結果を別の応答として学ばせない。
- 特徴は最大4個の公開カテゴリ。未知の知覚・ルール・因果構造を自動発見する機能ではない。ゲーム名による勝ち手分岐はないが、観測と効果の意味はゲーム側が定義する。
- 効果が欠測でも公開応答があれば更新する。応答が欠測なら効果だけから応答を捏造しない。更新前の予測を採点してから、新しい観測を取り込む。
- `record/restored` は所有者、ルール・観測意味の識別子、学習設定、時刻を照合する。`DecisionLoop` とモデル内の本人は同じ観測処理を使う。仮想分岐の学習が実本人の記憶へ戻ることはない。

`Reading` に第二候補を渡さず、現在の混合分布で同じ本人が行動を再評価する。既存の読み予算・進展監視・主義・選択差の基準は維持する。監査表示 `single_response_forecast` は一つの応答予測を使ったという意味で、確率の校正を認定するものではない。確率精度と本人の選択の良し悪しは別に測る。

記憶は最大128件（既定値）。長期間来ない状況の経験は失われうる。入力が128手より長い依存や未観測の変化を含む場合、その構造をこの履歴だけで復元できるとは限らない。Policyの1行動あたり8結果または読み予算を超える列挙は辞退し、通常判断を維持する。学習自体はより広い応答集合も扱える。

`HypothesisTracker` 固有の思い込み・慣性の設定を自動移植する接続ではない。今回の比較はそれらを無指定にした予測器同士で行う。固定人格と欲求の採点は維持するが、主観的信念の粘りと新しい学習器の合成は別途検証が必要。

実行例は `tools/run_prediction_recovery.py`、以前の失敗条件との接続は `tools/run_recovery_bridge.py`。先読みの世界法則は根元の実観測で学んだ記憶を固定し、仮想の公開特徴・合法集合で予測し直す。仮想の本人は各分岐の経験で再判断する。未知の真の変化を予見することや、外側の探索を将来も再帰的に行うことまでは再現していない。
