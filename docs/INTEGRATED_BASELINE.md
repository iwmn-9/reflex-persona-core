# 同じ本人で現在と予測を判断する

目的は、人格を残しながら判断の足場を安定させること。ゲーム名・特定の手・人格別の勝率補正を共通の判断器に加えない。ゲーム側は公開情報、合法手、効果の意味、進展・終局条件を供給する。未知のゲームでルールだけから評価関数が自動生成される契約ではない。

## 時間と採点単位

`intertemporal.forecast` の効果packetは従来どおり −1～1 の平均で運ぶ。新しい `ReturnFrame` が、人格・損失・主義の優先層を判定する前に割引累積量へ戻す。`goal_forecast` は目的の終局評価を一回だけ使い、欲求・価値・費用は累積量で比べる。単なる終局後のゼロ埋めで優先層や選択が変わらない。

`Policy.decide(..., return_frame=frame)` と `TiledPolicy.decide` が対応する。frame は本人と目的に結び付き、効果の倍率とroot項の倍率を保持する。即時判断でframeを指定しない経路は従来のまま。平均packetだけ取り出してframeを落とした採点は、累積量の採点と同じではない。

長い窓で実際の収入や損失が増えれば評価も変わる。未来が不明な場合のゼロ埋めは許諾しない。ゲーム側が時間単位と各効果の尺度を揃える責任は残る。一時的な資源支出と、取り返せない信条違反の意味もadapterが区別する。全ての価値効果を足すだけで普遍的な倫理が生まれるわけではない。

## 実行する本人の複製

`DecisionLoop(..., social=memory)` は既存の連続・三段階・撤退型の関係記憶を保持できる。`Request(..., social_bindings=bindings)` は現在の候補と相手を結び付ける。根の関係評価は即時判断・目的付き予測・修補の比較へ渡る。実行後の `observe/abandon(..., social_events=events)` は、その本人の実区間の事件を検証してから更新する。checkpointは記憶の内容だけでなく投影方式も保存する。

予測では以下の部品を `flow_rollout.rollout(..., observer=model, policy=actor.policy)` に渡す。

```python
model = ActorModel(
    actor,
    root_request=raw_request,
    request_for=request_from_public_context,
    feedback=revealed_modeled_feedback,
)
```

`raw_request` は、圧力や学習を適用する前の現在のRequest。未来のRequest生成器は公開contextを受け取る。feedbackは遷移後に判明した本人の即時効果・相手の公開行動・欲求進展・関係事件を `ModelFeedback` で返す。観測できない項目を捏造しない。

各候補・model seed が別の本人の複製を持つ。選んだ候補のモデル結果だけをその分岐で学び、次の判断へ渡す。分岐の経験を実本人の経験やconfidenceへ加えない。強制された根でも本人のintent遷移を保存する。人格、固執、残差方策が呼出側で食い違えば拒否する。

停滞防止で実行を止めた手も、比較用の最初の一手として分岐内で仮置きできる。その後は同じ本人の進展監視で判断する。実行する本人の制約は解除しない。違法手・既知の失敗手を比較用に通すこともない。単一案の偽プランナーを通して仮置きする旧経路は廃止した。

これは本人の**即時実行器**のモデルであり、実行時に呼ぶ重い外側の探索を無限に再帰するものではない。世界モデル、相手仮説、非終局proxyの正しさも保証しない。新しい実観測ごとに外側は再計画する。予測と実行の全てのずれが解消したとは扱わない。

`CategoricalReader` と有限support版は `belief_options` を受け取り、主観の初期仮説・慣性と独立した経験推定をcheckpointに保つ。以前の設定のないcheckpointを、強い思い込みを持つ人物へ黙って復元しない。コールバックが定義する仮説・supportの意味はゲーム側の責任。

## 省略と有限指示の検査

`ObservationSchema(name, facts)` は、計画の照合に必要な公開facts欄をゲーム側が宣言する任意機能。時刻、本人の状態・設定、候補効果、相手の証拠欄は保持する。無関係な表示注記だけなら計画を捨てずに済むが、意味を持つ状態変化は照合を外す。schemaは名前と欄の集合で版を固定し、計画・記憶・探索で共有する。学習による状態抽象化や、実ゲームの再利用率改善を実証した機能ではない。

`ExecutionWindow(scope, directives)` は有限の指示を監査する受動的な部品。ゲーム側が機会・実際の達成・失敗を申告する。反復指示は期限を延長せず、機会なし・中断・期限切れ・遅れた達成を成功へ混ぜない。記録は監査出力であり、人物checkpointの追加層にはしない。

指示の条件が満たされた回数と、その指示によって行動・物理結果が変わったかは別に比較する。指示選択器や強化学習の報酬を自動で作る部品ではない。戦略選択・実行能力を分けて測るという [MicroRTSの研究](https://arxiv.org/html/2610.11663v1) の考え方を利用した。論文の性能や学習済み方策をこのコアへ移植したわけではない。

予測枝数は既存の `rollout(..., seeds=...)` で1～8本を明示できる。今回の追加比較では、同じ８手でも２本から８本へ増やすと３系統で改善があった。全てのゲームの標準予算や、長く読めば常に強くなるという保証にはしない。サンプルを増やすことと、誤った相手モデルを直すことも別。

## 検証の正本

[実結果、得たもの・手放したもの、残る不足](../evidence/integrated_baseline/REPORT.md)。一般試合では２手、２model seedの予測を比較する。相手の読み、強CPU、全ゲームの接続について以前の試験を今回の新しい実試合数へ加算しない。

```sh
python -m unittest discover -s tests -p test_return_frame.py -v
python -m unittest discover -s tests -p test_actor_model.py -v
python -m unittest discover -s tests -p test_observation_schema.py -v
python tools/run_integrated_baseline.py --prepare --output FRESH_ROOT
# FRESH_ROOT内のlegacy/currentのrunnerを、preregisterのcohort別に実行する。
python tools/verify_integrated_contracts.py --output FRESH_CONTRACTS
python tools/verify_integrated_defaults.py --old-source FRESH_ROOT/legacy --output FRESH_PARITY
```

30接続の診断には任意の麻雀エンジンが必要。一般判断の必須依存は増やしていない。全比較の集計器は768試合・48世界の完了を要求し、一部だけを全体と呼ばない。
