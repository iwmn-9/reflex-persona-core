# 学ぶ本人と将来の評価

`closed_loop.owned_rollout` は、通常の `DecisionLoop` を使う本人を将来の各分岐へ複製する。固定した性格・価値観・Policyを変えず、分岐内の観測から予測、結果記憶、欲求圧力、進展監視を更新する。実行にも同じ `resolve_feedback` を使う。ゲーム側の合法手・公開状態・条件付き遷移・目的の定義は必要で、コアがルールや勝利条件を発明するものではない。

従来の終局評価の簡略化と異なり、本人の記憶を捨てない。将来の目的得点だけでなく、そこへ至る欲求・価値・様式・費用の相関した経路も運ぶ。目的点は最後に一度だけ評価し、既知の現在点との差で本人の損失評価へ渡す。人格の採点と最強主義の候補集合を維持し、その中で目的の許容幅を適用する。既存の `goal_forecast` と `ReturnFrame` を再使用した。輸送可能な経路長を256手へ拡張したが、分岐木の深さの既定値・通常の即時Policyは変更していない。

「完全な経路」は、宣言された効果列を途中で省かないという意味。経路全体の採点は現在の本人の尺度に従い、将来の欲求・経験・関係の変化は、その時点の選択とゲームから返る効果を通して反映する。未来の各時点の心理効用や関係appraisalを別々に足すモデルではない。効果スキーマと目的の意味は予測区間内で共通にする。この境界を越えた心理・価値の変化を自動的に扱えるとはしない。

## 世界と本人を分ける

実観測後の predictor を複製し、次の予測内の世界法則を作る。この世界法則は予測中は固定した経験を参照するが、公開状態に条件づけた応答確率を返せる。分岐中の本人は別の複製で、想像上の観測から学ぶ。本人の理解が変わっても、それだけで世界法則を変えない。次の実観測の後は必ず実本人から予測器を作り直す。古い本人の状態で作った予測器を再利用するとエラーにする。

入力は `events(public_state, action)` の `PublicEvent` 列。各イベントは公開応答記号、次の公開状態、実1手の効果を持つ。記号集合と予測器の応答集合は一致させる。相手の未公開意図、実世界の乱数、検証用の真確率を渡さない。

`posterior` / `persistent` では、各イベントの純粋な `feedback` が対応する `event={'revealed_action': symbol}` を必ず返す必要がある。公開応答の欠落や取り違えを黙って世界の条件付き履歴から消さないため、標本に選ばれなかった応答も含めて照合し、接続不整合は拒否する。応答を観測できないPOMDPの潜在履歴を積分するAPIではない。

既定の `world_model='frozen'` で固定するのは現在の予測分布であり、未知の相手タイプ・真の確率分布をベイズ事後から経路ごとに引く方式ではない。結果の偶然性は扱うが、「どの世界法則が正しいか」の不確かさを完全な信念状態として表してはいない。想像上の学習を実行できることと、情報獲得の価値を正確に予測することは別になる。

`EventReader(..., learning='segmented')` には、明示した区間生成モデルに基づく別の継続を用意した。`world_model='posterior'` は実観測時点の履歴と生成モデルを固定し、各経路で得る公開応答に条件づけて次の世界応答を予測する。本人の好みや効用を世界法則へ流し込まない。ある傾向が続く世界と、途中で変わる世界を周辺化するため、各手を現在の平均確率から独立に発生させる既定方式とは、未来の観測列の相関が異なる。世界の真確率は入力しない。

この区間モデルは、供給された条件付き仮説と未知のカテゴリ分布を区間ごとに比較する。未知分布は対称Dirichlet事前分布、変化頻度は不変を含む四つの仮説を用い、保持した公開履歴の周辺尤度で重み付けする。最大四つの観測特徴の部分集合も同じ証拠で比較する。ゲーム名・勝ち手・人格係数はモデルに入れない。実本人は直近128観測を保持する有限履歴の近似であり、無期限の厳密な事後分布ではない。予測中の世界モデルは、その予測を始めた時点の履歴と新しい仮想観測を保持し、本人が古い観測を忘れても世界側からは削らない。現在のAPIは予測長が本人の応答記憶に収まる場合だけ受け付ける。

比較用の `world_model='persistent'` は、次の応答を生む傾向の不確かさを同じ事後分布から引き継ぎ、そこから先の一回の先読みでは傾向が続くと仮定する。未知の確率を平均へ潰す方式とは異なり、観測列の相関と将来の本人の学習は残る。将来の未観測の変化まで外挿することだけを省く。実際に結果を観測した後は、ほかの方式と同様に変化を検出する本人からモデルを作り直す。どちらの継続仮定も全環境で優れる保証はない。

区間モデルの仮定、事前分布、公開特徴や途中の目的評価が間違う可能性は残る。BAMCPの探索・最適性保証を実装したものではなく、正しい仮想更新と実世界での判断改善は別に検査する。

この区別は、次の一手の平均予測が同じでも必要になる。例えば未知の二値確率にBeta(1/2, 1/2)の事前分布を置くと、最初の応答は五分五分だが、同じ傾向から次も応答が出るモデルでは最初と同じ記号が出る予測は3/4になる。平均の五分五分を毎手独立に使う世界では1/2のまま。将来の本人が一回目から学んで二回目を選ぶ利益が異なる。この有限例を全列挙と照合することと、実際の相手にその継続仮定が合うことは別である。

## 読みの省略と適用限界

先頭の全応答を列挙し、その後を共通の独立した予測用乱数列で試走する。既定の将来の本人は通常の即時判断で動き、経験に応じてその判断を変える。外側の長い探索を未来の全手で再帰的に実行するものではない。

`future_planner(person, request, public_state)` を渡せば、予測中にも実際と同じ予定の再計画を実行できる。引数は複製した本人と公開情報で、予測用乱数は渡さない。計画しない時は `None`、計画する時は通常の `DecisionLoop` 用plannerを返す。`future_planner_id` で方式を識別し、`future_planner_limit` で各経路の呼出回数を制限する。内側の探索にさらに同じ仕組みを無制限に入れない。全探索できる小問題の一致と、有限な将来予測の精度は別に判定する。追加探索を使わなかった前段の長期比較は [owned_value](../evidence/owned_value/REPORT.md) に保存している。

任意の `tail_recovery=2` は、予測中の本人が知る合法手のどれにも正の即時目的利益を見込めない場合だけ、同じ本人による二手の分岐評価を使う。二手目は更新済み本人の即時判断で、全応答を列挙する。通常の人格採点・学習・欲求更新は維持する。これは「補充」「投資」などの行動名を指定する仕組みではなく、正の即時利益だけでは理解できない準備を有限の範囲で評価するもの。各経路の追加計画は既定で最大16回。`tail_recovery_limit` で1〜256回を指定でき、`RecedingOwner` にも同じ引数を渡す。長い予測で全期間の回復能力を維持する場合は、未来の手数以上の予算を明示する。上限後の予測は即時判断へ戻るので、期間の延長と予算変更の品質効果は分けて検証する。外部future plannerの `future_planner_limit` も1〜256回。任意の外部future plannerとの同時指定と、未来のない `tail_steps=0` との併用は拒否する。実際の外側探索を将来の全時点で再帰実行する近似ではない。

`tree_steps` は先頭で列挙する深さ、`tail_steps` はその後に試走する長さで、合計は256以下。未来のない短い診断は `tail_steps=0` を使える。`future_choices='reflex'` では、根元以外の候補木をその本人が実際に選ぶ即時判断へ絞る。最後の結果の後に一度も判断しない場合に限り、未使用の本人更新を省く。物理的な結果、嗜好の効果列、目的点は省かない。

途中で打ち切った場合、目的点は明示的にゲーム側の途中評価である。終局の実結果や校正済み勝率と混同しない。未知の相手変化、標本誤差、途中評価の不備は残りうる。全人格の勝率を揃えるための係数変更、ゲーム名に応じた採点分岐は共通コアへ入れない。

## 標本誤差による候補除外の研究用補正

`sampling_standard_errors=2.` は、目的平均の僅差が有限の試走の揺れである可能性を、候補を除外する前に考慮する任意構成。現時点では長期診断中で、採用済みの標準構成ではない。先頭の条件付き応答は確率で積分し、その後の同じ予測用乱数に対応する完全経路同士の目的差から標準誤差を計算する。条件付き分岐を別々の独立標本として水増ししない。元の人格の優先層内で、既存の目的許容幅と指定倍の標準誤差を超えて劣る候補だけを除外する。進展監視が許さない候補は他の候補を排除する根拠にしない。

欲求・価値・費用・リスクの相関した分布と目的平均は変えない。判断材料の数値誤差を、本人の危険回避へ加点する方式ではない。`JointForecast.purpose_sampling` の `PurposeSampling` は、対応する標本ID、計画ごとの値列、予測元を明示し、平均が予測目的と一致することを検査する。`owned_rollout` / `RecedingOwner` では、独立した予測用乱数が2〜32本、`purpose_estimator='conditional'`、一手の根元と完全な本人の試走を持つ場合だけ指定できる。層別標本、多段の木、スカラーだけの終端推定へ同じ標準誤差を流用しない。

これは経験的なMonte Carlo誤差の目安で、世界モデルの正しさや校正済みの信頼区間、方策改善の保証ではない。少標本で誤差を過小評価する場合、予測中と実行時の本人が異なる場合、途中代理値が不適切な場合は残る。誤差のない二択で小さな目的差まで優先する既存の相対許容幅の性質も変えない。[既知局面の診断](../evidence/long_horizon/sampling_band_roots.json) を出発点として、検証条件は [sampling_band_plan](../evidence/long_horizon/sampling_band_plan.json) に固定し、局面単位の改善だけで採用しない。

## 接続する条件

### 瞬間の満足と、蓄積した成果の区別（研究中）

`CumulativePreference` は、価値の一つを「指定した目的点の累積変化」と明示する任意の契約。長期の達成を、一手ごとの上限付き満足として合計すると、大きな満期収入や準備の成果を数え落とす。また、瞬間の満足を時間で割り引くことと、最終的に残る成果を評価することは異なる。この宣言は両方の意味を区別するためのもので、単なる丸め誤差修正ではない。長期試行での採否はまだ決めていない。

```python
from reflex.cumulative_preference import CumulativePreference
basis = CumulativePreference(
    target="wealth", unit="declared-achievement-unit",
    channels=(("values.achievement", 100.),),
)
# owned_rollout / RecedingOwner に cumulative_preference=basis を渡す。
```

宣言した価値の**全体**を、各経路の `(予測結果の目的点 - 現在の実目的点) × 固定の単位換算` で置き換える。その経路に残る成果を一度数え、手ごとの上限と主観的な時間割引を適用しない。値は有界な転送形式と `ReturnFrame` の復元尺度で保持する。宣言しない欲求・他の主義・行動傾向・費用には従来の時間重みを残し、固定人格・主義の強さ・実際の学習イベントも変更しない。最も強い主義による候補層も維持する。利益を全人格の最優先事項にする機能ではない。

指定できるのは `values.*` の独立した全チャネルだけ。瞬間の快楽と累積成果を混ぜたチャネルを自動で解釈しない。目的識別子・現在の実目的点・完全な嗜好経路・明示した分岐確率を必要とし、従来の終局点専用 `PreferenceBasis` との同時指定は拒否する。換算の絶対値は256以下、転送の復元尺度は512以下。標本ごとの結果と他の好み・リスクの相関は残し、条件付き平均の補正を各標本へ加算することはしない。契約は外側の判断と内側の二手の回復計画へ渡し、予測識別子にも含める。

今回の四つの検証ルールでは、既存の `achievement` がすべて `目的点の変化 × 全ターン数` と定義されていた。その同一の単位だけを宣言する `cumulative_achievement=True` を試す。ルール・観測・実際の結果・人格係数を変えず、ゲーム名や行動名による判断分岐を追加しない。128/256ターンの検証範囲を、任意の長さの実ゲームへの実証と解釈しない。

これは目的点の代理評価自体を正しくする仕組みではない。目的の定義、打ち切り、相手予測、有限標本、将来の本人の近似が間違えば損は残る。[Ng, Harada, Russell (1999)](https://people.eecs.berkeley.edu/~russell/papers/icml99-shaping.pdf) の潜在値の差分という考え方を参照したが、非線形な人格の多目的判断に論文の方策不変性保証を主張しない。検査は `tests/test_cumulative_preference.py` にまとめる。

報酬を一定範囲へ切り詰めると目的や行動が変わる問題は [van Hasselt et al., Learning values across many orders of magnitude (2016)](https://arxiv.org/abs/1602.07714) にもある。同論文の PopArt は学習対象を正規化しつつ元の出力を保つ方法。本実装は学習器の PopArt を導入せず、既存の `ReturnFrame` で元の評価単位を復元する。正規化の都合で大きな成果を消さないという区別を参考にしている。

通常の本人は `DecisionLoop`、公開応答の予測器は `EventReader`（または同じ `frozen_forecaster()` 契約を満たす予測器）を使う。各候補に共通の応答分布を前提とする。選択した手で相手の応答確率が変わるモデル、未観測ルールの発見、独立readerによる後からの上書きはこのAPIの範囲外。単に制限を解除して対応済みにはしない。

```python
controller = owned_rollout(
    owner, observe=game.context, events=game.events,
    terminal=game.terminal, assess=lambda w: game.goal(w).record(),
    record=lambda w: w, request_for=game.request, feedback=game.feedback,
    model_id=game.model_id, target=game.target, tail_steps=15,
)
request = game.request(game.context(public_world, owner.state))
result = DecisionLoop.decide_batch(
    [(owner, request)], False, controller.planner(owner, request, public_world)
)[0]
# ゲームが実際に選択を実行し、その観測だけを返す。
resolve_feedback(owner, result['ticket'], actual_feedback)
# 次に読む時には、更新済みownerからcontrollerを再構築する。
```

この再構築を毎回の実行境界としてまとめた任意の入口が `reflex.receding_owner.RecedingOwner`。上記と同じゲーム側のコールバックを渡し、`result = controller.decide(public_world)`、実行後に `controller.resolve(result['ticket'], actual_feedback)` と呼ぶ。予測は解決済みの本人から毎回作り直す。未解決の結果を残した次の判断や、終局後の判断、不完全な予測から暗黙に即時判断へ戻る動作は拒否する。保存する本人は通常の `DecisionLoop.record()` で、復元後に同じ設定のwrapperを再構築する。

毎回読むこと自体が品質向上を保証するわけではない。未来の本人が立て直せない近似や標本誤差があると、繰り返すほど現在の選択を悪くする場合がある。下記の回復まで含む構成は、4ルールの独立確認と長期ストレスを終え、品質優先の任意構成として採用した。既定の予測方式・通常Policyは変更しない。改善範囲と残る損は [継続と長期判断の比較](../evidence/owner_continuity/REPORT.md) を参照。

この比較の `receding_recovery@1` は、`EventReader(learning='segmented')` を持つ本人に、`RecedingOwner(..., world_model='posterior', purpose_estimator='conditional', tail_recovery=2, tail_steps=15, seeds=(701, 702, 703))` を接続し、実結果を解決するたびに再計画する構成である。名前だけで同じ構成になるわけではなく、これらの引数とゲーム側の接続契約を合わせる。wrapperの既定値は引き続き `world_model='frozen'`、`purpose_estimator='sampled'`、`tail_recovery=0`。既定の呼出しがこの任意構成と同じ挙動をすると解釈しない。

`PolicyTail.evaluate()` は従来の目的点だけの推定経路として残す。完全な嗜好経路には `trajectories()` を使う。既存の終局点置換用 `PreferenceBasis` と完全な嗜好経路を同時に有効にすると、どの流れを終局点で置き換えるかが曖昧になるため拒否する。今回、その意味を自動推測する機能は足していない。

## 採った研究上の考え方

比較再現用に、旧窓予測器へ変化点予測を重ねる `EventReader(renewal=True)`、`owned_rollout(sampling='stratified')`、`purpose_estimator='conditional'` も保持する。いずれも標準には採用していない。区間予測器とrenewalの重ね合わせは拒否する。層化は各手の周辺確率を覆うための方法で、経路差の分散低減は保証しない。conditionalはモデル内の条件付き目的平均による制御変量で、実際の標本経路・リスク採点・学習イベントを替えず、世界モデルの誤りを直す機能ではない。詳細と失敗結果は [auction_recovery](../evidence/auction_recovery/REPORT.md)。

- Guez et al., [Bayes-Adaptive Simulation-based Search with Value Function Approximation](https://acguez.github.io/files/guez_nips2014.pdf): 物理状態だけでなく履歴・信念を意思決定の状態に含める考え方を参照した。既存のEventLearnerは厳密なベイズ事後分布ではないため、BAMCPの最適性・収束保証を主張しない。
- Adams & MacKay, [Bayesian Online Changepoint Detection](https://arxiv.org/pdf/0710.3742): 変化点と区間内分布を周辺化する考え方を区間予測器へ用いた。供給仮説との混合、観測特徴別の区間、有限履歴は本実装のモデル化で、論文のコードをそのまま移植したものではない。保持した履歴と宣言した事前分布に対する一致を小問題の全列挙で検査する。
- Bertsekas, [Model Predictive Control, and Reinforcement Learning: A Unified Framework Based on Dynamic Programming](https://web.mit.edu/dimitrib/www/Bertsekas_NMPC_IFAC.pdf): 現在の方策を実行して継続価値を見積もり、実観測後に再計画するrolloutを参照した。人格の多目的採点、不完全な相手モデル、有限標本を含む本実装には、厳密な単一目的方策改善定理をそのまま適用できない。
- Kastner et al., [Distributional Model Equivalence for Risk-Sensitive Reinforcement Learning](https://proceedings.neurips.cc/paper_files/paper/2023/file/b0cd0e8027309ea050951e758b70d60e-Paper-Conference.pdf): 平均の価値が合うモデルだけでは、リスクに敏感な判断の一致を保証できないという境界を確認した。人格の採点を平均目的点へ置き換える根拠にはせず、今回も相関した嗜好経路を残す。この論文の分布等価な学習器や保証を実装済みとはしない。

## 検証

`tests/test_owned_value_loop.py` は完全な小さい分岐木について、予測の選択、実行後の学習済み本人、復元後の本人、期待目的点を照合する。長い検証は `tools/run_owned_value.py`。競り、設備投資、弾薬と体力を持つ戦闘、賭けと満期のある資産を扱う。資源・負傷・経験は128ターン中に初期化しない。隠れた相手傾向は実世界のrunnerに置き、同じ人格・乱数列・ルールで即時判断、短い流れの評価、本人を含む目的評価を比較する。

採用の判定は部品のテスト件数ではなく、長い実行での目的得点・人格差・適応後の挙動・予測と実結果のずれによる。短い試験や一つの成功例を一般的な知能の証明にしない。

### 長い実行の保存・再開

実行環境の時間上限をまたぐ場合は、チェックポイント対応の同じ凍結ソースから `tools/run_owned_chunk.py` を使える。`--job-file` には登録の一つのjob配列を保存し、同じ `--arm`、`--checkpoint`、`--output` で再実行する。`--decisions` は今回進める判断数だけを制限する。ゲームの残り時間や計画範囲はjobの値のままで、資源・経験・人物を初期化しない。結果を解決済みの境界でのみ保存する。

途中では完走行を出力せず、全体が終わった時だけ元と同じ形式の一行を出力する。保存する世界の乱数状態は実行管理用で、NPCの観測や予測には渡さない。再開時にjob・方式・実行ソース・Python/NumPy・プラットフォームを照合し、一致しない場合は拒否する。既に終わったチェックポイントを新しい世界としてやり直すこともない。部分記録を完走扱いにしたり、異なるソースの実行を黙って結合するための機能ではない。

同一ホストで比較した二方式が時間上限に当たった場合は、`tools/resume_horizon_pair.py` に両方の元ディレクトリ、登録job、ソースmanifestとcheckpointのSHA-256を渡す。両方が経験済みの共通時点から、新しい同一ホストで続ける。先に進んだ側の余分な区間は元資料に保全し、比較用コピーでは共通時点まで戻して再計算する。所有者・世界・乱数・固定軸の検査数をその時点へ戻し、凍結ソースと実行環境を照合する。各ホストが担当した区間と保持した経験のhashを残し、集計でも照合する。二方式を別々のホストで無断で継ぎ足した結果は、同一ホスト比較として扱わない。クラウド実行は `owner-horizon-pair-resume.yml`、一致と拒否条件の検査は `tests/test_pair_continuation.py` を参照。

保存した予測を作り直すと、同じPython/NumPyでも確率の末尾に丸め差が出る場合がある。通常の監査は従来どおり完全一致を求める。これで完走後の監査だけが失敗した場合に限り、明示した `--recorded-predictions` により `tools/audit_recorded_forecasts.py` を使える。再計算した各確率と当時の記録を固定の8 ULP以内で照合し、**当時実際に用いた予測**への結果として学習を再生する。最終的な本人の各hash、物理結果、世界乱数、固定軸には近似一致を認めない。8 ULPは監査の数値誤差上限であり、予測精度や環境をまたぐ完全再現の保証ではない。

この別方式は行動を選び直す検証ではない。元の実行ソース・経験・軌跡を変更せず、監査方式・差の全記録・軌跡とソースのhashを `recorded_forecast_replay.json` に残し、集計でも照合する。方式を隠した集計、上限を超える予測差、本人のhash不一致は拒否する。プロセスを再起動した後の再計算が完全一致したと報告しない。検査は `tests/test_recorded_forecast_audit.py`。
## Optional refinement of coarse immediate ties

`owned_rollout(..., tail_goal_refinement=True)` lets a simulated future owner
use its own public-response beliefs to distinguish immediate actions whose
entire coarse effect vector is constant and identical. The selected action's
equivalence class must also share target, confidence, familiarity and reading
offset, with zero switching cost. Different intrinsic preferences remain
outside that comparison. The ordinary persona tier and progress mask still
decide eligibility; social appraisal and external advice currently decline
this refinement. Defaults are unchanged.

Only a first-order dominant one-step purpose lottery can resolve such a tie.
Crossing risk distributions are not ranked by their mean. This adopts the
finite-distribution comparison described in [MIT's stochastic dominance
notes](https://ocw.mit.edu/courses/14-123-microeconomic-theory-iii-spring-2015/875150f8deb05a910756a02a4f9f78a5_MIT14_123S15_Chap4.pdf).
Its application here is deliberately limited: dominance of the immediate
declared purpose is **not** dominance of future states, knowledge or options.
An inaccurate purpose proxy or response belief can still mislead it.

The callback runs only when the separately budgeted future recovery planner
declines or has exhausted its budget. It does not spend additional recovery
calls, change world dynamics, create empirical experience or update the real
owner. It adds conditional event enumeration to the imagined reflex and may
cost more CPU time. Cache identities and registered experiment configurations
include the option. Long-run adoption remains subject to separate validation.

累積チャネルを併用する場合、この比較を使うのは宣言した換算がすべて正のときだけである。負の換算には目的点の低下を好む意味があり、目的点の増加を根拠に同点を解消しない。

## 再計画器の設定を含む保存復元

`RecedingOwner.checkpoint(public)` は、実結果を解決済みの本人、読みの長さ・予測乱数・世界予測方式・回復予算・目的評価の設定、および公開された世界の境界を一つのJSON互換記録にする。`expected.restored(public, saved)` は、ホストが期待する構成で作った `expected` と照合して新しい `RecedingOwner` を返す。元の本人を上書きしない。読みの設定を保存ファイルの都合で変更したり、人格・学習器・目的・復元先の世界を取り違えたりした場合は拒否する。

```python
saved = controller.checkpoint(public_world)  # resolve() 済みであること
# ホスト側の世界・実世界の乱数も同じ境界で保存し、先に復元する。
controller = expected_controller.restored(restored_public_world, saved)
```

未解決の判断、破損、未知の形式、異なる設定は復元しない。合法手のない終局も、ホストの `terminal(public)` が真であれば保存・復元できる。`expected_controller` は同じ人物と構成で初期化済みのものを渡す。終局での新しい判断は引き続き拒否する。内部の `DecisionLoop` で終了状態だけを直接復元する場合は `allow_inactive=True` を明示する。通常の判断入力に合法手が必要な契約は変えない。

ゲームの世界そのものや実世界の乱数、コールバックのプログラムはこの記録に含めない。ホストはそれらを同じ境界で保存・復元し、`record(public)` が計画に必要な公開状態を欠かさず表すようにする。`model_id` はルール・評価・観測コールバックの意味が変われば変更する。予測器も自分の `restored()` でモデルの設定と記憶を照合する。ハッシュは破損検知であり、第三者が書き換えた保存データの真正性を保証しない。この接続改修はH32研究候補の採用や長期判断の合格を意味しない。

## 固定32手構成の安定版接続

`reflex.fixed_horizon.FixedHorizonOwner` は、登録40組・10,240実判断の完走・監査と悪化例の審査を経て採用した固定32手構成。`decide → resolve → checkpoint/restored` を安定版の接続入口とする。通常の `RecedingOwner` の既定を置き換えず、任意の研究部品の組合せまで安定版とはしない。[採否と全悪化例](../evidence/stable_release/adoption.json)、[長期結果](../evidence/long_horizon/REPORT.md)、[配布範囲と検査](../evidence/stable_release/RELEASE.md)。

```python
from reflex.fixed_horizon import FixedHorizonOwner

controller = FixedHorizonOwner(
    actor, observe=observe, events=events, terminal=terminal,
    assess=assess, request_for=request_for, feedback=feedback,
    model_id=model_id, target=target,
    cumulative_preference=declared_preference,
)
result = controller.decide(public_world)
# ゲーム側で実行し、実際に観測した結果だけを渡す。
controller.resolve(result['ticket'], actual_feedback)
saved = controller.checkpoint(next_public_world)
```

検証対象の構成は `horizon=32, seeds=(701,702,703), recovery_cache=64`。既存の条件付き平均、同じ本人による人格評価、準備の継続候補、将来の本人の二手回復をそのまま使う。評価の打ち切りは一つであり、複数期間による投票は実行しない。新しい準備ボーナスやゲーム名による分岐はない。私有モジュールへ移した研究コードの入口は、既存実験との互換用に残している。

公開モデル状態はJSON互換で、計画に必要な情報を欠かさず渡す。累積価値の換算 `CumulativePreference` はホストが明示する。主義・欲求をゲーム側の勝利スコアへ自動置換しない。保存にはコントローラーの種類と予算も含み、別構成への誤復元を拒否する。予測の未完了を即時判断へ黙って置き換えることもない。ホストの世界・実世界の乱数・コールバックの版を同じ境界で保つ責任は、上記の保存契約と共通である。

凍結した研究実装と配布物を別プロセスで照合し、判断・全予測経路・実際の学習記録の一致を確認した。接続確認に加えて登録済みの長期品質基準も満たしたが、個別の悪化5条件、世界予測の誤り、読み切れない準備、重い費用は残る。別の予算・未知ゲーム・多数NPCのリアルタイム性能へ、この限定結果をそのまま一般化しない。

## 独立した予測群での判断確認（任意・検証中）

`RecedingOwner(..., sampling_check=DecisionSampling())` は、同じ実際の本人を複製して、独立した予測乱数の組A・Bと、両組をまとめた予測で判断する。三つの選択が一致すれば統合した6標本を使い、食い違えば第三の独立した組Cを加えた9標本で判断する。初期値は各組3標本。組・標本数は明示的に指定でき、有限の上限を検査する。

統合するのは、欲求・価値・危険・費用の相関を保った経路そのもの。多数決や、人格により候補を落とした後の得点平均は使わない。元と同じ人格の優先順位と目的の許容幅で、まとめた経路を再評価する。判断の一致は標本に対する再現性の確認であり、統計的な信頼区間や正しさの保証ではない。同じ世界モデルの誤り、読みの打ち切り、想像した将来の本人の近似は残る。

同じ実際の判断中だけ、公開状態・将来の本人・予測組が一致する経路を再利用する。実際の経験更新をまたぐキャッシュは持たない。各試行は実際の本人から切り離し、最後に採用する予測の入力が実際の判断入力と完全一致することを検査して、一つの通常の未解決判断を作る。結果は従来どおり `resolve()` で学習する。未完の予測を黙って採用することもない。

得るものは、少ない標本に偶然有利だった手を一度確かめる機会。支払うものは通常2〜3組分の予測と再採点の計算量である。設定・各段階の選択・採用標本・追加した組・再利用数を実行記録に残す。既定は無効。`tests/test_sampling_check.py` で、新しく全標本を計算した場合との一致、4ルールの通常結果学習、途中再開、キャッシュ分離、試行失敗時の本人保全を検査する。

## 同じ累積利益の平均を目的と主義へ接続する比較版

`goal_forecast(..., coherent_cumulative_mean=True)` は明示した累積価値チャネルに限り、目的評価と同じ条件付き平均の補正を適用する。従来は目的の平均を補正しても、主義の優先候補を決める累積価値の平均は少数の標本のままで、同じ利益に二つの異なる期待値を与えていた。比較前に有利な候補が落ちる既知の局面を再現している。

`ReturnMeans` は人物・目標・候補に結びつけた平均補正を、`ReturnFrame` で復元した単位で保持する。線形の期待効用と主義の平均にだけ適用し、各標本の結果、損失への反応、内的な欲求・他の価値・個性・実際の観測学習は書き換えない。候補を絞る既存処理や人物ごとの分割採点でも同じ補正を使う。未対応の人物・候補・確信度・単位での使用を拒否する。既定は無効で、Rustの即時判断契約には追加しない。

得るのは同じ利益情報の一貫した受け渡し。捨てるのは、目的の標本誤差だけを減らし、同じ量を表す主義には別の標本平均を残す扱いである。損失リスクの推定には従来の標本誤差が残り、世界予測の誤り、切断された将来、有限標本の境界への射影による偏りも解消しない。補正後の各仮想結果を新造して危険を消す方法ではない。

研究用runnerでは `--purpose-estimator conditional --coherent-cumulative-mean` を使う。`--fresh-controls` により過去の異なる実行環境の対照を再利用せず、同じ環境で両方式を最初から比較できる。これも既存の凍結済み40/8/8試行を変更しない。採否と全損失は [長期比較報告](../evidence/long_horizon/REPORT.md) に残す。

## 小さな即時利益があっても準備を読む比較版

従来の `tail_recovery=2` は、将来の本人に即時の目的改善が一つもない場合だけ二手の回復を読む。この省略には、小さな改善を続けられると準備の価値を見逃す弱点がある。任意の `tail_recovery_mode='always'` はこの条件を外し、将来の各判断で同じ二手の本人判断を使う。`tail_recovery_limit` は `tail_steps` 以上が必要で、不完全な予測から黙って即時判断へ戻ることも拒否する。既定は従来の `'stalled'`。

各将来の本人は通常の結果観測で学び直し、その時点の欲求・感情・人格を通して選ぶ。準備行動への加点やゲーム名による分岐は加えない。現在の32手の判断器と将来の二手の判断器は依然として異なり、完全な自己予測ではない。また、この変更は世界予測や有限標本の誤差を解消しない。

実験runnerの `--tail-replanning` はこの設定と全tail分の回数を接続し、途中再開でも契約を固定する。比較計画は [tail_replanning_diagnostic_plan.json](../evidence/long_horizon/tail_replanning_diagnostic_plan.json)。得る狙いは、少量の即時利益で将来の準備を抑制しないこと。支払うものは全予測経路で二手の判断を繰り返す計算量。研究用の比較版であり、局面診断・単体テストだけでは採用しない。長期実行の採否は [長期比較報告](../evidence/long_horizon/REPORT.md) を正本とする。

## 将来の本人が読む期間を明示する比較

`tail_recovery_horizon` は既定の2を保ち、`tail_recovery_mode='always'` の場合だけ3〜128へ延ばせる。二手は従来と同じ全応答の列挙。長い比較では、先頭の全応答と、その後の学習する本人の試走を用い、内部でさらに同じ長い探索を再帰しない。通常の人格評価、累積価値、目的の許容幅、欲求更新を共用する。準備行動の名称やゲーム名による採点分岐はない。

内部の予測用乱数は `(1701,1702,1703)` に固定し、外側の評価用乱数との重複を拒否する。長さと内部の方式は予測の識別子へ含める。実験runnerは `--tail-replanning --tail-replanning-horizon 32` のように指定し、設定をcheckpoint・継続実行・収集・診断まで保持する。途中で二手へ戻したり、別の期間へ置換することは許さない。

これは基準方策を試走して一段改善するrolloutの考え方を、未来の本人の能力近似にも使った比較である（[Bertsekasの講義資料](https://web.mit.edu/dimitrib/www/RLTopics_2022_Lect5.pdf)）。一般的な改善保証を、非線形の人格評価・不完全な世界予測・有限標本へ移していない。内側の試走は依然として即時判断を基準とし、複数の準備操作の組合せをすべて発見する機構ではない。また、外側の評価窓で未回収の準備を過小評価する可能性も残る。

`tools/run_preparation_horizon.py --root <新しい出力先>` は、準備の回収が3・7・11手遅れる公開モデルで、同じ外側16手・128実行ターンを二手/16手の将来判断で比較する。64手後に見返りが下がる追加条件も用いる。その変更は実行側がその時点で公開し、未来の計画へ事前に教えない。準備に固執せず見送れるか、同じ長期準備を先送りし続けないかを検査するための診断であり、競り・戦闘などの登録済み長期比較や独立確認を置き換えない。

