# 共通判断経路の契約

任意の `goal_guard.choose_with_goal(context, names, returns)` は、ゲーム側が与えた合法手と同じ行動集合について、共通標本の正規化した目的成果を比較する。目的の見込み損が許容幅を明確に超える候補を除き、残りを固定の有限人格Policyへ戻す。人格や勝率を補正しない。現在の研究値は許容幅0.12・差の標本標準誤差2倍。これはモデル内の目的許容幅で、実勝率の保証でも厳密な同時信頼区間でもない。大きな目的犠牲の自由は狭まるため、通常コアへ自動接続しない。

`goal_progress.choose_with_progress(context, names, success_samples, progress_samples)` は、全候補・全標本の勝利持分が0の局面だけ、adapterが供給した正規化進捗へ比較軸を移す任意経路。平均勝率が同じだけでは発動せず、勝利が確保できる予測にも介入しない。`relative_progress(scores, viewer, direction=..., scale=...)` は参加人数によらない首位との差の一例で、得点方向と比較単位はゲーム側が決める。複合勝利などでは別の進捗標本を供給でき、共通コアが勝利条件を解釈する機構ではない。

返り値は調整済みcontext・選択・guardの三つ。人格・主義・欲求・根の費用を保持し、定数だったobjective/achievementの結果だけを進捗へ置換する。元の他軸の結果分布と進捗標本の積を作るため、入力の一行目だけで他の分岐を置き換えない。ただし両者の相関は未供給なので独立近似となる。最大8結果への圧縮は平均と最悪objectiveを保持するが、区間内の主観的リスクは近似。guardの`signal`は目的持分か進捗かを明記し、許容幅と標本誤差の意味を混ぜない。モデルが間違っていれば進捗比較も間違う。通常の`Policy`・バッチ反射や他ゲームへ自動適用しない。

`goal_progress.omit_expired_proxies(context, needs=(...), values=(...), style=(...))` は、ゲームが実際の期限切れを確認した資源・欲求の代理評価だけ外す共通部品。固定性格、主義の強度、物理費用、他の有効な欲求は保持する。無効になった主欲求への執着は解除する。探索の深さを使い切ったことを期限切れと呼ばない。No Thanks! adapterの `horizon_progress` は残り山札0に限り、次のカードに備える安全欲求・security・neuroticismの結果proxyを外す。現在のカードを拒否する支払いと強制取得は、引き続き終局シミュレーションで評価する。終盤の実証と採否は [goal_progress](../evidence/goal_progress/REPORT.md) が正本。

`strong_table.PlaybackStart` と `play(..., start=..., forced_root=(actor, action))` は開発用の分岐教師生成。現在の公開状態、本人別stateと複製した記憶から、一人の合法な根だけを置き換え、将来の全員の実際の再判断・公開学習を回す。山札は評価器の世界解決に属し、NPCの入力ではない。同時入札は全員の選択後にだけ公開する。シミュレーション教師を実際の観測記憶に混ぜない。小数の仮想世界から最良だった手を選んでも、真の期待最適性を証明したことにはならない。

`HypothesisTracker.observe(..., forecast=pre_reveal_distribution)` は、実際の意思決定で用いた更新前の予測を忘却・予測損失の基準にできる。全合法手に正の有限確率を持つ正規化分布を要求し、不正入力は状態更新前に拒否する。省略時は旧予測経路を保持する。人格の同定ではなく公開行動の仮説更新であり、仮想標本を実観測として学習しない。新しい二ゲーム経路の `PublicMemory` はこの分布を供給し、直近閾値と条件付き係数も未来標本へ渡す。実際の他者の内部状態は渡さない。

`DecisionLoop`は一人・一ゲーム・一エピソードの判断と結果を接続する任意経路。人格・主義を固定し、動く欲求、意図、ルート、経験、相手の仮説を分離する。ゲーム固有の戦術名や勝利名を共通コアへ持ち込まない。

```
本人の観測 → 任意のルート選択 → 条件別経験 → バッチ反射
                                      ↓ 必要な場合だけ bounded reader
                                選択ticket → 公開された結果
                                            ↓
                          更新前予測の比較 → 経験/仮説の更新
```

## 最小接続

```python
from reflex.decision_loop import DecisionLoop, Request
from reflex.judgment import Binding
from reflex.planning import vector

loop = DecisionLoop(initial_context)
request = Request(
    context=observed_context,
    bindings={
        a['id']: Binding(a['id'], public_condition_key, ('objective',))
        for a in observed_context['actions']
    },
    exact={a['id']: ('cost',) for a in observed_context['actions']},
    maintains_advantage=False,
    threatened=False,
)
result = loop.decide(request)
# この間にゲームが選択を実行し、本人が知れる結果を返す。
loop.observe(result['ticket'], vector(observed_effect))
```

`initial_context`/`observed_context`/`observed_effect`は既存の`core`契約に従う。スコープ、性格、主義、seedは個体の存続中に固定。tickは増加する。最初以外の行動intentはloopが引き継ぐ。欲求や観測はゲームが更新する。bindingsは全候補に必要で、estimatedとexactを重ねない。即時結果として観測できない効果はestimatedへ指定しない。

複数個体は`DecisionLoop.decide_batch([(loop, request), ...])`へ渡す。全個体の検証・採点・reader検査・ticketの準備が終わってから、全員の状態を確定する。最後の個体が不正でも先頭個体を先に進めない。ゲームの世界更新は全選択確定後に行う。コールバックは受け取った観測だけを使う純粋関数にする。外部世界やコールバック自身の副作用までロールバックする機構ではない。

一度の選択に一度の公開結果。二重結果、別個体ticket、非有限/範囲外の結果、exactとの矛盾は拒否する。結果が欠測なら`abandon(ticket)`を明示し、架空の0や失敗を学習しない。未解決ticketがあれば次の選択とcheckpointを拒否する。

`record()`はJSONへ保存可能。`DecisionLoop.from_record(context, record, predictor=...)`は所有者、人格、状態、時間軸、記憶容量を検査して再開する。Policy係数と個体の読み予算も保存/復元し、別設定へのすり替えを拒否する。predictorがある場合、再開時にも同じ純粋なモデル/ルール関数を渡す。関数や世界はcheckpointに保存しない。

## 目的の停滞と欲求不足（任意）

`DecisionLoop(context, pressure=NeedPressure(context['scope']))`で有界な不足を接続できる。性格・主義・手の効果は固定し、ゲームが実際に観測した欲求ごとの進展を`observe(..., need_progress={'growth': delta}, maintained=(), completed=())`へ渡す。各進展は[-1,1]。既定では進展がない3観測までは蓄積せず、その後0.1ずつ最大0.7を加える。正の進展で0.25を減らし、完了で清算する。これは心理学的に校正した数値ではない。

有効な保持・待機は`maintained`へ明示する。観測していない欲求を失敗扱いせず、無効な欲求を追加しない。ルート段階にも同じ不足を適用するが、実観測は選択結果につき一度だけ。個体別の5不足/5停滞回数をcheckpointへ含める。異常な結果・進展ラベルは記憶もticketも更新しない。

将来を含む効果ベクトルを即時結果として観測できない場合は、`abandon(ticket, need_progress=actual_goal_progress, maintained=..., completed=...)`を使える。効果の経験学習は行わず、別契約の実際の目的進展だけ不足へ返す。欠測を0の効果として学習しない。不足は最強主義を強制解除せず、何もしていない慎重な人格が必ず打開する保証にはならない。ゲーム側の進展尺度・有効待機の定義も検証対象。

共通4系統の再現は`python -m reflex.cli validation`。結果の正本は[evidence/validation/REPORT.md](../evidence/validation/REPORT.md)と[evaluation.json](../evidence/validation/evaluation.json)。ギャンブルは既知の確率・有限資金・借金なしの自作ルールで、好機/不利/途中変化を比較する。

## 基線を壊さない比較

経験は既存の`OutcomeMemory`をそのまま使う。2観測後から直近4結果を混ぜ、重み上限0.75。大きい予測支持外の結果によるリセットを維持する。経験の予測誤差は更新前の分布を保存して有限CDF閾値の二乗誤差で採点するが、この誤差だけで経験全体を無効にしない。`learning[*].predictive_warning`は診断情報。単一の不運や予測精度だけから、判断の正しさや学習の撤回を認定しない。

経験を混ぜる際も、`Binding.estimated`以外の効果は、現在のpriorが宣言した同時分布と確率を保つ。例えば確定ルールが「費用0または1」なら、記憶の圧縮で存在しない費用0.5を作らない。最大8結果へ収める際は、非推定効果が同じ分岐の中だけを圧縮する。同一の効果ベクトルは確率を合算してから、各分岐へ最低1枠を割り当て、残りは1枠あたりの確率質量が大きい分岐へ配分する。推定効果の平均と確率は保つが、分岐内の損失の裾や相関は近似され、1枠しかない分岐では推定効果が平均だけになる。これは未知の危険を正確に学べる保証ではない。推定効果と非推定分岐の独立性を置く従来の経験混合も維持する。

`CategoricalReader`は、ゲームが提供する有限の応答仮説と、公開応答ごとの既知の即時効果を接続する。3公開観測後から直近4手＋各応答1疑似観測の分布を適応基線に使う。追加の`HypothesisTracker`分布は、基線との更新前Brier比較に加え、両分布が選ぶ手の即時目的効果も比較する。各4件以上の比較で利益があり、直近の比較も悪化していない場合だけ切り替える。新しい悪い比較は採用を撤回し、適応基線へ戻す。

後者は「公開された同じ応答＋既知ルール」で導ける仮想比較。未選択手の実観測ではなく、`OutcomeMemory`へは入れない。相手の当手が自分の当手を見て反応する場合、この同じ応答による比較は使えない。ゲーム側のルール表は別に正しさを検査する。選択済みの公開結果が宣言した条件付きルールに矛盾すれば更新を拒否する。

読みを省く優位中も、安い応答確率の予測は公開後に採点できる。根×応答の効果評価は必要な時だけ。全評価をnode上限へ計上し、表全体を作れなければ反射へ戻る。実験の上限は16評価、循環対戦では12。確定効果、合法性、手のidentity、人格、元の観測はreaderで変更できない。

効果が未知の外部`Reading`は、予測源と各方法/状況の比較が必要。一つの手で正しくても、未試行の別手を信用しない。経験/予測/採用状態は有限容量の個体別記憶で、学習された万能価値関数ではない。

## ルート・時間幅・性能

`IntentRequest(route_context, build, uncertainty)`は、共通`RouteState`で選んだルートを`build(route)`へ渡し、同じtickの`Request`を作る。ルートと単発行動は別の記憶。同じPolicyを両方に使い、異常時はルートも進めない。

現在の経験・予測採点は即時結果だけ。`outcome_target!='immediate'`や、未来/終局のreaderを即時結果につなぐ接続は拒否する。任意の`JointForecast`による未来の選択は下記の別targetへ接続した。`rolling_plans`/終局MCとの統合と、遅延成果による未来モデルの経験学習は未完成。選択への接続と実経験への接続を区別する。

このJSON経路はコピー、検証、比較用採点の費用を増やす。数値の大量NPC経路`Population`/`AdaptivePopulation`へ自動置換しない。読みのnode上限はCPU時間上限ではない。全ゲーム対応、学習の統計的校正、戦闘の強さ、大量実時間NPC、疑似人間としての完成は未確認。

再現：`python -m reflex.cli decision-loop`。比較数値、棄却した案、検証範囲の正本は[evidence/decision_loop/REPORT.md](../evidence/decision_loop/REPORT.md)と[evaluation.json](../evidence/decision_loop/evaluation.json)。軌跡はローカルに作り、公開版へ同梱しない。


## 戦闘の即時経験候補と目的達成確認

`python -m reflex.cli purpose`。正本は`evidence/purpose/REPORT.md`と`evaluation.json`。
共通Policy/人格/主義の優先規則を保ち、戦闘側の`make_context(..., survival_security=True)`だけで生存損失と負傷の欲求を分ける。現在の相手全員が公開前の射程内から独立に一様な対象を選んで射撃する仮説を置く。死亡した敵も既に選んだ同時射撃を実行し、こちらの非公開の移動後に新しく見えた相手へ標的を変更できない。現在の実意図・実命中乱数・試験相手の制御型を入力しない。最大8結果の未校正モデルで、相手の方針変化を正しく予測できる保証はない。

`validation_experiment.Session(..., learn=True)`は、既存DecisionLoopの選択ticketで実行後に観測した同じ1ターンの`transition_effect`を返す。費用はルール確定、その他の効果は推定として扱い、方法を行動ID、条件を地形/体力/弾薬/射線内の数/選択ルートとする。共有記憶の既定の上限64条件・直近4結果・2観測後から混合を維持する。他者の同時行動も含む条件付き経験であり、自分の行動が共同成果を引き起こしたという因果的な功績ではない。予測と観測の進捗/特徴の定義はゲームが供給する。

他ジャンルは既存の効果定義と欲求不足の対照検証を使う。資源モデルの未来収入や静止H3脅威を即時実観測にせず、ギャンブルの既知確率を少数の運で上書きしない。この即時経験候補だけでは味方の競合と弱い人格を安定改善できなかった。後続の共同先読みは下記で比較する。候補の品質確認には最弱人格の勝利改善・各人格の勝利維持・各相手群で対条件悪化の超過なしを要求する。小標本を一般的な知性の証明にしない。

## 未来の選択と実経験の分離

`DecisionLoop.decide_batch(items, planner=callback)`は既存の即時判断の後、commit前に任意の共同予測を評価する。callbackはコピーされた現在の有効context群と反射decision群を受け、`JointForecast`または`None`を返す。対象は同じ時点で共同計画できる個体群。plannerを省略すれば従来の経路を使う。

`JointForecast(contexts, roots, purpose, horizon, max_regret=.15, audit=None, target='game-purpose')`の契約:

- `contexts`: 現在と同じ固定人格・値・欲求・状態・tickの所有者別context。actionsだけを共同案IDと将来効果へ置き換える。観測済みfactsを保持し、予測の説明を追加できる。
- `roots`: 共同案ID→各個体が今実行する合法な行動IDのtuple。既知失敗手を含めない。未来の仮想案IDを実行・記憶の方法IDにしない。
- `purpose`: 全共同案に対応した[-1,1]のゲーム側の目的達成見込み。校正済み勝率ではない。同じチーム目的が前提で、異なる私的目的の交渉モデルではない。
- `horizon`: 1〜16。具体的な時間単位はadapterが定義する。`max_regret`は0〜2、最良の見込みから許す差。固定0.15がどのゲームでも妥当と保証しない。
- `target`: ゲームが名付ける予測目的尺度の識別子。追加候補との比較では、同じ目的尺度・期間・人格の選択幅を維持する。

目的の帯に入る案を既存Policyで採点し、各本人の最大主義tierが全員一致して受け入れられる案から、本人別の主観的採点差の平均が最小の案を選ぶ。他者の多数決で一人のtierを押し切らない。ただし帯外の案を先に除外するため、元の全合法手に対する主義の自由は制限される。これは明示した能力下限の仮定であって、帯外の実行手がルール上違法になったわけではない。帯内の損失やこだわりは残す。

両立する案がなければ`adopted=False`として従来の反射へ戻る。そのtickの目的下限・味方無競合は保証しない。全体でcommitする前に予測/実root/人格の契約を検査し、異常なら全個体の状態を進めない。

採用時は選択を実rootへ戻し、現在contextのhash・実target・単発intentを保存する。`result['context']`とpendingのprior/bindingは即時のまま。予測された将来の改善、未選択結果、共同成果の因果功績を`observe(ticket, ...)`へ渡さない。結果の`deliberation`auditは`target='modeled-horizon'`を明記する。採用結果は現在contextだけを単体Policyで再採点した反射decisionとは異なる。

`IntentRequest(route_context, build, uncertainty=.35, route_planner=callback)`では、ルートの未来予測を先に選び、選択済みルートで`build(route)`を呼んで即時target/bindingを作る。単発行動を作った後でルートだけ変えない。route auditの従来`scores/eligible`は即時ルート評価で、未来選択は別の`deliberation`auditを参照する。

## ゲーム側の先読み接続と検証範囲

`combat_planning.TacticalControl(horizon=6, samples=4, max_plans=24, max_regret=.15, coordinate=True, recovery_margin=.02)`は、公開盤面から有限の共同案を作る。目的担当と射撃援護の組み合わせ、味方移動先の予約、goal-directed/aggressiveの相手仮説を使い、実ルールの同時解決で先を進める。相手の実制御型・現在の非公開意図・実命中乱数を渡さない。仮説は固定で、相手傾向を学習する予測器ではない。全共同手を探索せず、戦術的継続もゲーム側の近似。本人の人格評価は根の案に適用し、未来の全戦術手を人格Policyで選ぶとは主張しない。役割は実tickごとに再検討し、継続的な役割契約は未実装。

`resource_planning.EconomyControl(horizon=4, max_regret=.15)`の単位は本人の手番数。勝利ルートと単発行動を評価し、相手3人の手番・生産・維持費・領地干渉も実ルールで進める。本人の継続は固定人格Policyで選び、近い/遠い経済方策を相手仮説にする。現在公開されている供給を使い、未来の実イベント予定は渡さない。この仮説群には比較相手に近い自作方策も含み、未見相手の一般化は未証明。安価な貪欲な本人継続は判断を悪化させたため棄却した。

目的尺度、効果、合法手、世界モデル、戦術はゲーム側が実装する。共通の`deliberation.py`とDecisionLoopはゲーム名や人格→勝利名の固定対応を持たない。二つのゲームへの接続は確認したが、任意のゲームのルールを読み込むだけで正しい評価器が生まれる仕組みではない。

`python -m reflex.cli stability`で696局の固定候補確認を再生成する。戦闘576局（3方式）、資源48局（2方式）、既存の競り/ハゲタカ/ギャンブル対照72局。全18,308実遷移を保存軌跡から再生。戦闘/資源の任意候補は今回の品質ゲートを通過したが、2seedと自作小規模ゲームに限る。対照ゲームは新しい先読みを接続していない。速度費用は合否に使わず、既定の反射経路は維持。結果と得失の正本は[evidence/stability/REPORT.md](../evidence/stability/REPORT.md)と[evaluation.json](../evidence/stability/evaluation.json)。

残る不足は未来モデルの誤りを実際の遅延成果で学ぶ契約、主義の対立時の交渉、役割の継続/切替、未見ゲームの評価、人格の長期的一貫性。planningの`nodes`はモデル世界を実ルールで進めた回数で、候補生成/戦術採点の内部仕事を含む完全な計算費用ではない。

## 意味のある停止と停滞の観測

`DecisionLoop(..., progress=ProgressWatch(scope))`へ任意の進展監視を接続できる。`Request(..., purpose=PurposeRequest(level, readiness, activities, maintained=False))`でゲーム側が現在観測済みの目的進捗・準備度[-1,1]と、全rootの意味を供給する。`maintained=True`は前回の観測後に新しく確認した有効な準備/維持の実績。自分の手番間の生産などを拾うための信号であり、未来の期待や「まだ待ちたい」を渡さない。性格・有限価値・目的そのものを変更する層ではない。数ターン止まることだけを失敗と判定しない。

`Activity(kind, reason, condition='same', patience=3, release='', replacement=True)`の意味:

- `attempt`: 目的または準備への決定的試行。同じroot/conditionで進展のない結果が既定3回続いたら再検討する。
- `uncertain`: 成功確率が正で、目的に役立つ確率的試行。偶然の不成功だけで禁止しない。結果を実観測できた試行は待機の一続きを終え、新たな短い待機を許す。目的が進んだという偽の記録は作らない。
- `wait`: 理由と`release`を持つ有限の機会待ち。`patience`は1〜16、同じ待ちの見直しまでの選択回数。予測の更新・理由名の変更だけで猶予を更新しない。現在の予測から新しい猶予を繰り返し申請しても、実観測なしでは延びない。
- `maintain`: 有効な状態の保持を期待する停止。実際に意味があった保持の観測がなければ既定3回の猶予後に再検討する。
- `idle`: ゲーム側が現在の目的への意味を裏づけていない手段。目的/準備の停滞が既定3回続けば、支持される代案がある時に除外する。

`replacement`は、この手段を切替先として合理的に検討できるというゲーム側の宣言。残る候補が全て`False`なら、停止を除外して危険な行動へ追い込まない。合法性を変更せず、別の能力制約として候補を制限する。制約前の最大主義の自由は狭まるが、その範囲で固定人格Policyを使う。主義の交渉を実装したわけではない。

`observe`/`abandon`の`purpose_feedback=PurposeFeedback(level, readiness, maintained=False, completed=False)`は、実際の目的/準備度と有効な維持/完了の別契約。即時効果ベクトルへの学習とは分離する。`maintained`にはゲームが確認した有効な保持・準備の回復を使い、単なる静止や生存をそのまま進展としない。過去最大を超えない位置/準備の往復は新しい進展にならない。欠測を失敗として数えず、選択した待機回数だけは消費する。checkpointは有限の失敗記憶、目的/準備の最大値、待機猶予と個体所有者を保存する。

進展制約は、単独の反射・readerの候補・共同先読みの実rootへ同じように適用する。全候補が支持されない場合は`progress.unresolved=True`として合法候補を残す。意味のある切替先を持たない状態まで、コア単体で解決したと主張しない。

`ProgressConfig(proof_margin=.02)`を指定した場合、期限は強制移動の命令ではなく再検討のきっかけになる。共同plannerは制約前の反射を含む同一候補群を生成し、元の人格に沿った共同案と制約後の共同案を比較する。実際に選択を変更するには、採用可能な代案のモデル目的スコアが元の案より`proof_margin`を超えて高い必要がある。同時判断で閾値が異なる場合は、再検討対象全員の最大閾値を使い共同案を一括で採否する。元の案が共同採用されなかった場合は、その反射root組に一致するモデル案の値を比較基準にする。その案もない/代案を採用できない/予測の差が小さい場合は、元の選択を保ち`unresolved`と比較値を記録する。待機予算や実際の進展を偽って更新しない。

この設定では別のreader/predictorとの同時使用を明示的に拒否する。planner不在や予測が欠ける場合も、根拠なく動きを強制しない。`proof_margin=None`の既定はゲームの`replacement`宣言を切替根拠に使う反射経路。`.02`は今回の戦闘の工学上の切替基準で、人格の選択幅`max_regret=.15`とは別に制御する。人間の合理性や正しい予測を保証する定数ではない。未来比較は人格の数値を変えず、即時経験へも学習しない。

ゲーム側の`progress_adapters.py`では、戦闘の射撃、回復/装填、目的地点への移動、味方の道を空ける移動、拠点保持を区別する。チーム目的なので味方全体の準備度と実際の装填/回復も観測し、本人の静止だけで味方への支援を停滞と扱わない。敵の実意図を見ず、現在の公開盤面と未校正の即時生存損失モデルを使う。切替先のモデル死亡リスク上限0.35は、この試験のゲーム側の仮定であって汎用の安全保証ではない。戦闘ではさらに`proof_margin=.02`で未来の目的改善を比較する。資源管理は公表済み生産による科学/文化の未達閾値を有限の待機理由にし、前提が満たされたら同じ待機理由を使わない。先行ルートに隠れる別ルートの準備と、他者の手番後に起きる実生産も、前回の観測との差から拾う。余剰生産は新しい待機根拠にしない。ゲームの保存/復元は、コアのcheckpointと共にこの前回観測の公開世界状態も保存する必要がある。

`python -m reflex.cli purposeful-wait`は、追加前の共同先読み方式と進展監視ありを同条件で比較する。静的な期限診断の意味は両方式で同じ。戦闘の未来比較が裏付けられず残した停止は別に数え、解決済みにしない。勝率維持は保守的な回帰診断で、採否の唯一の条件ではない。人格/環境/相手の相性による得意不得意は残し、判断の不具合と人格・目的に沿った負けを区別する。根拠切れの選択数だけを減らして勝率が落ちた案を成功にしない。速度/費用は合否に使わない。[結果・得失・未解決](../evidence/purposeful_wait/REPORT.md)を正本とし、データには実世界の遷移と監視状態の再生を分けて記録する。

## 打開候補と先送りの診断

共同plannerに任意の `recover(contexts, original_decisions, root_allowed)` 属性を付けると、進展制約後の通常候補で共同案を採用できない時だけ追加案を求める。追加のJointForecastは元の `target`・`horizon`・`max_regret`を厳密に維持し、同じ観測・固定人格・合法な実rootを持つ必要がある。新案も各本人の最大主義tierの一致と元の判断に対する改善幅を通す。比較を成立させるために主義を解除しない。検査の失敗は全個体のcommit前に拒否する。採用できない理由と支持候補数、本人別のtier候補数を残す。

戦闘の `TacticalControl(recovery_options=True, recovery_plans=48)` は、通常の最大24案に加えて、支持された実rootを組み合わせた目的案・人格採点上位の組み合わせ・支持rootを一つずつ含む補完案を最大48案作る。同じ実ルール・相手仮説・モデル乱数で比較する。全共同手の探索ではない。資源管理の単独手番はもともと全合法rootをモデル評価しており、この追加共同候補を必要としない。

共通 `horizon.purpose_return(path, progress_weight=0.)` は、同じ意味の各手番の目的スコア[-1,1]を受け、終点と途中平均の加重和を返す。0なら従来の終点だけ。今回の任意比較は0.3（終点70%・途中平均30%）で、先読みの「あとで進む」案と早く目的へ進む案を区別する。これは目的評価の明示的な変更で、性格パラメータではない。長期の終点価値の一部を早期進展へ配分する得失があり、全ゲームに適切な重みと保証しない。相手待ち・準備・生産の価値をコアが自動発見するわけでもない。

戦闘は実tick、資源は相手の手番と生産まで完了した本人1周を単位として予測経路を保存する。早期にルール上の終局になったモデルの目的値は、残り期間も同じ終局値として比較する。`compare_paths` は選択済み予測の次観測・期間終点を、その同じ目的・単位の実観測と比較する。資源の勝利ルートが途中で変わっても、診断は元のルート尺度で実状態を評価する。終局でない欠測を未来結果にしない。

予測誤差には、モデルの誤りだけでなく実際の再判断で継続手段が変わる影響も入る。ここでは診断に留め、即時OutcomeMemoryや自動の信頼更新へ投入しない。「将来の進展を期待して待ったがその期間に実現しなかった」件数も、因果的な失敗や無意味な待機の確定判定ではない。

## 予測と実行の対応（任意のオフライン診断）

`TacticalControl(trace_execution=True)` は、各候補の各モデル標本について実際にモデル内で進めた遷移だけを記録する。既定は `False`。公開盤面の前後のfingerprint、実tick、本人チームと相手のactor/action対応、終局を保存する。標本は混ぜず、平均目的経路に対応する単一の架空の行動列を作らない。終局後に目的値を補う処理は従来どおりだが、行動列を補完しない。

検証harnessは最終採用auditの `selected_plan` に属する標本だけを実軌跡へコピーする。追加案が棄却され元の案に戻った場合も、最終auditを参照する。実験条件を識別するrun IDはここで付け、planner・人格判断・乱数へ渡さない。map、persona、相手条件、設定が違う実験の同じseed/tickを混ぜない。traceを有効にするとコピー/ハッシュ/保存の費用が増えるため、実時間経路の既定にはしない。

`alignment.compare_execution(predicted, observed)` はゲーム名を持たない純粋な比較関数。run/group/target/unit/start/horizonの一致を検査する。1〜16観測単位、明示的なindex、連続した状態fingerprint、actor文字列→action文字列、終局ラベルが必要。観測していない行動は `None`、個体が残っていない側は空辞書。欠測tickは `None` の枠として残し、後の観測を前に詰めない。予測は全期間または終局まで必要、実観測は途中まででもよい。

比較の意味:

- 公開された直前状態が同じ時だけ、本人と相手それぞれの行動一致/不一致を数える
- 状態がずれた後の別の行動は `not-comparable`。状態が同じに戻った時は再び記述的な比較ができる
- 同じ状態・同じ両者の行動から異なる後状態になった件数も分ける。乱数や特定モデルへの因果的な責任とは呼ばない
- 未観測の将来、終局の時点の違い、両方の終局後の値補完を分け、架空の行動一致を加算しない
- 件数はモデル標本ごとの遷移。複数標本が同じ実観測を参照するため、独立な実試行数や校正済み確率ではない

`alignment_experiment.audit_combat_execution(run)` は全状態/解決後の行動が公開される自作戦闘だけに接続する。他ゲーム、とくに不完全情報ゲームでは、本人の観測履歴・行動公開範囲を含む同等性のadapter契約が別途必要。観測外の情報をfingerprintへ混ぜない。

この診断は `OutcomeMemory`、`HypothesisTracker`、信頼更新へ接続しない。行動列が一致しても、平均予測の誤差をそのまま学習してよい条件が成立したとは限らない。非一致の予測から敗北の原因を断定しない。


## 任意の人格帯による自己継続

戦闘だけの `TacticalControl(self_continuation='persona-band')` は実験用。既定の `objective` と共通Policy/Populationは変更しない。将来の公開モデル盤面から生存・負傷の即時効果を再構成し、元の人格の許容階層と0.025近接帯の中で戦術の都合を比較する。目標は各個体の根で選んだルートを保持し、援護役割を本人の目標へすり替えない。衝突回避のために階層や帯を下げることはない。

根ごとの仮の意図を引き継ぐが、未来の経験・欲求不足の蓄積・進捗監視・ルート変更は再現しない。予測した結果を経験記憶へ書かず、相手の実制御や当ターンの選択・実乱数を参照しない。今回の適用範囲は既定Policyと即時survival-security効果の戦闘実験であり、完全な将来のDecisionLoopではない。

[比較と非採用条件](../evidence/persona_continuation/REPORT.md)。同じ公開直前状態での自己継続不一致を減らすことと、実際の目的達成を助けることを別の条件で評価する。

## 任意の根の衝突補完

戦闘の `TacticalControl(root_completion=True)` は既定offの小さなablation。`coordinate=True`、`self_continuation='objective'` の組合せだけで有効にする。予測する未来の自己継続・相手仮説・目的の採点・plan生成は変更しない。既存の共同plannerが最終的に辞退した時だけ、progress proofの再考を終えた実行直前の根を補完する。別reader/predictorとの同時使用は、未検証rootの承認を迂回しないよう未対応として拒否する。

共通境界は、同じownerのeffective contextから既存Policyを再採点する。最終progress maskとexact-waste guardを使った元の最強主義階層・0.025近接帯を先に決める。その後、変更先として既存ProgressWatchがblockedと記録した手段を除く。proof不成立によって元の停止を維持する許可を、新しい期限切れ停止へ変更する許可に広げない。代案を除いたあとで階層を下げて採点し直すことはない。元の根がこの帯の外なら全体を変更せず、原因を記録する。

戦闘adapterだけが、公開盤面・episode・unitと明示された同じ味方チームを照合し、同じ移動先へ向かう本人たちを対象に最大3人の有限組合せを調べる。衝突していない個体の選択は固定する。変更人数が最少の案から、元のPolicy得点合計、canonicalな行動ID順で選ぶ。許容された全根を同時に非衝突にできなければ元の衝突を残す。危険な行動一般を避ける規則や、別陣営・独立世界を協調させる仕組みではない。

`result['root_completion']` はbefore/after、変更人数、近接帯内のregret、検討数・採否を持つ。`deliberation.adopted` はfalseのままなので、この実行上の補完を未来予測の採用と取り違えない。変更された根のnext_state、intent age、本人の実経験ticketだけを通常の原子的commitへ渡す。仮想結果を記憶へ加えない。共通高速Policy/Populationには接続しない。

再現: `python -m reflex.cli root-collision`。新規32対、元の条件によるserial trace-off費用、既存mainとの既定経路一致、実ルール・purpose再生を分ける。[機構・結果・制限](../evidence/root_collision/REPORT.md)。

## 目前の利得と後の見返り

任意の `intertemporal.forecast(context, branches, horizon=..., unit=..., target=...)` は、1個体の全実行可能rootについて、同じ1..16ステップの**増分効果**を受け取り `JointForecast` を返す。`branches[root]` は最大8個の `Branch(probability, effects, confidence)`。各flowは既存effect契約で `p=1`、分岐確率はBranch側に置き全分岐の合計を1とする。分岐は各ステップの独立な平均ではなく、一貫した将来仮説の経路である。

各効果を `gamma**t` で重み付けし、全候補共通の `sum(gamma**t)` で割る。途中の支出・損害も残し、収入と終局報酬は獲得時に一度だけ計上する。蓄積stockや状態potentialの水準を毎回の収入として渡してはいけない。状態potentialを使う場合は差分として渡す。この増分契約は、従来 `purpose_return` が受け取る「各時点の状態価値」と異なる。終局後はゼロ増分で揃え、未予測の非終局の未来をゼロで補完しない。期間外の末端価値はこの部品に存在しない。

`gamma` は既定で `clip(.45 + .45*勤勉性 + .1*(1-神経症傾向) - .35*切迫度, .25, 1)`。切迫度は有効な生理/安全欲求の最大不足で、成長・主義の評価は従来Policyが担う。これはゲーム設計上の仮の写像で心理測定の主張ではない。`discount=...` を明示してゲーム側で差し替えられる。時間の好みと、探索期間/賢さは別であり、同じ期間を見ても目先を重くする人格があり得る。

各stepのconfidenceは正の利益だけを縮め、予測損失・費用を消さない。元rootのconfidence/familiarityはPolicyで引き続き扱い、現在のintentに一致するrootの切替費用を二重に課さない。`max_regret` は同じ割引目的に対するゲーム側の許容幅（既定.15）。十分な幅を与えるかどうかはゲーム設計の判断であり、これを狭めて全人格に最長期の利益を強制しない。

一度だけ発生するfamiliarity加点とswitch_costも、効果と同じ分母で正規化する。期間を延ばしただけで切替費用が相対的に増えることを避ける。Policyの最大主義tierの許容差.02とpurpose corridorは正規化した期間平均の尺度に対するものなので、異なる期間を横断して同じ総額の許容差と解釈しない。長期化に伴う主義tierの感度は追加検証が必要で、既存Policyを変更してはいない。

`DecisionLoop.decide_batch(..., planner=...)` へ接続できる。コールバックはその時点の `contexts[0]` を使って作成し、古い人格状態や観測へ戻さない。即時RequestとBindingsはそのままで、実行時は実rootだけ確定する。未来flowを即時経験へ学習してはいけない。採用は各本人の現在のmodeと、期間全体で評価した最大主義tierに従う。固定人格の数値は変えないが、価値の評価対象を即時から期間全体へ広げる変更である。明示的に禁止されたrootの解禁や、無条件に主義を薄める処理はない。

`python -m reflex.cli intertemporal` は既存の資源/競り/戦闘ルールに接続する条件付き判断試験。初手後の継続は公開・固定して比較し、実際の人格付き再判断との不一致を解決した試験とはしない。新部品は1個体向けで、同時多人数の長期共同計画や、既存の標準戦闘plannerへの自動適用は未実装。[結果・得失](../evidence/intertemporal_integrated/REPORT.md)。

## 毎手の再判断と有限の回収

任意の `flow_rollout.rollout(c, initial, observe=..., advance=..., terminal=..., horizon=..., seeds=...)` は、全実行可能rootに対して同じ1..16stepのflowを構成し、上記forecasterに渡すBranch辞書と継続行動の監査情報を返す。ゲームadapterの `observe(state, actor_state)` はモデル盤面の即時context、`advance(state, root, model_seed)` は次盤面と一度だけ獲得/支払うeffect、`terminal(state)` は既知の終局・本人死亡などを定義する。所有者の性格・主義・scope・seed・objectiveは固定し、各モデル分岐の仮のmode/need/intentを引き継ぐ。モデル盤面から欲求を再構築する責務はadapterにある。

2手目以降は同じ人格Policyの**即時判断**を継続モデルにする。実行時は毎手intertemporal.forecastで選び直すため、これは未来の実行を完全に再現した自己モデルでも、再帰的な最適探索でもない。継続予測と実際の次の選択のずれを記録し、ずれそのものを失敗・学習根拠とは扱わない。変更された状況では選び直すことが正しい場合もある。将来の経験学習・進捗監視・相手の潜在方針変更はこの自己モデルに含まない。

最大8本の異なるモデルseedを等確率の一貫経路として使う。実seedや未観測のイベント予定は渡さない。終局後だけゼロ増分で埋める。未知の非終局の計算失敗は例外として残し、ゼロ未来に置換しない。入力盤面・人格状態を分岐ごとに複製し、仮想結果を実経験へ書かない。任意の `continuation(state, context)` は固定継続の対照実験用で、実行可能rootのみ選べる。

再現: `python -m reflex.cli payback-cycle --output FRESH_ROOT`。`FRESH_ROOT/closed_loop_payback` に事前条件・源hashと集計、ローカルの実軌跡を保存し、既存結果の上書きを拒否する。今回は常時有効化せず、[得失と限界](../evidence/closed_loop_payback/REPORT.md)を残す。標準Policy/Population、既存共同planner、実ルールは変更しない。公開は集計だけで、軌跡を持つローカル環境では `python tools/audit_payback_cycle.py FRESH_ROOT/closed_loop_payback` で実ルール再生・集計照合ができる。

## 目的の決着と準備完了の接続

任意の `purpose_plan.Goal(target, status, value)` は、`running` のゲーム側代理値と、`success=1` / `failure=-1` / `draw=0` の決着を区別する。得点最大化では `scored` と有界の実得点を使える。途中までの進捗を終局時の引き分け得点として残してはいけない。誰が勝ったか、相打ちや期限切れが引き分けかはゲーム側の規則が決める。本人の死傷・資源消費・主義の利得は別のflowに残る。共通部品に戦闘・経済・競りのルールは内蔵しない。

`flow_rollout.rollout(..., assess=callback)` の任意callbackは、各モデル経路の最後の状態を一度だけ受け取り、監査用のassessmentを返す。`goal_forecast(context, paths, rollout_audit, horizon=..., unit=..., target=..., max_regret=.15)` はそれを受け取る。全合法root・各一貫分岐に同じtargetのGoalと、実際にモデルが進めたaction列を必要とし、終局とstatusの不一致を拒否する。

目的の許容幅には各分岐の終点valueの期待値を使い、期間平均の分母で勝ち/引き分けの差を薄めない。一方、本人の主観objectiveはモデル内の決着/打切り時点まで `gamma**(steps-1)` で割り引くので、目前と先の利得に対する人格の時間選好を残す。正の終点見込みはその実行経路の最低confidenceでも縮める。予測された負の結果はconfidenceの低さで消さない。元root confidenceは従来Policyで扱う。needs/values/style/costは従来の割引flowを保持し、未来を即時経験へ書かない。

まず全モデル候補で本人の現在の最大主義tierを確定して残し、その中で目的の許容幅を評価する。目的の点が高いだけで下位の主義へ移らせない。許容幅はゲーム側のcompetence設定で、強く狭めれば欲求側の譲歩を制限する。終点proxyは成功確率に校正された値ではなく、先読みが正しい保証でもない。実行時と未来モデルの判断手順は依然異なり、将来の経験更新や進捗監視まではモデル化しない。

準備/待機には既存の `ProgressWatch` / `NeedPressure` と、ゲームが定義するrequest/feedbackを接続する。経済では研究/記念碑の実必要量に達したら待機をidleへ解放し、戦闘では実際の回復/装填や位置準備を観測する。競りでは公知の後続景品を待つ有限Activityを使う。観測されていない準備や、予測が変わっただけの状態を実進捗に加点しない。未達による欲求圧力は固定性格の学習ではない。

`ProgressConfig(preserve_persona_tier=True)` は停滞maskで元の最大主義tier全体が失われる場合に強制変更を留保する。`PersonaProgressWatch` はその設定付きの通常watchで、設定を通常checkpointへ保存し、DecisionLoopがbase watchとして復元しても挙動を維持する。既存設定は既定falseで、古いcheckpointもfalseとして読み込める。元の独立ゲーム経路を自動でこの設定へ変更しない。

比較: `python -m reflex.cli purpose-recovery --output FRESH_ROOT`。旧flow方式、目的定義/終点評価の修正、修正+実観測の準備解除/停滞圧力を分ける。`FRESH_ROOT/purpose_recovery` を凍結し、既存出力の上書きは拒否する。源hash・条件・control/confirmationとseedを比較開始前に保存する。[得失と採否](../evidence/purpose_recovery/REPORT.md)。ローカル実軌跡がある場合は `python tools/audit_purpose_recovery.py FRESH_ROOT/purpose_recovery` で実ルールと実観測更新を再生できる。過去の源hash照合は、その報告を作ったcommitで行う。

## 観測状態の分岐内更新と、改善根拠付きの停滞解除

任意の `ModelObserver(scope, purpose=..., feedback=..., progress=..., pressure=..., previous=...)` は、現在の実watch/pressureをコピーして持つ。`flow_rollout.rollout(..., observer=observer)` はさらにroot/seedごとにforkし、モデル内の遷移からだけ仮のwatch/pressureを更新する。実ownerの状態・経験・相手仮説には書き戻さない。ゲームcallbackは `PurposeRequest` と `ModelFeedback(purpose, needs, maintained, completed)` を返し、実進捗とモデル進捗の意味を同じにする責任を持つ。

rolloutのroot contextはDecisionLoopが既にpressureを適用したcontextを渡す。最初の手で二重適用せず、2手目以降のraw observationへ仮のpressureを適用する。仮のprogress maskも2手目以降の候補へ適用する。最初の強制rootは、実行時には選べないmask外の手も比較用にモデル化できるが、実際の選択はDecisionLoopのroot_allowedが制限する。identity/性格/主義/seed/目標は継承する。

これは即時Policyに監視/圧力を接続した継続モデルで、将来の本人が同じ先読みを再帰的に実行する完全な自己モデルではない。経験更新・相手学習・別reader・複数人共同計画もモデル化しない。callbacksは公開モデル状態だけを読み、実イベント予定や実乱数を参照してはいけない。任意observer未指定の既存rolloutは従来の経路を維持する。

今回の `verified` 比較は、新しい重みではなく、既存 `ProgressConfig(proof_margin=0.)` の回復検証をgoal_forecast経路で有効にしたもの。単に停滞回数を超えたから候補を削るのではなく、元の方針と制限後の候補を同じ目的・期間・人格条件で比較し、目的見込みに厳密な正の差があるときだけmaskを採用する。同値/悪化/比較不能なら元の方針を残す。目的はゲーム側の代理値でもあり、実勝利や安全を保証する条件ではない。

verifiedはモデル内observer更新を加えない分離比較で、将来の回復検証を再帰的に再現するとは主張しない。`python -m reflex.cli observed-transfer --output FRESH_ROOT` で凍結比較を作成できる。[別環境での得失と採否](../evidence/observed_transfer/REPORT.md)を参照。ローカル実軌跡がある場合は `python tools/audit_observed_transfer.py FRESH_ROOT/observed_transfer` で実ルールと選択されたモデル分岐の状態更新を再生できる。新納品環境の到達可能性探索は診断だけに使用し、判断入力や教師正解として採用しない。
## 任意の複数手継続候補

`continuation_search.search_forecast(context, world, observe=..., advance=...,
terminal=..., assess=..., horizon=6, seeds=(...), width=2, depth=2,
target=...)` は、既存の `JointForecast` を返す。通常の `Policy` と高速バッチ経路は変更しない。

各合法初手について、従来の反射継続を必ず候補に残し、将来の行動列を有限幅のbeamで追加する。同じ行動列を全モデルseedへ適用し、分岐を平均してから既存の人格評価で比較する。分岐ごとに最も都合のよい列を選び直すことはしない。予定した行動が観測上実行できない場合だけ、その分岐の人格反射へ戻る。実際に実行するのは初手だけで、次の実観測から再計画する。

順位付けは既存の終点目標評価・主義の優先層・目的許容幅・人格スコアを使う。追加の報酬係数やゲーム名別の行動優先表はない。ゲーム側は依然として合法手、公開観測、遷移モデル、目標と効果の意味を提供する必要がある。

`width` は1..4、`depth` は0..horizon-1、`horizon` は1..16。depthは初手の後に列を探索する長さで、残りは人格反射で補う。候補容量は256で、初手を捨てて収めることはせず超過をエラーにする。各層も同じ候補容量に従う。目的が同じでも人格の価値を薄めず、強い主義に沿った不利な選択も許す。

これは完全な条件分岐付き戦略や最適探索ではない。未来の途中で柔軟に方針を変える計画や、beamで早期に落ちる長い準備を見落とす。反射で補った先の評価と、実際の再計画が一致する保証もない。`flow_rollout.rollout` の `schedule` は初手後の行動列、`roots` は明示的な初手部分集合で、一般のforecastへ渡す前には全初手の被覆が必要。

`intertemporal.forecast` / `purpose_plan.goal_forecast` の任意 `plan_roots` は「候補ID→実初手ID」。複数候補が同じ初手を持てるが、全合法初手を少なくとも一度含める。各候補の分岐確率・効果・終点は従来と同じ契約で検査する。省略した既存呼び出しは従来どおり。

## 任意の計画保持と到達時刻の比較

`plan_continuity.PlanIntention(context)` は個体と固定人格に所有された有限の意図。実際に選択・実行した初手の後に `remember(context, forecast, selected_plan, actual_root)` を呼び、全モデル分岐に共通する残りの行動接頭辞だけを保存する。分岐が分かれた所で止め、都合のよい分岐の列を選ばない。最大15手で、観測結果を学習した記憶ではない。`record()` / `from_record()` で所有者付きcheckpointを保存・復元する。

次の実観測では `offer(context, target=..., unit=..., horizon=...)` の列を `search_forecast(..., retained=...)` に渡す。offerは意図を変更せず、対象目的や時間単位が変われば空列を返す。同じtickの再実行や別個体・別人格への転用は拒否する。残りの初手が現在実行不能なら候補へ加えず、実行可能なら現在の盤面から全モデルseedで採点し直す。未来の途中で予定手が違法になった場合の反射への復帰は従来どおり。

残りの計画が現在の主義tierに残る時だけ、`JointForecast.continuity` に `ArrivalPreference` を付ける。selectはさらに現在のprogress maskと目的許容幅を確認する。保持案が全モデル分岐で成功し、新案が目的期待値を増やさず、対応する全分岐で同時刻以降・少なくとも一つで遅く成功するなら、その新案を除外する。早い案・同時刻の案・分岐ごとに早さが逆転する案・未達や未確定の案はこの規則では除外しない。保持案が現時点の人格/進捗/目的に支持されない時も無効になる。新しい報酬係数は使わない。

`ArrivalPreference.arrivals` は候補ごとに対応する同一モデル分岐順の成功stepまたはNoneを指定する。外部plannerが直接構築する場合、順序・分岐の意味の整合はそのplannerの責任。計算された成功を現実の確実な成功や校正済み確率とは扱わない。現在は1個体のforecastだけに対応し、共同計画への暗黙の適用は拒否する。

これは「目的が増えない完了遅延」を一部抑える優先規則で、後回しを一律に禁止する規則ではない。既に主義tierから落ちた達成案を復活させず、選んだ列の最適性や将来の再計画との一致も保証しない。`continuity` を省略した既存forecastの選択・監査形式は従来どおり。[固定比較・得失・採否](../evidence/continuity_transfer/REPORT.md)。

再現は `python -m reflex.cli continuity-transfer --output FRESH_ROOT`。既存反射継続、共有列探索、探索＋保持の3方式を比較し、既存の凍結資料を上書きしない。ローカル完全軌跡があれば `python tools/audit_search_transfer.py FRESH_ROOT/continuity_transfer` で共有列・初手選択・現在観測からの保持案再評価・実ルール・所有者のcheckpointを再生できる。監査はbeamが最適な候補を残すことの証明ではない。


## 有限の主義優先と、同じ方策の接続

`Policy(principle_priority='finite')` は、既存の主義強度・mode・欲求・危険・費用・目的の採点係数を変えず、最大主義の効果が.02以内の候補だけを残す前処理を使わない。principle modeの最強主義への追加重み1.6×強度は残る。したがって小さな主義利益だけで任意の大損を拒否することはなく、大きい主義利益なら目的の損失を引き受ける選択もできる。全候補の合法性・既知失敗・乱数の近接帯は従来どおり検査する。これは数値化した主義の優先と、絶対禁止を区別する変更であり、新しい人格軸ではない。

既存呼び出しの互換用に `Policy()` は `principle_priority='lexicographic'` を維持する。設定を暗黙に混ぜない。`DecisionLoop(context, policy=policy)`、`Population(contexts, policy)`、`TiledPolicy(..., principle_priority='finite')` で同じ選択規則を使う。DecisionLoopは異なる優先規則の一括判断を拒否し、finite設定をcheckpointへ保存する。設定のない旧checkpointは旧規則として復元し、異なる規則を指定した復元は拒否する。

未来の採点だけを変えるのではなく、`flow_rollout.rollout(..., policy=policy)`、`purpose_plan.goal_forecast(..., policy=policy)`、`continuation_search.search_forecast(..., policy=policy)` へ同じpolicyを渡す。検索のbeam順位、未来の反射継続、forecastの主義候補、実際の初手選択が同じ規則を使う。plannerはDecisionLoopが渡した最新contextを使い、未来の結果を実経験へ保存しない。ゲーム側が初期に選んだ目標/ルートはこの変更では作り直さない。

目的許容幅は別の制約。`max_regret=.15` は主義が有限でも目的の最低水準を優先し、大きな主義のための目的犠牲を除外する場合がある。`max_regret=2.` は[-1,1]の全目的結果を比較に戻し、既存の有限な主観スコアで判断する。後者は「勝てる手を必ず選ぶ」設定ではない。厳格なゲーム規則や絶対禁止まで主義の数値で代用しない。

[有限優先の固定比較・得失・残る不足](../evidence/finite_transfer/REPORT.md)。`python -m reflex.cli finite-transfer --output FRESH_ROOT` は、旧規則・有限優先+.15目的幅・有限優先+全目的比較を比較する。通常の反射継続を使った4ジャンルの比較と、以前失敗した共有列探索の診断を分離する。計画の保持や新しい評価係数を追加する比較ではない。
## 探索手法の組み合わせ

`search_forecast(..., depth=2, width=2, samples=12, policy=policy)` は、既存のビーム探索にランダムシューティングの候補を加える。`depth=0` と正の `samples` はサンプリングだけ、`samples=0` は従来の探索。期間・幅・深さ・サンプル数はゲーム側の知能予算で、性格の値と独立する。

各根について、モデル分岐を順番に使って合法な行動列を一様サンプリングし、その列を固定して全モデル分岐で評価する。分岐ごとに勝つ列を選び直さず、現在の実行は根の一手のみ。モデル内の合法手の発見には仮説状態を使うため、列のサンプリングは独立の検証標本ではない。実乱数・実相手の非公開方策は入力しない。

途中で不利な案を捨てるビームと、途中の採点で捨てず最後まで試すサンプリングは、同じ人格・目的評価で候補を共有する。通常反射の候補は全ての根に残すが、それより良い判断になる保証ではない。モデル誤りや標本への過適合は別の問題。サンプリングは実行のランダム化や経験学習ではない。仮想の欲求・意図・進捗は実個体へ書き戻さない。

`flow_rollout.rollout` の `schedule_sampler` は合法候補のtupleと未来stepだけを受ける、単一モデル分岐の候補生成用callback。別のschedule/continuationとの併用は拒否する。通常の反射やバッチ経路では使わない。新たな外部依存は不要。性能・得失は [hybrid_transfer](../evidence/hybrid_transfer/REPORT.md) を参照。


## 目的の最低水準と、候補の独立再評価

人格に合う採点と、目的をどれだけ犠牲にしてよいかは別の設定。既存の `max_regret=.15` は、最良のモデル目的見込みから.15以内の候補に人格採点を適用する。勝率15%という意味ではなく、ゲームが定めた[-1,1]の目標評価の差である。小さな目的犠牲は許せるが、大きな目的犠牲を伴う主義の貫徹は制約する。`max_regret=2.` はその制約を外す。いずれもBig Five、主義強度、欲求やPolicy係数を変更しない。目的見込み自体が間違っていれば最低水準も保証できない。

`search_forecast(..., validation_seeds=(...))` は、beamとランダムシューティングで保持した候補を、生成に使わなかったモデルseedでもう一度評価する。全合法根の反射baselineを含む全保持候補を再評価し、その結果で最終的な目的制約と人格採点を行う。候補の生成・途中の間引き・サンプリングは独立bankを見る前に完了する。予定手が実行不能なら、その時点の観測に基づく人格反射へ戻る規則は同じ。

任意のbankは1..MAX_OUTCOMES個の相異なる整数で、生成bankとの重複を拒否する。実行の未来乱数を渡さない責任はcallerにある。省略は従来と同じ選択・metadata。`continuation_search.validation` は両bank、再評価件数、保持候補の生成側の軌跡を記録し、通常のproposals/endpointには再評価側の軌跡を保存する。追加評価を元の `evaluated` やpilot件数へ混ぜない。

これは独立モデル標本による候補の再点検で、信頼区間・校正された勝率・未知の相手の正しいモデルを作る機能ではない。最終選択には再評価bankを使うので、そのbankも最終性能の未使用テストではない。別の実seedと未使用条件で実行結果を比較する必要がある。仮想経験は実記憶へ学習せず、人格・実観測のcheckpointを変更しない。継続計画の再評価にも適用できるが、その併用の実対戦成績は今回の比較外。

再現は `python -m reflex.cli competence-transfer --output FRESH_ROOT`。本比較では有限Policy、幅2、深さ2、サンプル12、期間6を維持し、全目的比較・.15目的制約・.15目的制約と独立bankの3方式を分離する。実行証拠がある環境では `python tools/audit_search_transfer.py FRESH_ROOT/competence_transfer`。[採否・人格差・費用・限界](../evidence/competence_transfer/REPORT.md)。


## 終局結果の時間選好を分離する任意比較

`purpose_plan.goal_forecast(..., settlement_weight='discounted')` が既定。モデル内の終局の目標値は、その決着stepに応じて既存のpatience由来の割引を掛け、一度だけ主観objectiveへ入れる。`'absolute'` はsuccess/failure/draw/scoredの終局値の時間割引だけを外す。正の結果へのconfidence、非終局runningの代理値、欲求/価値/行動style/費用の流れの時間割引は維持する。到達した報酬を残り期間の毎ターン収入にはしない。purpose corridorに使う終点評価の意味も変えない。

`continuation_search.search_forecast(..., settlement_weight=...)` は、途中のbeam間引き、shooting候補の順位、保持案・独立bankの最終再評価まで同じモードを渡す。未来の反射Policyを別モードに取り替えない。省略時のcontext、採点、metadataは従来と同じ。absolute指定だけ `subjective_settlement_semantics` を監査へ追加し、未知の文字列を拒否する。モデルの仮想経験を本人の実経験へ書き戻さない。

これはBig FiveやPolicy係数の変更ではないが、**終局結果への時間選好と報酬の表現を変える**。max_regret=2.を併用すれば大きな信条利益のための目的犠牲も許せる。一方、同じ勝利なら早く完了する圧力が弱まり、途中利益を得るために必要手を先送りする危険がある。資源が増えた遅い勝利を、自動的に賢さの改善と数えない。

再現は `python -m reflex.cli settlement-transfer --output FRESH_ROOT`。4既知ルール系の未使用パラメータ条件と、開発に使った物流の診断を分けて比較する。48局で診断の未達は直ったが、留保戦闘の勝利と生存が悪化したため一律採用しない。[得失・不採用理由・費用・限界](../evidence/settlement_transfer/REPORT.md)。完全軌跡を生成した後の `python tools/audit_search_transfer.py FRESH_ROOT/settlement_transfer` は保持案の分岐・最終選択・実行とcheckpointを再生する。最適な候補の発見や正しい相手モデルの証明ではない。
