# Reflex Persona Core

異なるゲームへ接続する、人格を条件としたNPC/CPU判断の実験的な土台。通常の判断はCPUの数値バッチ処理で行い、実行時のLLMを必須にしない。

目標は、同じルールと観測情報で、本人の目的を追いながら人間らしい利害やこだわりを見せる相手。相手への対策自体を報酬にしない。方針変更は、本人の目的への利益と変更の費用で判断する。

開発中のアルファ版。元のColab作業環境から、人格AIのソース・検査・再現用の集計を独立して切り出した公開プロジェクト。認証やモデル本体は不要で、ローカルCPUから試せる。書き出したソースの由来とハッシュは`SOURCE_SNAPSHOT.json`に記録する。

## 公開相手行動の予測診断

[固定軌跡での観測記憶の検査](evidence/observed_opponent/REPORT.md)：敵競合の確率0は仮説の行動範囲不足だった。個体別・相手別の有限記憶は全合法セルでの集計誤差を下げたが、一部人格の軌跡と局平均では悪化し、対照の全行動予測は一様より悪い。意思決定には未接続、既定値と人格コアは変更していない。

## 動かす

Python 3.10以上とNumPy。別の仮想環境で使う（import名`reflex`は他の同名パッケージと混在させない）。現在のWindows検証はPython 3.12。

```sh
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
python -m pip install -e .
python -m unittest discover -s tests -v
python -m reflex.cli demo
python -m reflex.cli judgment
python -m reflex.cli resource-world
python -m reflex.cli victory-routes
python -m reflex.cli contests
python -m reflex.cli combat
python -m reflex.cli npc-scale
python -m reflex.cli decision-loop
python -m reflex.cli validation
python -m reflex.cli purpose
python -m reflex.cli stability
python -m reflex.cli purposeful-wait
python -m reflex.cli recovery-options
python -m reflex.cli execution-alignment
python -m reflex.cli self-continuation
python -m reflex.cli root-collision
```

`judgment`は自作3ルール系の384エピソードを再生成する。結果は`reflex_artifacts/judgment`。大量の軌跡を作るため、Gitへは自動追加しない。麻雀は任意依存で、`python -m pip install -e '.[mahjong]'`の後に`python -m reflex.cli mahjong`。基本テストで未導入の麻雀検査はskipする。

`validation`はオークション/ハゲタカ、複数資源・複数勝利条件、戦闘、ギャンブルの共通検証。実際の目的進展に応じた有界な欲求不足を追加し、人格固定のまま既存判断と比較する。[結果・得失・未解決](evidence/validation/REPORT.md)を参照。慎重型の停滞や勝率差を解決済みとは扱わない。

`purpose`は戦闘で主義の安全と負傷の欲求を分け、選択後の実際の1ターン結果を既存の有界経験記憶へ返す候補を比較する。2地形・4目的・固定/攻撃/途中で変化する相手で実勝敗を確認し、他ゲーム性の対照も行う。候補は任意で、最低限の賢さを解決済みとは扱わない。[結果・採否・残る不足](evidence/purpose/REPORT.md)。

`stability`は、目的達成の見込み・共同先読み・人格選択を同じDecisionLoopへ接続する任意の候補。戦闘では味方の共同移動と目的担当/援護担当を評価し、資源管理では勝利ルートを先に選んで相手の手番・生産・維持費も予測する。696局の確認では戦闘の勝利64/192→130/192、資源管理19/24→24/24。4人格すべての勝利維持/改善、最弱人格改善、各相手群で対条件の悪化超過なしを満たした。速度費用は合否に使わず、既定の反射経路は変更しない。[結果・得失・限界](evidence/stability/REPORT.md)、[接続契約](docs/DECISION_LOOP.md)。

`purposeful-wait`は、目的のある停止と根拠の薄い停滞を区別する任意候補。待機理由・解除条件・有限の見直し猶予を持ち、実際の目的/準備の進展を確認する。戦闘では味方の準備も観測し、期限後の切替には元の案より目的見込みが改善する比較を求める。有効な防衛や確率的試行は残し、支持された代案がなければ無理に動かさず未解決として記録する。資源管理では先行ルートに隠れた別勝利ルートの有効準備も数える。人格・環境・相手の相性による得意不得意は残し、勝率の均等化を目指さない。[検証と得失](evidence/purposeful_wait/REPORT.md)、[接続契約](docs/DECISION_LOOP.md)。

`recovery-options`は、通常候補で打開できない時の追加案と、終点だけでなく途中の目的進展も評価する候補を分けて比較する任意実験。人格・主義を緩めず、支持された代案との比較を成立させる。戦闘と資源管理で予測と数手後の実績を同じ尺度で照合するが、実際の継続手段の変更も含むため、誤差をそのまま学習には使わない。720局の比較と得失は[結果](evidence/recovery_options/REPORT.md)、APIの境界は[接続契約](docs/DECISION_LOOP.md)。

今回の確認では追加案で期限到達停止1096→1021、戦闘勝利115→114/192。途中評価も加えた版は984停止・115勝で改善9/悪化9条件、移動失敗110→145。全面採用は見送り、追加提案/診断の部品と任意実験を保持する。合理的な待機と人格の得意不得意を残し、停滞数だけを減らして賢さの安定を達成したとしない。

`execution-alignment` は既知のseed180条件で予測した行動と実行を照合する診断。公開状態が同じ時の本人/相手の不一致、状態のずれ、終局、欠測を分ける。traceの既定はoffで、学習や選択は変えない。[診断の結果と限界](evidence/execution_alignment/REPORT.md)、[六つの開発基準と次の検証](docs/PROJECT_DIRECTION.md)。

`self-continuation` は、目的だけに沿う自己継続を、既存人格Policyの主義階層と近接得点帯に制限する任意候補の比較。実際には将来も先読みを使う本人を、即時反射だけで近似する限界を調べる。本人と相手の同一状態での不一致、実勝敗・停滞・移動失敗、trace off費用を分け、既定は変更しない。[検証結果と採否](evidence/persona_continuation/REPORT.md)。

`root-collision` はobjective自己継続をそのままに、共同plannerが辞退した時の味方の根の衝突だけを補完する既定offのablation。元の主義階層・0.025近接帯・進捗制約を保ち、衝突していない個体や別世界を変更しない。[機構と検証](evidence/root_collision/REPORT.md)。

## 依存関係

| 用途 | 依存 | 扱い |
| --- | --- | --- |
| 判断コア・数値バッチ | NumPy `>=1.26,<3` | 唯一の必須外部実行依存。[BSD 3-Clause](https://numpy.org/doc/stable/license.html) |
| 麻雀のルール環境 | RiichiEnv `==0.4.10` | 任意。[Apache 2.0](https://github.com/smly/RiichiEnv/blob/v0.4.10/LICENSE) |
| パッケージのビルド | setuptools `>=68` | ビルド時のみ |

麻雀の任意依存を導入すると、RiichiEnv経由でPyYAML・IPython・Pygmentsとその依存も入る。通常の人格判断にはLLM、PyTorch、Colab、Google Drive、外部APIを必要としない。外部ライブラリのソースやバイナリはこのリポジトリに同梱していない。それぞれのライセンスは外部ライブラリ自身に適用され、このプロジェクトの利用許諾を意味しない。

## 接続の境界

- コア：固定Big Five 5軸、動的欲求5領域、有限の価値10分類、候補採点と個体別の状態。
- ゲーム：本人が知る情報、合法手、本人の目的、欲求/価値の対象と効果、実行後に本人が観測できた結果。
- 推定：確定ルールとは区別する。効果の対象・時間幅・状況の同等性はゲームが指定する。未観測結果を0や成功確定へ置き換えない。

`reflex.core.Policy`と`reflex.runtime.Population`が通常の採点/数値バッチ。`reflex.judgment.OutcomeMemory`は選択済み実観測だけの条件別記憶。`AdaptivePopulation`は固定方法/状況・単一prior分岐の数値経験更新。`ReadControl`は読みの予算要求で、探索完了を意味しない。

記憶は既定64方法条件・直近4結果。2観測後から経験を混ぜ、重み上限0.75。大きい予測支持外の結果では古い証拠を弱める。これは未校正の工学的ヒューリスティックで、学習の正しさを保証する採用ゲートではない。状態チェックポイントの所有者・選択ticketを検査し、他個体や未選択の結果を混ぜない。

## 確認できた範囲

自作の循環同時対戦、4人資源干渉、分岐配送で、同じ人格Policyを維持して経験更新を比較。見込みを更新しない方式との96条件比較では、経験方式で78改善/18同じ/0悪化。相手の非公開な現在手や未観測の封鎖は判断入力にしない。全23,040選択は元ワークスペースで再生・再採点した。

経路の情報取得には時間と得点を払うが、その未来価値はゲーム側の代理評価。人間同等の知性、汎用的な情報価値学習、未見商用ゲームの強さ、万能なハメ防止は未実証。心理尺度はゲーム上の軸の参照で、心理学的妥当性を実証した人格モデルではない。

数値と費用の正本は[evidence/judgment/REPORT.md](evidence/judgment/REPORT.md)と[evaluation.json](evidence/judgment/evaluation.json)。コア/数値更新の速度にはゲーム進行・JSON変換・任意読み・描画を含めない。

## 複数資源・複数勝利条件の検証

`resource-world`はGUIなしの自作4人経済シミュレーション。食料/木材/鉱石/資金と研究/文化/軍事、設備投資、維持費、貯蔵上限、技術前提、有限領地への競争を扱う。科学/文化/勢力の勝利条件を別に定義し、全経路有効/科学のみ/文化のみで同じ人格Policyを比較する。Civや商用ボードゲームの再現ではない。

本人の1/4自己手番モデルと、専用経済方策の相手3人による実対戦。先読みでは相手を停止し、現在公開された供給だけを使う貪欲な自己継続案を作る。相手応答を含む探索ではなく、経済代理評価もゲーム側の手設計。勝率と勝利条件への進捗を分けて記録する。数値・得失の正本は[evidence/resource_world/REPORT.md](evidence/resource_world/REPORT.md)と[evaluation.json](evidence/resource_world/evaluation.json)。大量の再生用軌跡は公開版へ同梱しない。

## 勝利ルートと公開入札の読み合い

`victory-routes`は同じ経済ゲームで、人格・公開資源による有限勝利ルート選択と、選んだルートの専用進捗を接続する。32実対戦。特性と勝利名の固定対応を共通コアに追加しない。ルート選択に小さい差での変更を抑える記憶を持たせ、単発行動の記憶と分離する。ゲーム側で機会と価値を定義する必要は残る。[数値・得失](evidence/victory_routes/REPORT.md)。

`contests`はハゲタカと自作有限予算オークションの576実対戦。固定傾向/劣勢で変更/公開過去入札へ応答する相手に、読みなし・学習予測を常用・利益ゲートを比較する。全員が同一の公開前盤面で決め、公開後のみ既存の有限仮説を更新。真の相手制御型・当ラウンドの非公開入札・山順は判断へ渡さない。現在ラウンドの32共同標本と札/資金の機会費用代理評価で、全ゲーム勝率の探索ではない。[数値・得失と検証規約](evidence/contests/REPORT.md)。コア/runtimeは両方とも変更せず、LLM/学習済み汎用評価モデル/GPUの追加はない。軌跡は公開版に同梱しない。

## GUIなしの戦闘への再利用

`combat`は自作3対3の公開グリッド戦闘。3地形、射線/遮蔽物、射程/体力/弾薬/救急品、移動/射撃/装填/防御/味方回復を扱う。撃破・拠点確保・どちらか(OR)・両方(AND)の4条件を120対戦で比較する。拠点確保は3tick多数占有の実績。ANDは敵全滅と確保実績を両方満たし、自軍の生存も必要。味方最大3体は同じ選択前盤面で1つのPolicyバッチとして採点し、全員確定後に同時解決する。

経済ゲームの勝利名/評価を戦闘へ流用せず、共通`Policy`/`Population`/`routes.choose_route`をそのまま使う。射撃位置へ近づく意味や予想被害、生存/援護/目標進行の評価は戦闘側で定義する。相手の現在選択/実命中乱数は人格判断へ渡さない。この初期実験は公開ターン戦闘の接続試験。[結果・得失](evidence/combat/REPORT.md)。後続の`stability`で任意の共同先読みを追加したが、実時間の物理/照準/霧情報/戦闘相手モデルの学習は未実装。

## 大量NPC性能

`python -m reflex.cli npc-scale`で数値更新100/1000/10000体と、60/600/1800体を独立3v3戦闘に分けた全処理を測定する。[結果・費用内訳](evidence/npc_scale/REPORT.md)は、人格採点とゲーム状況変換の性能を区別する。巨大な一つの戦場や描画/物理/相手学習の性能を保証するものではない。

任意の`TiledPolicy`は共通Policyの中間配列を小分けにする。通常の候補数・人格・判断頻度・乱数を維持し、Populationの個別状態も同じ結果になる。入力/出力と更新コピーは人数に比例し、JSONによるゲーム接続の費用は別に残る。NumPyの逐次バッチ処理で、マルチコアやGPUの実装ではない。

ローカルi7-9700Fでは、1万体/12候補×2結果の数値step中央値は通常121.64ms、256体分割98.81ms（配列生成は別）。16候補×4結果では分割で速くならない。独立3v3戦闘の600体は地形距離の共有で1.52秒→1.29秒、まだ大量実時間戦闘の速度を満たすものではない。共有するのは静的地形だけで、人格・占有・体力・目的進捗は再評価する。

```python
from reflex.runtime import Population
from reflex.scaling import TiledPolicy
population = Population(contexts, policy=TiledPolicy(tile_size=1024))
decision = population.step(needs=needs, effects=effects, confidence=confidence)
```

ゲーム側は正規化した数値配列を渡す。行動ID・容量・人格の変更時はPopulationを再構築する。計測の再実行はCPUのみ、速度をテストの合格条件にしない。

## 次の検証

1. 更新前に予測し、結果の公開後に予測誤差を測る。仮説を作った同じ標本だけで正しさを認定しない。
2. 観測事実と、その事実から作った仮説を分ける。偶然の負け、推測の誤り、環境変化を区別し、疑わしい仮説の重みを下げる/保留する/失効させる。
3. 仮説の正しさと、方針変更の利益を別に評価する。変更に利益がない、または優位を維持しているなら今の方針を続ける。
4. 過去の公開行動から攻略を変える相手と比較し、失敗反復/過剰反応/目的停滞/人格の軸を同時に検査する。

任意の`DecisionLoop`で、観測→経験/予測→バッチ反射→公開結果→訂正を接続した。既存の適応方式を基線として維持し、追加の相手仮説は予測精度と即時判断利益を分けて検査する。初案の成績低下は棄却し、経験全体を予測誤差だけで止めない。所有者/ticket、欠測、checkpoint、読みの予算、ルートを同じ境界で扱う。現在は任意の未来選択も接続済み。想像した未来を実経験として保存せず、遅延成果による未来予測の学習は未完成。[接続契約と未完成部分](docs/DECISION_LOOP.md)、[比較・棄却した案](evidence/decision_loop/REPORT.md)、[安定性の確認](evidence/stability/REPORT.md)を参照。未見ゲームでの評価、対立した主義の交渉、汎用の賢さの保証は残る。[近い取り組みと位置づけ](docs/RELATED_WORK.md)を参照。

## 配布の扱い

現在は閲覧・検討のための公開で、利用ライセンスは付与していない。このプロジェクトのコードについて、利用・改変・再配布の一般的な許諾を出していない。公開されていることだけを理由に利用可能とは扱わない。GitHubの利用規約に基づく閲覧・forkや、適用法が認める例外は別に扱われる。[GitHubのライセンス説明](https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features/customizing-your-repository/licensing-a-repository)を参照。

この公開版にはColab認証、モデル本体、Drive個別ID、会話履歴、ローカルの個人ログを入れない。他プロジェクトのコードを取り込んだ版ではない。

### 事前固定した混雑条件の診断

既存default-offの根補完を、3種の対照配置48対と通常開始16対で比較。介入は9回あり局所衝突は減ったが、曝露条件不足と人格内の目的・停滞悪化のため昇格しない。[結果と再現手順](evidence/controlled_congestion/REPORT.md)。`python -m reflex.congestion_experiment` は味方の根、敵占有仮説、選択中の目的を別々に記録する評価用コマンドで、通常方策を変更しない。

### 公開履歴からの目的別相手予測

`GoalOpponentMemory` は公開盤面に対するeliminate/secureの仮説を分け、相手個体ごとの公開行動から有限の尤度重みを更新する。汎用の `GoalBeliefs` は戦闘規則を持たず、別ゲームadapterでも利用できる。新しい56局では目的の多様性が移動先予測を改善した一方、学習による全行動予測の改善と衝突先予測の悪化が併存した。[数値・費用・境界](evidence/goal_opponent/REPORT.md)。この予測比較は通常判断の変更でも、実対戦の強さの証明でもない。

### 目的別相手モデルを判断に使う限定実験

`TacticalControl(opponent_model='goal-uniform')` で、学習なしの目的別相手仮説を既存4標本・6stepの先読みに接続できる。通常は `'fixed'` のまま。新しい96実対戦では移動失敗は減ったが、元方式より勝利が減り留保の停止が増えたため、既定へ昇格しない。`'uniform-coverage'` との支持拡張対照、人格ごとの得失、因果照合、別枠の費用を[実対戦報告](evidence/goal_gameplay/REPORT.md)に保存した。
