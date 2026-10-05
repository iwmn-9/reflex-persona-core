# 目的別の相手予測：行動の読みは改善、衝突先への学習効果は逆方向

基線はPR6統合済みmain `60b202b3902202a2e5ed32397fbf7a474c05bdd1`。PR6の `aedb130` とtree `89d51b141aac7ee377281fc0a111d2adfda9f932` が完全一致することを取得後に確認した。旧64局を再利用せず、新しい56局を実行した。**この予測比較では通常判断・人格・本人の経験更新・既定値を変更していない。**

## 作った能力と今回の判断

前周期は二つの仮説がともに射撃しか予測せず、中央確保への移動を見落とした。新しい `GoalOpponentMemory` は同じ公開盤面からeliminate/secureの両目的を評価し、各相手の過去の公開手がどちらの仮説に合うかを個別に蓄積する。ゲーム非依存の `GoalBeliefs` は既存 `HypothesisTracker` を所有者・局・相手・時間の境界で包む。戦闘の距離や勝利条件はadapterだけが持つ。

- 目的を等重みで混ぜるだけの方式は、敵が届く合法目的セルのBrier/log lossについて、一様分布と元仮説＋一様coverageの双方を、対照・留保、全事象平均・局等重み平均の全16比較で上回った
- さらに目的の重みを学習すると、全敵行動のBrier/log lossは等重みより28局改善・28局同値・悪化0。ただし同値28局は公開上の有効目的が一つしかなく、目的推定の試験になっていない
- **学習を加えると、目的セルの全事象平均は対照・留保の両方で等重みより悪くなる**。行動全体をよく説明する目的が、衝突という特定の判断に最適とは限らない
- 対照の全行動予測は依然一様分布より悪く、事前の次段階gateは32比較中4件で不成立。この凍結試験ではlive判断に接続しない。目的別予測の利益を勝率改善と読み替えない

これは供給した二つの手設計モデル内の尤度重み付けであり、相手の真の目的や人格を同定したという主張ではない。別の、事前固定するdefault-off実対戦で「学習なしの目的多様性」が判断に役立つかを試す余地は残る。今回の結果を後から調整してその試験の成功扱いにはしない。

## 予測と隔離

各目的の既存 `tactical_scores` に温度0.04のsoftmaxを使い、0.08の一様平滑化を行う。過去の公開手の尤度を使う既存trackerはretention=0.9、responsive=False。trackerのheuristic confidenceは予測に使わない。公開の達成済み条件で有効目的を絞り、その中で重みを正規化する。

Fは元の二つの**共同**仮説、Uは各敵の合法手一様分布の積。Gは目的別分布を各敵について混ぜた後、その積を取る。敵同士の独立性はG/Uの明示的な仮定であり、協調を学習していない。

|方式|共同分布|
|---|---|
|uniform|U|
|fixed|F|
|fixed_coverage|0.8F + 0.2U|
|goal_uniform|0.8G（有効目的等重み）+ 0.2U|
|goal_adaptive|0.8G（公開履歴の尤度重み）+ 0.2U|

固定目的にも平滑化があるため、固定仮説＋coverageとの違いは目的仮説の支持だけではなく分布形状も含む。等重みとの比較が学習分の切り分けである。目的重みを実controllerの種類・seed・現在手・将来結果から計算する経路はない。モデル内の仮想結果を学習していない。

局・観測者・相手ごとに状態を分離。全員の予測を作ってから現在手を読み、採点してから一括revealする。公開pre-stateのhashと連続tickが一致しないrevealは拒否する。無効な部分公開、tracker更新中の例外でも全相手の更新をrollbackする。返した予測は後から変わらない。死亡した観測者の更新は止まる。

## 新しい56局

[design.json](design.json)、[preregister.json](preregister.json) を2026-10-05 21:29:03 UTCに凍結してから実行した。

- 対照24局：既存3合流配置 × secure/both × 4人格。1701/1702を条件と人格で交互に割り当て、各人格・各条件で陣営を均衡。24tick上限
- 留保32局：open/secure/reference、choke/eliminate/switch、open/either/raider、choke/both/reference × 4人格 × 1701/1702の両陣営。40tick上限
- 旧64局とはIDが完全に異なる。配置・相手方策の族は既知であり、未見環境への一般化や外部検証ではない
- 元のobjective自己継続、recovery_options、実行後だけの経験更新を使う。全方式を同じbaseline軌跡で予測比較し、候補による実対戦はこの試験に含めない

1,011tick、全合法移動候補8,859件、敵行動2,377件。占有ラベルは実移動の成功位置ではなく、相手がそのセルへ進もうとした意図。対照3,256候補中33陽性、留保5,603中49陽性。敵が届く候補は235/353件。構造的な陰性を含む全候補だけで利益を判断しない。各敵行動は最小IDの生存観測者で一度だけ数える。

## 敵が届く目的セルの結果

Brier/log lossは低い方が良い。ここは全事象平均。局内のtickや重なる合法目的セルは独立標本ではない。

|分割|方式|Brier|log loss|
|---|---|---:|---:|
|対照|uniform|0.120668|0.403827|
|対照|fixed_coverage|0.109925|0.398416|
|対照|goal_uniform|0.097384|0.324635|
|対照|goal_adaptive|0.100261|0.333918|
|留保|uniform|0.091484|0.327878|
|留保|fixed_coverage|0.103408|0.322994|
|留保|goal_uniform|0.083021|0.275838|
|留保|goal_adaptive|0.090818|0.294032|

局を等重みにすると、対照のreachable Brierは等重み0.122863→学習0.120992に改善し、全事象平均と向きが逆になる。留保は0.079781→0.079958に悪化、log lossは0.268181→0.260390に改善する。敵が届く候補がない3局ずつはこの平均から除き、誤差0の局として水増ししていない。全合法候補の局等重みBrierは対照・留保とも学習で悪化する。

集計の改善で人格内の悪化を隠さない。対照careのreachable Brierはfixed_coverage 0.100429、等重み0.102720、学習0.119722。留保steadyは一様0.088220、等重み0.120536、学習0.134667。留保egoの学習0.055585も一様0.046073より悪い。全人格・条件・局の値を保存した。

校正も一様より常に良いわけではない。対照reachable ECEは一様0.02058、等重み0.06530、学習0.07914。留保は0.16156、0.11135、0.10642。固定10区間の記述値であり、校正済みの認定ではない。全行動のone-vs-rest ECEは一様だと区間内で機械的に相殺し得るため、それだけで識別の質を比較しない。

## 全敵行動と4件の不成立

多クラスBrierは予測ごとに全合法手の二乗誤差を合計する。対照776予測、留保1,601予測。

|分割|方式|Brier|log loss|
|---|---|---:|---:|
|対照|uniform|0.795547|1.635314|
|対照|fixed_coverage|1.247698|2.574719|
|対照|goal_uniform|0.906867|1.963518|
|対照|goal_adaptive|0.885069|1.909770|
|留保|uniform|0.795154|1.618914|
|留保|fixed_coverage|0.852593|1.727092|
|留保|goal_uniform|0.648154|1.281667|
|留保|goal_adaptive|0.632232|1.246198|

不成立の4件はすべて対照の全行動対一様比較。上表の全事象Brier/log lossの2件と、局等重みBrier 0.899625対0.796288、log loss 1.991603対1.643226の2件。その他28件のgate比較は通過した。結果後の重み変更・seed追加・有利な除外はしていない。

元のFの全行動log lossは対照595件・留保817件が支持0で無限。JSONでは真の平均をnull、無限件数を別欄に残す。epsilon=1e-12でclipした値は別名であり真のlog lossではない。

全行動について学習は等重みより各分割で改善した。ただし切替相手はeliminate単一目的の条件なので、goal_adaptiveとgoal_uniformはそこで同じ。今回の結果を、切替検知や任意の新戦術への追従の実証とは呼ばない。非戦闘の合成2行動adapterでは反復・反転・再出現に応じた重み変化を確認したが、実ゲームの汎用的な戦術学習の証明ではない。

## 費用と独立確認

- 500 tests、486 passed、14 optional skips、失敗0。新規34検査。compileall・diffチェック成功
- 全56局・1,011ルール遷移・2,915進捗更新・5,292合法実手・3,488保存仮説先頭を照合
- 独立レビューは全8,859占有予測と2,377行動予測、1,011belief遷移を公開軌跡から別の尤度・混合・積の式で再計算。最大確率誤差2.22e-16。集計・校正・局等重み・32gate比較一致。別の合成3,196確率検査と6,000共同占有列挙も成功。[independent_review.json](independent_review.json)
- 元のreflexファイルは変更せず、新規3モジュールのみ。共通Policy/runtime/DecisionLoopから新予測器をimportしていない。source/tests/design/inputは凍結前後で一致
- 全生存観測者の予測tick中央値1.480ms、reveal0.536ms。別の同じ公開fixture・観測者3人・全合法移動照会では元Fのp50 0.353ms/p95 0.601ms、目的予測1.622ms/2.412ms。中央値比4.59倍。元Fには目的expertやstate hashがなく、全ゲームoverhead比ではない
- 相手3人×目的2個のlog重みは6数値。ただし既存trackerの最大192観測ID/18履歴も保持。16合成reveal後のPython再帰sizeは観測者1人11,068 byte、pending予測あり16,930 byte。JSON最大689 byteは実heapではない
- 汎用推定部だけの独立観測者1/32/128人のp50は0.052/1.837/7.591ms。公開expertは事前供給。巨大な同一戦場、大量NPCの全処理、60Hzの保証ではない。[memory_cost.json](memory_cost.json)

## 再現

    python -m unittest discover -s tests -v
    python -m reflex.goal_opponent_experiment preregister --output reflex_artifacts/new_goal_prediction
    python -m reflex.goal_opponent_experiment run --output reflex_artifacts/new_goal_prediction --workers 4
    python -m reflex.goal_opponent_experiment evaluate --output reflex_artifacts/new_goal_prediction
    python tools/goal_opponent_cost.py --output reflex_artifacts/new_goal_prediction/memory_cost.json
    python tools/summarize_goal_opponent.py --input reflex_artifacts/new_goal_prediction --output /path/to/new_summary

凍結したsource snapshotで実行する。既存preregisterや原始出力は上書きしない。公開[evaluation.json](evaluation.json)と[per_game.json](per_game.json)は、繰返しの校正binだけを省いた投影で、損失・分母・無限件数は残す。全体の校正binは保持。元出力と投影のhashは[projection.json](projection.json)、約26MBの原始軌跡・予測・belief等のhashは[verification.json](verification.json)に保存した。
