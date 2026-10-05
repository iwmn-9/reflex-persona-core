# 人格を保った目的達成と経験接続の確認

480 games; 11132 actual transitions replayed. Seeds [70, 71].

戦闘候補は、主義の安全を生存損失、負傷を欲求/リスクに分け、現在の同時射撃の不確実性を評価する。実際に選んだ手の1ターン後の公開結果だけを既存の有界記憶へ返す。未来成果や未選択の結果は学習しない。

|種別|方式|人格|勝利持分/局|敗北|経験使用判断|失敗移動|
|---|---|---|---|---|---|---|
|combat|baseline|growth|31/48|9|0|208|
|combat|baseline|steady|3/48|24|0|123|
|combat|baseline|care|21/48|12|0|205|
|combat|baseline|ego|32/48|7|0|134|
|combat|candidate|growth|27/48|7|950|329|
|combat|candidate|steady|16/48|16|1021|244|
|combat|candidate|care|23/48|11|909|193|
|combat|candidate|ego|31/48|7|879|128|
|auction|baseline|growth|3/3|0|0|0|
|auction|baseline|steady|0/3|0|0|0|
|auction|baseline|care|1/3|0|0|0|
|auction|baseline|ego|1/3|0|0|0|
|auction|pressure|growth|3/3|0|0|0|
|auction|pressure|steady|0/3|0|0|0|
|auction|pressure|care|1/3|0|0|0|
|auction|pressure|ego|1/3|0|0|0|
|hagetaka|baseline|growth|0/3|0|0|0|
|hagetaka|baseline|steady|3/3|0|0|0|
|hagetaka|baseline|care|0/3|0|0|0|
|hagetaka|baseline|ego|0/3|0|0|0|
|hagetaka|pressure|growth|0/3|0|0|0|
|hagetaka|pressure|steady|2/3|0|0|0|
|hagetaka|pressure|care|0/3|0|0|0|
|hagetaka|pressure|ego|0/3|0|0|0|
|resources|baseline|growth|2/3|0|0|0|
|resources|baseline|steady|3/3|0|0|0|
|resources|baseline|care|3/3|0|0|0|
|resources|baseline|ego|0/3|0|0|0|
|resources|pressure|growth|2/3|0|0|0|
|resources|pressure|steady|3/3|0|0|0|
|resources|pressure|care|3/3|0|0|0|
|resources|pressure|ego|1/3|0|0|0|
|gambling|baseline|growth|1/3|0|0|0|
|gambling|baseline|steady|1/3|0|0|0|
|gambling|baseline|care|1/3|0|0|0|
|gambling|baseline|ego|1/3|0|0|0|
|gambling|pressure|growth|1/3|0|0|0|
|gambling|pressure|steady|1/3|0|0|0|
|gambling|pressure|care|1/3|0|0|0|
|gambling|pressure|ego|1/3|0|0|0|

対条件での勝利持分（改善/悪化/同じ）：
- combat/raider: 6/7/51。
- combat/reference: 10/5/49。
- combat/switch: 9/3/52。
- auction/game-owned: 0/0/12。
- hagetaka/game-owned: 0/1/11。
- resources/game-owned: 1/0/11。
- gambling/game-owned: 0/0/12。

確認ゲート: {'default_changed': False, 'confirmation_gate_passed': False, 'weakest_persona': 'steady', 'weakest_improved': True, 'no_persona_win_drop': False, 'no_opponent_family_paired_regression': False}。既定は変更しない。最弱人格の改善、各人格の勝利維持、各相手群の対条件で悪化超過なしを同時に要求する。

得たもの: 戦闘で即時の実経験を判断/方針と接続した比較可能な候補。途中で相手の方針が変わる対戦、同時移動の失敗、各人格の実勝敗を含め、評価の誤りと経験接続を一緒に検証できる。共有Policy/人格/主義の優先規則は変更しない。

捨てた案: 停滞時に優先モードだけを周期的に再抽選する案と、目標進行を欲求へ足すだけの案は開発試験で安定改善せず、今回の候補から外した。防御を違法扱いしたり、慎重な人格を強制解除したりはしない。開発試験の60/61は確認用の70/71と分けた。

費用: 即時の不確実な結果が最大8分岐、個体別の有界記憶、JSON/契約検査、実観測の更新。大量NPCの性能合格を意味しない。

未解決: 本人の目的への長期価値、経験からの相手行動モデル改善、味方の選択の競合、遅延成果の帰属、どの人格でも最低限賢いという安定性。勝率差だけを縮める修正は採用しない。

他ゲーム性の確認は共通コアを保つ対照試験。戦闘の生存評価をオークション等へコピーしていない。正確なギャンブル確率を少数の実運で上書きせず、資源モデルの未来収入を即時実観測にしない。

Colab/GPU/Drive・LLM教師・訓練は使わず、ローカルCPUで実行。

確認結果の読み方: 戦闘の総勝利は87/192→97/192、慎重型は3/48→16/48。成長型31→27、利己型32→31と退行があり、攻撃型相手の対条件も6改善/7悪化。したがって全面採用の確認ゲートは不合格。人格を保って慎重型が勝つ局が増えたことは収穫だが、賢さの安定を解決したとは扱わない。

対照ゲーム: オークションの勝利12対は同じ。資源では1改善/0悪化、ハゲタカでは0改善/1悪化、ギャンブルは12対同じで負EV賭け/破産0。既存の欲求不足もゲームをまたいで常に改善するわけではなく、任意のままとする。戦闘候補の改善が他ジャンルへ自動転移したという結果ではない。

次に満たすべき条件: 生き残るための防御と目的を捨てる停滞を、将来の目標達成可能性で区別する。同時移動では、味方が同じ進路を選んで互いに止まることを予測と選択へ返す。実経験は共同結果の相関なので、自分の手の効き目を誤って学ぶ危険が残る。これらを改善し、最弱人格の改善と他人格の強さ維持を同時に確認する。

今回の戦闘判断+実観測更新時間（対戦ごとの中央値の中央値、最大3味方/1tick）はbaseline 14.59ms、candidate 32.21ms。ゲーム進行・監査用の再採点・描画は含めず、数値Populationの大量NPC性能とは別の測定。
