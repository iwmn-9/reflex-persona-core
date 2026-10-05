# 賢さの安定：目的の下限・共同予測・経験との分離

696 games; seeds [90, 91]; 18308 real transitions replayed.

現在は機能を足して成立させる段階。速度・費用は合否に使わない。勝利/人格/手段の失敗を評価する。

|種別|方式|人格|勝利/局数|敗北|味方の移動先競合|実移動失敗|共同案採用tick|辞退tick|
|---|---|---|---|---|---|---|---|---|
|combat|baseline|growth|27/48|10|131|260|0|0|
|combat|baseline|steady|1/48|24|192|192|0|0|
|combat|baseline|care|10/48|19|134|142|0|0|
|combat|baseline|ego|26/48|15|90|116|0|0|
|combat|experience|growth|17/48|13|247|348|0|0|
|combat|experience|steady|11/48|21|189|214|0|0|
|combat|experience|care|12/48|21|191|220|0|0|
|combat|experience|ego|20/48|16|143|180|0|0|
|combat|deliberation|growth|31/48|8|0|30|879|0|
|combat|deliberation|steady|33/48|5|12|36|839|79|
|combat|deliberation|care|32/48|6|55|99|631|316|
|combat|deliberation|ego|34/48|5|0|35|905|0|
|resources|baseline|growth|5/6|0|0|0|0|0|
|resources|baseline|steady|6/6|0|0|0|0|0|
|resources|baseline|care|6/6|0|0|0|0|0|
|resources|baseline|ego|2/6|0|0|0|0|0|
|resources|deliberation|growth|6/6|0|0|0|0|0|
|resources|deliberation|steady|6/6|0|0|0|0|0|
|resources|deliberation|care|6/6|0|0|0|0|0|
|resources|deliberation|ego|6/6|0|0|0|0|0|
|auction|baseline|growth|3/6|0|0|0|0|0|
|auction|baseline|steady|0/6|0|0|0|0|0|
|auction|baseline|care|2/6|0|0|0|0|0|
|auction|baseline|ego|1/6|0|0|0|0|0|
|hagetaka|baseline|growth|1/6|0|0|0|0|0|
|hagetaka|baseline|steady|3/6|0|0|0|0|0|
|hagetaka|baseline|care|0/6|0|0|0|0|0|
|hagetaka|baseline|ego|1/6|0|0|0|0|0|
|gambling|baseline|growth|4/6|0|0|0|0|0|
|gambling|baseline|steady|3/6|0|0|0|0|0|
|gambling|baseline|care|4/6|0|0|0|0|0|
|gambling|baseline|ego|4/6|0|0|0|0|0|

対条件の勝利（改善/悪化/同じ）：
- combat/experience/raider: 8/8/48。
- combat/experience/reference: 6/11/47。
- combat/experience/switch: 9/8/47。
- combat/deliberation/raider: 27/3/34。
- combat/deliberation/reference: 24/6/34。
- combat/deliberation/switch: 29/5/30。
- resources/deliberation/game-owned: 5/0/19。

品質ゲート: {'combat': {'no_persona_win_drop': True, 'weakest_persona': 'steady', 'weakest_improved': True, 'no_opponent_family_paired_regression': True, 'passed': True}, 'resources': {'no_persona_win_drop': True, 'weakest_persona': 'ego', 'weakest_improved': True, 'no_opponent_family_paired_regression': True, 'passed': True}}。全人格の勝利維持・最弱人格改善・各相手群で悪化超過なし。差を縮めるだけでは合格にしない。

得たもの: 共通DecisionLoopに共同先読みを接続し、未来の想定と即時の実経験を分離。目的達成見込みが最良案から大きく外れる候補を避け、その帯の中では既存人格Policyの最大主義/欲求/リスクで選ぶ。味方の進路競合を予測し、目的担当と援護担当を組み合わせる。

制限したもの: 目的の見込みが設定した帯から外れる手段の自由。帯内での損失やこだわりは残す。最大主義を他者の多数決で押し切る協力案は採用しない。人格別の勝率補正はない。

維持したもの: 固定Big Five/有限価値、現在の欲求、合法手/同一情報、個体別の経験と選択ticket。未来の仮説・未選択結果・共同成果の因果功績を実観測として学ばない。

検証ゲーム専用の部分: 戦闘の移動/射撃/拠点と、経済の生産/資源/勝利を評価するアダプタ。共通の選択/人格/目的帯/現在経験との分離はdeliberation.pyとDecisionLoopで共用。具体的な戦術/価値尺度はゲームが供給する。

残る不足: 未見ゲームの評価、未来予測の誤りを遅延結果で学ぶ契約、相手の傾向の学習、互いに両立しない人格の交渉、役割の継続と切替。今回の品質ゲートと人格別結果を超えて完成を主張しない。

Colab/GPU/Drive/LLM教師/訓練なし。ローカルCPUの独立対戦を複数workerで進行。速度費用を理由に機能を削る合否判定はしていない。

## 今回の確認結果の読み方

戦闘は元の反射64/192勝→共同先読み130/192勝。成長27→31、慎重1→33、配慮10→32、自己優先26→34（各48局）。
単なる即時経験候補は60/192勝であり、経験の更新を足すだけでは改善しなかった。共同先読み候補も経験を含むため、個々の追加機能の独立した寄与を証明する要因実験ではない。

味方の移動先競合547→67体回、実際の自軍移動失敗710→200体回。共同案が辞退された慎重型79tick、配慮型316tickがあり、競合を完全になくしたとは扱わない。
reference相手24改善/6悪化、raider相手27/3、途中で方針を変えるswitch相手29/5（各64条件）。全条件で悪化なしという意味ではない。

資源管理は19/24→24/24勝。成長5→6、慎重6→6、配慮6→6、自己優先2→6（各6局）。この相手と小標本では上限に達しており、より厳しい相手や長い局面で賢さが保たれるかは未確認。

棄却した開発案: 資源管理で本人の未来行動を単純な貪欲代理で進めた案は、開発seed61/mixedの4人格で元3勝→1勝。ルート予測を足すだけでも直らなかった。固定人格Policyによる本人継続と、近い/遠い相手の経済仮説へ置換した案は同条件4勝。開発条件の結果は留保seed90/91の確認結果へ混ぜない。

人格差の補足確認: 同じ公開盤面、同じseed、同じprinciple/safety優先モードを固定し、人格5軸と有限価値だけを変えた6場面中、5場面で選択した根の組み合わせが異なった。本人別に候補を作ることも含む層全体の差であり、共通候補だけでの独立した性格効果、長期的一貫性、人間らしさの証明ではない。

|場面|tick|成長|慎重|配慮|自己優先|
|---|---|---|---|---|---|
|choke/both|0|move:2:0, move:2:2, guard|move:2:0, move:2:2, guard|move:1:1, move:2:2, move:1:3（反射へ復帰）|move:2:0, move:1:3, move:2:4|
|choke/both|6|move:2:0, move:3:2, move:1:4|move:2:0, move:3:2, move:1:4|move:2:0, move:3:2, move:1:4|move:2:0, move:3:2, move:1:4|
|open/either|0|move:2:0, move:2:2, move:2:4|move:2:0, move:2:2, move:2:4|guard, move:2:2, guard（反射へ復帰）|move:2:0, move:2:2, move:2:4|
|open/either|6|move:3:0, move:0:3, move:1:4|move:3:0, move:0:3, move:1:4|guard, guard, guard（反射へ復帰）|move:3:0, move:0:3, move:1:4|
|open/eliminate|0|move:2:0, move:2:2, guard|move:2:0, move:2:2, move:1:3|move:2:0, move:2:2, move:2:4（反射へ復帰）|move:0:0, move:2:2, move:2:4|
|open/eliminate|6|move:3:1, move:1:2, move:1:3|move:3:1, move:1:2, move:1:3|move:2:0, move:1:2, move:2:4（反射へ復帰）|move:2:2, move:1:2, move:2:4|

方向性: 固定人格を変えて勝率をそろえるのではなく、本人の目的を果たせる手段の見通しを足した。合理性の帯の中で価値と欲求を使う。共通層はゲーム固有の戦術や目的尺度を自動発見しない。

次の急所: 共同案を受け入れられない時も不合理な停滞へ戻らない仕組みと、未来予測が実際の目的達成に役立ったかを遅延結果で検証・訂正する接続。
