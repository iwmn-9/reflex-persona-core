# GUIなし戦闘への人格コア接続

120 actual 3v3 battles, 7088 decisions replayed; core/runtime unchanged.

公開9×5盤面、射程4、壁の射線遮断、遮蔽物/防御の命中率低下、HP9/弾3/救急品1。移動/射撃/装填/防御/隣接味方への回復を同時に選ぶ。撃破・拠点確保・どちらか(OR)・両方(AND)の4条件。拠点は3tick多数占有で確保実績を保存、ANDは確保実績と敵全滅を両方満たし、自軍生存も必要。

|goal|profile|games|wins|losses|draws|survivors|team p50 ms|
|---|---|---:|---:|---:|---:|---:|---:|
|eliminate|growth|6|4|1|1|2.33|7.93|
|eliminate|steady|6|0|0|6|2.67|6.90|
|eliminate|care|6|4|0|2|2.17|7.19|
|eliminate|ego|6|3|2|1|1.17|6.04|
|eliminate|neutral|6|2|0|4|2.00|6.69|
|secure|growth|6|4|2|0|3.00|7.16|
|secure|steady|6|0|6|0|3.00|7.22|
|secure|care|6|2|4|0|3.00|7.14|
|secure|ego|6|2|4|0|3.00|7.11|
|secure|neutral|6|2|4|0|3.00|7.09|
|either|growth|6|5|1|0|3.00|7.88|
|either|steady|6|1|5|0|3.00|6.99|
|either|care|6|1|5|0|3.00|7.36|
|either|ego|6|4|2|0|3.00|7.26|
|either|neutral|6|1|5|0|3.00|7.02|
|both|growth|6|4|0|2|2.83|7.08|
|both|steady|6|0|0|6|3.00|7.27|
|both|care|6|2|0|4|2.67|7.44|
|both|ego|6|4|0|2|2.50|7.11|
|both|neutral|6|2|0|4|2.83|7.23|

## 得たものと追加費用

経済ゲームの科学/文化/勢力を参照せず、既存の共通routes.choose_routeとPolicyをそのまま再利用。ルートの選択/切替と、合法な戦闘行動の人格評価を分離。生存/安全/承認の欲求、援護/権力/達成の価値をゲーム側から渡す。
味方最大3人の判断は同じ選択前盤面から1つのPolicyバッチで採点。全員が確定した後に移動/回復/同時射撃を解決。相手の現在選択や命中乱数はコアへ渡さない。全実判断でバッチ/個別が一致。
撃破の未達代理値は敵HP減少.65＋可射撃位置への近さ.35。射撃地点への静的地形距離は歩行できる場所と射程/射線から求める。接近を撃破達成には数えず、移動占有/相手移動/正しい終局EVは未計算。
追加したのは戦闘ルールと接続評価。射程/射線/地形距離/生存価値/拠点進行のゲーム別計算は必要。物理演算/描画/リアルタイム照準/霧情報/読み学習/チーム連携計画は未追加。

## 同一盤面の人格差

1/9 actor-scenes have action differences across four profiles; 36 batch/single matches.

|scene|actor|profile|route|action|
|---|---:|---|---|---|
|threatened_ally|0|growth|secure|move:3:3|
|threatened_ally|0|steady|secure|move:3:3|
|threatened_ally|0|care|secure|move:3:3|
|threatened_ally|0|ego|eliminate|move:3:3|
|threatened_ally|1|growth|secure|heal:1|
|threatened_ally|1|steady|secure|heal:1|
|threatened_ally|1|care|secure|heal:1|
|threatened_ally|1|ego|eliminate|heal:1|
|threatened_ally|2|growth|secure|move:2:4|
|threatened_ally|2|steady|secure|move:2:4|
|threatened_ally|2|care|secure|move:2:4|
|threatened_ally|2|ego|eliminate|move:2:4|
|safe_ally|0|growth|secure|guard|
|safe_ally|0|steady|secure|guard|
|safe_ally|0|care|secure|heal:1|
|safe_ally|0|ego|eliminate|guard|
|safe_ally|1|growth|secure|heal:1|
|safe_ally|1|steady|secure|heal:1|
|safe_ally|1|care|secure|heal:1|
|safe_ally|1|ego|eliminate|heal:1|
|safe_ally|2|growth|secure|move:2:4|
|safe_ally|2|steady|secure|move:2:4|
|safe_ally|2|care|secure|move:2:4|
|safe_ally|2|ego|eliminate|move:2:4|
|loaded_fight|0|growth|secure|move:3:3|
|loaded_fight|0|steady|secure|move:3:3|
|loaded_fight|0|care|secure|move:3:3|
|loaded_fight|0|ego|eliminate|move:3:3|
|loaded_fight|1|growth|secure|reload|
|loaded_fight|1|steady|secure|reload|
|loaded_fight|1|care|secure|reload|
|loaded_fight|1|ego|eliminate|reload|
|loaded_fight|2|growth|secure|move:2:4|
|loaded_fight|2|steady|secure|move:2:4|
|loaded_fight|2|care|secure|move:2:4|
|loaded_fight|2|ego|eliminate|move:2:4|

## 限界

- authored fully public turn-synchronous 3v3 test, not commercial/real-time combat
- rules/objective/LOS/path and effect proxies are combat-owned; not rules-only automatic generalization
- stationary/posture rival assumption and uniform-target incoming damage proxy can be wrong
- no opponent learning or coordinated joint action planning in this adapter
- simultaneous destination conflicts and ally kits can be wasted; measured instead of hidden
- route desirability mapping is game-authored, not a validated psychological trait-to-tactic mapping
- small development maps/seeds and fixed comparator; not broad combat intelligence evidence
- team timings include JSON/schema/route conversion but exclude enemy decisions/world/logging

## 射撃位置の評価追加前後（同じ開発条件）

|goal|games/version|before wins|after wins|before draws|after draws|
|---|---:|---:|---:|---:|---:|
|eliminate|30|0|13|30|14|
|secure|30|10|10|0|0|
|either|30|9|12|0|0|
|both|30|4|12|26|18|

接近/可射撃位置の評価をPolicyと参照相手の両方へ追加した比較。相手も行動が変わるため、人格側だけの単独改善と断定しない。初版120戦の資料/ソースはローカルバックアップへ保全。
安定志向は慎重すぎて1/24勝に留まる。移動先競合は全員合計1427体回。安全や目的進行の代理値、編隊の連携が品質面の残課題。コア再利用の接続試験を、人間同等/実ゲーム戦闘AIの完成とは扱わない。
