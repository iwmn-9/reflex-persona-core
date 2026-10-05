# 判断・経験・相手読みを接続する共通ループ

800 episodes; 19200 integrated selections independently replayed.
固定人格4種、development seed0/1/10/11/20/21と未使用seed30/31。10/11と20/21は棄却した案で確認済みなので開発条件へ移した。循環対戦・4人資源干渉・配送、さらに反応型/雑音型の相手。選択前の予測を保存し、実際の公開結果で比較してから経験/仮説を更新する。新方式は固定方式だけでなく既存の適応方式と比較し、経験だけ/読みありを対応させる。

|split|game|mode|episodes|mean purpose score|losses|late losses|read adopted|revoked|
|---|---|---|---:|---:|---:|---:|---:|---:|
|development|cyclic_arena|fixed|24|-0.642|391|97|0|0|
|development|cyclic_arena|legacy_reflex|24|15.512|75|13|0|0|
|development|cyclic_arena|legacy|24|17.070|75|13|16|0|
|development|cyclic_arena|integrated_reflex|24|15.512|75|13|0|30|
|development|cyclic_arena|integrated_read|24|17.070|75|13|16|30|
|development|four_player_commons|fixed|24|5.300|600|225|0|0|
|development|four_player_commons|legacy_reflex|24|24.092|66|0|0|0|
|development|four_player_commons|legacy|24|24.092|66|0|0|0|
|development|four_player_commons|integrated_reflex|24|24.092|66|0|0|0|
|development|four_player_commons|integrated_read|24|24.092|66|0|0|0|
|development|branching_delivery|fixed|24|-6.400|480|168|0|0|
|development|branching_delivery|legacy_reflex|24|19.940|120|0|0|0|
|development|branching_delivery|legacy|24|19.940|120|0|0|0|
|development|branching_delivery|integrated_reflex|24|19.940|120|0|0|0|
|development|branching_delivery|integrated_read|24|19.940|120|0|0|0|
|development|cyclic_reactive|fixed|24|-0.670|427|122|0|0|
|development|cyclic_reactive|legacy_reflex|24|32.722|95|0|0|0|
|development|cyclic_reactive|legacy|24|33.089|93|0|5|0|
|development|cyclic_reactive|integrated_reflex|24|32.722|95|0|0|0|
|development|cyclic_reactive|integrated_read|24|33.089|93|0|5|0|
|development|cyclic_noisy|fixed|24|-0.533|367|97|0|0|
|development|cyclic_noisy|legacy_reflex|24|0.338|270|52|0|0|
|development|cyclic_noisy|legacy|24|5.639|219|36|77|0|
|development|cyclic_noisy|integrated_reflex|24|0.338|270|52|0|4|
|development|cyclic_noisy|integrated_read|24|5.639|219|36|77|6|
|held_out_seeds|cyclic_arena|fixed|8|3.100|106|25|0|0|
|held_out_seeds|cyclic_arena|legacy_reflex|8|23.250|13|3|0|0|
|held_out_seeds|cyclic_arena|legacy|8|23.742|13|3|3|0|
|held_out_seeds|cyclic_arena|integrated_reflex|8|23.250|13|3|0|11|
|held_out_seeds|cyclic_arena|integrated_read|8|23.742|13|3|3|11|
|held_out_seeds|four_player_commons|fixed|8|7.500|160|60|0|0|
|held_out_seeds|four_player_commons|legacy_reflex|8|22.590|20|0|0|0|
|held_out_seeds|four_player_commons|legacy|8|22.590|20|0|0|0|
|held_out_seeds|four_player_commons|integrated_reflex|8|22.590|20|0|0|0|
|held_out_seeds|four_player_commons|integrated_read|8|22.590|20|0|0|0|
|held_out_seeds|branching_delivery|fixed|8|-6.400|160|56|0|0|
|held_out_seeds|branching_delivery|legacy_reflex|8|19.940|40|0|0|0|
|held_out_seeds|branching_delivery|legacy|8|19.940|40|0|0|0|
|held_out_seeds|branching_delivery|integrated_reflex|8|19.940|40|0|0|0|
|held_out_seeds|branching_delivery|integrated_read|8|19.940|40|0|0|0|
|held_out_seeds|cyclic_reactive|fixed|8|-1.317|145|36|0|0|
|held_out_seeds|cyclic_reactive|legacy_reflex|8|33.560|27|0|0|0|
|held_out_seeds|cyclic_reactive|legacy|8|33.660|28|0|2|0|
|held_out_seeds|cyclic_reactive|integrated_reflex|8|33.560|27|0|0|0|
|held_out_seeds|cyclic_reactive|integrated_read|8|33.660|28|0|2|0|
|held_out_seeds|cyclic_noisy|fixed|8|1.875|131|36|0|0|
|held_out_seeds|cyclic_noisy|legacy_reflex|8|5.652|80|19|0|0|
|held_out_seeds|cyclic_noisy|legacy|8|3.365|87|22|11|0|
|held_out_seeds|cyclic_noisy|integrated_reflex|8|5.652|80|19|0|2|
|held_out_seeds|cyclic_noisy|integrated_read|8|3.365|87|22|11|2|

## 実際の目的得点の比較

|split|game|reference|mode|better|same|worse|mean delta|
|---|---|---|---|---:|---:|---:|---:|
|development|cyclic_arena|fixed|integrated_reflex|19|5|0|16.155|
|development|cyclic_arena|fixed|integrated_read|19|5|0|17.712|
|development|cyclic_arena|legacy_reflex|integrated_reflex|0|24|0|0.000|
|development|cyclic_arena|legacy_reflex|integrated_read|9|8|7|1.558|
|development|cyclic_arena|legacy|integrated_reflex|7|8|9|-1.558|
|development|cyclic_arena|legacy|integrated_read|0|24|0|0.000|
|development|four_player_commons|fixed|integrated_reflex|15|9|0|18.792|
|development|four_player_commons|fixed|integrated_read|15|9|0|18.792|
|development|four_player_commons|legacy_reflex|integrated_reflex|0|24|0|0.000|
|development|four_player_commons|legacy_reflex|integrated_read|0|24|0|0.000|
|development|four_player_commons|legacy|integrated_reflex|0|24|0|0.000|
|development|four_player_commons|legacy|integrated_read|0|24|0|0.000|
|development|branching_delivery|fixed|integrated_reflex|24|0|0|26.340|
|development|branching_delivery|fixed|integrated_read|24|0|0|26.340|
|development|branching_delivery|legacy_reflex|integrated_reflex|0|24|0|0.000|
|development|branching_delivery|legacy_reflex|integrated_read|0|24|0|0.000|
|development|branching_delivery|legacy|integrated_reflex|0|24|0|0.000|
|development|branching_delivery|legacy|integrated_read|0|24|0|0.000|
|development|cyclic_reactive|fixed|integrated_reflex|21|3|0|33.392|
|development|cyclic_reactive|fixed|integrated_read|21|3|0|33.759|
|development|cyclic_reactive|legacy_reflex|integrated_reflex|0|24|0|0.000|
|development|cyclic_reactive|legacy_reflex|integrated_read|3|20|1|0.367|
|development|cyclic_reactive|legacy|integrated_reflex|1|20|3|-0.367|
|development|cyclic_reactive|legacy|integrated_read|0|24|0|0.000|
|development|cyclic_noisy|fixed|integrated_reflex|10|6|8|0.872|
|development|cyclic_noisy|fixed|integrated_read|18|6|0|6.172|
|development|cyclic_noisy|legacy_reflex|integrated_reflex|0|24|0|0.000|
|development|cyclic_noisy|legacy_reflex|integrated_read|15|7|2|5.301|
|development|cyclic_noisy|legacy|integrated_reflex|2|7|15|-5.301|
|development|cyclic_noisy|legacy|integrated_read|0|24|0|0.000|
|held_out_seeds|cyclic_arena|fixed|integrated_reflex|6|2|0|20.150|
|held_out_seeds|cyclic_arena|fixed|integrated_read|6|2|0|20.642|
|held_out_seeds|cyclic_arena|legacy_reflex|integrated_reflex|0|8|0|0.000|
|held_out_seeds|cyclic_arena|legacy_reflex|integrated_read|2|5|1|0.492|
|held_out_seeds|cyclic_arena|legacy|integrated_reflex|1|5|2|-0.492|
|held_out_seeds|cyclic_arena|legacy|integrated_read|0|8|0|0.000|
|held_out_seeds|four_player_commons|fixed|integrated_reflex|4|4|0|15.090|
|held_out_seeds|four_player_commons|fixed|integrated_read|4|4|0|15.090|
|held_out_seeds|four_player_commons|legacy_reflex|integrated_reflex|0|8|0|0.000|
|held_out_seeds|four_player_commons|legacy_reflex|integrated_read|0|8|0|0.000|
|held_out_seeds|four_player_commons|legacy|integrated_reflex|0|8|0|0.000|
|held_out_seeds|four_player_commons|legacy|integrated_read|0|8|0|0.000|
|held_out_seeds|branching_delivery|fixed|integrated_reflex|8|0|0|26.340|
|held_out_seeds|branching_delivery|fixed|integrated_read|8|0|0|26.340|
|held_out_seeds|branching_delivery|legacy_reflex|integrated_reflex|0|8|0|0.000|
|held_out_seeds|branching_delivery|legacy_reflex|integrated_read|0|8|0|0.000|
|held_out_seeds|branching_delivery|legacy|integrated_reflex|0|8|0|0.000|
|held_out_seeds|branching_delivery|legacy|integrated_read|0|8|0|0.000|
|held_out_seeds|cyclic_reactive|fixed|integrated_reflex|7|1|0|34.877|
|held_out_seeds|cyclic_reactive|fixed|integrated_read|7|1|0|34.977|
|held_out_seeds|cyclic_reactive|legacy_reflex|integrated_reflex|0|8|0|0.000|
|held_out_seeds|cyclic_reactive|legacy_reflex|integrated_read|1|6|1|0.100|
|held_out_seeds|cyclic_reactive|legacy|integrated_reflex|1|6|1|-0.100|
|held_out_seeds|cyclic_reactive|legacy|integrated_read|0|8|0|0.000|
|held_out_seeds|cyclic_noisy|fixed|integrated_reflex|5|1|2|3.777|
|held_out_seeds|cyclic_noisy|fixed|integrated_read|6|1|1|1.490|
|held_out_seeds|cyclic_noisy|legacy_reflex|integrated_reflex|0|8|0|0.000|
|held_out_seeds|cyclic_noisy|legacy_reflex|integrated_read|3|2|3|-2.288|
|held_out_seeds|cyclic_noisy|legacy|integrated_reflex|3|2|3|2.288|
|held_out_seeds|cyclic_noisy|legacy|integrated_read|0|8|0|0.000|

## 得たもの・制限

DecisionLoopへ人格Policy、証明された無駄の除外、個体の意図、OutcomeMemory、採用/撤回EvidenceGate、ReadControl、所有者別HypothesisTracker、RouteStateを接続。個体の入力/モデル検証を全て終えてからバッチを確定し、誤った最後の個体が他個体を先に進めない。
既存の経験更新/忘却方式は維持する。予測誤差を根拠に経験全体を止める案は成績を悪化させたため棄却し、経験の比較誤差は診断情報に留める。個体のcheckpoint再開/未観測の明示破棄/公開手掛かりでの条件別失効を用意。
任意の読みは同じ結果対象だけを変更できる。未来・終局の値を即時結果で学習する接続は拒否する。未知の効果モデルは方法条件別の比較を必要とし、ある方法で正しいことから未試行の別方法を信用しない。条件付き効果が既知のルールである場合、相手の実際に公開された選択の予測を既存の直近公開履歴予測と比較する。一様予測だけを弱い基線にしない。
削ったもの/費用：全ての経験を採用待ちにする初案は、既存方式より対応が遅れたため廃棄した。既存の2観測後・上限0.75の経験混合を適応基線として残し、過去4件による忘却と支持外の大きな観測によるリセットを維持する。経験の比較誤差だけで基線を無効にしない。既知の条件付きルールは直近4公開手の控えめな応答分布で使い、追加の有限仮説は基線より予測がよい比較4件以上に加え、同じ公開応答での即時目的効果も比較4件以上でよい場合に切り替える。ここでは既知のルールによる仮想比較を明示し、未選択手の実観測として記憶へ入れない。追加仮説が悪化すると基線へ戻す。優位中も安い応答予測は公開後に採点するが、根の効果評価は読みが必要な場合だけ。常時探索を必須にせず1個体16root/response評価上限。CPUと入力契約は増える。部品の接続を、強さ/人間味の完成とは扱わない。
戦闘のRouteState→候補効果→Policy→公開結果も同じ経路へ接続。96対戦・5717実判断の結果が既存方式と一致。戦闘の将来価値を即時報酬で誤学習しないよう、この接続では経験更新対象を宣言せず、戦闘の賢さを改善したとは扱わない。
初案の既存読み方式との比較はseed0/1で9改善・9同じ・22悪化、10/11で10改善・6同じ・24悪化だった。固定方式だけに勝つ案は採用しない。結果を見た10/11を未見のまま扱わず、予測比較だけの第2案も600 episodeで既存方式より8条件悪化した。反射基線を変更せず、追加読みの判断利益も検査する第3案の未使用条件は30/31にした。
今回の対応した既存適応方式との有限条件ゲート：PASS。反射は既存経験のみ、読み付きは既存経験＋読みと比べる。FAILなら既定方式の置換を認めない。未見ルール/商用ゲームや戦闘の弱さを解決した証拠とは扱わない。

## 残る課題

- three rule mechanics, two extra rival behaviors; authored proxies, not broad human-level intelligence
- held-out seeds repeat mechanisms/profiles, are correlated, and do not prove novel-game transfer
- online one-step predictions only; delayed/terminal planning feedback is intentionally rejected
- CDF/Brier diagnostics and extra-model adoption/revocation thresholds are finite engineering controls, not universal accuracy/strength guarantees
- known conditional rules must be independently correct; game supplies equivalence, exact masks and public cues
- JSON lifecycle/state copies add CPU cost; existing Population fast path is unchanged
- future information value CHECK is game-authored, excluded from instant-outcome learning
- poor combat proxy/over-caution and real-time integration remain separate unresolved quality issues

## 最終ソースでの再確認

19200判断を最終ソースで公開結果の更新/checkpointまで再生した。測定時に実際に使った既定Policy/ReadControlをcheckpointへ明記してから再検査。追加仮説の実採用は0回、既知ルールで選択差を比較できたのは80回。既存適応方式との対応した320条件は全て同じ得点で、悪化を防いだ接続の確認であり、賢さが増した証拠ではない。改善のない基線維持を強さの向上とは報告しない。
