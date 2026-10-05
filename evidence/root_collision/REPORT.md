# 根の衝突補完：機構は分離できたが、留保標本は介入0

基線はPR3統合後のmain `3ca35a3f0eccc8665c7f39596b52abe8e3ff3149`。`TacticalControl(root_completion=True)` は既定offの比較用部品。objective自己継続、相手仮説、目的採点、共通Policy/Population、実経験の学習規則は変えない。

## 結論

**既定へ昇格しない。** 新しく予約した32対では両方式の全実軌跡が一致し、21勝ずつ、味方衝突0、全移動失敗4/864だった。planner辞退は32tickあったが、その根に味方衝突がなかったため補完を一度も採用しなかった。この標本から「害がない」「効く」「従来のpersona-bandの利益を再現した」とは判断できない。0件を見て好都合なseedへ差し替えず、検出対象への曝露不足として報告する。

旧標本からは、根の衝突が起こる実際の経路と制約内の代案を確認できた。それは開発用の機構診断であり、新しい留保性能の証拠と混ぜない。

## 先に確認した機構

PR3の保存軌跡にあるobjective方式では、味方衝突は12tick・25移動、すべてcare。全12tickで共同plannerは「purpose corridor内に全個体の主義階層が両立する案がない」と辞退していた。採用した共同案の根には味方衝突がなかった。

共同案生成は重複移動先を除く。しかし最終的に共同案を辞退すると、DecisionLoopは別々に選んだ本人の即時根をcommitする。その組合せには衝突確認がなかった。旧tickの再構成では、衝突した個体25件は原則modeであり、最大benevolence効果は0だった。この即時主義階層は通常のlegal rootsを狭めていない。これらの衝突を「careには衝突が必要」と解釈しない。

- 11/12tickは、衝突していない個体を固定したまま、元の即時主義階層・0.025帯・最終progress/waste mask内で非衝突にできた
- 既存progress証拠がblockedとする別の手段を新しく導入しない制約も加えて、同じ11tickで可能。必要な変更は合計12個体
- 残る `choke/both/care/seed282/tick24` は、progress maskが個体0と2をともに `move:3:2` へ制限する。この衝突を消すために階層や進捗制約を下げない
- proof不成立で旧根を維持した局面では、literalな最終maskだけを使うと、別の期限切れguardを新しく選べる罠があった。候補では元のtier/bandを決めてからblocked代案を除き、採点し直して階層を下げない

全衝突tickの元の反射選択と48件の保存contextを厳密に再現した。非衝突tickの6件はproof前のbaseline drawを保持する経路とfinal-maskだけの再採点が異なるため、すべての辞退tickを再構成できたとは主張しない。詳細とsource hashは [diagnosis.json](diagnosis.json)。persona-bandでの旧衝突減少25→2はこの機構発見の手掛かりであり、この局所補完の因果的な性能実証ではない。

## 候補の境界

最終planner辞退・progress proof再考後だけで動く。元のPolicy tierと0.025近接帯を、本人のeffective context、最終progress mask、exact-waste guardから決める。元の根がその帯の外なら全体を変更しない。progressがblockedと記録した新しい根は候補から除く。

同じ公開盤面・episode・明示された味方チームだけを対象に、現在衝突している個体の根を組み合わせる。非衝突の本人の選択は固定。変更人数を最小にし、その中で元のPolicy得点合計、canonicalな行動ID順で選ぶ。全体を非衝突にできなければ元の選択を残す。最大3個体の現在legal rootsに限り、モデル遷移を追加しない。

相手の未公開現在手・実乱数・実controllerを読まない。危険な移動一般を抑制せず、別陣営や独立世界のownerを協調させない。変更した本人のintent/ageと実際に選んだ即時rootだけを通常の原子的commitへ渡す。仮想経験は保存しない。別reader/predictorとの組合せは、未検証rootの承認を迂回しないよう拒否する。

## 事前に固定した新しい32対

4人格 × seeds391/392 × 4条件群。条件群はopen/secure/reference、choke/eliminate/switch、open/either/raider、choke/both/reference。両陣営・全4目的・2地形・3相手群を含むが、全因子の総当たりではない。地形と目的・相手を独立に推定する設計でもない。

方式間で同じ世界乱数・相手・即時学習・追加recovery案を使う。原案 [design.json](design.json) は実装前、最終候補・64件のjob・source hashは結果前の [preregister.json](preregister.json) に固定した。旧180/281/282は開発条件で、391/392とは別にする。留保結果を見た閾値調整や追加seed探索は行わない。

|留保各32局|baseline|root-completion|
|---|---:|---:|
|勝利 / 敗北 / 両者未勝利|21 / 1 / 10|21 / 1 / 10|
|実tick / 個体判断|654 / 1,876|654 / 1,876|
|最終eliminate進捗の局平均|0.521803|0.521803|
|最終secure進捗の局平均|0.899740|0.899740|
|移動失敗 / 移動試行|4 / 864 (0.463%)|4 / 864 (0.463%)|
|味方だけ / 相手だけ / 両者競合の失敗移動|0 / 4 / 0|0 / 4 / 0|
|期限切れ停止 / 未解決停止の個体選択|268 / 268|268 / 268|
|適用中progress maskに反する根|0|0|
|planner辞退tick|32|32|
|衝突補完の採用tick / 変更個体|該当なし|0 / 0|

停止268件は個体判断の14.286%。ここでの「未解決」は既存progress/proof診断で、実際に全手段が無意味だと証明したものではない。未解決停止の0件化や勝率の均等化を目的に、本人の主義・正当な意図を変えない。friendly_collision_rateは味方だけの競合を指し、両者競合は別欄にする。この標本では両者競合0なので総味方競合と同じになる。

|人格：各8局ずつ、両方式で同一|勝利|未解決停止|移動失敗 / 試行|最長実観測stall|
|---|---:|---:|---:|---:|
|growth|5|61|0 / 194|11|
|steady|5|75|0 / 199|8|
|care|5|71|2 / 256|9|
|ego|6|61|2 / 215|14|

条件群別ではopen/secureとopen/eitherが各8/8勝、choke/eliminateが3/8勝、choke/bothが2/8勝。後者はsecureの進捗が高くてもeliminateを満たせない未完了が残る。二つの目的を単一の最大値へ潰して成功と扱わない。全条件群・相手群・各対の目的進捗と分母は [evaluation.json](evaluation.json) に保存。

## 検証

- 435 tests成功、14 optional skips。compileall、git diff --check成功
- 新規28 contract testsと6集計・対照検査。元の主義階層、0.025帯、blocked代案、進捗、exact waste、非衝突個体固定、intent age、owner隔離、atomic rollback、別reader拒否、既定off、no-cheatを検証
- 独立コードレビューでmust-fixなし。味方だけの競合と両者競合の率を取り違えない注記を反映
- 64局・1,308実ルール遷移・3,752目的更新を独立再生
- 全32対で、同じ公開pre-state・同じ実行手・同じ実遷移をtickごとに照合。first-divergence検査でも差は0
- 実験開始前後で全reflex Python source hashが一致。結果を見た方策・閾値変更なし

全trajectoryはローカル `reflex_artifacts/root_collision/` に生成し、公開evidenceへ同梱しない。保存した64局のSHA-256は `265833434e6d899091930aec923d33c6248ff98cec7a6842c45bfa8ed59a57e5`。

## 費用と既知条件

既知careのchoke/eliminate/seed281とchoke/both/seed282を、留保標本とは別にserialで各方式3反復した。全反復で結果は同一。4条件でtrace on/offの全実軌跡・既存診断・学習件数が一致し、baseline2局は旧mainの保存軌跡とも一致した。保存JSONとの比較ではmap key/tuple/list表現を正規化し、trace-offにないexecution診断payloadを除く。表は3反復の中央値。

|既知care、両方式とも40tick・未勝利|全局秒 baseline→補完|判断+feedbackのtick中央値ms baseline→補完|
|---|---:|---:|
|choke/eliminate、281|10.924→9.576|226.789→220.271|
|choke/both、282|13.368→21.330|258.854→453.068|

**既知局でも利益は一様ではない。** eliminate281では味方だけの失敗4→0、相手だけ6→14で、全移動失敗10/72 (13.89%)→14/78 (17.95%)。未解決停止22→19、eliminate進捗0.3333→0.3917だが、最長実観測stallは4→7。味方衝突だけを減らして万能の改善とはしない。

both282は最初の根補完1回・1個体の変更後、別の4tickでは制約内の非衝突組合せがなく、衝突を残した。味方失敗8/68→8/76、未解決停止31→17、最長stall11→4、最終eliminate/secure進捗0.3917/1.0と未勝利は同じ。全局は約1.60倍、tick中央値は約1.75倍重くなった。補完自体は追加rolloutを呼ばないが、後続状態の変化によって全局model transitionsが14,205→24,392へ増えた。同じ40tickでも同量のplanner仕事ではなく、これを補完関数だけの実装overheadとはみなさない。eliminate281では12,868→10,288遷移だった。

2既知対の最初の実行差は同じ公開状態の根補完に一致した。合計3採用tick・4個体変更、帯外選択0・新しいblocked rootの導入0。これは開発条件の局所因果切り分けで、留保性能の代用ではない。

値はwarmな3v3 adapterの費用であり、共通数値反射や大量NPCの速度ではない。並列留保行列のplanner時間差は性能推定に使わない。留保では両方式とも705 forecast呼出し・244,324 model transitionsだった。費用は [reference.json](reference.json)、既知局の分解と原始データhashは [verification.json](verification.json)。

## 再現と次に残る問い

    python -m unittest discover -s tests -v
    python -m reflex.cli root-collision
    python tools/root_completion_reference.py

既存mainとの保存軌跡比較も行う場合は、reference scriptへ旧persona_continuationの `trajectories.jsonl` を `--prior` で渡す。旧大規模軌跡は公開版に含めない。

今回の到達点は、採用した共同案ではなく辞退後の独立根で衝突が起こる機構と、それを本人の既存制約内で扱う小さな部品・検査である。人間らしい人格の長期的一貫性、役割交渉、未知世界への汎用性、未来も再計画する本人の自己予測は実証していない。

次に評価するなら、機構診断とは独立に事前定義した自然な混雑条件で、介入機会を十分含む確認設計が必要。今回の32対を成功試験と呼び替えず、この周期の範囲で追加の都合のよい条件を探さない。既定変更、遅延誤差の自動学習、相手モデル更新はしない。
