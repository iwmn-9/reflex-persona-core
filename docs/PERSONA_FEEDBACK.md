# 判断する本人と、学習を評価する本人を揃える

`DecisionLoop(..., decision_metric='persona')` は即時の既知条件付きルールに限る任意経路。従来の `purpose` が目的点だけで追加予測の判断利益を採点するのに対し、欲求・主義・性格・損失回避・切替費用を含む、判断時の本人の評価を用いる。人格を学習で書き換えない。採否と実験の正本は [報告](../evidence/persona_feedback/REPORT.md)。

## 適用できる情報

既存の `Reading(known_conditionals=True)` / `CategoricalReader` と同じ契約を使う。

- 相手の応答が自分の**同時刻の封じた選択に反応しない**。過去の自分の行動への反応はよい。
- 応答が実際に公開され、全候補について「この応答ならこの即時効果になる」というルールが既知。
- 選んだ手の観測効果と条件付きルールが矛盾した場合は、経験・予測・判断の採点を確定しない。

単に最後の勝敗だけが見えるゲーム、行動ごとの未公開の運、選んだ手に応じて変わる相手の応答には、この比較をそのまま接続しない。未選択の結果は**既知ルールの評価**であり、実経験として `OutcomeMemory` へ追加しない。先読みの終点を即時経験に混ぜない。

## 同じ価値基準での採点

`Policy.decide(..., outcome_audit={})` は任意の診断出力。各結果の効用に損失回避を反映し、判断時に計上した慣れ・切替費用・残差・読み・関係の加点を固定する。その確率付き平均が通常の最終 `scores` に一致する。候補の適格性や最上位主義の層は独立した制約であり、後から得点へ換算しない。

助言比較は、基線の予測で凍結した**同一の評価枠**を、基線と追加予測の選んだ両方の手に適用する。追加予測が自分に都合よく残差や関係を採点し直すことを防ぐ。予測によって残差などの定額部分が変わる場合、この比較は追加予測自身の総合得点そのものではない。

実際に公開された応答での本人評価の差を２で割り、[-1,1]へ制限して既存の `EvidenceGate` に入れる。大差が一回出ただけで履歴全体を支配させないための工学的な上限で、心理学的な校正値や統計保証ではない。元の差は監査記録に保持する。既存の最低比較数・最近の失敗による撤回・予測精度のgateを引き続き使う。精度gateを通るだけでは採用しない。

## 予定した費用と予測外れ

比較記録には以下を分けて残す。

- `purpose_delta`: 同じ公開応答で評価した２つの手の目的点差。
- `planned_purpose_delta`: 各助言が事前に見込んだ、選択した手の目的点の差。
- `purpose_surprise_delta`: それぞれの事前予測からの外れの差。上の２項を加えると実応答での差になる。
- `candidate_rationale`: **追加予測と同じ確率分布**で、目的点最大の手に比べてどれだけ目的点を支払い、その代わりに本人評価がどれだけ高いか。目的点最大の手が主義の適格層に入るかも残す。
- `baseline_choice` / `candidate_choice` / `executed_choice`: 採点した助言と、実際に選んだ手。

これは完全な因果診断ではない。予測同士の差だけから、損の全てを意図的な自己犠牲と断定しない。一回の不運もモデル誤りの証明にしない。遅延成果、相手への長期的影響、計画変更の帰属は未解決。

## 保存と再実行

目的点の旧gateと新gateは別key。checkpointに `decision_metric` を保存する。指定のない旧checkpointは `purpose` として復元し、その履歴を人格評価の確信として再利用しない。明示的に異なるmetricを指定した復元は拒否する。

```python
loop = DecisionLoop(context, predictor=reader, decision_metric='persona')
saved = loop.record()  # 実観測済み・未解決ticketなし
restored = DecisionLoop.from_record(context, saved, predictor=reader,
                                    decision_metric='persona')
```

```sh
python -m unittest discover -s tests -p test_persona_feedback.py -v
python tools/run_persona_feedback.py --output FRESH_DEV --seeds 2 --seed-start 9100
python tools/run_persona_feedback.py --output FRESH_CONFIRM --seeds 4 --seed-start 10100
python tools/run_persona_feedback.py --output FRESH_STRICT --seeds 2 --seed-start 11100 --priority lexicographic --regimes stationary switch noisy
python tools/analyze_persona_feedback.py --cohorts FRESH_DEV FRESH_CONFIRM FRESH_STRICT --output analysis.json
```

共通コアにゲーム名・相手controller名・実行時の乱数は追加しない。試験は既存ルールを使った反復の局所判断実験。競りは毎ロットの資金支給、交換は１相手への既知の即時効果であり、有限予算の通し対戦や自律的な多人数社会、長期の読みを試験したとは扱わない。ギャンブルの非公開抽選へ未選択結果の採点を接続しない。
