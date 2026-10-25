# 独立確認20組時点の追加監査

前の18組を再試行せず、新たに完走した戦闘1組と賭け1組の512実判断を監査した。配布入力からの再監査も一致した。先の18組と重複なく合わせた20組・5,120判断が `combined.json`。全40組の完了・採用ではない。

競り8組と戦闘8組が全件揃った。戦闘の最終目的点は5改善・3悪化で、目的点・条件付き増分の平均差はともに正。賭け4/16・資源管理0/8は途中で、その平均から分野合格とは判断しない。得点が増えた場合も含め、条件付き増分や参照選択が悪化した例を保持する。

元の結果・実本人・世界乱数を含む `inputs.zip` と固定ソースは、前の18組と同じ仕様で保存した。Python 3.12 / NumPy 2.3.5で次のように再現する。出力先には未使用フォルダを指定する。

```text
python -I evidence/long_horizon/selected_confirmation_review_20/reproduce.py --output results/reproduce_selected_review_20
```

新しい対戦は実行せず、新たな2組の履歴を再生する。先の18組の証明は `../selected_confirmation_review_18/verification.json`。全体の採用基準は `../continuation_selected_confirmation_acceptance.json` のままである。

戦闘全8組の期間別会計と、参照選択差が悪化した3条件から機械的に選んだ6局面は `combat_diagnosis.json`。各条件で基準との差が最大の2局面を使った。6局面すべてで、選択は「現在の予測分布が続く」と仮定した参照の最良手に含まれた。参照は人格・将来の学習を省いており、標本の影響がないと証明するものではない。真の法則との不一致を、根の人格優先や採点ミスだけで説明しないための診断である。

この計算は次のコマンドで、固定ソース・配布済みの18組と追加2組の資料から再現する。出力先には新しいファイルを指定する。初回の7組の計算は、全8組を含めた再計算でも一致し、6局面の合法手30件について参照値と実際の遷移からの再計算を照合した。証明は `combat_diagnosis_verification.json`。

```text
python -I evidence/long_horizon/selected_confirmation_review_20/diagnose_combat.py --output results/reproduced_combat_diagnosis.json
```
