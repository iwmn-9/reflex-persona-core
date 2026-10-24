# 完走済み4組の途中監査

独立40組のうち、2026-10-25の確認時点で完走していた競り4組を対象とする。全体の合格・採用を示す資料ではない。事後の原因診断を独立確認へ加算しない。

- `plan.json`：対象4組、元の40組登録、全16入力ファイルのハッシュ。
- `inputs.zip`：基準・候補の実行結果と実際の所有者・世界乱数を含むチェックポイント。悪化例と増益例を両方保持する。
- `review.json.gz`：実履歴1,024判断の再生監査と収支の分解。真の世界法則による参照評価は評価器だけが使用する。人格・情報制約を含む差なので、そのまま知能の不備とは呼ばない。
- `sampling_plan.json` / `sampling_probe.py` / `sampling_results.json.gz`：参照評価で選択の寄与が悪化した全条件から、追加損失が最大の2局面ずつを固定した12比較。元から使っていた3組の予測標本をすべて残す。実際の人物・履歴・世界乱数は変えない。得点が上がった条件も対象に含む。
- `diagnosis.json`：本人の候補制限、許容した予測上の目的損失、標本を変えたときの選択、参照評価との相違。先の行動方針と世界予測が異なる参照値は、本人が本来知っている正解ではない。
- `reference_depth.py` / `reference_depth.json`：同じ4局面を、32手と終局までの両方で評価した既存の参照計算。目的の単位を元ゲームへ揃え、44,660個のBellman等式を実際の遷移・目的点と照合した。人格と将来の学習を省くため、NPCの正解や長期改善の証明とは扱わない。
- `verification.json`：全入力・出力との照合。`complete` はこの限定された監査の完了を意味し、40組全体の完了は別項目で偽のまま保持する。

凍結ソース383ファイルは既存の `../continuation_horizon_ablation/source.zip` を共用する。実行中の40組、予測乱数、人格、採用条件を変更しない。診断で良く見えた標本への差し替えはしない。

リポジトリのルートで、元の実行環境と同じ Python 3.12 / NumPy 2.3.5 を使い、以下で実履歴の監査を再現できる。出力先には未使用のフォルダを指定する。

```text
python -I evidence/long_horizon/selected_confirmation_review/reproduce.py --output results/reproduce_selected_review
```

このコマンドは新しい対戦や12局面の重い再予測を行わない。保存した実際の行動結果・学習・乱数を再生し、全1,024判断と会計値の一致を確認する。予測診断の再実行には、同じ凍結ソース、`sampling_plan.json`、`sampling_probe.py`、候補側チェックポイントを、元の入力名で別フォルダに配置する。全12比較を保持し、独立40組の進行とは分離する。

上の再現用フォルダを作成後、同じ長さの参照計算は次で再確認できる。

```text
python -I evidence/long_horizon/selected_confirmation_review/reference_depth.py --source results/reproduce_selected_review/trial/_source --output results/reproduce_selected_review/reference_depth.json
```
