# 任意のチーム協調契約

所属、共同目的、公開状態、合法な行動、世界の遷移、何を目撃できるかはゲームが定義する。人格コアが全NPCを一つの勝利目的へ統合する仕様ではない。通常の独立判断は変更していない。

## 所属と合意

`reflex.team_choice.select(base, forecast, group=..., mode='consent', outside=...)` は合意に参加した所有者だけの共同候補を選ぶ。`JointForecast`に全員分の同じ候補ID、各人の結果、合法な根の組合せ、共同目的の評価を渡す。game/episode/tickの一致とnpcの相違を検査し、性格・価値・公開事実・本人の記憶を維持する。採点には既存の有限主義Policyを使う。

`sum`は既存の共同平均選択との比較用。`consent`は共同目的の許容域内で全員の譲歩限度を満たす候補を探す。限度は既存の調和性・善意・力から投影した工学的な式で、心理学的な実測値や新しい性格軸ではない。死や長期損失を必ず避ける保証もない。

`outside`は各人が先に提出した独立行動のtuple。他者の独立行動を固定し、本人だけが変更できる有限候補内の代案を基準にする。全員が自分だけに尽くす想像上の最善案を、独立行動の価値と混同しない。独立案は候補群へ必ず残す。狭い候補群では基準も不十分になる。合意がなければ`None`を返し、ゲーム側が独立案へ戻す。

所属するだけの仲間は投票・効用計算に参加しない。命令したことにもならない。実行は別の本人制御器が行う。敵と仲間の未実行の現在手、他者の私的記憶、実世界の乱数は予測へ渡さない。

## 仲間の観測

`PartnerMemory(game, episode, group, actors)`は既存の有限仮説学習器を仲間ごとに隔離して使う。ゲームが合法行動上の仮説分布を作り、`begin(scope,tick,models)`で観測前の予測を固定する。公開実行後に`observe(..., witnessed=True)`へ目撃した行動を渡す。観測欠落は`abandon`。想像したロールアウトを経験に昇格させない。

本人の性格・主義や他者の実人格は更新しない。材料不足で行動が失敗しても、実行しようとした行動の傾向だけを学ぶ。「裏切り者」などの動機を付与しない。仮説外の行動は仮説が正しいという証拠にならない。

検証adapterの仮説は「待機」と「独立した能力方策」の2つ。観測が増えても候補にない行動・動機は発見できない。重みは相対支持で、校正済みの未来確率ではない。少量の一様混合を残し、既存の保持・変化対応を使う。

## 二種類の世界

`team_combat`は既存の3対3の公開戦闘。移動・射撃・回復・拠点占有、6手・24共同候補・4標本の有限予測。実敵は既存のreference/switchで、最適な協調CPUや人間ではない。

`team_projects`は2～4人ずつの公開資源競争。共通材料、個別の体力・成長・名誉、共同の建設達成を持つ。建設/訓練が開始時の材料を超えて重複予約すると全予約が失敗する。同時に集めた材料はその手では使えない。6手・最大64候補。市販ゲームの再現ではない。

将来の継続は宣言された能力方策。実際の人格判断・共同再交渉と同一ではない。ゲーム固有の目的評価は引き続き必要。通常の心理モデルや優先順位は変更していない。

## 再実行

```powershell
python tools/run_team_cooperation.py --root FRESH_PATH --start 9800 --seeds 8 --workers 4 --stage confirmation --modes independent sum consent bargain sum_uncertain bargain_uncertain
python tools/analyze_team_cooperation.py --root FRESH_PATH --replay-policies --workers 4
python tools/summarize_team_cooperation.py --root FRESH_PATH
```

観測比較は`--participation partial --modes independent sum_uncertain bargain_uncertain sum_observed bargain_observed`。人数比較は`--genres projects --members 2`または`4`。新しい保存先へ実行前に数値コードと条件を凍結する。

世界照合は別実装の同時解決台帳。判断照合は凍結ソースによる完全記録再現。援助診断は他者の現在手を固定した一手の介入で、追加試合や長期利益の証明ではない。

性能・採否・停滞の正本は[検証報告](../evidence/team_cooperation/REPORT.md)。標準切替や性能保証は行っていない。
