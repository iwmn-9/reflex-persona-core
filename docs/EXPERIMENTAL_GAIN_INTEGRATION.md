# 未採用候補のPython接続

この開発ブランチの `0.1.1.dev0` は統合確認用。安定版v0.1.0やRustを更新する配布ではない。
長期比較と独立確認の通過は別途必要で、この入口が動くことを採用根拠にしない。

既存の研究構成を `reflex.experimental_gain` へ接続した。
`transfer_event_reader(...)` は公開応答を学ぶ新しい本人を作り、
`GainHorizonOwner(actor, ...)` は既存の長期判断を実行する。
新しい学習則、目的ボーナス、人格変更、ゲーム名による分岐は加えていない。
`tools` や研究runnerを実行時にimportする必要はない。

ホストが用意する情報とcallback、実際の世界・乱数の所有は
[安定版の接続契約](STABLE_INTEGRATION.md#ホストが決めるもの)と同じ。
この応答モデルは候補行動に共通の公開応答分布を前提とする。
相手がこちらの行動を見て別の確率で応じる環境まで対応したとは扱わない。
転移事前分布は4応答までを扱い、それより多い場合は元の分節モデルへ戻る。

1. 新規の `transfer_event_reader` を `DecisionLoop` のpredictorとして渡す。
2. その本人と同じホストcallbackを `GainHorizonOwner` へ渡す。
3. `decide(public)` → ホストで実行 → `resolve(ticket, actual_feedback)` の順に呼ぶ。
4. 解決後に世界・実乱数・`checkpoint(public)`をまとめて保存する。
5. 同じモデル・人格・設定で作った新しいtemplateの `restored(public, saved)`で再開する。

既存の学習履歴を新しい学習器へ自動変換しない。異なるcontroller、学習器、
予測設定や世界境界の保存は拒否する。未解決の判断も保存できない。
予測失敗時に即時判断へ黙って戻らず、本人に判断を確定する前に例外を返す。
再実行や欠測の扱いは安定版のホスト契約を参照する。

研究で使う構成は `horizon=32, seeds=(701,702,703), recovery_cache=64`。
別の予算を受け付けることは、その設定の判断品質の保証ではない。
通常の判断とともに、将来の公開目的を評価する判断、利益が出た時点で本人へ戻る
準備反復案を比較する。予測した複数手を実世界で一括実行せず、実観測の後に選び直す。

[動く接続例](../examples/experimental_gain_host.py)は
`stable_host.py`と同じ場所に置く。新しいpackageを使って以下を実行できる。

```powershell
python examples/experimental_gain_host.py --save full.json
python examples/experimental_gain_host.py --stop-after 3 --save prefix.json
python examples/experimental_gain_host.py --resume prefix.json --save resumed.json
```

6手の例は接続と保存復元の確認であり、長期判断や強さの試験ではない。
Rust所有者APIはこの転移学習器を拒否する。現在の安定版Rustの対応範囲は変更していない。

配布確認と同じ606 Pythonファイルで、全体回帰1,374件を実行した。
1,355件成功、任意の麻雀環境17件とLinux専用プロセス制御2件を除外し、失敗・エラーは0。
[ソース照合・全ログと除外理由](../evidence/gain_integration/full_regression.json)を保存した。
これは接続と既存機能の回帰確認であり、未完走の長期比較や独立40条件の代わりにはしない。
