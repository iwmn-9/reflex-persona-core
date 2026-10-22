# 長期実験の実行区間と完走

判断器の凍結ソース・人格・世界乱数・学習履歴・登録条件は変更せず、実行管理だけを更新する。
実験の完走と、1台の実行枠を正常に引き継いだことを区別する。

| 状態 | GitHubの区間ジョブ | 実験完走 | 回収後の処理 |
|---|---|---|---|
| 完走・監査済み | success | 両側の既存監査と`contract_pair.complete`で判定 | 正規集計へ |
| 引き継ぎ待ち | success、進捗チェックはneutralで「experiment incomplete」 | false | 保存証拠を再検証して同じ登録条件を継続 |
| 実行・監査・保存検証・アップロード失敗 | failure | false | 原因を検査。既知の監査復旧以外を無条件に再送しない |

`success`だけでは完走に数えない。正規集計の`tools/summarize_value_contract.py`は従来どおり、
両側の終局・監査・固定ソース・同一履歴と`return_codes == [0, 0]`を要求する。

Linuxの継続ワーカーは330分で、そのワーカーが作った子プロセス群だけへ停止要求を送る。
345分のステップ上限と360分のジョブ上限は、停止失敗や回収失敗を検出する最後の上限として残す。
固定判断器の4手ごとの原子的保存を使い、途中計算を完了した判断として加算しない。
子の異常終了や標準エラーがある場合は正常な停止と認定しない。

`segment_handoff.json`は、両側の終了状態・元の接続点からの前進・保存checksum・凍結ソース・
登録設定・本人の時系列・実世界乱数・既存履歴との一致を検査して初めて作る。
欠損、改変、前進なし、完走偽装、別runの記録、片側の実エラーは引き継ぎとして拒否する。
監査失敗と相方の時間枠停止が同時に起きた場合は失敗を維持する。
既知の丸め差監査の復旧が未完了のまま時間枠に達しても、正常な引き継ぎにはしない。

実行管理の正本は次の通り。

- `tools/observe_horizon_pairs.py`: 実行中の進捗・区間結果の公開。
- `tools/resume_horizon_pair.py`、`tools/resume_horizon_study.py`、`tools/horizon_segment.py`: 凍結ソースの実行、時間枠の停止、保存証拠。
- `tools/collect_value_contract.py`: 完走結果と引き継ぎ待ちの回収。途中データは完走集計へ入れない。
- `tools/finish_registered_worlds.py`: `results/registered_world_completion/state.json`を正本に、回収・検証・継続を管理。
- `tools/prepare_active_value_continuation.py`、`tools/dispatch_registered_continuation.py`: 検証済み入力を準備し、送信記録を先に保存して重複送信を防ぐ。
- `tools/inspect_registered_progress.py --watch`: 保存手数・CPU稼働・失敗・引き継ぎ待ちを読み取り専用で監視。

旧`results/`と外部コレクターの起動ファイルは正本を呼ぶ互換入口。
監督処理の再起動前に実行中PIDと子処理を確認し、既存の送信・回収と重ねない。
実行中の古いworkflowには後から変更が入らないため、自然な区間終了後の次回継続から新方式になる。

会話の返答終了、クラウド計算、ローカルの回収・再送、AIによる分析は別の実行境界を持つ。
GitHubで開始済みの区間はローカル監督の停止後も進めるが、次区間の送信は現在ローカル監督に依存する。
`status: active`という台帳や古いPIDだけを継続の根拠にしない。実プロセス・更新時刻・GitHubの現在状態を確認する。
AIによる分析・改修はチャットの実行時に行う。定期呼び戻しは有効なスケジュールとローカル実行環境が必要で、
スケジュール有効だけでは常時稼働の保証にならない。中断時は元の保存履歴を検証してから継続する。
現在はユーザーの意図により定期呼び戻しを停止している。有限の登録済み実験を継続する監督処理と区別し、
「継続してよい」という許可だけで定期呼び戻しを再度有効にしない。

監督のGitHub読取は一時的なHTTP障害、タイムアウト、および実際に観測した`error connecting to api.github.com`を
最大5回で再試行する。認可エラーや上限超過は停止し、実験の再送・失敗判定にはこの再試行を適用しない。

検査: `python -m unittest tests.test_horizon_segment tests.test_horizon_observer tests.test_pair_continuation tests.test_recorded_forecast_audit -v`。
Linuxの実プロセス停止検査はLinuxで実行する。Windowsではその2件を明示的にスキップする。
