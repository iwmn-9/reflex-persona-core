# 目的別相手試験の再現とWindows互換性

PR7の最初のWindows CIで、事前登録manifestのfile_hashesのキーがOS依存の区切り文字になる問題を確認した。Linuxでは `tests/test_goal_opponent_experiment.py`、Windowsではbackslash区切りとなり、二つのmanifest契約試験が失敗した。

互換修正は二つの実験harnessの `frozen_files()` で、相対Pathを `str(...)` から `.as_posix()` へ変える2行。PureWindowsPath/PurePosixPathを使い、両harnessが同一の相対キーを作る回帰試験を追加した。相手予測・rollout・人格・実経験・目的評価のコードは変えていない。新しい方式やseedは実行せず、凍結結果も上書きしない。

再現用source snapshotはmergeで保持する次のcommitを使う。

- 56局の予測比較: [affef064](https://github.com/iwmn-9/reflex-persona-core/commit/affef0641d2884d6374ff9e8be5e477a54b1ab81)、tree `2b74b9dbb55c3134689fc0d5270d782d37251d22`
- 96局＋4費用局の実対戦: [220b15e](https://github.com/iwmn-9/reflex-persona-core/commit/220b15ebea81b1819b96bf996477786d032e3cef)、tree `7c36b4ff302dc194f6f0aafaf993495c5be1c225`

既存のpreregister/source hashは当時のexact snapshotを表す。この後の互換修正commitと、当時の凍結sourceが同一だとは主張しない。新しい環境で新規試験を始める場合は新規出力先へ新しい事前登録を作る。元の凍結試験を再現する場合は対応するsnapshotをcheckoutしてLinuxで実行する。
