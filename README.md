# Reflex Persona Core

LLMを使わず、人格を保って学習・予測・再計画するゲームNPCの判断基盤。
ゲーム側が観測・合法手・結果モデル・目的を与える。

- [現在地](docs/STATUS.md)
- [完成基準と指標](docs/COMPLETION.md)
- [目的と制約](docs/PROJECT_DIRECTION.md)
- [失敗から残す教訓](docs/LESSONS.md)
- [Python導入](docs/STABLE_INTEGRATION.md)・[ホスト例](examples/stable_host.py)
- Rust：[即時判断](rust/reflex-immediate/README.md)・[世界との接続](rust/reflex-runtime/README.md)・[実学習と長期候補](rust/reflex-owned/README.md)
- [安定版の採用根拠](evidence/stable_release/RELEASE.md)

Python安定版はv0.1.0。Rustの長期候補生成はPythonに残る。
公開wheelは後続のRust接続を含まない。連携時は両側のソース版を揃える。
未知ゲームでの人間水準や面白さは未実証。
利用・改変・再配布の一般的な許諾は未付与。
