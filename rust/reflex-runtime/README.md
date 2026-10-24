# reflex-runtime

複数の人格NPCを、一つのゲームエピソードで動かす実行・保存の入口。
採点は採用済み `reflex-immediate` に委譲し、その数式・人格・乱数・v1保存形式は変えない。
本体にゲーム名・行動名による分岐はない。ルールと効果推定は `World` アダプターへ置く。
`publish = false`。利用許諾や公開レジストリへの配布は追加しない。
検証結果と認定範囲は [統合検証](../../evidence/rust_port/runtime/REPORT.md)。

## 一つの入口で動かす

```toml
[dependencies]
reflex-runtime = { path = "../reflex-persona-core/rust/reflex-runtime" }
```

| `World` の操作 | ゲーム側の責務 |
| --- | --- |
| `revision()` | 世界が変わるたびに増える版。NPC個別の判断tickとは別 |
| `observe(config)` | 本人が知る事実、動的欲求、合法候補、今回の結果見込み。休眠・死亡などは明示的に `None` |
| `advance(choices)` | 同一の選択前世界でルールを適用し、次世界と実行結果を返す。共有資源の競合もここで解決 |
| `SAVE_FORMAT` | 世界データの意味を識別する版。意味を変えた保存を旧版で読み込まない |

```rust
use reflex_runtime::{GameSession, SessionId, World, immediate::Config};

fn run<W: World>(world: W, id: SessionId, actors: &[Config])
    -> reflex_runtime::Result<GameSession<W>> {
    let mut game = GameSession::new(id, actors, world)?;
    for r in game.step(true, 2)? { // 確率選択あり、最大2ワーカー
        println!("{}: {:?}", r.npc, r.execution);
    }
    Ok(game)
}
```

共有資源を取り合う6人の経済世界と、弾薬・回復・防御・対象選択を持つ6人戦闘の
[接続例](examples/headless.rs)を用意した。どちらも同じ `GameSession::step` を呼ぶ。
ゲーム側の見込みは例示用の手設計で、強さ・学習・面白さのベンチマークではない。

```sh
cargo run --release --locked --manifest-path rust/Cargo.toml -p reflex-runtime --example headless
```

## 実行・取消・途中参加

`step` は観測、一括判断、次世界の作成、全NPCの反映をまとめる。
世界の計算や一人分の入力・実行報告が不正なら、世界も全NPCも進めずエラーを返す。
世界更新は `&self` から新しい世界を作る契約。アダプターが共有可変状態・外部エンジン・
ファイルを直接変更した場合まで巻き戻せるわけではない。不可逆な副作用は成功後に行う。

- `Executed { action_id, target }`：本人が実際に行動した。命中しなかった攻撃もこちら。
- `NotExecuted { reason }`：行動そのものが起きなかった。NPCの意図・判断tickは進めない。
- 提案された全NPCに一件ずつ必要。未報告、重複、別の対象・行動は拒否する。
- 次の観測には実世界から求めた欲求・見込みを渡す。未実行・欠測を成功やゼロの経験へ変換しない。
- `insert` / `remove` は途中参加・退出用。世界の時刻は巻き戻らない。再参加時に経験を維持したい場合は
  名簿を維持し、`observe` の `None` で休眠させる。remove/newは明示的に新しい個体状態を作る。
- `reset_intent` は有効な欲求の種類が変わったときの意図リセット。人格・主義は変えない。
- プレイヤーの操作などを反映した新世界は `replace_world` で渡す。版が進まない置換は拒否する。

## 非同期エンジンへつなぐ場合

低水準の `Runtime` を直接使える。`prepare(revision, inputs, stochastic, workers)` は純粋で、
不透明な `Batch` を返す。案は捨てれば取消。並列数・入力順で判断乱数や結果は変わらない。
実行直前に `validate_batch` を呼び、世界とruntimeの排他を保って実行する。
`resolve(batch, next_revision, receipts)` は全員分を検査してから一括反映する。
世界変更・名簿変更・意図リセット・別バッチの確定・保存からの復元で以前の案は無効になる。
他runtimeへ案を移せない。同じ人物IDを削除・再登録しても旧案は復活しない。

`resolve` の原子性はNPC状態の範囲。既に外部エンジンで実行した操作は巻き戻さない。
外部エンジンのトランザクションや実行ログと結び付ける責務はホストにある。
遅い読みを別スレッドで走らせる場合も、古い世界の案を現在の世界へ無条件に反映しない。

## 世界とNPCを一緒に保存する

`World` にSerdeのSerialize/Deserializeを実装すると `GameSession::checkpoint` / `restore` が使える。
世界の乱数、動的欲求、知覚・推定の状態も世界側の保存へ含める。
保存byte列は世界・全NPC・世界版を一つの検査対象として持つ。
ファイル/DBへの原子的な永続化は利用側が担当する。

復元時には保存内容とは独立して、期待するセッションと全NPCの `Config` を渡す。
名簿、人格、目的、主義方式、seed、世界形式、版、チェックサムを検査する。
世界の版とNPC群の版が違う保存、暗黙の人格の置換、不足するNPCは拒否する。
チェックサムは破損検知であり、改ざん認証や古いセーブへの巻戻し防止ではない。
未確定の判断案は保存しない。復元後に現在の世界から判断し直す。

一つのruntimeは最大10,000人、1回最大64ワーカー。世界データ最大4MiB、保存全体最大64MiB。
検査した境界であり、1万NPCを毎フレーム動かせる速度を保証する数ではない。
スレッドはバッチの寿命内に終了する。外部サービス、GPU、可変な共有乱数器は使わない。

## 全体への接続と未移植の部分

| 層 | 現在の実装 |
| --- | --- |
| 世界・本人の知覚・合法性・効果の見込み | ゲームの `World` 実装 |
| 全員の観測・判断・実行照合・進行・保存 | `GameSession` / `Runtime` |
| 固定人格・主義、動的欲求への即時判断 | `reflex-immediate` v1 |
| 経験の検査・学習、感情/関係更新、長期評価・32手読み | Pythonで継続検証。今回のruntimeへ組み込んでいない |

今回の統合は、実装済みの通常判断と世界のライフサイクルをつなぐこと。
将来価値を即時効果に押し込んだり、学習したふりの記憶欄を増やしたりはしない。
提供済みの相手予測は既存の `Observation.opponent` で受け取れるが、その予測器を移植したわけではない。
未移植の機構が安定したら、同じ所有者・観測・実行結果・保存の境界へ接続して再検証する。

```sh
cargo fmt --all --manifest-path rust/Cargo.toml -- --check
cargo clippy --locked --manifest-path rust/Cargo.toml --workspace --all-targets -- -D warnings
cargo test --locked --manifest-path rust/Cargo.toml --workspace
python tools/verify_rust_runtime.py --output FRESH_RUST_RUNTIME
```

Python比較は4ルール×4人格×3主義方式×2世界乱数。各NPCの物理世界は独立し、
四つを一つのruntimeに入れる。共有世界の競合は別のRust統合試験で検査する。
これらを長期知能の独立40組へ合算しない。Windows/Linux CIも同じ入力の移植検証である。
