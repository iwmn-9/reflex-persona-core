# reflex-immediate

通常の人格判断を独立して実行するRust部品。候補の見込みを受け取り、本人の性格・欲求・価値に沿う行動を選び、実行済みの判断状態を保存・復元する。Pythonの `Policy.choose` の通常経路を参照とする。実験・学習・長期評価はPython側に残る。

本リポジトリの利用許諾は追加していない。crateは `publish = false` であり、公開レジストリへの配布は行わない。

昇格判断と確認済み環境は [検証報告](../../evidence/rust_port/immediate/REPORT.md) が正本。APIはこの用途のまとまりとして固定するが、全人格AIの完成を意味しない。

複数NPCをゲーム世界へまとめて接続する入口は [reflex-runtime](../reflex-runtime/README.md)。
このcrateの判断を変更せず、世界の実行照合・並列処理・一体の保存復元までを扱う。

## 接続

Rustの利用側からpath dependencyを指定する。

```toml
[dependencies]
reflex-immediate = { path = "../reflex-persona-core/rust/reflex-immediate" }
```

`Config` は人物・乱数系列・目的・採点方式、`Observation` は今回の観測・欲求・合法候補・結果の見込みを表す。配列の次元名と順番は公開定数 `TRAITS` / `NEEDS` / `VALUES` が正本。各候補は1〜8結果、候補数は1〜256。結果の効果は[-1,1]、費用と確率は[0,1]で、確率の合計は1。人格・価値・欲求の不足度は[0,1]。

```rust
use reflex_immediate::{Agent, Config, Observation};

fn decide_and_save(
    config_json: &[u8],
    observation_json: &[u8],
    saved: Option<&[u8]>,
    execute: impl FnOnce(&str) -> bool,
) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
    let config = Config::from_json(config_json)?;
    let mut actor = match saved {
        Some(bytes) => Agent::restore(&config, bytes)?,
        None => Agent::new(config)?,
    };
    let observation = Observation::from_json(observation_json)?;
    let prepared = actor.preview(&observation, true)?;
    let action = prepared.decision().action_id.clone();
    if execute(&action) {
        actor.commit(prepared, &action)?;
    }
    Ok(actor.checkpoint()?)
}
```

返るbyte列をゲームの保存システムへ渡し、次回 `restore` する。ファイル・DBへの書き込みと、世界側の実行を含むトランザクションはゲーム側が担当する。保存内容は判断の状態であり、盤面・欲求・学習記憶・予測器は含まない。

- `preview` は純粋。採点、選択、次状態を確認しても本人を更新しない。提案を捨てれば取消になる。
- `commit` は実行した行動IDと、提案時の人物・設定・状態・時刻を照合して一度だけ反映する。他の行動を実行した場合や古い提案は拒否する。
- `checkpoint` / `restore` は版番号・チェックサム・型と範囲・人物・採点設定を検査する。人物や設定は呼び出し側から明示し、保存データに書かれたものへ自動で切り替えない。チェックサムは破損検知であり改ざん認証ではない。
- `from_state` はPython等からの明示的な初期状態の移入用。通常のセーブ読込みで `restore` の代わりに使用しない。
- 有効な欲求の種類が変わり、前の主欲求を無効化する時は `reset_intent` を明示する。人格・採点・時刻は保ち、戦術的な意図だけを初期化する。観測の変化だけで暗黙の人物変更を行わない。

同じ人物の同じtickを繰り返し反映できない。tickは飛び番号を許すが、反映済みtickより大きく、2^63未満とする。seedも2^63未満。行動IDはソートして扱うため入力順によらない。乱数はseed・scope・tickから導出し、共有の可変乱数器を持たない。`preview_batch` や独立したスレッドで同じ結果を得られる。バッチの部分反映は自動で行わない。

`commit` はゲーム世界そのものを所有しない。判断案の作成後に世界が変わった場合、合法性の再確認と案の破棄・再判断はゲーム側が行う。同じscopeを持つ意図的な複製は同じ人物の複製として扱う。

## 責務の境界

含むものは通常の欲求選択、主義3方式、損失評価、慣れ、切替費用、有界な補正係数、提供済み相手予測の採否、近接候補の確率選択と状態更新。`Config` の範囲で人格・採点設定は固定する。

学習、先読み、社会関係の加点、`ReturnFrame` による長期の単位換算、人格の成長、ゲームの効果推定は含まない。未来の価値を即時効果へ丸めて渡すAPIではない。JSON入力で未知のフィールドは拒否するので、未対応の機能を指定したまま無視する経路もない。

学習を実装するPythonの `DecisionLoop` 全体の代替ではなく、ゲーム側から効果を与えて使う即時判断のまとまり。人格の心理学的な正しさやNPC全体の賢さは、この移植の認定対象ではない。

## 検証と更新

```sh
cargo test --locked --manifest-path rust/Cargo.toml
cargo clippy --locked --manifest-path rust/Cargo.toml --all-targets -- -D warnings
python tools/verify_rust_immediate.py --output FRESH_RUST_IMMEDIATE
```

Python比較は `examples/verify.rs` 経由で公開API・保存・復元を使う。元の数値試作の5,124判断に加え、4ルール×4人格×3主義方式×2世界乱数の96継続実行を比較する。各判断でcheckpointを復元し、64手目にはRustプロセスを作り直す。死亡などの正当な終端を含め、行動・状態・乱数・盤面の一致を要求する。固定の元検証40/8/8組とは別の、移植同等性の検査である。

同じtickと入力に対するRust内の保存復元は完全一致を要求する。Pythonとの数値差には2e-12の上限を置くが、選択・乱数・離散状態の不一致は許容しない。浮動小数の全境界、未検証のCPU・コンパイラー、速度向上は保証しない。

参照はNumPy 2.3.5のWindows/Linux x86-64版。通常の単独 `Policy.choose` と比較する。採点の加算順序を固定し、環境任せのSIMDやFMAによる並べ替えをしない。Pythonで異なる人物を一緒に配列化する場合の全ての丸め境界まで一致すると約束するものではない。

依存はSerde/serde_json（型付き入出力）とSHA-256（seedと破損検知）。lockfileを保持する。既存のPython実験には依存を追加しない。状態の意味・採点の意味を変える際はsnapshot版を分け、旧保存データを暗黙に読み替えない。
