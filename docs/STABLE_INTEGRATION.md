# 安定版をゲームへ接続する

対象は公開済み **Python v0.1.0 / `FixedHorizonOwner`**。
推奨設定は `horizon=32, seeds=(701,702,703), recovery_cache=64`。
人格・主義を固定し、実際の結果で経験・予測・欲求圧・目的進捗を更新する。
品質を確認した範囲と個別の悪化は [安定版の採否](../evidence/stable_release/RELEASE.md) にある。

この文書と動くホスト例を接続の入口とする。研究用runnerや過去の実験を読む必要はない。
Rustとの接続は別の [所有者API](../rust/reflex-owned/README.md) が対象。
公開v0.1.0 wheelはRust接続モジュール追加前のPython版であり、mainの全機能を含まない。
Rust連携を使う場合はPython側も同じmainのソース版に揃える。

## 配布物を入れて、保存と再開まで動かす

Python 3.12/3.13、NumPy 2.3.5、Windows/Linuxが確認済み。以下はPowerShell。
[v0.1.0の公開ページ](https://github.com/iwmn-9/reflex-persona-core/releases/tag/v0.1.0)から
`reflex_persona_core-0.1.0-py3-none-any.whl` をダウンロードする。
wheelのSHA256は `27fa081e83f3ed269e46efa69fd7ec3796cf0199fb236682b094827fdd2fcb43`。
`Get-FileHash -Algorithm SHA256 <wheelのパス>` で照合できる。

```powershell
python -m venv .venv-stable
.venv-stable/Scripts/python.exe -m pip install "numpy==2.3.5" ./reflex_persona_core-0.1.0-py3-none-any.whl
.venv-stable/Scripts/python.exe examples/stable_host.py --save full.json
.venv-stable/Scripts/python.exe examples/stable_host.py --stop-after 3 --save halfway.json
.venv-stable/Scripts/python.exe examples/stable_host.py --resume halfway.json --save resumed.json
```

[ホスト例](../examples/stable_host.py)は一つのファイルで動く。リポジトリ外へコピーしてもよい。
Linuxでは仮想環境のPythonを `.venv-stable/bin/python` に読み替える。
`full.json` と `resumed.json` の世界・乱数・判断列・本人の全学習状態が一致する。
例は6手で終わるため予測も終局で止まる。長期品質やゲームの面白さを測る例ではない。
`--save` は指定したファイルを更新するので、実ゲームのセーブとは別の名前を使う。

リポジトリからの誤importも禁止して検証する場合は、別ディレクトリへwheelを
`pip install --no-deps --target <installed>` し、次を実行する。
出力先は未作成のものを指定する。実行環境にはNumPyが必要。

```powershell
python -I tools/verify_stable_host.py <installed> --output <新しい検証結果フォルダ>
```

## ホストが決めるもの

| 入力・関数 | 責任と戻り値 |
| --- | --- |
| 人物の初期context | 人物ID、固定性格・主義、目的、乱数seed、現在の欲求、合法手を明示する。人物を別NPCへ使い回さない |
| `observe(public, intent)` | その人物が観測できる世界と現在の意図からcontextを作る。実世界の秘密や乱数器は渡さない |
| `models(context)` / `outcome(context, action, response)` | 公開応答の仮説確率と、各応答が起きた場合の既知の効果を返す。仮説は真の相手状態ではない |
| `request_for(context)` | 全候補の`Binding`と、目的・準備の現在値、各候補が試行・待機・維持などのどれかを`Request`へ入れる |
| `events(public, action)` | 公開ルール上の各応答について`PublicEvent(symbol, next_public, effect)`を返す。実際の乱数を消費しない |
| `assess(public)` / `terminal(public)` | 共通目標の評価と終局判定。`Goal`の値は−1〜1。状態は`running/success/failure/draw/scored`。勝利・敗北・引分は値1/−1/0 |
| `feedback(before, after, action, context, effect)` | その遷移から観測できた結果を`ModelFeedback`で返す。実行・想像のどちらでも使う純粋関数 |
| 実際の実行処理 | 本物の世界・乱数を進め、得られた観測だけを本人へ返す。予測内の経験をここへ合流させない |
| `model_id` | ルール・効果・目的・観測・callbackの意味を版管理する。意味を変えたら別IDにする |

effectには目的への即時効果、欲求、主義、行動様式、費用を載せる。
`CumulativePreference`では、累積成果と同じ意味を持つ価値チャネルと単位換算を宣言する。
主義のすべてを勝利得点に変えない。例のachievementは実得点差そのものなので換算1を使う。
目的点を意味のない即時ボーナスにすると、コアだけでは長期利益を復元できない。

詳細なcontext・effect・Bindingの契約は [DECISION_LOOP](DECISION_LOOP.md)、
現在の安定構成の内部と比較用の任意機能は [OWNED_VALUE](OWNED_VALUE.md) を参照。
`examples/stable_host.py` の `make_controller()` が、採用構成の初期化をまとめた例。
ゲーム側で選んだ人格と主義は維持し、学習時に成績へ合わせて書き換えない。

## 実行・欠測・失敗・保存の順序

1. 現在の公開世界で `controller.decide(public)` を呼ぶ。予測を作れなければ例外となり、即時判断へ勝手に戻らない。
2. 返された `decision` の実行をホストで確定する。`ticket`は今回の本人の未解決判断に結び付く。
3. 実観測を `controller.resolve(ticket, actual_feedback)` へ一度だけ渡す。
4. 解決済みの世界・実乱数・`controller.checkpoint(next_public)`を一つの保存として確定する。
5. 復元ではホストの世界と乱数を先に戻し、同じルール・人物・設定を持つ新しいtemplateの
   `template.restored(public, saved_owner)`から本人を得る。callbackのコードはセーブから復元しない。

欠測は失敗や成功として捏造しない。`ModelFeedback.observed/event/needs/purpose`は別の観測経路で、
得られないものは`None`で表す。返せる観測だけを渡す。公開条件付きの予測内では、
列挙した応答記号に対応する仮想event・必要な進捗/欲求結果を必ず返す。
実行したが失敗した行動は、その失敗を観測として学習する。

`FixedHorizonOwner.decide()`は未解決判断を作る。実行が取り消された場合、取り消しを
悪い結果として学習させず、ホストの実行前保存と本人を一緒に復元する。
実行済みなのに結果の受け渡しだけ失敗した場合は、実行を重複させず未解決ticketへ
正しい観測を渡し直す。解決前の本人は保存できない。
途中でプロセスが落ちうる外部ゲームでは、ホスト側で実行結果の永続記録と再送管理が必要。

不正入力や違う設定のセーブは拒否する。チェックサムは破損検出であり、改ざんへの認証ではない。
端末にある未信頼のセーブを安全なものへ変換する仕組みではない。
保存失敗時は最後の確定済み保存を残す。例は一時ファイルから置き換える方法を示す。

## 対応範囲

明示した公開ルールと効果があるゲームへ接続する判断基盤。
ルールの自動発見、隠れた情報の完全推定、人間水準、全人格の勝率改善、
大量NPCの毎フレームH32実行は認定していない。
公開v0.1.0のライセンス方針は従来どおりで、利用許諾を新たに付与していない。
