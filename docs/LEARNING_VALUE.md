# 学んだ後の判断を含めた計画

`reflex.belief_planning.BeliefKernel` は、複数の仮説と、その後に本人が観測できる結果を既存の `ClosedLoop` へ接続する任意部品。情報量に報酬を付けない。観測で見込みが変わり、同じ人格の次の選択が変わることを、元の目的・欲求・主義で評価する。実測・不採用案・限界は [検証報告](../evidence/learning_value/REPORT.md) に置く。

## 入力と責務

ゲーム側は、有限の仮説名、事前確率、`model(hypothesis, public, action)`、公開状態の保存形式、目的の意味を提供する。コアはゲーム名や勝ち手を持たない。仮説は敵の行動様式、資源の収量、未知の成功率などを表せるが、ゲームのルールから自動的に仮説を発明・学習する機能ではない。

モデルは仮説ごとに、確率・観測後の公開状態・観測した効果・次の仮説を `HiddenOutcome` で返す。次の仮説を変えれば、相手や環境が変化する過程も宣言できる。実際に選ばれた隠れた仮説、未来の乱数、未公開の相手の手をモデル入力へ渡さない。

`public_record` は本人が区別できる情報を過不足なく記録する。`flow` も本人が観測する情報として扱う。未観測の報酬や内部状態は、アダプターで先に周辺化する必要がある。この公開性は、一般的なPythonコールバックの中身から自動で証明できない。

## 観測と更新

同じ公開状態・同じ観測効果に到達する隠れた分岐を、**次の意思決定より前に合流**する。次の本人へ渡すのは、それらを合計して更新した事後確率。一つの試走で選ばれた隠れた仮説を、本人が知ったことにしない。

`InformationState` は公開状態、仮説確率、モデル・観測者の識別子、観測回数を保持する。`kernel(state, action)` は想像上の公開分岐を返す純粋な処理。実行時は `kernel.reveal(state, action, observed_public, observed_flow)` で、実際の観測だけを反映する。入力や別の分岐は変更しない。

実観測がモデルの支持範囲に存在しなければ `UnsupportedObservation` とし、事後確率を捏造しない。呼出側がその実観測を保持してモデルを改訂する必要がある。これは自動復旧や、未知の敵への対応完成を意味しない。静的モデルで確率ゼロにした仮説は自然には復活しない。変化があるならその遷移を宣言する。

`record` / `restore` はモデル・観測者が一致する状態だけを復元する。実ゲームでは `DecisionLoop` とこの知識状態を同じ実行時点で保存する。想像上の更新を本人の実体験の履歴へ保存しない。モデル・知覚・効果の意味を変えたら `model_id` を更新する。

## 接続と比較

`ClosedLoop` の `transitions` にカーネルを渡し、`observe` は `InformationState` の公開状態と確率から候補効果を作る。`record` にはカーネルの保存形式を使う。既存の同一本人の複製・フィードバック・将来の選択をそのまま利用する。

`update=False` は検証用の比較条件。隠れた状態の変化は予測するが、探索内では観測による確率の更新を止める。実行時の更新は通常どおり行う。したがって「今までの情報は使えるが、次に学べることを今の計画に入れない」という差を比較できる。実用の忘却・頑固さの実装としては使わない。

`PolicyTail` の公開状態にも接続できる。その場合、短い探索の先でも仮説を更新し、本人の反射で次を選ぶ。ただし反射の継続は、将来も探索する本人とは同一でない。短い区間の人格効果しか評価しないという [既存の限界](FRONTIER_VALUE.md) も残る。

## 採用範囲

- 通常の反射や従来の相手記憶の既定は変えない。既存の主観的な思い込み・頑なさを、この厳密な有限モデルの確率更新へまだ合成していない。
- 事後確率は**宣言したモデル内**の確率。モデルそのものの正しさや、未知の敵に対する校正済みの信用ではない。
- 全分岐を評価する有限モデルでは、終局まで同じ本人が判断する予測と実行を照合できる。短い探索、標本モデル、欠落した仮説への一般保証はない。
- 同じ資源の同じ増減には、行動名によらず同じ効果の意味を与える。情報取得の費用だけ欲求・主義への損失から外すと、学習の利益を水増ししてしまう。検証用の資源・賭けゲームはこの不均衡を修正した。
- この部品の役割は、学習による将来の選択変更を評価可能にすること。過去の `TerminalMemory` の汎用的な判断利益や、長い計画の打切り問題を解決したという意味ではない。

## 研究との対応

[Poole / Mackworth の Value of Information](https://artint.info/2e/html2e/ArtInt2e.Ch9.S4.html) は、情報の価値を、それを使って将来の選択を変えられることによる効用の差として扱う。本実装も、情報を得たことへの固定加点を避ける。完全に合理的な選択者なら情報を無視できるが、有限の人格方策について目的値が必ず改善するとは主張しない。

[Guez / Silver / Dayan の BAMCP](https://proceedings.neurips.cc/paper_files/paper/2012/hash/35051070e572e47d2c26c241ab88307f-Abstract.html) と [Arumugam / Singh の情報獲得期間の研究](https://proceedings.neurips.cc/paper_files/paper/2022/hash/80b7bec60081f95d900973509744a306-Abstract-Conference.html) は、知識の変化を状態へ含めて探索と活用を扱う参考。本実装はBAMCPやその近似アルゴリズムの再実装ではなく、小さい有限の条件付きモデルを既存の人格選択器へ接続する。

## 再現

```sh
python tools/run_learning_value.py --root FRESH_DEVELOPMENT --stage development
python tools/run_learning_value.py --root FRESH_CONFIRMATION --stage confirmation
python tools/run_learning_value.py --root FRESH_CONTINUATION --stage continuation
python tools/probe_learning_connections.py --root FRESH_CONNECTIONS
python -m unittest discover -s tests -p test_belief_planning.py -v
```

32個の実行乱数は同じ有限条件の反復。全分岐での期待値の比較件数と、実行エピソード数は分けて報告する。オークションは既存の3人用決済ルール、他の3系統は新しい自作診断ルールであり、既存の大型戦闘・帝国ゲームでの勝率改善とは数えない。
