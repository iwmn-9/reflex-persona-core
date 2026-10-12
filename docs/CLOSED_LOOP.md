# 将来の本人も選び直す先読み

`reflex.closed_loop` は任意の条件付き探索器。未来の各公開局面で、複製した同じ `DecisionLoop` が残りの先を読んで選ぶ。その選択後の結果分布を現在へ戻す。通常の反射・人格係数・主義の強さは変更しない。採否と検証結果は [実験報告](../evidence/closed_loop/REPORT.md)。

従来の rollout は、最初の候補の後を主に反射で進めて見積もる。実行時には再び先読みするため、予測内と実行時の本人の選択が違い得る。新方式はその差を有限の木の中で扱う。結果を平均した一つの架空結果にせず、損失を含む分布のまま既存の人格評価へ渡す。

## ゲーム側が供給するもの

`ClosedLoop` へ次の純粋なモデル関数を渡す。ゲーム名・特定の手への補正はコアに入れない。

- `observe(world, owner_state)`、`request_for(context)`：公開観測と同じ本人の要求。
- `transitions(world, action)`：`Transition(probability, successor, realized_flow)` の有限分布。確率は正で合計1。公開ルールから正しく列挙できる場合だけ exact と呼べる。
- `terminal(world)`、`assess(world)`：終局判定と `Goal` の記録。途中値はゲーム側の代理評価であり、成功確率とは限らない。
- `record(world)`：モデル上の完全な公開Markov状態、または信念状態の安定した記録。省略した情報が将来の遷移や評価を変えるなら、不正なキャッシュや分岐統合になる。
- `feedback(before, after, action, effective_context, flow)`：`ModelFeedback`。実行側も同じ意味の実フィードバックを `reveal` / `observe` / `abandon` で確定する。
- `unit`、`target`、`model_id`：累積の単位、目的、モデル識別子。ルール・確率・観測・評価を変えたら識別子を変え、新しい制御器を作る。

`SampledKernel(advance, seeds)` は公開モデルから同じシナリオ群を標本化する代替。実ゲームの未来seedを渡してはいけない。隠れた持続状態は公開信念に保持する必要がある。各節点で新しく標本化するだけで、非公開カードや相手の内面を正しく推定できるわけではない。

## 接続と所有

```python
request = request_for(context)
capture = {}
proposal = tree.planner(owner, request, public_world,
                        remaining=remaining_owner_decisions, captured=capture)
result = DecisionLoop.decide_batch([(owner, request)], False, proposal)[0]
# 実ゲームで result['decision']['action_id'] を実行する。
reveal(owner, result['ticket'], actual_feedback)
```

モデル内の本人は毎枝複製する。経験・欲求・関係・進展監視を実体験へ混ぜない。採用された手の後について、強制した仮想rootと通常の計画採用を経た本人の全保存状態が一致することを検査する。実観測が変われば次の判断を作り直す。モデルと異なる結果が出た場合まで予測の一致を保証するものではない。

現在は**単独の意思決定所有者、決定論的選択**が対象。相手はゲーム側の公開遷移モデルに含む。チーム全員が同時に再交渉する探索器ではない。`reader`、`value_proposal`、独立 `predictor` は予測後の別選択をモデル化していないため拒否する。通常の学習、関係、進展監視との接続は検査済みだが、全任意部品を同時に有効化した推奨構成ではない。

## 計算上限と打切り

`TreeBudget(horizon=4, nodes=4096, outcomes=256)`。上限はそれぞれ16、100000、4096。上限超過は予測**全体**を辞退し、元の判断へ戻す。計算できた候補だけで比較しない。キャッシュ済みの節点も論理的な計算予算に数え、温まったキャッシュだけで予算を通過させない。記録と実効果が同じ分岐だけ確率を合算する。

`compile_batch` の既定結果数8は維持。新経路だけ明示した `outcome_limit` を使う。他の旧合成器も任意幅の分布へ対応した、という意味ではない。

`DecisionWindow` は本人の実行回数で残りの評価区間を減らす。観測tickの差やモデル内の仮想手では進めない。`remaining` は読み取りのみ、実結果を確定してから `advance(context)` を一度呼ぶ。保存復元可能。同じ終点を比較する機構として有用だが、短い固定窓は終局直前の利益を見落として悪化したため**標準方策にはしない**。必要なら毎手同じ長さを読む、終局まで読む、明示した固定窓を使う、という実験条件を分ける。

全分岐の列挙は行動数と深さに対して重い。速度は今回の採否基準にしていない。目的値の誤った代理評価、相手モデルの誤り、少数標本の偏りは、深い木にしても自動では直らない。

`max_regret` は各節点で、本人が保つ主義の階層内の候補に適用する既存の目的損の許容幅。全生涯の損の上限や、目的だけの最善方策との差の保証ではない。目的値と、人格を含む採点と、実際の達成結果を分けて検査する。

## 再現

```text
python -m unittest discover -s tests -p test_closed_loop.py -v
python tools/run_closed_loop.py --root <new-path> --stage graph_confirmation --workers 3
python tools/measure_closed_loop_gambling.py --root <new-path> --workers 3
python tools/run_closed_loop.py --root <new-path> --stage confirmation --workers 3
python tools/probe_closed_loop_frontier.py --root <new-path> --workers 2
python tools/probe_closed_loop_connections.py --root <new-path>
python tools/analyze_closed_loop.py --cohorts <complete-paths> --output <new-file.json>
```

各実験はソースと条件を開始前に凍結し、全登録結果・辞退・失敗を保存する。多人数ゲームの全実行は長時間を要する。終局まで全確率を積分した評価と、実際の1試合の結果は別の証拠として扱う。

`probe_closed_loop_frontier.py` は確認群のGoofspielの失敗を見てから同じ条件で行った追試。残り4手を終局まで読む場合と、その同じ木で未来だけ反射にする場合を分ける。新しい未見条件の成績には数えない。

`probe_closed_loop_connections.py` は共通Bridge全25系統で1手を評価して実行し、次の局面でもう一度判断する。2回ずつの実判断・保存復元・実遷移の再生を確認し、深い読みの強度検証には数えない。別形式の観測系4種とclosed mahjongは新探索に未接続。

生成モデルから有限の先を探索する設計は [Kearns, Mansour & Ng (2002)](https://www.cis.upenn.edu/~mkearns/papers/sparsesampling-journal.pdf) を参考にした。ここでは人格による非線形採点、主義の階層、動く内部状態を使う。同論文の最適性保証がこの実装へそのまま適用されるとは主張しない。
