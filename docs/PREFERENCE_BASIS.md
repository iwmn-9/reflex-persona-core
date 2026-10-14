# 人格の代償と代理評価の誤り

`PreferenceBasis` は、人格係数を変えずに「本人が欲しいもの」と「それを得るための近似的な助言」を分ける任意の接続。`decision_audit.policy_loss_audit` は同じ公開モデル・目的・局面で、人格、予測の誤り、代理点の効果を別々に評価する。[検証結果と採否](../evidence/preference_basis/REPORT.md)。

## 入力の意味

入力の同じ `security` でも、資産を持つ安心自体を好む場合と、将来の勝ち筋を残すために資源を温存する場合がある。数値だけからこの違いを推定しない。ゲームの接続側が意味を宣言する。

- 宣言しない軸は、そのまま残す。身体の損傷、他者への利益、今の達成感、行動の大胆さなどを勝率へ吸収しない。
- `instrumental` は、宣言した目的へ到達するためだけの将来見積り。終局まで同じ目的を計算できる場合、その古い近似点を別の心理罰として重複加算しない。
- `purpose` は、同じ目的の達成を指す代理点。各結果分岐の終局目的値と、宣言した単位変換で置き換える。平均値だけにして危険な結果を消さない。

各宣言は**軸全体の意味**。一つの軸で内在的な楽しみと手段的な助言を混ぜた入力は、この契約で安全に分離できない。接続側で先に分ける必要がある。成績が悪いという理由だけで軸を `instrumental` に指定してはいけない。

```python
from reflex.preference_basis import PreferenceBasis
from reflex.opportunity_graph import persona_choices

basis = PreferenceBasis(
    target="declared-purpose",
    instrumental=("values.security",),  # この接続では将来の目的損失の近似だけ
    purpose=(("values.achievement", 1.0),),
)

def choose(states, alternatives):
    return persona_choices(
        [observe(s) for s in states], alternatives, policy,
        preference_basis=basis, target=basis.target,
    )

solution = graph.solve(law, law_id=belief_id, choose=choose,
    chooser_id=basis.identity)
```

`observe` は人物、目標、記憶の十分な状態を渡す。`chooser_id` は実際にはbasisに加えて人格・方策・状態解釈の変更も識別すること。通常の `Policy`、性格、主義の強さ、固執度、危険の評価、局所的な目的許容幅は同じものを使う。

## 実行への接続

`ClosedLoop(..., budget=TreeBudget(horizon=1), frontier=solution, preference_basis=basis)` と、内部の `goal_forecast(..., preference_basis=basis)` にも同じ意味を渡す。未来の選択にだけ補正を入れ、現在の実行を古い採点へ戻すことを避ける。

現在の対応範囲は**当該一手の心理効果＋終局目的分布**。複数手の心理フローを累積する用途や、目的の原点・単位を別変換する組合せは拒否する。分岐の不確かさは明示した結果確率で扱い、追加の branch confidence は1に限定する（根のaction confidenceは従来通り）。途中の未完了見積りで代理点を消すことも拒否する。実際の終局、または `EndpointEstimate.settled_fraction == 1` のモデル内終局被覆が必要。これをモデルの正しさ・予測校正の保証と混同しない。

元の実体験の効果、観測、学習対象は書き換えない。補正したものは判断用の予測入力だけ。basisを指定しない従来の挙動は維持する。

## 損失の比較

四つの方策を、一つの参照確率で終局まで評価する。

|記号|方策を作る条件|
|---|---|
|O|参照確率で目的のみ最大化|
|I|同じ人格・修正した意味・参照確率|
|R|同じ人格・修正した意味・本人の確率予測|
|L|同じ人格・従来の代理点・本人の確率予測|

`O − L = (O − I) + (I − R) + (R − L)`。

- `O − I`：宣言した人格方策の目的上の代償。危険回避、主義、欲求、スタイル、現在と未来の選択、局所的な許容幅を含む。「人間心理として正しい」という判定ではない。
- `I − R`：予測を替えたことで変わる方策の効果。
- `R − L`：同じ本人の予測のまま、代理入力を修正した効果。

後二つは負にもなる。間違った近似が偶然よい目的結果をもたらしていた場合を隠さない。この分解は順序依存で、唯一の因果配分ではない。参照確率＋従来の代理点も渡せば、予測誤差と代理点の交互作用を別途出す。

人格の代償はさらに、同じ本人の未来を前提にした現在の妥協と、未来も本人が選ぶことによる差に分ける。局所許容幅 `.15` は、全行程の目的最適方策に対する損の上限ではない。

比較方策の作り方は呼び出し側の実験契約。ラベルだけでは同じ人格・情報条件を保証できないため、入力とソースを固定して保存する。参照確率が実環境で未知なら、診断もその仮定に条件付けられる。事後の真の相手確率を実行本人へ渡してはいけない。

## 検証経路

```sh
python -m unittest discover -s tests -p test_preference_basis.py -v
python tools/run_preference_basis.py --root FRESH_DEV --cohort development --workers 4
python tools/run_preference_basis.py --root FRESH_HOLDOUT --cohort holdout --workers 4
python tools/analyze_preference_basis.py --root FRESH_HOLDOUT --output analysis.json
```

全条件を実行前に登録し、複製したソースで実行する。Goofspielは以前からの2手全列挙器とも照合する。全比較で選択・目的点・主義階層・許容幅・悪化例を保存し、各人物の3通りの公開結果列では実 `DecisionLoop` を最後まで通して、予測中と実行時の選択を照合する。全分岐の実行再生、対人戦の勝率、人間水準の証明ではない。
