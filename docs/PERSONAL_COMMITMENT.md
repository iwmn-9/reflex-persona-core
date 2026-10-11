# 主義への固執と、手段の見込みを分ける

主義の強さは既存の10価値、性格は既存のBig Fiveの数値を保つ。新しい任意設定は二つ。主義を譲る範囲を制御する `rigidity` と、予測の改訂速度を制御する `inertia`。独立なので、信条は頑なだが戦術をよく学ぶ人物も、妥協はするが見込みを修正しにくい人物も作れる。心理学上の測定尺度やBig Fiveからの実証済み換算ではない。ゲーム作者が人物設定として与える。

## 固執度を人物ごとに設定する

```python
from reflex.core import Policy, character_key

policy = Policy(principle_priority='personal', commitments={
    character_key(first_context): 0.2,
    character_key(second_context): 1.0,
})
```

0は元の有限優先と同じ。1は最強の主義の評価が最大値から0.02以内の候補を残し、切迫した欲求でも主義モードを維持する。中間値は主義モードにおける許容差 `0.02 + 1.98 * (1-rigidity)**2` を使う。既存の欲求モードへの遷移は中間値で残る。この曲線は効果の契約範囲[-1,1]を使う設計上の仮定で、人物の心理を測った式ではない。1でも違法手・理解済み失敗を許す設定にはならない。価値が全て0なら固執する価値はない。

固執の対象は候補の価値への寄与であり、特定の行動IDではない。同じ主義を守れる別の手段は、利益・費用・欲求で選び直せる。具体的な禁忌を自動解釈するものではなく、ゲームが結果の意味を値へ定義する。資源を一時的に失うことと信条を破ることを、adapterが同じ値で表すと判断にもその混同が残る。

`character_key`はscope・seed・固定性格・固定価値から作り、目標変更で固執度を失わない。計画の`owner_key`は目的も含む別のキーなので、別目的の計画を黙って再利用しない。未設定の人物はエラーにし、都合のよい固執度を推測しない。

Population/TiledPolicyは人物別の値を一つの数値バッチで使える。DecisionLoopも異なる人物設定を同じ反射バッチへ接続できる。共同の計画器には全参加者を設定したPolicyを明示して渡す。各DecisionLoopのcheckpointはその人物の固執設定一つだけを書き出し、他者の設定を変更しても本人の設定が同じなら復元できる。

現在の投票、未来の反射、目的候補の採点、進展監視の主義保護へ同じPolicyを渡す。`contingent_trials.play(..., policy=policy)`、`observed_transfer.run(..., rigidity=...)`、`contingent_mahjong.play(..., rigidity=...)`が接続例。目的の許容差`max_regret`は別のゲーム設定として残り、固執度の数字を全場面の損失許容率と読み替えない。

## 思い込みと実経験を分ける

```python
from reflex.opponent_beliefs import HypothesisTracker

belief = HypothesisTracker(('old', 'new'),
    prior={'old': 0.95, 'new': 0.05}, assumption_trust=0.9, inertia=0.8)
before = belief.snapshot()
predicted = before.predict(public_models)
# 実際に公開された、選んだ対象の結果だけを後から更新する。
feedback = belief.observe(public_models, revealed_action, observation_id, forecast=predicted)
```

ゲームは仮説ごとの同じ公開応答集合の確率を供給する。主観の初期仮説を保存する推定と、初期仮説なし・標準の更新速度で同じ観測を評価する推定を別に持つ。`evidence_confidence/evidence_weights`は後者、`weights`は本人の主観。観測だけの推定も仮説集合・忘却・予測品質に依存するheuristicで、真相や校正済みの確率ではない。

経験の強さを主観の正しさへ丸ごと流し込まず、両者の重なりも確認する。`confidence`は予測で使う主観と裏付けの混合度、snapshotの`assumption_trust`はその時点の有効な仮説への信用で、constructorの初期設定値とは区別する。初期仮説を経験件数へ加算しない。仮説を区別できない応答だけでは観測の推定は偏らない。

慣性の最大値でも更新率は0.2を残す。予測が大きく外れた際の古い見込みの弱め方も慣性で変わる。最初の不運一件だけで初期仮説を消さず、4件以上の観測後は最近の予測品質も信用へ反映する。4件を魔法の正しさの閾値とは扱わない。強制された一択は仮説の裏付けを増やさず、同一観測を重複更新しない。

状態は`subjective-belief-v2`として保存・復元できる。v1の主観と経験が混ざった形式を暗黙に読み替えない。仮想rolloutで`observe`を呼んで実経験へ混ぜず、予測側はsnapshotを使う。actor/観測対象ごとの所有とモデルの意味づけはゲーム側の責務。

既定のPolicyと、追加設定なしのHypothesisTrackerは維持する。新しい人物設定が全ゲームの強さを増すとは扱わない。[今回の実行結果と得失](../evidence/personal_commitment/REPORT.md)。
