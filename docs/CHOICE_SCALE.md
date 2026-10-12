# 目的の尺度と公開された相手予測

目的は、人格のために払った損と、採点の基準点・単位・誤った確信から生じた損を分けること。ゲーム側が目的の意味と観測範囲を供給する。同じ数の範囲に入れただけでは、勝利・資源・価値観が同じ単位になるとは扱わない。検証と採否の正本は [報告](../evidence/choice_scale/REPORT.md)。

## 起点・単位・確かさを揃える任意経路

`goal_forecast` に `objective_origin=Goal(...)` を渡すと、主観的な目的評価を終点の水準から、現在の評価との差へ変える。現在の値を使うことも、差を使うことも設計上の選択で、既定は元の水準評価。`ForecastBudget(objective_gain=True)` はadapterの `assess(initial)` を起点としてこの経路を使う。

起点は現在の公開状態から求めた同じ目的座標の値。runningの代理点数を既知の勝率と呼ばない。終局の±1へ切り替わる評価器の意味が同じかもadapterの責務で、単に数値を引けば全ゲームに正しい価値関数が生まれるわけではない。

- 正の増分だけを、rootと経路の支持度の最小値で弱める。両方に同じ支持度が含まれていても二重に掛けない。支持度は確率として校正されていない。
- 既にある起点には、この未来の増分の支持度を掛けない。負の増分は弱めない。
- 主観的な危険評価だけでなく、目的の許容帯・独立候補の比較も、この同じ支持度付き終点を使う。
- 増分は最大２になるので半分で輸送し、`ReturnFrame`で評価単位へ戻す。クリップによる情報損失を避ける。

`objective_unit` は目的の数値座標で１評価単位に相当する量（0.125–1、既定1）。目的の主観得点と `max_regret` の許容量を同じ単位へ接続する。たとえば単位0.5・許容量0.15は、輸送された目的値の差0.075までを許す。単位を変更するだけで目的の重要さを変更することもできるので、ゲームを跨ぐ自動校正とは呼ばない。意味を保った数値の再表現では、起点・終点・単位を対応させる必要がある。

この経路でも欲求・主義・性格・費用の実際のflowを維持する。勝率を全人格で揃える係数は加えない。未来の本人の完全な再帰計画、モデル誤差の自動修正、遅延成果の因果学習を達成したという機能でもない。

## 採用を見送る候補間の正規化

`opportunity_regret=True` は研究対照。人格の全優先層と実行できるrootの中で、候補の最高・最低目的差に `max_regret` を掛けて許容帯にする。未来の評価窓を伸ばすだけで全ての手を同じ尺度へ変換する機能ではない。

決定的な二択では、割合が１未満なら低い目的の択が全て消える。最良でない小さな人格的妥協まで排除し、選択肢を増減すると許容幅も変わる。既定の人格NPCには採用しない。検査の負の対照と凍結した比較を再現するために残す。

## 相手の内部情報を予測へ使わない

`PublicResponses` は、受け取ったpeer packetの `seed`、`state`、`opponent` を実際の相手の内部情報として使わない。仮説の乱数種（既定271・811）、未知の内的状態、相手の私的読みを持たない状態へ置き換える。仮説側でもtraits/valuesを宣言した有限profileへ置換する。合法手・身体資源・公開された行動の見込みから仮説の分布を作る。

仮説の乱数種は相手の真の内部モードを知ったものではなく、未知の判断状態の少数サンプル。実際の公開行動で仮説と検証gateを更新し、架空の行動で実bankを学習させない。未来のbankは実観測時点の複製であり、相手の私的関係記憶や持続した内部モードを正確に再現したわけではない。

チェックポイントは `public-response-v2`。旧v1は入力に相手の乱数種が残る推定器から得たlikelihood/gateなので、自動復元して新推定器の確信として使わない。公開行動の元記録を保存していれば、新推定器で再観測する。旧試験ではharnessのpeer stateは初期状態であり、相手の実際の記憶を受け取っていたという確認ではない。ただし汎用APIへ実mode/intentを入れれば予測を変えられる欠点も修正した。

前段のBrier改善の結果は、乱数種を含む旧推定器の結果として残す。純粋な公開情報から相手を読む能力の証拠として引用しない。[旧試験](../evidence/forecast_reliability/REPORT.md)は履歴として保全し、修正後の結果を新しい報告へ分ける。

## 再実行と検査

```sh
python -m unittest discover -s tests -p test_choice_scale.py -v
python -m unittest discover -s tests -p test_public_responses.py -v
python -m unittest discover -s tests -p test_choice_scale_cache.py -v
python tools/run_choice_scale.py --output FRESH_DEV --stage development --seeds 2 --arms level gain opportunity gain_opportunity
python tools/run_choice_scale.py --output FRESH_CONFIRM --stage confirmation --seeds 4 --arms level gain
python tools/run_choice_scale.py --output FRESH_TRANSFER --stage transfer --seeds 4 --arms level gain
python tools/run_choice_scale_social.py --output FRESH_SOCIAL --seeds 2 --workers 4 --paired-cache
python tools/analyze_choice_scale.py --cohorts FRESH_DEV FRESH_CONFIRM FRESH_TRANSFER --social FRESH_SOCIAL --output FRESH_ANALYSIS
```

runnerは source・条件を開始前に凍結する。`--partial` は監視用で、未完の試験を最終結果へ数えない。身体資源・生存目的と、実際に得た主義のflowを別に集計する。同じ世界のNPCや同じ条件の再実行を独立した試合として水増ししない。

社会試験の `forecast_cache` は比較のための任意な省略。モデル化する世界、本人の全checkpoint、公開context、相手の仮説bank、窓・方式が同じ時だけscenario生成を共有し、価値評価は別に行う。入力が変われば再生成する。512件を超えるとcacheを消す。選んだ手・物理世界・学習・checkpoint・選択済みのモデル記録まで、省略なしの経路と一致することを別に検査する。一般のNPC性能改善として報告しない。

多目的の逐次判断では、目的を一つの数へまとめる関数と方策の種類によって問題が変わる、という整理が参考になる。この実装は、その最適性保証や解法を移植したものではない。[Roijersらのサーベイ](https://arxiv.org/abs/1402.0590)。
