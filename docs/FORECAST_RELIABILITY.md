# 予測を信用する前に比較する

目的は人格を残した判断の安定。ゲーム別の手・勝率・係数を共通コアへ追加せず、同じ人格とルールで読み方だけを比較する。標準の反射判断は置き換えない。実結果の正本は [検証報告](../evidence/forecast_reliability/REPORT.md)。

## 有限の計画修正と独立した比較

`ActorModel(..., future_planner=callback, future_planner_limit=1)` は、予測内の本人にも公開観測から一度計画を修正させる。callback は `(detached_actor, raw_request, public_state)` を受け、通常の `DecisionLoop` のplannerを返す。分岐のmodel seedは渡さない。最初の仮置きrootでは呼ばず、実行後の次の公開観測から使う。経験・関係・圧力・内的intentはその本人の複製が保持する。

`ForecastController` はこの接続を使う任意の一人用controller。ゲーム側は公開観測、モデル遷移、終局評価、Request生成、観測できるfeedbackを供給する。`ForecastBudget` が読みの窓・独立した乱数bank・未来の読みの深さ・回数を明示する。

今回の修正版は次の有限な比較をする。

1. ４手・４モデル世界の即時継続で候補を選び、２手・同じ４世界で参照候補を選ぶ。
2. 候補を固定してから、別の４世界でその２候補を評価する。この段階で未来の本人に２手・さらに別の２世界の読みを一度行わせる。
3. 独立評価内の人格の優先層と目的の許容帯を守り、両候補の目的差とその標準誤差を確認する。各枝が絶対に悪化しないことは要求しない。人格のための許容された損と、推定の不確かさを区別する。

主義を守るため、参照候補が最初の全候補の優先層に入らなければ、参照を強制しない。独立評価は固定された候補の比較であり、全ての可能な手の再探索でも普遍的な安全保証でもない。標準誤差１個分の差し引きは検証対象のheuristicで、校正された信頼区間ではない。２bankが同じ誤ったモデルを使えば、どちらも誤る。

実本人は新しい観測ごとに外側の比較をやり直す。予測中の本人は一度の２手読みで、その後は反射へ戻る。**外側controllerを全未来で完全に再現したわけではない。** 有限の近似が有用かを測り、人格や目的を揃えただけで計画全体の整合が保証されるとは扱わない。

追試の `extra_validation_seeds` は、固定候補の比較が不確かで逆転し得る場合だけ評価世界を追加する。候補を後から増やさない。`purpose_challenger=True` は発見段階の人格優先層の最高目的候補も固定し、別の `audit_seeds` で選択を監査する。前者は48条件中１条件改善、後者は48条件中２条件悪化・改善ゼロだった。標準のrunnerでこの２方式を自動選択しない。未来の窓を伸ばした８手追試は既知の確認条件を再使用した探索的比較で、未知ゲームへの独立検証ではない。

## 公開された他者の傾向

`PublicResponses` は既存の `HypothesisTracker` と `EvidenceGate` を使う、本人に所有された有限の他者仮説bank。adapterは公開された合法手・資源・履歴と、宣言した仮想profileを供給する。相手の本当のtraits/values、私的なDecisionLoop、未公開の現在の行動は供給しない。

観測前に仮説の応答分布を凍結し、実際に公開された選択を同じ応答空間で採点する。宣言した中立の予測より最近の予測が良ければ採用し、悪化すれば採用を撤回する。過去の強い証拠だけで現在の外れを隠さない。これは真の人格型の同定ではなく、次の公開選択を見積もるための仮説。

社会試験では本人の関係記憶は予測内でも更新する。相手の仮説bankは実観測時点の複製を固定して各モデル世界で使い、架空の相手から実bankを学習させない。予測内の相手まで仮想学習させる機能は今回含めない。採点が良いことと、その分布を使って有益な判断ができることは実世界の目的値・健康・休息行動で別に測る。

`PublicResponses.record()` / `from_record(scope, record)` は同じ所有者の仮説・最近の検証gateを復元する。別の所有者へ移植しない。凍結した初回社会試験の内部診断recordはこの後のcheckpoint形式とは異なる。予測精度の改善と、本人の目的値・主義の実成果を別に集計した結果、標準採用に足る意思決定の利益は確認できなかった。

## 遅れて判明した予測の採点

`ForecastLedger` は選んだrootの「目的値＠宣言した本人手数」を保存し、その手数が経過した実観測で採点する。実終局が先に来た場合は吸収状態として解決する。割引した選好、即時効果、違う時点の目的値を同じ教師値へ混ぜない。

未選択の手の結果は観測したことにしない。実際の後続手も保存し、想定した手順と違ったことを隠さない。ただしモデル世界の手順と実手順の不一致には乱数や公開状況の違いもあるので、不一致率をそのまま「本人の判断器のずれ」とは呼ばない。

比較用の採点は有限CDFの誤差と、現在の目的値がそのまま続くという参照予測の誤差。runningの目的値はゲームが定義したproxyであり、勝率ではない。異なる読みの長さは採点対象の時点も違う。予測が当たることだけを根拠に行動を採用・却下せず、遅延した成果の因果的な学習責任も自動で割り当てない。

## 再実行

```sh
python -m unittest discover -s tests -p test_forecast_control.py -v
python -m unittest discover -s tests -p test_public_responses.py -v
python tools/run_forecast_reliability.py --output FRESH_DEV --stage development --seeds 4 --games auction_contest hagetaka gambling
python tools/run_forecast_reliability.py --output FRESH_CONFIRM --stage confirmation --seeds 4 --arms short more_samples feedback checked
python tools/run_forecast_reliability.py --output FRESH_SETTLEMENT --stage transfer --seeds 4 --horizon 8 --games auction_contest hagetaka gambling --arms short checked adaptive
python tools/run_forecast_reliability.py --output FRESH_CHALLENGER --stage transfer --seeds 4 --horizon 8 --games auction_contest hagetaka gambling --arms checked challenged
python tools/run_public_peer_reliability.py --output FRESH_SOCIAL --seeds 1
python tools/analyze_forecast_reliability.py --cohorts FRESH_DEV FRESH_CONFIRM --social FRESH_SOCIAL --output FRESH_ANALYSIS
```

runnerは開始前にsourceと条件を凍結する。失敗・退行・同じ行動列も保持し、途中結果は確認済みの全体へ数えない。社会の同一世界のNPCを独立した試合として数えない。新しい心理層・LLM・Colab/GPU・必須依存は追加しない。

上の道具は現在のsourceを凍結して再実行する。過去の凍結sourceと全byte同じ試験になるとは限らない。原比較の権威は公開provenanceのsource hashと条件であり、後のオプション・checkpoint形式・省略ガードの変更はimplementationに明記した。

`direct_continuation=True` は検証のための明示的な純Policy継続。学習・相手reader・圧力・進展監視・社会記憶・root修補を持たないRequestだけを受け、モデルfeedbackも空であることを検査する。それ以外は通常のActorModelを使う。未来のnested planner自体は同じDecisionLoopで選ぶ。この省略の等価性は別に検証し、人格や計画修正を省いて高速化したという成果にはしない。
