# 公開観測に応じて手順を変える計画

`reflex.contingent_search.search_forecast` は、同じ本人の数値Policy、欲求・主義・目的尺度を使い、後の観測ごとに異なる手を選べる有限の候補を作る。通常の反射経路と既存の共通行動列探索は変更しない。任意部品であり、全ゲームの推奨制御器に昇格したという意味ではない。実結果と採用判断の正本は `evidence/contingent_coverage/REPORT.md`。

## ゲーム側から渡すもの

ゲームは、本人の観測 `observe(state, actor_state)`、モデル遷移 `advance(state, action, model_seed)`、終了条件、`Goal`形式の目的評価、時間単位を供給する。遷移の効果は増減を一度だけ加えるFLOW。残高・盤面価値を毎ターンの収入として繰り返してはいけない。終局の勝敗と途中の代理値を区別する。

本人が知る現在の事実を、`context.facts`へ完全に渡す。`with_public`は明示的に公開とした構造を長さ制限のある文字列へ分ける補助で、公開範囲を自動判断する機能ではない。相手の未公開人格、山札の真の順序、伏せ手、実乱数、将来の事件を渡さない。モデル内の仮想世界が私有状態を持つことと、本人の判断に私有状態を見せることは分離する。

```python
from reflex.contingent_search import search_forecast
from reflex.observation_plan import PlanMemory
from reflex.deliberation import select
from reflex.core import Policy

policy = Policy()  # 呼出元と未来の本人に同じ設定を使う
memory = PlanMemory(initial_context, "game-goal", "owner-decisions")
held, prefix = memory.offer(current_context)
forecast = search_forecast(
    current_context, perceived_model_state,
    observe=observe, advance=advance, terminal=terminal,
    assess=assess, horizon=3, seeds=(401, 402),
    target="game-goal", unit="owner-decisions", policy=policy,
    retained=held, prefix=prefix, width=1, revisions=2,
    patch_budget=4,
)
decisions, vote = select([current_context], forecast, policy=policy)
action = decisions[0]["action_id"]
# 実行が確定した根だけを記憶する。実世界の遷移もゲーム側で行う。
memory.remember(current_context, forecast, vote.get("selected_plan"), action)
```

`DecisionLoop`へ接続する場合は、その準備済みcontextで予測を作り、実際の根マスクを含む通常の投票を使う。`observed_transfer.run(..., search=True, contingent=True, continuity=True)` が進展・欲求圧力の既存経路との接続例。`ModelObserver`は枝ごとの影の進展・圧力を持ち、未来でも同じ公開の許可を守る。仮想の経験を実学習へ混ぜない。

## 有限プログラムと修正

ノードのキーは、実行済みの公開履歴、並べ替えを正規化した観測context、許可された合法手から作る。同じ公開入力には同じ手を使う。モデルseedやworld objectは制御器の引数に入らない。`View`はcontext・allowed・prefix・nodeの四項目だけを渡す。

`ObservationPlan`は最大128個のノードと手の対応を持つ不変のプログラム。該当しない観測、古い観測、許可されない手は、その本人の反射へ戻る。`PlanMemory`は所有者、目的、時間単位、実行済みprefixを確認し、保存・復元できる。失敗した記憶更新は部分的にprefixを進めない。

現在の各合法手には、必ず未変更の反射による継続を残す。そのうえで未来の訪問ノードを部分修正し、全モデル枝へ同じプログラムを適用して評価する。評価には既存の最優先価値層と目的許容幅を使う。勝利だけで候補を作る診断オラクルへ人格を置換しない。

呼び出し側の同じ `Policy` を現在の投票、未来の本人、目的候補の採点へ渡す。`contingent_trials.play(..., policy=policy)`もこの契約を保つ。既定の `principle_priority='lexicographic'` は最優先主義の途中評価で候補を除外できる。既存の `'finite'` は主義を有限の重みとして比較する設定で、人格軸自体は変えないが絶対優先は手放す。長期の打開に有利な局面があっても、全ゲームでの品質改善を確認せず既定へ変更しない。

`horizon`は1～16、`width`は1～4、`revisions`は0～8、`patch_budget`は1～32。修正候補は観測時点ごとに回して選び、遅い情報開示も候補になる。探索は有限で、良い手順を必ず発見する保証はない。数値候補は最大256。多くの合法手があるときは全ての根の基線を保ち、残りの枠を公開IDのハッシュ順で公平に配る。合法手の削除で上限をごまかさない。枠数・探索量はauditへ残す。

修正に使うモデル履歴で到達できなくなった古いノードは圧縮する。モデルにない実観測では反射へ戻り、次の実際の再計画がその現実を取り込む。この圧縮も、珍しい未知状況の対応を完全に保つ保証ではない。

## 検証の意味と限界

`validation_seeds`は候補生成と重ならないモデル枝で、凍結した全候補を採点し直す。生成枝の都合のよい結果だけを採用しないための比較手段で、現実への信頼区間やモデルの正しさを保証しない。未来の乱数リスクと、予測モデルが間違っている不確かさは同じものではない。

予測中は選んだ有限プログラムと本人の反射で続け、実行時は次の観測で再探索する。未来の再探索自身を無限に再帰して正確に予測したものではない。相手仮説が外れること、途中評価が終局の利益を誤ること、実行時の読み直しで選択が変わることは残る。

一人の計画を所有する部品。仲間を含む世界で動作することと、複数人の条件付き合意を交渉・維持することは別。共同案の条件付き契約は今回の一般化に含めない。人格軸の値を維持することと、人間として面白い判断や人格ごとの成績を保ったことも区別する。

過去の[調査と容量診断](../evidence/observation_capacity/REPORT.md)を実装の起点にした。POMCP/DESPOTそのものや、その最適性・安全性の保証を実装したとは呼ばない。

## 再現する

`tools/run_contingent_coverage.py --output FRESH_DIRECTORY` は先にソース・条件を凍結し、そのソースから独立CPU試合を実行する。`--stage confirmation`は別の盤面・人数・供給・条件を使う。負けた条件や未対応の範囲を削除しない。`--validate`は一般ゲームbridgeで独立モデル枝の比較を行う。生履歴とソースはローカル、公開は集計と来歴。再実行は新しい性能標本に数えない。

`contingent_games.py`は検証用ゲームadapter、`contingent_trials.py`は一試合の接続であり、判断コアへゲーム固有の分岐を持ち込まない。麻雀の`contingent_mahjong.py`は実際の4人閉じ手対局を進めるが、条件付きモデルは自分の手牌2判断の代理値だけ。対局全体、役・打点、相手の鳴き、放銃確率を予測したものではない。

現在の登録は一般bridge 25件、進展・欲求圧力を含む既存observer接続4件、麻雀1件の30個の名前付き接続。複数の接続が同じルールを使うため、30種類の独立ゲームと呼ばない。移動・循環対戦・工程、4目並べ、リバーシ、コリドール、拒否カード、同時入札、競り、賭け、資源と複数勝利条件、チーム資源/戦闘、暮らし、取引/共同警戒、旧小型adapter、公開生成グラフを含む。`BRIDGES`へ追加したものは集計時に全4人格の比較が必要になり、未試験の接続を自動的に成功へ数えない。

全ルールの基本接続と、全ての制御機能を統合したことは別。一般bridgeは本人一人を制御し、関係記憶・感情評価・実経験学習・共同合意を新経路へ一律に接続したものではない。既存の強い対戦相手向け制御器も全面置換しない。生成グラフbridgeはソースcheckout中の既存診断toolを再利用するため、wheelだけの環境では使わない。

通常の多数NPCの反射は既存の数値バッチ。計画探索を追加したCPU実験を、そのまま多数NPCの並列速度の測定と読み替えない。ゲームは知能予算を明示する。比較バッチも、コリドールは2判断・幅1・修正1回・提案2個、一般接続は3判断・幅1・修正2回・提案4個、麻雀は自分の手牌2判断・修正1回という別の予算を記録する。

計算基盤の中断で終了したバッチは、`tools/resume_contingent_coverage.py --root STUDY --workers 4`で未完の試合だけを凍結ソースから再開できる。完了済みJSONLのprefixと登録SHAを確認し、旧集計と再開の来歴を保存する。数値・ルールのエラーは再開対象にしない。元のソースの修正が必要な不具合は、別の登録で影響する条件全体をやり直す。
