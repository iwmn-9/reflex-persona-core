"""Export completed native games as one offline behavioral replay."""
import argparse,json
from collections import Counter,defaultdict
from pathlib import Path

PAGE=r'''<!doctype html><html lang="ja"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>人格AI — 実対戦リプレイ</title><style>
*{box-sizing:border-box}body{margin:0;background:#f4f5f2;color:#202a29;font:15px/1.6 system-ui,sans-serif}main{max-width:1280px;margin:auto;padding:24px}h1{font-size:24px;margin:0}p{margin:6px 0 18px;color:#596763}header{border-bottom:1px solid #cbd3cd;padding-bottom:16px;margin-bottom:20px}.controls{display:flex;gap:12px;align-items:center;flex-wrap:wrap}select,button{font:inherit;padding:8px;border:1px solid #abbcb4;background:white;border-radius:5px}select{max-width:100%}button{cursor:pointer}input[type=range]{flex:1;min-width:180px}section{display:grid;grid-template-columns:1.1fr 1fr;gap:20px;margin-top:20px}.panel{background:white;border:1px solid #d1dbd5;border-radius:8px;padding:18px;overflow:auto}.cards{display:grid;grid-template-columns:1fr 1fr;gap:10px}.actor{padding:12px;border-left:4px solid #668b78;background:#f4f7f5}.actor h3{margin:0 0 4px;font-size:16px}.muted{color:#66766f;font-size:13px}.action{font-size:17px;font-weight:600}svg{width:100%;max-height:440px}table{border-collapse:collapse;width:100%}td,th{padding:7px;text-align:left;border-bottom:1px solid #e1e6e2}.hand{font-size:12px;word-spacing:4px}.events{margin-top:12px}#history{max-height:270px;overflow:auto}.event{border-bottom:1px solid #e5e9e7;padding:6px 0}details{margin-top:18px}pre{white-space:pre-wrap;font-size:12px}@media(max-width:860px){section{grid-template-columns:1fr}.cards{grid-template-columns:1fr}main{padding:14px}}</style>
<main><header><h1>人格AI — 実対戦リプレイ</h1><p>全員が同じ公開局面から選択し、実結果で学習する。表示された意図はモデルの予測。行動が人間らしく賢いかは、実際の展開で確かめる。</p>
<div class="controls"><select id="match" aria-label="対戦を選択"></select><label><input id="identity" type="checkbox">人格名を表示</label><label><input id="before" type="checkbox">行動前の盤面</label></div></header>
<div class="controls"><button id="prev">←</button><button id="play">再生</button><button id="next">→</button><input id="turn" type="range" min="0" value="0" aria-label="手番"><b id="clock"></b></div>
<section><div class="panel"><h2 id="title"></h2><div id="world"></div><div id="events" class="events"></div></div><div class="panel"><div id="actors" class="cards"></div></div></section>
<section><div class="panel"><b>直近の行動</b><div id="history"></div></div><div class="panel"><b>対戦情報</b><div id="meta"></div><p class="muted" id="model"></p><details><summary>この手の数値を表示</summary><pre id="raw"></pre></details></div></section></main>
<script>const matches=__DATA__;
const $=id=>document.getElementById(id),esc=v=>String(v).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const labels={combat:'3対3戦闘',hagetaka:'4人ハゲタカ',exchange:'4人の協力・取り分競争'},modes={reflex:'即時判断',planned:'全員が計画',challenger:'一方に計画を導入'};
const reasons={reflex:'即時判断',keep:'中間目標を継続',replan:'計画を比較',reached:'中間目標に到達',withdraw:'計画を撤回',agreement:'本人が評価し、参加者全員が同意'};
function action(s){const [a,b,c]=s.split(':');const n={guard:'防御',reload:'装填',rest:'休息',alone:'単独活動',finished:'行動終了'};return n[a]||({move:`移動 (${b},${c})`,shoot:`P${b}を射撃`,heal:`P${b}を回復`,cooperate:`P${b}へ協力`,claim:`P${b}へ取り分を要求`}[a])||`札 ${s}`;}
let timer=null;matches.forEach((m,i)=>{let o=document.createElement('option');o.value=i;o.textContent=`${labels[m.game]} / ${m.scene} / ${modes[m.mode]} / ${m.seed}`;$('match').append(o)});
function render(){const m=matches[+$('match').value],t=+$('turn').value,r=m.trace[t],w=$('before').checked?r.before:r.after;
$('turn').max=m.trace.length-1;$('clock').textContent=`${t+1} / ${m.trace.length} 手`;$('title').textContent=labels[m.game];
if(m.game==='combat'){let cells='';for(let y=0;y<5;y++)for(let x=0;x<9;x++)cells+=`<rect x="${x*52}" y="${y*52}" width="52" height="52" fill="${x===4&&y>=1&&y<=3?'#e3edc9':'#f8faf7'}" stroke="#d4ddd6"/>`;for(let [x,y] of w.walls)cells+=`<rect x="${x*52+3}" y="${y*52+3}" width="46" height="46" fill="#66716e"/>`;for(let [x,y] of w.covers)cells+=`<path d="M${x*52+5} ${y*52+46}L${x*52+46} ${y*52+5}" stroke="#b5be95" stroke-width="8"/>`;w.units.forEach((u,i)=>{if(u.hp>0)cells+=`<circle cx="${u.x*52+26}" cy="${u.y*52+23}" r="19" fill="${u.team?'#be744d':'#467896'}"/><text x="${u.x*52+26}" y="${u.y*52+28}" text-anchor="middle" fill="white" font-size="13">P${i}</text><text x="${u.x*52+26}" y="${u.y*52+48}" text-anchor="middle" font-size="11">HP${u.hp}</text>`});$('world').innerHTML=`<svg viewBox="0 0 468 260" role="img" aria-label="戦闘盤面">${cells}</svg><div>拠点の連続占有：${w.hold.map((n,i)=>`チーム${i} ${n}/3`).join(' / ')}　確保：${w.secured.map(v=>v?'確保済み':'未確保').join(' / ')}</div>`;}
else if(m.game==='hagetaka'){$('world').innerHTML=`<p>場札 <b>${r.before.prize}</b> ＋ 繰越 ${r.before.carry}</p><table><tr><th>NPC</th><th>得点</th><th>残り手札</th></tr>${w.hands.map((h,i)=>`<tr><td>P${i}</td><td>${w.scores[i]}</td><td class="hand">${h.join(' ')}</td></tr>`).join('')}</table>`;}
else{$('world').innerHTML=`<table><tr><th>NPC</th><th>健康</th><th>活動力</th><th>備蓄</th><th>累計収入</th></tr>${w.people.map((p,i)=>`<tr><td>P${i}</td><td>${p.health.toFixed(2)}</td><td>${p.energy.toFixed(2)}</td><td>${p.stock.toFixed(2)}</td><td>${p.score.toFixed(2)}</td></tr>`).join('')}</table>`;}
let events=[];if(r.agreements?.length)events.push(`合意 ${r.agreements.length} 件`);if(m.game==='exchange')r.actors.forEach(a=>{let [kind,id]=a.action.split(':');let other=r.actors[+id];if(other&&other.action.split(':')[1]===String(a.player)&&a.player<+id)events.push(`P${a.player} ${action(a.action)} ／ P${id} ${action(other.action)}：接触成立`)});
if(m.game==='combat')r.after.units.forEach((u,i)=>{let d=u.hp-r.before.units[i].hp;if(d)events.push(`P${i} HP ${d>0?'+':''}${d}`)});
if(m.game==='hagetaka')r.after.scores.forEach((v,i)=>{let d=v-r.before.scores[i];if(d)events.push(`P${i} 得点 ${d>0?'+':''}${d}`)});$('events').textContent=events.join(' / ');
$('actors').innerHTML=r.actors.map(a=>{const p=a.planning,q=p.milestone;const learned=m.individual_beliefs?`他者の公開行動を観測 ${a.public_behavior_observations} 回`:`相手モデルを識別できた観測 ${a.prediction_observations} 回`;return `<article class="actor"><h3>P${a.player}${$('identity').checked?' · '+esc(a.name):''}</h3><div class="action">${esc(action(a.action))}</div><div>${esc(reasons[p.reason]||p.reason)}</div>${q?`<div class="muted">目標値 ${q.value.toFixed(3)} ／ 期限 ${q.deadline}手<br>予定：${q.remaining.map(action).map(esc).join(' → ')}</div>`:''}<div class="muted">実目的値 ${a.purpose_before.toFixed(3)} → ${a.purpose_after.toFixed(3)}${p.modeled_purpose===undefined?'':`<br>試走の目的値 ${p.modeled_purpose.toFixed(3)}`}<br>実結果 ${a.learning_observations} 回<br>${learned}</div></article>`}).join('');
$('history').innerHTML=m.trace.slice(Math.max(0,t-5),t+1).map(h=>`<div class="event"><b>${h.tick+1}手</b> ${h.actors.map(a=>`P${a.player} ${esc(action(a.action))}`).join(' ／ ')}</div>`).join('');
$('meta').innerHTML=`<p>${esc(modes[m.mode])}<br>世界乱数 ${m.seed} ／ 読みの上限 ${m.horizon} 手<br>${m.factual_decisions} 実判断 ／ ${m.seconds.toFixed(2)} 秒で完走<br>個体別予測 ${m.individual_beliefs?'あり':'なし'} ／ 任意合意 ${m.coordination?'あり':'なし'}</p>`;$('model').textContent=(m.individual_beliefs?'他者ごとの公開行動から予測を更新する。':'3種類の全員行動モデルで予測する。')+(m.coordination?'通信できる相手とは今の意図を共有し、参加者全員が受け入れた案だけ実行する。':'')+'相手の人格や未公開の札順は参照しない。観測回数は予測精度ではない。人間水準を確認した版ではない。';$('raw').textContent=JSON.stringify({actors:r.actors,agreements:r.agreements||[]},null,2);}
function step(n){let m=matches[+$('match').value];$('turn').value=Math.max(0,Math.min(m.trace.length-1,+$('turn').value+n));render();}
$('match').onchange=()=>{$('turn').value=0;$('turn').max=matches[+$('match').value].trace.length-1;render()};$('turn').oninput=render;$('identity').onchange=render;$('before').onchange=render;$('prev').onclick=()=>step(-1);$('next').onclick=()=>step(1);$('play').onclick=()=>{if(timer){clearInterval(timer);timer=null;$('play').textContent='再生';}else{timer=setInterval(()=>{if(+$('turn').value>=+$('turn').max){clearInterval(timer);timer=null;$('play').textContent='再生';}else step(1)},900);$('play').textContent='停止';}};render();</script></html>'''

def summarize(matches,sources):
    """Same selected matches as the viewer; scores are not humanlike-quality labels."""
    index={(r['game'],r['scene'],r['seed'],r['mode']):r for r in matches}
    if len(index)!=len(matches):raise ValueError('summary requires one configuration per match')
    cases=[]
    for r in matches:
        if r['mode']!='challenger':continue
        b=index.get((r['game'],r['scene'],r['seed'],'reflex'))
        if b is None:continue
        if any(r.get(k)!=b.get(k) for k in ('horizon','individual_beliefs','coordination','profiles','focal_players')):
            raise ValueError('paired controllers require equal settings and personalities')
        if r.get('second_moves',8)!=b.get('second_moves',8):
            raise ValueError('paired controllers require equal second-move budgets')
        if r.get('max_regret',.15)!=b.get('max_regret',.15):
            raise ValueError('paired controllers require equal purpose filters')
        if r.get('intent_sharing',False)!=b.get('intent_sharing',False):
            raise ValueError('paired controllers require equal communication')
        if r.get('intention_protocol')!=b.get('intention_protocol'):
            raise ValueError('paired controllers require equal intention protocols')
        if r.get('frozen_reference',False)!=b.get('frozen_reference',False):
            raise ValueError('paired controllers require equal reference design')
        if r.get('scenario_batches',1)!=b.get('scenario_batches',1):
            raise ValueError('paired runs require equal candidate sample settings')
        if r.get('frozen_reference'):
            for i,c in enumerate(r['controllers']):
                if i not in r['focal_players'] and c!=b['controllers'][i]:
                    raise ValueError('frozen opponents must use identical controller settings')
        focal=r['focal_players'];delta=sum(r['purpose'][i]-b['purpose'][i] for i in focal)/len(focal)
        cases.append(dict(game=r['game'],scene=r['scene'],seed=r['seed'],focal_players=focal,delta=delta,
            classification='equivalent' if abs(delta)<=.005 else 'better' if delta>0 else 'worse',large_negative_diagnostic=delta<=-.02))
    groups={}
    for game in sorted(set(c['game'] for c in cases)):
        cs=[c for c in cases if c['game']==game]
        groups[game]=dict(pairs=len(cs),mean_purpose_delta=sum(c['delta'] for c in cs)/len(cs),
            classification=dict(Counter(c['classification'] for c in cs)),large_negative_cases=[c for c in cs if c['large_negative_diagnostic']])
    social=Counter();actions=defaultdict(Counter)
    for r in matches:
        if r['game']!='exchange' or r['mode']!='planned':continue
        social['matches']+=1
        for t in r['trace']:
            social['ticks']+=1;social['all_alone_ticks']+=all(a['action']=='alone' for a in t['actors'])
            for a in t['actors']:
                actions[a['profile']][a['action'].split(':')[0]]+=1
                if not a['action'].startswith('cooperate:'):continue
                p=a['player'];q=int(a['action'].split(':')[1]);other=t['actors'][q]['action']
                if p<q and other==f'cooperate:{p}':
                    social['mutual_pairs']+=1
                    social['mutual_pairs_with_both_gaining']+=all(t['after']['people'][i]['score']>t['before']['people'][i]['score'] for i in (p,q))
                if other==f'claim:{p}':social['cooperator_facing_claim']+=1
    return dict(format='native-arena-summary-v1',complete=True,stable_adoption=False,sources=sources,selected_matches=len(matches),
        paired_comparisons=len(cases),all_planned_matches=sum(r['mode']=='planned' for r in matches),
        factual_decisions=sum(r['factual_decisions'] for r in matches),seeds=sorted(set(r['seed'] for r in matches)),
        horizon=sorted(set(r['horizon'] for r in matches)),
        second_moves=sorted(set(r.get('second_moves',8) for r in matches)),
        max_regret=sorted(set(r.get('max_regret',.15) for r in matches)),
        intent_sharing=sorted(set(r.get('intent_sharing',False) for r in matches)),
        intention_protocol=sorted(set(r.get('intention_protocol','sequential' if r.get('intent_sharing') else 'none') for r in matches)),
        frozen_reference=sorted(set(r.get('frozen_reference',False) for r in matches)),
        scenario_batches=sorted(set(r.get('scenario_batches',1) for r in matches)),
        seconds=dict(min=min(r['seconds'] for r in matches),max=max(r['seconds'] for r in matches)),
        by_game=groups,exchange_all_planned=dict(**social,profile_actions=dict(actions)),focal_comparisons=cases,
        limits=['Purpose deltas are adapter-defined, not win-rate deltas.',
            'Single-match losses are diagnostic cases, not by themselves reproducible condition-level regressions.',
            'Humanlike appeal and benefit from learning alone are not established.'])

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('folders',nargs='+',type=Path);p.add_argument('--out',type=Path,required=True);p.add_argument('--summary',type=Path);a=p.parse_args()
    matches=[]
    seen=set()
    sources=[]
    for folder in a.folders:
        selected=set()
        for f in sorted(folder.glob('*.json')):
            r=json.loads(f.read_bytes())
            if not (r.get('complete') and r.get('game') and r.get('trace')):continue
            identity=(r['game'],r['seed'],r['mode'],r['horizon'],r.get('second_moves',8),r.get('max_regret',.15),r.get('scenario_batches',1),r.get('intent_sharing',False),r.get('intention_protocol','sequential' if r.get('intent_sharing') else 'none'),r.get('frozen_reference',False),r.get('individual_beliefs',False),r.get('coordination',False),json.dumps(r['trace'][0]['before'],sort_keys=True))
            if identity in seen:continue
            seen.add(identity)
            selected.add(r['game'])
            r['scene']=' / '.join(f.stem.split('-')[1:3])
            matches.append(r)
        if selected:
            plan=json.loads((folder/'plan.json').read_bytes()) if (folder/'plan.json').exists() else {}
            sources.append(dict(folder=str(folder),games=sorted(selected),binary_sha256=plan.get('binary_sha256')))
    matches.sort(key=lambda r:(r['mode']!='planned',('exchange','combat','hagetaka').index(r['game']),r['seed'],r['scene']))
    if not matches:raise ValueError('completed actual matches required')
    if a.summary:
        for source in sources:
            campaign=json.loads((Path(source['folder'])/'summary.json').read_bytes())
            if not campaign.get('complete') or campaign.get('failures'):
                raise ValueError('finish the source campaign before exporting a complete summary')
        summary=summarize(matches,sources)
    a.out.parent.mkdir(parents=True,exist_ok=True)
    data=json.dumps(matches,ensure_ascii=False,separators=(',',':')).replace('<',chr(92)+'u003c')
    a.out.write_text(PAGE.replace('__DATA__',data),encoding='utf8')
    if a.summary:
        a.summary.parent.mkdir(parents=True,exist_ok=True)
        a.summary.write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf8')
    print(json.dumps(dict(matches=len(matches),output=str(a.out))))

if __name__=='__main__':main()
