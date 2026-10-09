"""Generate a finite, reproducible demonstration and trim previously audited runs."""
from pathlib import Path
import argparse
import copy
import hashlib
import json
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.persona_review import immediate, forecast_frame, unchanged_streak, render
from reflex.core import Policy, TRAITS, digest
from reflex.examples import context, action, effect
from reflex.laboratory import Laboratory, profiles

BASE = '9b33a806b9f09fd6b56254587cd79d87a28f3458'
FIRST = '64b542f0e4b04f7385576d12f75d1cf72ab0c469'


def read_verified(folder, audit_path, commit):
    audit = json.loads(audit_path.read_text(encoding='utf-8-sig'))
    path = folder/'trajectories.jsonl'
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    if not audit['source_freeze_verified'] or not audit['actual_checkpoints_replayed'] or sha != audit['trajectories_sha256']:
        raise ValueError('matching completed replay evidence required')
    registration = json.loads((folder/'preregister.json').read_text(encoding='utf-8'))
    for name, expected in registration['source_hashes'].items():
        blob = subprocess.check_output(['git', 'show', commit+':reflex/'+name], cwd=ROOT)
        if hashlib.sha256(blob).hexdigest() != expected: raise ValueError('source mismatch: '+name)
    rows = [json.loads(line) for line in path.open(encoding='utf-8')]
    if len(rows) != audit['episodes']: raise ValueError('episode count mismatch')
    return rows, dict(commit=commit, trajectories_sha256=sha, episodes=audit['episodes'],
        actual_rule_transitions=audit['actual_rule_transitions'], replay_completed=True)


def roster():
    return {p['id']: dict(id=p['id'], name=p['name'],
        personality=dict(zip(TRAITS,p['traits'])), values=p['values']) for p in profiles()}


def abstract_cases():
    acts = [action('take', effect(.8, {'physiology':.2}, {'benevolence':-.5,'power':.6})),
            action('share', effect(.45, {'belonging':.4}, {'benevolence':.65,'power':.1})),
            action('recover', effect(.15, {'physiology':.75}, {'security':.5}, cost=.03))]
    cases = []
    for label, needs in [('取り分と協力',{'physiology':.2,'growth':.2}),
                         ('活動余力が切迫',{'physiology':.9,'growth':.2})]:
        tracks = []
        for p in profiles():
            c = context('review-allocation', copy.deepcopy(acts), needs, p['values'], dict(zip(TRAITS,p['traits'])), mode=None)
            c['scope']['npc'] = p['id']; c['seed'] = 1010
            r = immediate(c)
            tracks.append(dict(id=p['id'], profile=p['id'], variant='finite',
                final=None, frames=[dict(turn=1, action=r['choice'], state=r['state'],
                    immediate=r, actual_goal=None, verdict='pending_human_review')]))
        cases.append(dict(id='allocation-'+str(len(cases)), kind='snapshot', title=label,
            description='四人へ同じ選択肢と不足を渡した独立判断。相互作用する対戦ではない。効果の数値は作者設定で、実行後の結果はまだない。',
            tracks=tracks, turns=1, fact='自分の取り分・協力・回復の得失が公知。人格と主義だけが異なる。'))
    return cases


def laboratory_cases():
    result = []; checks = dict(decisions=0, illegal=0, understood_failure_repeats=0, snapshot_replay=0, reading_changes=0)
    for scenario, title in [('ordinary','四人の暮らし：通常'),('scarcity','四人の暮らし：供給損失'),('contest','四人の暮らし：取り分の競合')]:
        lab = Laboratory(seed=1010, scenario=scenario, reading=True)
        lab.population.policy = Policy(principle_priority='finite')
        rows = lab.run(24, capture=True); tracks = []
        for p in profiles():
            fs = []
            for r in rows:
                if r['npc'] != p['id']: continue
                calculated = immediate(r['context'])
                if calculated['choice'] != r['action']: raise ValueError('snapshot choice does not replay')
                score = next(x['score'] for x in calculated['choices'] if x['action']==r['action'])
                if abs(score-r['choice_score']) > 1e-9: raise ValueError('snapshot score does not replay')
                checks['snapshot_replay'] += 1
                fs.append(dict(turn=r['round']+1, action=r['action'], before=r['before'], after=r['after'],
                    state=dict(primary_need=r['primary'], mode=r['mode']), immediate=calculated,
                    reading=dict(used=r['reading_used'], changed=r['action_changed_by_reading'],
                        reflex_action=r['reflex_action'], confidence=r['read_confidence'], partner=r['partner']),
                    cause=r['cause'], verdict='pending_human_review', actual_goal=None))
            final = next(a for a in lab.summary(rows) if a['npc']==p['id'])
            tracks.append(dict(id=p['id'], profile=p['id'], variant='finite', frames=fs, final=final))
        checks['decisions'] += len(rows); checks['illegal'] += lab.illegal
        checks['understood_failure_repeats'] += lab.failure_after_known
        checks['reading_changes'] += sum(r['action_changed_by_reading'] for r in rows)
        extra = ('13手目の判断前に、供給の80%または2の大きい方を失う（下限0）。'
            if scenario=='scarcity' else '挑戦志向と自己主張、安定志向と協力志向がそれぞれ相互作用する。'
            if scenario=='contest' else '挑戦志向と安定志向、協力志向と自己主張がそれぞれ相互作用する。')
        result.append(dict(id='lab-'+scenario, kind='laboratory', title=title, turns=24,
            description='四人が先に同時判断し、その後に結果を解決する。性格・主義は固定、欲求と相手の観測履歴は変化する。有限Policy、seed1010の固定実演。',
            fact='供給は毎手0.10維持費。協力は相手にも供給を渡す。取り分は相手と競合すると減る。休息・成長には活動余力が必要。'+extra, tracks=tracks))
    return result, checks


def recorded_case(rows, spec, variants, title, key, seed, description):
    selected = [r for r in rows if r['spec']==spec and r['variant'] in variants and r['seed']==seed]
    if len(selected) != 4*len(variants): raise ValueError('complete four-profile comparison required')
    if any(r['trace'][0]['before'] != selected[0]['trace'][0]['before'] for r in selected):
        raise ValueError('common initial world required')
    tracks = []
    for p in profiles():
        for v in variants:
            r = next(x for x in selected if x['profile']==p['id'] and x['variant']==v)
            frames = [forecast_frame(r['trace'], i) for i in range(len(r['trace']))]
            for f, n in zip(frames, unchanged_streak(frames)): f['unchanged_streak'] = n
            tracks.append(dict(id=p['id']+'-'+v, profile=p['id'], variant=v, frames=frames,
                final={k:r[k] for k in ('status','goal_value','steps','surviving','shortages','solvable_route_lost')}))
    rules = {'resources':'目標：技術段階3かつ研究蓄積14以上。研究は蓄積・鉱石・資金を使って技術を上げる。',
        'delivery':'目標：自分の納品6または援助4を期限内に達成。援助も正式な達成ルート。',
        'combat':'目標：本人が生存して相手を倒す。相打ちは引き分け。射撃は同時に解決する。'}
    return dict(id=key, kind='recorded', title=title, tracks=tracks, turns=max(len(t['frames']) for t in tracks),
        description=description+' 四人格・両設定を省かず収録。開始状態は同じで、選択後は別々の世界になる。',
        fact=rules[spec['genre']]+' モデルの目的値は校正済みの勝率ではない。', spec=spec, seed=seed)


def build(evidence, verification):
    second, b = read_verified(evidence/'settlement_transfer', verification/'settlement-transfer-audit.json', BASE)
    first, a = read_verified(evidence/'competence_transfer', verification/'competence-transfer-audit.json', FIRST)
    cases = abstract_cases(); lab, checks = laboratory_cases(); cases += lab
    cases.append(recorded_case(second,dict(genre='resources',route='science',opening='conversion',limit=6),
        ('discounted','absolute'),'準備と妥協：資源を残す遅い勝利','resources',400,
        '終局を割り引く基準と、割引を外した比較案。遅い勝利の備蓄は、勝利後の長期利益を証明しない。'))
    cases.append(recorded_case(first,dict(genre='delivery',limit=9,capacity=2,supply=3),
        ('hybrid','guarded'),'目的を取り逃す：物流の修正','delivery',380,
        '目的の全結果を比較する基準と、最良見込みから.15以内に制限する設定。後者は大きな信条の犠牲を狭める。'))
    for seed in (400,401):
        cases.append(recorded_case(second,dict(genre='combat',hp=8,limit=11,distance=4),
            ('discounted','absolute'),'失敗も見る：戦闘 '+str(seed),'combat-'+str(seed),seed,
            '終局割引を外した案は一律不採用。勝利・生存・目的のない先送りが残る比較。'))
    result = dict(format='persona-review-v1', source_commit=BASE, profiles=roster(), cases=cases,
        proof=dict(previous_runs=[a,b], laboratory=checks, snapshot_decisions=8,
            recorded_episodes=sum(len(c['tracks']) for c in cases if c['kind']=='recorded'),
            recorded_frames=sum(len(t['frames']) for c in cases if c['kind']=='recorded' for t in c['tracks'])),
        consent=dict(training_approved=False, human_verdicts=[], default_policy_changed=False),
        demonstration_only=True)
    result['digest'] = digest(result)
    return result


def write(data, output):
    output.mkdir(parents=True,exist_ok=True)
    template=(ROOT/'tools/persona_review.html').read_text(encoding='utf-8')
    (output/'data.json').write_text(json.dumps(data,ensure_ascii=False,allow_nan=False,separators=(',',':'))+'\n',encoding='utf-8',newline='\n')
    (output/'review.html').write_text(render(data,template),encoding='utf-8',newline='\n')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    p.add_argument('--evidence-root',type=Path);p.add_argument('--verification-root',type=Path);p.add_argument('--from-data',type=Path)
    args=p.parse_args()
    if args.from_data:
        data=json.loads(args.from_data.read_text(encoding='utf-8'));original=data.pop('digest')
        if digest(data)!=original:raise ValueError('review data digest mismatch')
        data['digest']=original
    elif args.evidence_root and args.verification_root:data=build(args.evidence_root,args.verification_root)
    else:p.error('supply --from-data OR both evidence and verification roots')
    write(data,args.output);print(json.dumps(data['proof'],ensure_ascii=False))
