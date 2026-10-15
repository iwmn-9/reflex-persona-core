"""Replay physical rules and summarize lifecycle evidence without choosing moves."""
import argparse,collections,hashlib,json,subprocess,sys,types
from pathlib import Path
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from reflex import progress
from reflex.examples import context,action,effect
from reflex.laboratory import PROFILES
from tools.run_lifecycle_soak import Adapter,distribution,sample,summarize


def recovery_probe(module):
    c=context('recovery-probe',[action(k,effect()) for k in ('restore','other','wait')])
    acts={'restore':module.Activity('attempt','restore-capacity'),
          'other':module.Activity('uncertain','different-route'),
          'wait':module.Activity('wait','opening',patience=3,release='observed-opening')}
    watch=module.ProgressWatch(c['scope'])
    watch.observe(0,'other',module.PurposeRequest(0.,1.,acts),module.PurposeFeedback(0.,1.))
    blocked=0;wait_renewals=0;working_while_blocked=0
    # Capacity suffers a known external loss, then demonstrably recovers.
    # Neither productive recovery nor a sawtooth cycle is a new global peak.
    tick=1
    for cycle in range(24):
        for i in range(8):
            req=module.PurposeRequest(0.,i/10,acts)
            allowed,audit=watch.mask(c,req)
            blocked+=int('restore' not in allowed)
            working_while_blocked+=int('restore' in audit['blocked'])
            watch.observe(tick,'restore',req,module.PurposeFeedback(0.,(i+1)/10));tick+=1
        for i in range(3):
            req=module.PurposeRequest(0.,.8,acts)
            watch.observe(tick,'wait',req,module.PurposeFeedback(0.,.8));tick+=1
        if cycle>0:wait_renewals+=int(watch.wait_spent<3)
    # A recovery method that now stops working must still lose permission.
    for i in range(3):
        req=module.PurposeRequest(0.,.8,acts)
        watch.observe(tick,'restore',req,module.PurposeFeedback(0.,.8));tick+=1
    allowed,_=watch.mask(c,req)
    return dict(productive_updates=192,productive_method_blocked=blocked,
        productive_method_labeled_failure=working_while_blocked,wait_renewals=wait_renewals,
        stalls=watch.stalls,ready_peak=watch.ready_peak,
        ineffective_method_blocked='restore' not in allowed)


def baseline_progress(ref,source=None):
    data=Path(source).read_bytes() if source is not None else subprocess.check_output(['git','show',ref+':reflex/progress.py'],cwd=Path(__file__).resolve().parents[1])
    module=types.ModuleType('reflex._baseline_progress');sys.modules[module.__name__]=module
    exec(compile(data,module.__name__,'exec'),module.__dict__)
    return module,hashlib.sha256(data).hexdigest()


def audit(root,baseline,baseline_source=None):
    snapshots=json.loads((root/'source_snapshot.json').read_text())
    for name,sha in snapshots.items():
        assert hashlib.sha256((root/'_source'/name).read_bytes()).hexdigest()==sha,name
    rows=[json.loads(line) for line in (root/'trajectories.jsonl').read_text().splitlines()]
    reg=json.loads((root/'registration.json').read_text())
    choice_study=rows[0]['arm'] in ('persona_joint','persona_choice','purpose_joint','legacy_support')
    support_study=rows[0]['arm'] in ('global','legacy-support','support')
    arms=('global','legacy-support','support') if support_study else ('coupled','channels')
    if choice_study:arms=('purpose_joint','legacy_support','persona_joint','persona_choice') if reg['holdout'] else ('persona_joint','persona_choice')
    assert len(rows)==len(arms)*len(reg['jobs'])
    expected={(g,PROFILES[p]['id'],s,mode,visibility,arm) for g,p,s,mode,t,visibility,pe in reg['jobs'] for arm in arms}
    actual={(r['game'],r['profile'],r['seed'],r['schedule'],r['visibility'],r['arm']) for r in rows}
    assert len(actual)==len(rows) and actual==expected
    profiles={p['id']:p for p in PROFILES};replayed=0
    phases=collections.defaultdict(list)
    for r in rows:
        a=Adapter(r['game'],profiles[r['profile']],r['seed']%2);s=a.start();scores=[];seen=0
        rng=np.random.default_rng(r['seed']);uniforms=rng.random(r['turns'])
        assert len(r['trace'])==r['turns']
        for tick,t in enumerate(r['trace']):
            if a.terminal(s):scores.append(a.goal(s));s=a.start(s['round']+1,tick)
            assert t['tick']==tick and t['round']==s['round'] and t['step']==s['world']['tick']
            assert t['action'] in a.keys(s)
            dist=distribution(a.symbols(),tick,r['turns'],r['schedule'])
            s,flow,symbol=a.transition(s,t['action'],sample(dist,uniforms[tick]))
            assert t['response']==symbol and abs(t['goal']-a.goal(s))<1e-12
            assert t['effect_visible']==(r['visibility']=='complete' or tick%4==0)
            seen+=int(t['effect_visible'] or r['arm']=='channels' or support_study or choice_study)
            assert seen==t['predictor_observations']
            assert max(t['need_debt'].values())<=.7 and t['learned_entries']<=64
            stage=min(3,tick*4//r['turns'])
            phases[(r['game'],r['schedule'],r['visibility'],r['arm'],stage)].append(t['response_brier'])
            replayed+=1
        if a.terminal(s):scores.append(a.goal(s))
        assert scores==r['scores'] and abs(float(np.mean(scores))-r['score'])<1e-12
    summary=json.loads((root/'summary.json').read_text()) if support_study or choice_study else summarize(rows)
    if support_study or choice_study:
        result=collections.defaultdict(list)
        for pair in summary['pairs']:
            if choice_study:
                for target in pair['scores']:
                    if target!='persona_choice':result[(pair['game'],target)].append(pair['scores']['persona_choice']-pair['scores'][target])
            else:
                for target in ('global','legacy_support'):
                    result[(pair['game'],target)].append(pair['versus_'+target])
        field='choice_comparisons' if choice_study else 'support_comparisons'
        summary[field]=[dict(game=k[0],against=k[1],pairs=len(v),
            better=sum(x>1e-12 for x in v),equal=sum(abs(x)<=1e-12 for x in v),worse=sum(x< -1e-12 for x in v),
            mean_delta=float(np.mean(v))) for k,v in sorted(result.items())]
        summary.update(physical_replay_decisions=replayed,source_hashes_verified=True)
        (root/'audit.json').write_text(json.dumps(summary,indent=2)+'\n',encoding='utf-8')
        print(json.dumps(dict(decisions=replayed,comparisons=summary[field]),indent=2));return
    groups=collections.defaultdict(list)
    for p in summary['pairs']:groups[(p['game'],p['visibility'],p['schedule'])].append(p)
    aggregated=[]
    for key,ps in sorted(groups.items()):
        aggregated.append(dict(game=key[0],visibility=key[1],schedule=key[2],pairs=len(ps),
            better=sum(p['score_delta']>1e-12 for p in ps),equal=sum(abs(p['score_delta'])<=1e-12 for p in ps),
            worse=sum(p['score_delta']< -1e-12 for p in ps),mean_score_delta=float(np.mean([p['score_delta'] for p in ps])),
            mean_brier_delta=float(np.mean([p['response_brier_delta'] for p in ps])),
            action_changes=sum(p['changes'] for p in ps)))
    old,sha=baseline_progress(baseline,baseline_source);new_probe=recovery_probe(progress);old_probe=recovery_probe(old)
    assert old_probe['productive_method_blocked']>0 and new_probe['productive_method_blocked']==0
    assert new_probe['ineffective_method_blocked'] and new_probe['wait_renewals']==0 and new_probe['stalls']==268
    summary.update(physical_replay_decisions=replayed,by_condition=aggregated,
        phase_brier=[dict(game=k[0],schedule=k[1],visibility=k[2],arm=k[3],stage=k[4],mean=float(np.mean(v))) for k,v in sorted(phases.items())],
        recovery=dict(baseline_ref=baseline,baseline_sha256=sha,before=old_probe,after=new_probe),
        limitations=['One common real-world RNG seed across the matrix; profiles are not independent random samples.',
            'Repeated short bouts retain one owner; this is not one 240-step physical world or 240-step search.',
            '3-step sampled planning uses frozen root response law and immediate owned continuation.',
            'Censoring is an API-delivery test; some missing values could be reconstructed from the public rules.',
            'Conditional response is exogenous to the current chosen action; reactive opponents require another contract.',
            'Readiness/progress labels and binding support are adapter responsibilities.'])
    (root/'audit.json').write_text(json.dumps(summary,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(decisions=replayed,by_condition=aggregated,recovery=summary['recovery']),indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--baseline',default='53f7fe59ebab2cff4174898c7308bf02cd596446')
    p.add_argument('--baseline-source',type=Path,help='saved original progress.py for distributions without Git history')
    a=p.parse_args();audit(Path(a.root),a.baseline,a.baseline_source)
