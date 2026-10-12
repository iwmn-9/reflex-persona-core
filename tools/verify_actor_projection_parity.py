"""Check old/new forced-root checkpoints across existing public contexts."""
import argparse,copy,hashlib,importlib.util,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))

def main():
    p=argparse.ArgumentParser();p.add_argument('--old-source',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    from reflex.actor_model import ActorModel
    from reflex.core import Policy,character_key,digest
    from reflex.decision_loop import DecisionLoop,Request
    from reflex.judgment import Binding
    from reflex.model_observer import ModelFeedback
    from reflex.planning import vector
    from reflex.social import SocialBinding,SocialEvent
    from reflex.social_withdrawal import WithdrawalSocialMemory
    from tools.verify_integrated_contracts import contexts
    legacy=Path(a.old_source)/'reflex/actor_model.py'
    spec=importlib.util.spec_from_file_location('reflex._frozen_actor_model',legacy);old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
    rows=[]
    for case,profile,c in contexts():
        for social in (False,True):
            sb={x['id']:SocialBinding(x['target'],1.) for x in c['actions'] if x['target'] is not None} if social else None
            policy=Policy(principle_priority='personal',commitments={character_key(c):.5})
            actor=DecisionLoop(c,policy=policy,social=WithdrawalSocialMemory(c) if social else None)
            raw=Request(c,{x['id']:Binding(x['id'],'public-rule',('objective',)) for x in c['actions']},{x['id']:() for x in c['actions']},social_bindings=sb)
            for action in c['actions']:
                if not action['legal'] or action['known_failure']:continue
                key=action['id'];peer=action['target'];events=() if not social or peer is None else (SocialEvent(peer,'seen',c['tick'],-.5),)
                fb=lambda *args:ModelFeedback(None,observed=vector(action['outcomes'][0]),social_events=events)
                one=old.ActorModel(actor,root_request=raw,request_for=lambda cc:raw,feedback=fb)
                two=ActorModel(actor,root_request=raw,request_for=lambda cc:raw,feedback=fb)
                for model in (one,two):
                    cc,req,_,_=model.prepare(None,c,prepared=True);model.preview();model.commit_choice(key);model.advance(None,None,key,cc,req)
                assert one.record()==two.record(),(case['game'],profile['id'],social,key)
                rows.append(dict(game=case['game'],profile=profile['id'],social=social,root=key,checkpoint_sha256=digest(two.record())))
    out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
    result=dict(format='actor-projection-parity-v1',checks=len(rows),connections=len({r['game'] for r in rows}),rows=rows,
        old_helper_sha256=hashlib.sha256(legacy.read_bytes()).hexdigest(),current_helper_sha256=hashlib.sha256((ROOT/'reflex/actor_model.py').read_bytes()).hexdigest(),
        limitation='old/new helper with current shared loop; no active progress or predictor in the published game cohorts; blocked-progress handling is separately tested')
    (out/'result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');print('equal checkpoints',len(rows),result['connections'])
    return 0

if __name__=='__main__':raise SystemExit(main())
