"""Counterfactual check with one future public-observation replanning event.

The caller supplies the future controller without this check, avoiding an
unbounded recursion. Its independent forecast bank is not the trajectory seed.
This narrows one source of model/controller mismatch; it does not eliminate it.
"""
from dataclasses import dataclass,replace
import copy
from .core import MAX_OUTCOMES
from .flow_rollout import rollout
from .purpose_plan import goal_forecast
from .deliberation import JointForecast,select


def common_tail(rows):
    out=[]
    for column in zip(*(r['actions'][1:] for r in rows)):
        if len(set(column))!=1:break
        out.append(column[0])
    return tuple(out)


@dataclass(frozen=True)
class ReplanCheck:
    initial: object
    observe: object
    advance: object
    terminal: object
    assess: object
    planner: object
    seeds: tuple
    observer: object=None
    settlement_weight: str='absolute'

    def apply(self,c,expanded,policy,allowed):
        reference=expanded.stability_reference
        if not isinstance(reference,JointForecast) or reference.replan_check is not None:
            raise ValueError('independent unexpanded reference required')
        if not isinstance(self.seeds,tuple) or not 1<=len(self.seeds)<=MAX_OUTCOMES or any(type(s) is not int for s in self.seeds) or len(set(self.seeds))!=len(self.seeds):
            raise ValueError('distinct bounded feedback check seeds required')
        discovery=expanded.audit['continuation_search'].get('validation',{})
        if set(self.seeds)&set(discovery.get('seeds',[])+discovery.get('discovery_seeds',[])):
            raise ValueError('feedback check bank must be separate from candidate discovery/selection')
        masks=None if allowed is None else [allowed]
        new_records,new_vote=select([c],expanded,policy,masks)
        old_records,old_vote=select([c],reference,policy,masks)
        report=dict(kind='one future feedback replan, then observed-legal shared tail/reflex',
            accepted=True,protected=False,checked=False,seeds=list(self.seeds),
            guarantee='finite counterfactual model comparison; not a real-result or human-level guarantee')
        selected=None
        if new_records is None or old_records is None:report['reason']='no supported comparison'
        else:
            new_key=new_vote['selected_plan'];old_key=old_vote['selected_plan']
            def candidate(f,key,record):
                proposal=f.audit['plans'][key]['proposal'];row=f.audit['continuation_search']['proposals'][proposal]
                return dict(proposal=proposal,root=row['root'],tail=common_tail(row['branches']),state=record['next_state'])
            a=candidate(reference,old_key,old_records[0]);b=candidate(expanded,new_key,new_records[0])
            report.update(reference_proposal=a['proposal'],candidate_proposal=b['proposal'])
            if (a['root'],a['tail'],a['state'])==(b['root'],b['tail'],b['state']):
                report['reason']='same executable root, common intention and next owner state'
            else:
                local=copy.deepcopy(c);heads={a['root'],b['root']}
                local['actions']=[v for v in local['actions'] if v['id'] in heads]
                paths={};audits={}
                def next_choice(state,cc,observer,retained,permitted):
                    f=self.planner(cc,state,observer,retained)
                    if not isinstance(f,JointForecast) or f.target!=expanded.target or f.replan_check is not None:
                        raise ValueError('same-purpose bounded nonrecursive future controller required')
                    records,vote=select([cc],f,policy,None if permitted is None else [permitted])
                    if records is None:raise ValueError('future controller must provide supported decision')
                    k=vote['selected_plan'];p=f.audit['plans'][k]['proposal']
                    tail=common_tail(f.audit['continuation_search']['proposals'][p]['branches'])
                    return dict(root=records[0]['action_id'],remaining=tail,state=records[0]['next_state'],
                        audit=dict(target=f.target,selected_proposal=p,selected_purpose=vote['selected_purpose']))
                for label,row in (('reference',a),('candidate',b)):
                    p,tr=rollout(c,self.initial,observe=self.observe,advance=self.advance,terminal=self.terminal,
                        assess=self.assess,horizon=expanded.horizon,seeds=self.seeds,roots=(row['root'],),
                        schedule=row['tail'],record_choices=True,policy=policy,observer=self.observer,
                        root_state=row['state'],replan_once=next_choice)
                    paths[label]=p[row['root']];audits[label]=tr[row['root']]
                checked=goal_forecast(local,paths,audits,horizon=expanded.horizon,unit=expanded.audit['unit'],
                    target=expanded.target,max_regret=expanded.max_regret,policy=policy,
                    plan_roots={'reference':a['root'],'candidate':b['root']},settlement_weight=self.settlement_weight)
                _,vote=select([c],checked,policy,masks)
                winner=checked.audit['plans'][vote['selected_plan']]['proposal']
                by={checked.audit['plans'][k]['proposal']:k for k in checked.roots}
                delta=[g['assessment']['value']-h['assessment']['value'] for g,h in zip(audits['candidate'],audits['reference'])]
                protected='reference' in by
                accept=winner=='candidate' and (not protected or all(v>=-1e-12 for v in delta))
                report.update(checked=True,protected=protected,accepted=accept,branch_deltas=delta,
                    controller_vote=winner,branches=audits,reason='candidate accepted by feedback forecast' if accept else 'retain reference after feedback forecast')
                if not accept:
                    selected=next((k for k in expanded.roots if expanded.audit['plans'][k]['proposal']==a['proposal']),None)
                    if selected is None:
                        report.update(accepted=True,protected=False,reason='cannot override expanded strongest-principle tier')
        metadata=copy.deepcopy(expanded.audit);metadata['replan_check']=report
        if new_records is not None and 'repair_stability' in new_vote['metadata']:
            metadata['repair_stability']=copy.deepcopy(new_vote['metadata']['repair_stability'])
        result=replace(expanded,audit=metadata,stability_reference=None)
        if selected is None:
            # Preserve any first-stage paired protection, including its actual
            # root mask; return the voted bank with an ID-stable restriction.
            if new_records is not None:selected=new_vote['selected_plan']
        if selected is not None:
            cc=copy.deepcopy(result.contexts[0]);cc['actions']=[v for v in cc['actions'] if v['id']==selected]
            result=replace(result,contexts=(cc,),roots={selected:result.roots[selected]},
                purpose={selected:result.purpose[selected]},continuity=None)
        return result
