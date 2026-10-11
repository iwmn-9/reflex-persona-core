"""Paired model checks for an optional expansion of a planning bank.

The reference is independently selected without repairs on the SAME current
observation and validation branches. This is model dominance, not a statistical
or real-world safety guarantee. The owner's strongest value tier is preserved.
"""
from dataclasses import replace
import copy
from .deliberation import select


def safeguard(c, expanded, reference, policy, allowed=None):
    permission=None if allowed is None else [allowed]
    _,new_vote=select([c],expanded,policy,permission)
    _,old_vote=select([c],reference,policy,permission)
    new=new_vote.get('selected_plan');old=old_vote.get('selected_plan')
    report=dict(rule='paired endpoint nondeterioration against separately selected unexpanded bank',
        protected=False,accepted=True,reference_selection=old,expanded_selection=new,
        guarantee='sampled model dominance only; no calibrated probability or actual-result guarantee')
    restricted=None
    if old is None or new is None:
        report['reason']='no comparable supported decision'
    else:
        old_proposal=reference.audit['plans'][old]['proposal']
        new_proposal=expanded.audit['plans'][new]['proposal']
        compatible=any(expanded.audit['plans'][k]['proposal']==old_proposal for k in expanded.roots)
        if not compatible:
            report['reason']='reference outside expanded strongest-principle tier; protection cannot override values'
        else:
            def endpoints(f,key):
                proposal=f.audit['plans'][key]['proposal']
                traces=f.audit['continuation_search']['proposals'][proposal]['branches']
                goals=f.audit['endpoint'][key]
                if len(traces)!=len(goals):raise ValueError('complete paired endpoint coverage required')
                root=f.roots[key][0]
                confidence=next(a['confidence'] for a in c['actions'] if a['id']==root)
                return [(t['seed'],g['target'],g['value']*(g['positive_confidence']*confidence if g['value']>0 else 1.))
                    for t,g in zip(traces,goals)]
            before=endpoints(reference,old);after=endpoints(expanded,new)
            if [(s,t) for s,t,v in before]!=[(s,t) for s,t,v in after]:
                raise ValueError('same target and aligned independent validation branches required')
            delta=[b[2]-a[2] for a,b in zip(before,after)]
            accept=all(d>=-1e-12 for d in delta)
            report.update(protected=True,accepted=accept,reference_proposal=old_proposal,
                expanded_proposal=new_proposal,branch_deltas=delta,
                reason='no modeled branch worsened' if accept else 'expanded choice worsened a paired model branch')
            if not accept:
                restricted=next(k for k in expanded.roots if expanded.audit['plans'][k]['proposal']==old_proposal)
    metadata=copy.deepcopy(expanded.audit)
    metadata['repair_stability']=report
    if restricted is None:return replace(expanded,audit=metadata)
    # Preserve the original plan ID/audit bank for actual intention storage.
    # This restriction is made after common-tier and actual-root support checks.
    cc=copy.deepcopy(expanded.contexts[0]);cc['actions']=[a for a in cc['actions'] if a['id']==restricted]
    return replace(expanded,contexts=(cc,),roots={restricted:expanded.roots[restricted]},
        purpose={restricted:expanded.purpose[restricted]},continuity=None,audit=metadata)
