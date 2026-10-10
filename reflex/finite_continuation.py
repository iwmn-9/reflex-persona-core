"""Complete finite public decision chains with consistent future owner choices.

Each node has one settling action and at most one continuing action. Adapters
provide terminal vectors, public opponent forecasts and an owner selector. No
game rules, win conditions, personalities or hidden world sampling live here.
Backward evaluation uses the SAME selector at every future owner opportunity.
"""
import math


def resolve_chain(nodes,owner,choose):
    """Return weighted root outcomes and every owner decision, without sampling.

    ``choose(index, outcomes)`` selects a legal action. Outcomes map action IDs
    to tuples of (probability, terminal vector). Terminal vectors are opaque
    numeric data; the adapter determines what success and progress mean.
    """
    if not nodes or len(nodes)>256:raise ValueError('1..256 complete public chain nodes required')
    for i,node in enumerate(nodes):
        if len(node['terminal'])<2 or any(not math.isfinite(v) for v in node['terminal']):raise ValueError('finite terminal participant vector required')
        names=tuple(node['forecast'])
        required=(node['settle'],) if i==len(nodes)-1 else (node['settle'],node['continue'])
        if set(names)!=set(required) or any(not isinstance(n,str) or not n for n in required) or len(set(required))!=len(required):raise ValueError('complete legal chain actions required')
        masses=list(node['forecast'].values())
        if any(not math.isfinite(v) or v<0 for v in masses) or abs(sum(masses)-1)>1e-8:raise ValueError('normalized public branch probabilities required')
    suffix=None;roots=[None]*len(nodes);choices={}
    for i in range(len(nodes)-1,-1,-1):
        node=nodes[i];outcomes={node['settle']:((1.,tuple(node['terminal'])),)}
        if suffix is not None:outcomes[node['continue']]=suffix
        roots[i]=outcomes
        if node['actor']==owner:
            chosen=choose(i,outcomes)
            if chosen not in outcomes:raise ValueError('owner selector returned an illegal chain action')
            choices[i]=chosen;suffix=outcomes[chosen]
        else:
            merged={}
            for action,distribution in outcomes.items():
                for mass,terminal in distribution:
                    merged[terminal]=merged.get(terminal,0.)+node['forecast'][action]*mass
            suffix=tuple((p,t) for t,p in merged.items() if p>0)
        if abs(sum(p for p,_ in suffix)-1)>1e-8:raise ValueError('incomplete model probability mass')
        total=sum(p for p,_ in suffix)
        suffix=tuple((p/total,t) for p,t in suffix)
    return roots,choices
