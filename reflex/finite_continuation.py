"""Complete finite public decision chains with consistent future owner choices.

Each node has one settling action and at most one continuing action. Adapters
provide terminal vectors, public opponent forecasts and an owner selector. No
game rules, win conditions, personalities or hidden world sampling live here.
Backward evaluation uses the SAME selector at every future owner opportunity.
"""
import math


def resolve_chains(models,owner,choose):
    """Aligned public chains, ONE future owner policy across model families.

    The public states/actions/terminal vectors must agree; only opponent branch
    probabilities differ. This cannot align arbitrary divergent game trees.
    ``choose(i, model_outcomes)`` selects one action in every family.
    """
    if not isinstance(models,dict) or not 1<=len(models)<=16 or any(not isinstance(k,str) or not 0<len(k)<=128 for k in models):raise ValueError('bounded aligned model families required')
    reference=next(iter(models.values()))
    # Reuse the single-chain contract validation without executing caller code.
    for nodes in models.values():
        resolve_chain(nodes,owner,lambda i,outcomes:next(iter(outcomes)))
        if len(nodes)!=len(reference):raise ValueError('aligned public chain length required')
        for i,(a,b) in enumerate(zip(reference,nodes)):
            keys=('actor','settle','terminal') if i==len(reference)-1 else ('actor','settle','continue','terminal')
            if any(a[k]!=b[k] for k in keys):
                raise ValueError('same public opportunities and terminal vectors required')
    suffix={key:None for key in models};roots={key:[None]*len(reference) for key in models};choices={}
    for i in range(len(reference)-1,-1,-1):
        outcomes={}
        for key,nodes in models.items():
            node=nodes[i];outcomes[key]={node['settle']:((1.,tuple(node['terminal'])),)}
            if suffix[key] is not None:outcomes[key][node['continue']]=suffix[key]
            roots[key][i]=outcomes[key]
        if reference[i]['actor']==owner:
            action=choose(i,outcomes)
            if any(action not in o for o in outcomes.values()):raise ValueError('one legal owner action in every model required')
            choices[i]=action
            for key in models:suffix[key]=outcomes[key][action]
        else:
            for key,nodes in models.items():
                merged={}
                for action,distribution in outcomes[key].items():
                    for mass,terminal in distribution:merged[terminal]=merged.get(terminal,0.)+nodes[i]['forecast'][action]*mass
                suffix[key]=tuple((p,t) for t,p in merged.items() if p>0)
        for key in models:
            total=sum(p for p,_ in suffix[key])
            if abs(total-1)>1e-8:raise ValueError('incomplete model probability mass')
            suffix[key]=tuple((p/total,t) for p,t in suffix[key])
    return roots,choices


def resolve_chain(nodes,owner,choose):
    """Return weighted root outcomes and every owner decision, without sampling.

    ``choose(index, outcomes)`` selects a legal action. Outcomes map action IDs
    to tuples of (probability, terminal vector). Terminal vectors are opaque
    numeric data; the adapter determines what success and progress mean.
    """
    if not nodes or len(nodes)>256:raise ValueError('1..256 complete public chain nodes required')
    size=len(nodes[0]['terminal'])
    for i,node in enumerate(nodes):
        if len(node['terminal'])!=size or size<2 or any(not math.isfinite(v) for v in node['terminal']):raise ValueError('matching finite terminal participant vectors required')
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
