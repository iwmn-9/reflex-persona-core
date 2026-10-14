"""Finite public opportunities, compiled once and revalued as beliefs change.

Known rules form a DAG. A solve propagates terminal-purpose distributions under
an explicitly selected continuation. It never invents a price for a resource:
spending matters through the successors and legal options it leaves available.
Future preference flows are omitted; future choices can still use the persona.
No guarantee applies to an incorrect model or an insufficient state record.
"""
from dataclasses import dataclass, replace
import copy
import math
import numpy as np

from .core import Policy, compile_batch, digest
from .purpose_plan import Goal
from .frontier_value import EndpointEstimate


class GraphLimit(Exception):
    """Incomplete opportunity coverage must decline the entire proposal."""


@dataclass(frozen=True)
class Outcome:
    probability: float
    value: float
    flow: dict


@dataclass(frozen=True)
class Alternative:
    key: str
    outcomes: tuple


def persona_choices(contexts, alternatives, policy, max_regret=.15):
    """Batch the existing one-step persona chooser with terminal purpose atoms.

    Equivalent to horizon-one goal_forecast/select, without building diagnostic
    path dictionaries at every internal state. Only current preference flows
    are priced. Every context must contain the continuation's sufficient owned
    state. This function does not advance or reconstruct DecisionLoop memory.
    """
    if len(contexts)!=len(alternatives) or not contexts:
        raise ValueError('one context per continuation choice required')
    if isinstance(max_regret,bool) or not math.isfinite(max_regret) or not 0<=max_regret<=2:
        raise ValueError('bounded purpose corridor required')
    expanded=[];purposes=[];limit=1
    for c,choices in zip(contexts,alternatives):
        current={a['id']:a for a in c['actions'] if a['legal'] and not a['known_failure']}
        if [a.key for a in choices]!=sorted(current):
            raise ValueError('complete ordered viable action coverage required')
        fc=copy.deepcopy(c);fc['actions']=[];values=[]
        for alt in choices:
            action=copy.deepcopy(current[alt.key]);out=[]
            for atom in alt.outcomes:
                row=copy.deepcopy(atom.flow);row.update(p=atom.probability,objective=atom.value);out.append(row)
            action['outcomes']=out;fc['actions'].append(action);limit=max(limit,len(out))
            values.append(math.fsum(r.probability*r.value*(action['confidence'] if r.value>0 else 1.) for r in alt.outcomes))
        expanded.append(fc);purposes.append(values)
    if limit>4096:raise GraphLimit('persona outcome capacity')
    batch=compile_batch(expanded,outcome_limit=limit)
    first=policy.decide(batch,False)
    purpose=np.full(first.eligible.shape,-np.inf)
    for i,values in enumerate(purposes):purpose[i,:len(values)]=values
    best=np.max(np.where(first.eligible,purpose,-np.inf),axis=1)
    allowed=first.eligible & (purpose>=best[:,None]-max_regret)
    second=policy.decide(replace(batch,legal=batch.legal&allowed),False)
    return tuple(batch.ids[i][j] for i,j in enumerate(second.action))


class OpportunityGraph:
    """Known, finite, acyclic PUBLIC rule graph with complete action coverage.

    events(state, action) returns PublicEvent-compatible rows: symbol, public,
    flow. record must preserve every variable affecting rules, observations and
    the chosen continuation. Its projection is an adapter contract, not an
    automatic state abstraction. No probabilities/hidden process are compiled.
    """
    def __init__(self, *, actions, events, terminal, assess, record, model_id,
                 max_nodes=100000, max_edges=1000000, max_depth=64):
        if any(not callable(f) for f in (actions,events,terminal,assess,record)) or not isinstance(model_id,str) or not model_id:
            raise ValueError('identified public rule callbacks required')
        for n,cap in ((max_nodes,1000000),(max_edges,10000000),(max_depth,256)):
            if type(n) is not int or not 1<=n<=cap:raise ValueError('bounded graph capacity required')
        self.actions,self.events,self.terminal,self.assess,self.record=actions,events,terminal,assess,record
        self.model_id=model_id;self.limits=(max_nodes,max_edges,max_depth)
        self.nodes=[];self.lookup={};self.root=None;self.identity=None;self.last={}

    def compile(self, initial):
        nodes=[];lookup={};active=set();edges=0;target=None
        def visit(state,depth):
            nonlocal edges,target
            if depth>self.limits[2]:raise GraphLimit('graph depth')
            key=digest(self.record(copy.deepcopy(state)))
            if key in active:raise ValueError('cyclic state; a finite public deadline is required')
            if key in lookup:return lookup[key]
            if len(nodes)+len(active)>=self.limits[0]:raise GraphLimit('graph nodes')
            active.add(key)
            ended=bool(self.terminal(copy.deepcopy(state)));goal=Goal(**self.assess(copy.deepcopy(state)))
            if target is None:target=goal.target
            if goal.target!=target or ended!=(goal.status!='running'):
                raise ValueError('consistent goal identity and actual terminal status required')
            branches={};height=0
            if not ended:
                keys=self.actions(copy.deepcopy(state))
                if not isinstance(keys,(tuple,list)) or not 1<=len(keys)<=256 or len(set(keys))!=len(keys) or any(not isinstance(k,str) or not k for k in keys):
                    raise ValueError('all viable action identities required')
                for action in sorted(keys):
                    rows=self.events(copy.deepcopy(state),action)
                    if not isinstance(rows,(tuple,list)) or not 1<=len(rows)<=256:
                        raise ValueError('bounded explicit observable outcomes required')
                    symbols=[r.symbol for r in rows]
                    if len(set(symbols))!=len(symbols) or any(not isinstance(s,str) or not s for s in symbols):
                        raise ValueError('unique public event symbols required')
                    events=[]
                    for row in sorted(rows,key=lambda r:r.symbol):
                        if not isinstance(row.flow,dict) or row.flow.get('p')!=1.:
                            raise ValueError('conditional realized flow required')
                        edges+=1
                        if edges>self.limits[1]:raise GraphLimit('graph edges')
                        child=visit(copy.deepcopy(row.public),depth+1)
                        events.append((row.symbol,child,copy.deepcopy(row.flow)))
                        height=max(height,1+nodes[child]['height'])
                    branches[action]=tuple(events)
            active.remove(key);index=len(nodes);lookup[key]=index
            nodes.append(dict(state=copy.deepcopy(state),key=key,goal=goal,branches=branches,height=height))
            return index
        self.nodes=[];self.lookup={};self.root=None;self.identity=None;self.last=dict(complete=False)
        try:root=visit(copy.deepcopy(initial),0)
        except GraphLimit as error:
            self.last=dict(complete=False,reason=str(error),nodes=len(nodes),edges=edges);return False
        self.nodes,self.lookup,self.root=nodes,lookup,root
        self.identity=digest(['opportunity-graph-v1',self.model_id,target,[(n['key'],n['goal'].record(),n['branches']) for n in nodes]])
        self.last=dict(complete=True,nodes=len(nodes),edges=edges,height=nodes[root]['height'])
        return True

    def solve(self, law, *, law_id, choose=None, chooser_id='expected-purpose', max_atoms=256, batch_size=32, initial=None):
        """Revalue every reachable option; probability snapshot fixed per solve.

        choose(states, alternatives) returns one action per state, after children
        have been solved. None maximizes expected terminal purpose (comparison
        policy, not a model of a personality). Every action is evaluated fully.
        A capacity failure yields no solution, never partially valued roots.
        """
        if self.root is None or not callable(law) or (choose is not None and not callable(choose)):
            raise ValueError('complete graph and explicit prediction snapshot required')
        if not all(isinstance(x,str) and x for x in (law_id,chooser_id)):
            raise ValueError('identified prediction and continuation required')
        if type(max_atoms) is not int or not 1<=max_atoms<=256 or type(batch_size) is not int or not 1<=batch_size<=256:
            raise ValueError('bounded exact distribution and batch required')
        entry=self.root if initial is None else self.lookup[digest(self.record(copy.deepcopy(initial)))]
        reachable=set();pending=[entry]
        while pending:
            i=pending.pop()
            if i in reachable:continue
            reachable.add(i)
            pending.extend(j for events in self.nodes[i]['branches'].values() for _,j,_ in events)
        self.last=dict(self.last,solve_complete=False,solve_nodes=len(reachable))
        values=[None]*len(self.nodes);chosen=[None]*len(self.nodes);laws=[]
        groups={}
        for i,n in enumerate(self.nodes):
            if i in reachable:groups.setdefault(n['height'],[]).append(i)
        try:
            for height,indices in sorted(groups.items()):
                if height==0:
                    for i in indices:values[i]=((1.,self.nodes[i]['goal'].value),)
                    continue
                for start in range(0,len(indices),batch_size):
                    batch=indices[start:start+batch_size];options=[]
                    for i in batch:
                        node=self.nodes[i];alternatives=[]
                        for key,events in node['branches'].items():
                            symbols=tuple(s for s,_,_ in events)
                            p=law(copy.deepcopy(node['state']),key,symbols)
                            if (not isinstance(p,dict) or set(p)!=set(symbols) or
                                any(type(v) not in (float,int) or not math.isfinite(v) or v<0 for v in p.values()) or abs(math.fsum(p.values())-1)>1e-8):
                                raise ValueError('normalized observable prediction required')
                            laws.append((i,key,tuple(p[s] for s in symbols)));atoms=[]
                            for s,j,flow in events:
                                if not p[s]:continue
                                for q,v in values[j]:atoms.append(Outcome(p[s]*q,v,flow))
                            alternatives.append(Alternative(key,tuple(atoms)))
                        options.append(tuple(alternatives))
                    if choose is None:
                        keys=tuple(max(alts,key=lambda a:math.fsum(o.probability*o.value for o in a.outcomes)).key for alts in options)
                    else:keys=choose([copy.deepcopy(self.nodes[i]['state']) for i in batch],copy.deepcopy(options))
                    if not isinstance(keys,(tuple,list)) or len(keys)!=len(batch):raise ValueError('one continuation per state required')
                    for i,key,alts in zip(batch,keys,options):
                        if key not in {a.key for a in alts}:raise ValueError('continuation must select a viable action')
                        mass={}
                        for atom in next(a.outcomes for a in alts if a.key==key):mass[atom.value]=mass.get(atom.value,0.)+atom.probability
                        if len(mass)>max_atoms:raise GraphLimit('terminal distribution capacity')
                        total=math.fsum(mass.values())
                        values[i]=tuple((p/total,v) for v,p in sorted(mass.items()));chosen[i]=key
        except GraphLimit as error:
            self.last=dict(self.last,solve_complete=False,solve_reason=str(error));return None
        identity=digest([self.identity,entry,law_id,chooser_id,laws,chosen])
        self.last=dict(self.last,solve_complete=True,solve_identity=identity)
        return OpportunitySolution(self,tuple(values),tuple(chosen),identity,self.identity)


@dataclass(frozen=True)
class OpportunitySolution:
    graph: OpportunityGraph
    distributions: tuple
    choices: tuple
    identity: str
    graph_identity: str

    def index(self,state):
        if self.graph.identity!=self.graph_identity:raise ValueError('solution belongs to a different compiled graph')
        i=self.graph.lookup[digest(self.graph.record(copy.deepcopy(state)))]
        if self.distributions[i] is None:raise ValueError('state was outside the solved continuation')
        return i

    def action(self,state):return self.choices[self.index(state)]

    def evaluate(self,state,owner=None):
        return EndpointEstimate(self.distributions[self.index(state)],self.identity,1.)
