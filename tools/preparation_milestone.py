"""Research connection: an owned intention to a publicly observable benefit.

Reuses ArrivalPreference. It may sacrifice a same-tier subjective preference
for delay; it does not alter traits, value tiers, progress or the purpose band.
Model samples do not guarantee real-world arrival. Unrepresented intentions
and unknown arrivals never constrain a fresh choice.
"""
import copy,math
from dataclasses import replace
from reflex.core import digest
from reflex.plan_continuity import PlanIntention,ArrivalPreference

class Milestone:
    def __init__(self,context):
        self.intention=PlanIntention(context);self.value=None;self.deadline=None
        self.last_audit={}

    def record(self):
        return dict(intention=self.intention.record(),value=self.value,deadline=self.deadline)

    @classmethod
    def from_record(cls,context,row):
        if set(row)!= {'intention','value','deadline'}:raise ValueError('milestone checkpoint required')
        out=cls(context);out.intention=PlanIntention.from_record(context,row['intention'])
        v,d=row['value'],row['deadline']
        if (v is None)!=(d is None):raise ValueError('value and deadline must be present together')
        if v is not None and (type(v) not in (float,int) or not math.isfinite(v) or abs(v)>1 or
            type(d) is not int or out.intention.last_tick is None or not out.intention.last_tick<d<=out.intention.last_tick+16):
            raise ValueError('bounded public milestone required')
        out.value=v;out.deadline=d;return out

    def apply(self,context,forecast,modeled,origin):
        self.last_audit=dict(active=False)
        remaining=self.intention.offer(context,target=forecast.target,unit='owner-decisions',horizon=min(16,forecast.horizon))
        if self.value is None or not remaining or context['tick']>=self.deadline or origin>=self.value-1e-12:
            self.last_audit['reason']='no live unfulfilled intention';return forecast
        arrivals={};candidates=[]
        for name in forecast.roots:
            proposal=forecast.audit['plans'][name]['proposal'];rows=modeled[proposal]
            times=[next((i+1 for i,p in enumerate(r['prefixes']) if p['assessment']['value']>=self.value-1e-12),None) for r in rows]
            arrivals[name]=times
            if rows and all(tuple(r['actions'][:len(remaining)])==remaining for r in rows) and all(t is not None for t in times):
                candidates.append(name)
        if not candidates:
            self.last_audit['reason']='retained continuation is absent or no longer reaches the goal';return forecast
        incumbent=max(candidates,key=lambda k:(forecast.purpose[k],-max(arrivals[k]),k))
        # Single conservative bounds: no correspondence between sampled branches
        # is assumed. A single unknown branch disables that plan's comparison.
        bounds={k:(None if any(t is None for t in ts) else min(ts),) for k,ts in arrivals.items()}
        bounds[incumbent]=(max(arrivals[incumbent]),)
        self.last_audit=dict(active=True,incumbent=incumbent,value=self.value,deadline=self.deadline,
            remaining=list(remaining),comparison='latest incumbent arrival versus earliest alternative arrival')
        metadata=copy.deepcopy(forecast.audit);metadata['public_milestone']=copy.deepcopy(self.last_audit)
        return replace(forecast,continuity=ArrivalPreference(incumbent,bounds),audit=metadata)

    def remember(self,context,forecast,modeled,paths,selected,actual_root,origin,actual_goal,*,resolved_owner):
        """After real feedback only. Retain a maximum 16-step common prefix."""
        self.intention._check(context)
        owner=resolved_owner.record()  # Reject an unresolved decision ticket.
        if (owner['last_tick']!=context['tick'] or owner['state']['intent_action']!=actual_root or
            digest(owner['scope'])!=self.intention.owner or digest([owner['personality'],owner['values']])!=self.intention.preferences):
            raise ValueError('matching resolved actual owner required')
        proposal=forecast.audit['plans'][selected]['proposal'];rows=modeled[proposal]
        if forecast.roots[selected]!=(actual_root,) or any(r['actions'][0]!=actual_root for r in rows):
            raise ValueError('selected factual root required')
        old=self.intention.remaining
        continuing=(self.value is not None and self.intention.target==forecast.target and self.intention.unit=='owner-decisions' and
            context['tick']<self.deadline and actual_goal<self.value-1e-12 and
            old and all(tuple(r['actions'][:len(old)])==old for r in rows))
        if continuing:
            common=list(old[1:]);value=self.value;deadline=self.deadline
        else:
            common=[];value=None;deadline=None
            ordinary=[]
            for key,policy in forecast.audit['continuation_portfolio']['policies'].items():
                if policy['kind']!='reflex':continue
                mass=math.fsum(b.probability for b in paths[key])
                ordinary.append([math.fsum(b.probability*r['prefixes'][min(t,len(r['prefixes'])-1)]['assessment']['value']
                    for b,r in zip(paths[key],modeled[key]))/mass for t in range(min(16,forecast.horizon))])
            shared=[]
            for column in zip(*(r['actions'] for r in rows)):
                if len(set(column))!=1:break
                shared.append(column[0])
            for t in range(min(len(shared),16,forecast.horizon)):
                floor=min(r['prefixes'][t]['assessment']['value'] for r in rows)
                if t and ordinary and floor>origin+1e-12 and floor>max(v[t] for v in ordinary)+1e-12:
                    common=shared[1:t+1];value=floor;deadline=context['tick']+t+1;break
        if not common or (value is not None and actual_goal>=value-1e-12):
            common=[];value=None;deadline=None
        adapted=copy.deepcopy(forecast.audit)
        adapted.update(unit='owner-decisions',continuation_search=dict(proposals={proposal:dict(branches=[dict(actions=[actual_root]+common)])}))
        self.intention.remember(context,replace(forecast,audit=adapted),selected,actual_root)
        self.value=value;self.deadline=deadline
