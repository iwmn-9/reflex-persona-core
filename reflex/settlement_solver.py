"""No Thanks final-card adapter for finite policy-consistent continuation.

Every TAKE settles. PASS decreases the finite chip ledger, so repeated passing
eventually forces TAKE. All branches are public: no hidden deck/profile enters.
Opponent distributions remain hypotheses. Exact arithmetic is not exact skill.
"""
import copy
import numpy as np
from .finite_continuation import resolve_chain
from .strong_search import persona_context
from .goal_progress import relative_progress, omit_expired_proxies, choose_with_weighted_progress
from .tabletop_trials import score


def _arrays(outcomes,viewer):
    names=tuple(outcomes);size=max(map(len,outcomes.values()));players=len(next(iter(outcomes.values()))[0][1])
    finals=np.zeros((len(names),size,players));weights=np.zeros((len(names),size));shares=np.zeros_like(weights)
    for i,name in enumerate(names):
        for j,(mass,terminal) in enumerate(outcomes[name]):
            finals[i,j]=terminal;weights[i,j]=mass
            best=min(terminal);shares[i,j]=float(terminal[viewer]==best)/terminal.count(best)
    return names,finals,shares,weights


def _context(s,viewer,p,seed,tick,episode,state,names,finals,shares,weights):
    c=persona_context('no_thanks',s,viewer,p,seed,tick,episode,state,names,finals,shares,sample_weights=weights)
    if s.chips[(viewer+1)%len(s.chips)]==0:
        c=omit_expired_proxies(c,needs=('safety',),values=('security',),style=('neuroticism',))
        c['facts']['expiry_certificate']='TAKEは終局、PASSは次席のチップ0による強制TAKEで終局。本人の再判断は全合法根で不存在'
    c['facts']['forecast']='最後のカードの全公開分岐を重み付きで終局まで評価。未来の本人も同じ人格選択器で再判断。相手の公開行動仮説には誤りがありうる'
    c['facts']['continuation']='本人の全再判断と状態更新を同じ選択器で解決。探索打切り、未知の山札、相手の本当の人格は入力しない'
    c['facts']['estimate']=c['facts']['forecast']
    return c


def solve(s,viewer,p,seed,encounter,tick,memory,state,*,adaptive=True):
    if type(adaptive) is not bool:raise ValueError('explicit boolean learning control required')
    if s.remaining!=0 or s.card is None or s.turn!=viewer:raise ValueError('public final-card owner opportunity required')
    if (memory.game,memory.viewer,memory.players)!=('no_thanks',viewer,len(s.chips)):raise ValueError('game/observer support mismatch')
    episode=f'series-{seed}-encounter-{encounter}'
    # A single path of hypothetical PASS observations reaches every possible
    # later opportunity; TAKE ends a branch immediately. Real memory is detached.
    virtual=copy.deepcopy(memory);current=s;owner_state=copy.deepcopy(state);nodes=[];owner_states=[]
    snapshots=[];forced_pass_states={}
    while True:
        i=len(nodes);actor=current.turn;legal=current.legal();settled=current.play('TAKE')
        prediction=virtual.predict(current,actor,adaptive=adaptive) if actor!=viewer else {a:1/len(legal) for a in legal}
        nodes.append({'actor':actor,'settle':'TAKE','continue':'PASS','terminal':settled.scores(),'forecast':prediction})
        snapshots.append(current);owner_states.append(copy.deepcopy(owner_state))
        if len(legal)==1:break
        if actor==viewer:
            # Primary/mode updates depend on needs, fixed traits and existing
            # state, not forecast returns. The assertion below verifies this
            # before accepting any continued owner decision.
            dummy={a:((1.,settled.scores()),) for a in legal}
            names,finals,shares,weights=_arrays(dummy,viewer)
            c=_context(current,viewer,p,seed,tick+i,episode,owner_state,names,finals,shares,weights)
            d,_=score(c);next_state=d['next_state'];next_state['intent_action']='PASS'
            next_state['age']=min(owner_state['age']+1,1000000) if owner_state and owner_state['intent_action']=='PASS' else 0
            owner_state=next_state;forced_pass_states[i]=copy.deepcopy(next_state)
        elif adaptive:virtual.observe(current,actor,'PASS',f'encounter-{encounter}-tick-{tick+i}-actor-{actor}')
        current=current.play('PASS')
        if len(nodes)>=256:raise ValueError('ledger exceeds finite chain capacity')
    decisions={}
    def choose(i,outcomes):
        names,finals,shares,weights=_arrays(outcomes,viewer)
        c=_context(snapshots[i],viewer,p,seed,tick+i,episode,owner_states[i],names,finals,shares,weights)
        progress=relative_progress(finals,viewer,direction=-1,scale=35)
        c,d,guard=choose_with_weighted_progress(c,names,shares,progress,weights)
        if d['action_id']=='PASS':
            assert d['next_state']==forced_pass_states[i],'owner state update depends on future values; chain contract invalid'
        decisions[i]=(c,d,guard,names,finals,shares,weights,progress)
        return d['action_id']
    roots,choices=resolve_chain(nodes,viewer,choose)
    c,d,guard,names,finals,shares,weights,progress=decisions[0]
    stats=dict(method='adaptive',action=d['action_id'],variant='settlement',guard=guard,sample_count=0,
        finite_chain_nodes=len(nodes),future_owner_opportunities=len(choices)-1,
        validation_scenarios=0,training_scenarios=0,
        continuation=c['facts']['continuation'],actions={a:dict(win_share=float((shares[i]*weights[i]).sum()),standard_error=0.,
            mean_score=float((finals[i,:,viewer]*weights[i]).sum()),goal_progress=float((progress[i]*weights[i]).sum())) for i,a in enumerate(names)})
    return c,d,stats
