import copy
from dataclasses import replace
import unittest
from reflex.core import Policy,digest
from reflex.examples import context,action,effect
from reflex.flow_rollout import rollout
from reflex.continuation_search import search_forecast
from reflex.deliberation import select
from reflex.replan_check import ReplanCheck
from test_plan_repair import corridor


class ReplanCheckTests(unittest.TestCase):
    def test_one_public_feedback_event_can_replace_whole_remaining_plan(self):
        c=context('feedback',[action('start',effect())]);calls=[]
        def observe(s,m):
            cc=copy.deepcopy(c);cc['state']=m;cc['tick']=s
            cc['actions']=[action(k,effect()) for k in ('old','new')];return cc
        def controller(s,cc,observer,remaining,allowed):
            calls.append((s,remaining,digest(cc)))
            state=copy.deepcopy(cc['state']);state['intent_action']='new';state['age']=0
            return dict(root='new',remaining=('new','new'),state=state,audit={'public':True})
        _,audit=rollout(c,0,observe=observe,advance=lambda s,k,seed:(s+1,effect()),terminal=lambda s:s>=4,
            horizon=4,seeds=(30,31),schedule=('old','old','old'),replan_once=controller,record_choices=True)
        self.assertEqual([r['actions'] for r in audit['start']],[['start','new','new','new']]*2)
        self.assertEqual(len(calls),2);self.assertEqual(calls[0],calls[1])
        self.assertTrue(all(len(r['feedback_replans'])==1 for r in audit['start']))

    def test_selected_owner_memory_enters_future_observation(self):
        c=context('memory',[action('start',effect())]);memory=copy.deepcopy(c['state']);memory['mode']='principle'
        seen=[]
        def observe(s,m):
            seen.append(m['mode']);cc=copy.deepcopy(c);cc['state']=m;cc['tick']=s;return cc
        rollout(c,0,observe=observe,advance=lambda s,k,seed:(s+1,effect()),terminal=lambda s:s>=2,
            horizon=2,root_state=memory)
        self.assertEqual(seen,['principle']);self.assertEqual(memory['mode'],'principle')

    def test_illegal_feedback_choice_is_rejected(self):
        c=context('legal',[action('start',effect())])
        def observe(s,m):cc=copy.deepcopy(c);cc['state']=m;cc['tick']=s;return cc
        callback=lambda *args:dict(root='forbidden',remaining=(),state=c['state'],audit={})
        with self.assertRaises(ValueError):rollout(c,0,observe=observe,advance=lambda s,k,z:(s+1,effect()),
            terminal=lambda s:s>=2,horizon=2,replan_once=callback)

    def test_replan_gate_uses_fixed_independent_banks_and_common_owner_tail(self):
        c,args=corridor();old=('start','left','left','left','finish');saved=digest(c)
        f=search_forecast(c,(0,False),retained=old,repairs=24,repair_guard=True,validation_seeds=(3,4),**args)
        def future(cc,state,observer,retained):
            return search_forecast(cc,state,retained=retained,repairs=24,repair_guard=True,
                **{k:v for k,v in args.items() if k!='horizon'},horizon=5)
        checked=replace(f,replan_check=ReplanCheck((0,False),args['observe'],args['advance'],args['terminal'],args['assess'],future,(5,6)))
        d,a=select([c],checked)
        self.assertEqual(d[0]['action_id'],'start');self.assertTrue(a['metadata']['replan_check']['checked'])
        for rows in a['metadata']['replan_check']['branches'].values():
            self.assertEqual([r['seed'] for r in rows],[5,6])
            self.assertTrue(all(r['feedback_replans'][0]['audit']['target']=='finish-goal' for r in rows))
        self.assertEqual(saved,digest(c))
        with self.assertRaises(ValueError):select([c],replace(f,replan_check=replace(checked.replan_check,seeds=(3,4))))

    def test_feedback_check_requires_guarded_owned_planning(self):
        from reflex.observed_transfer import run
        from reflex.laboratory import profiles
        with self.assertRaises(ValueError):run({'genre':'delivery','limit':5},profiles()[0],'verified',1,replan_check=True)


if __name__=='__main__':unittest.main()
