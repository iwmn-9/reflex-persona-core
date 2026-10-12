import copy
from dataclasses import replace
import unittest
from reflex.core import Policy,NEEDS,digest
from reflex.examples import context,action,effect
from reflex.actor_model import ActorModel
from reflex.decision_loop import DecisionLoop,Request
from reflex.judgment import Binding
from reflex.model_observer import ModelFeedback
from reflex.deliberation import JointForecast,select
from reflex.forecast_control import ForecastBudget,ForecastController,ForecastLedger,checked_choice
from reflex.flow_rollout import rollout
from reflex.purpose_plan import Goal
from reflex.pressure import NeedPressure
from reflex.social import SocialBinding,SocialEvent
from reflex.social_withdrawal import WithdrawalSocialMemory
from reflex.loop_predictor import CategoricalReader
from reflex.planning import vector


def request(c):
    return Request(c,{a['id']:Binding(a['id'],'public',()) for a in c['actions']},{a['id']:() for a in c['actions']})


def scene():
    return context('feedback-controller',[action('rest',effect(.2)),action('work',effect(.1))],needs={k:0. for k in NEEDS})


def force_work(cs,ds):
    cc=copy.deepcopy(cs[0]);cc['actions']=[action('plan',effect(1.))]
    return JointForecast((cc,),{'plan':('work',)},{'plan':1.},2)


class ForecastControlTests(unittest.TestCase):
    def test_modeled_future_uses_real_planner_state_without_seed_or_owner_mutation(self):
        c=scene();actor=DecisionLoop(c);saved=actor.record();calls=[]
        def future(own,req,public):
            calls.append((public,req.context['tick'],own.owner));return force_work
        model=ActorModel(actor,root_request=request(c),request_for=request,feedback=lambda *a:ModelFeedback(None),future_planner=future)
        def observe(t,state):
            cc=copy.deepcopy(c);cc.update(tick=t,state=state);return cc
        _,audit=rollout(c,0,observe=observe,advance=lambda t,k,s:(t+1,effect()),terminal=lambda t:t==3,
            horizon=3,seeds=(9,10),observer=model,policy=actor.policy)
        for rows in audit.values():
            for row in rows:self.assertEqual(row['actions'][1:],['work','work'])
        self.assertEqual({x[0] for x in calls},{1,2})
        self.assertEqual(actor.record(),saved)
        actual=copy.deepcopy(actor);actual.decide(request(c),False);actual.abandon(actual.pending['ticket'])
        cc=observe(1,actual.state)
        result=DecisionLoop.decide_batch([(actual,request(cc))],False,force_work)[0]
        state=audit['rest'][0]['modeled_observations'][1]['checkpoint']['actor']['state']
        self.assertEqual(state,result['decision']['next_state'])

    def test_public_planner_exception_leaves_live_owner_untouched(self):
        c=scene();actor=DecisionLoop(c);saved=actor.record()
        def fail(*a):raise RuntimeError('modeled controller failed')
        model=ActorModel(actor,root_request=request(c),request_for=request,feedback=lambda *a:ModelFeedback(None),future_planner=fail)
        def observe(t,state):
            cc=copy.deepcopy(c);cc.update(tick=t,state=state);return cc
        with self.assertRaises(RuntimeError):
            rollout(c,0,observe=observe,advance=lambda t,k,s:(t+1,effect()),terminal=lambda t:t==2,
                horizon=2,observer=model,policy=actor.policy)
        self.assertEqual(actor.record(),saved)

    def test_planned_future_matches_composed_real_learning_pressure_and_relationships(self):
        c=scene();c['actions'][0]['target']='friend';c['values']['benevolence']=1.
        models=lambda cc:{'old':{'A':.9,'B':.1},'new':{'A':.1,'B':.9}}
        reader=CategoricalReader(c['scope'],('old','new'),models,lambda *a:effect(.1),
            belief_options=dict(prior={'old':.8,'new':.2},inertia=.5,assumption_trust=.4))
        original=DecisionLoop(c,policy=Policy(principle_priority='finite'),predictor=reader,
            pressure=NeedPressure(c['scope']),social=WithdrawalSocialMemory(c));saved=original.record()
        def req(cc):
            return Request(cc,{a['id']:Binding(a['id'],'public',('objective',)) for a in cc['actions']},
                {a['id']:() for a in cc['actions']},maintains_advantage=True,
                social_bindings={'rest':SocialBinding('friend',1.)})
        def observe(t,state):
            cc=copy.deepcopy(c);cc.update(tick=t,state=state);return cc
        def feedback(before,after,key,cc):
            t=cc['tick']
            return ModelFeedback(None,observed=vector(effect(.3 if t<3 else -.2)),
                event={'revealed_action':'A' if t<3 else 'B'},needs={'growth':.2},
                social_events=(SocialEvent('friend',str(t),t,.4 if t<3 else -.6),))
        actual=copy.deepcopy(original);states=[];actions=[]
        for t in range(6):
            cc=observe(t,actual.state or c['state']);r=DecisionLoop.decide_batch([(actual,req(cc))],False,None if t==0 else force_work)[0]
            key=r['decision']['action_id'];fb=feedback(t,t+1,key,cc)
            actual.observe(r['ticket'],fb.observed,event=fb.event,need_progress=fb.needs,social_events=fb.social_events)
            states.append(actual.record());actions.append(key)
        model=ActorModel(original,root_request=req(c),request_for=req,feedback=feedback,future_planner=lambda *a:force_work)
        _,audit=rollout(c,0,observe=observe,advance=lambda t,k,s:(t+1,effect()),terminal=lambda t:t==6,
            horizon=6,observer=model,policy=original.policy)
        row=audit[actions[0]][0];self.assertEqual(row['actions'],actions)
        self.assertEqual([o['checkpoint']['actor'] for o in row['modeled_observations']],states)
        self.assertEqual(original.record(),saved)

    def test_budgets_reject_seed_leakage_and_unbounded_recursion(self):
        for kw in (dict(seeds=(1,1)),dict(validation_seeds=(401,501)),dict(future_seeds=(401,)),dict(future_levels=3),
            dict(extra_validation_seeds=(505,)),dict(validation_seeds=(501,502),extra_validation_seeds=(502,503))):
            with self.assertRaises(ValueError):ForecastBudget(**kw)

    def test_additional_information_allows_beneficial_risk_without_always_reading_more(self):
        c=scene();c['needs']['growth']['deficit']=1.;actor=DecisionLoop(c,policy=Policy(principle_priority='finite'))
        controller=ForecastController(observe=None,advance=None,terminal=None,assess=None,request_for=request,feedback=None,unit='turns',target='goal')
        calls=[];uncertain=[True]
        def evaluate(cc,initial,own,req,budget,*,roots=None):
            calls.append(budget.seeds);fc=copy.deepcopy(cc);endpoint={};actions=[];purpose={}
            for key in ('rest','work'):
                values=[0. if key=='rest' or s<500 else (.5 if s>=505 or not uncertain[0] or s%2 else -.5) for s in budget.seeds]
                growth=.7 if key==('rest' if budget.horizon==2 else 'work') else .2
                actions.append(action(key,*[effect(v,needs={'growth':growth},p=1/len(values)) for v in values]))
                endpoint[key]=[dict(target='goal',status='scored',value=v,positive_confidence=1.) for v in values]
                purpose[key]=sum(values)/len(values)
            fc['actions']=actions
            return JointForecast((fc,),{k:(k,) for k in purpose},purpose,budget.horizon,.15,
                audit=dict(unit='turns',endpoint=endpoint,scenario_seeds=list(budget.seeds)),target='goal'),{}
        controller.evaluate=evaluate
        budget=ForecastBudget(validation_seeds=(501,502,503,504),extra_validation_seeds=(505,506,507,508))
        f=controller.planner(actor,request(c),None,budget=budget)([c],[])
        self.assertEqual(select([c],f,actor.policy)[0][0]['action_id'],'work')
        self.assertEqual(f.audit['adaptive_validation']['total_scenarios'],8);self.assertEqual(len(calls),4)
        uncertain[0]=False;calls.clear()
        f=controller.planner(actor,request(c),None,budget=budget)([c],[])
        self.assertEqual(select([c],f,actor.policy)[0][0]['action_id'],'work')
        self.assertNotIn('adaptive_validation',f.audit);self.assertEqual(len(calls),3)

    def test_goal_challenger_can_correct_agreement_on_the_same_bad_root(self):
        c=scene();c['actions'].append(action('invest',effect(.1)));c['needs']['growth']['deficit']=1.
        actor=DecisionLoop(c,policy=Policy(principle_priority='finite'));calls=[]
        controller=ForecastController(observe=None,advance=None,terminal=None,assess=None,request_for=request,feedback=None,unit='turns',target='goal')
        def evaluate(cc,initial,own,req,budget,*,roots=None):
            keys=tuple(roots or ('rest','work','invest'));calls.append((budget.seeds,keys));fc=copy.deepcopy(cc);actions=[];purpose={};endpoint={}
            for key in keys:
                v=(.6 if key=='invest' else -.6) if budget.seeds[0]>=500 else (.1 if key=='invest' else 0.)
                actions.append(action(key,effect(v,needs={'growth':.8 if key=='rest' else .1})))
                purpose[key]=v;endpoint[key]=[dict(target='goal',status='scored',value=v,positive_confidence=1.) for s in budget.seeds]
            fc['actions']=actions
            return JointForecast((fc,),{k:(k,) for k in keys},purpose,budget.horizon,.15,
                audit=dict(unit='turns',endpoint=endpoint,scenario_seeds=list(budget.seeds)),target='goal'),{}
        controller.evaluate=evaluate
        budget=ForecastBudget(validation_seeds=(501,502,503,504),audit_seeds=(701,702,703,704),purpose_challenger=True)
        f=controller.planner(actor,request(c),None,budget=budget)([c],[])
        self.assertEqual(select([c],f,actor.policy)[0][0]['action_id'],'invest')
        self.assertEqual(f.audit['purpose_challenger']['goal_root'],'invest')
        self.assertEqual(f.audit['purpose_challenger']['final_audit_roots'],['invest','rest'])
        self.assertEqual(len(calls),4)
        self.assertEqual(actor.personality,c['personality']);self.assertEqual(actor.values,c['values'])

    def test_direct_continuation_cannot_silently_drop_modeled_learning(self):
        c=scene();actor=DecisionLoop(c);saved=actor.record()
        def observe(t,state):
            cc=copy.deepcopy(c);cc.update(tick=t,state=state);return cc
        controller=ForecastController(observe=observe,advance=lambda t,k,s:(t+1,effect()),terminal=lambda t:t==2,
            assess=lambda t:Goal('goal','scored' if t==2 else 'running',0.).record(),request_for=request,
            feedback=lambda *a:ModelFeedback(None,observed=[0.]*22),unit='turns',target='goal')
        with self.assertRaisesRegex(ValueError,'feedback'):
            controller.evaluate(c,0,actor,request(c),ForecastBudget(horizon=2,direct_continuation=True))
        self.assertEqual(actor.record(),saved)

    def forecasts(self,c,values,bank):
        fc=copy.deepcopy(c);fc['actions']=[action(k,effect(v)) for k,v in values.items()]
        endpoint={k:[dict(target='goal',status='scored',value=x,positive_confidence=1.) for x in xs] for k,xs in bank.items()}
        return JointForecast((fc,),{k:(k,) for k in values},{k:sum(bank[k])/len(bank[k]) for k in values},2,.15,
            audit=dict(unit='turns',endpoint=endpoint),target='goal')

    def test_independent_check_rejects_lucky_discovery_but_allows_a_personal_cost(self):
        c=scene();p=Policy(principle_priority='finite')
        candidate=self.forecasts(c,dict(rest=0.,work=1.),dict(rest=[0.,0.],work=[.6,.6]))
        reference=self.forecasts(c,dict(rest=1.,work=0.),dict(rest=[.6,.6],work=[0.,0.]))
        held=self.forecasts(c,dict(rest=0.,work=1.),dict(rest=[.6,.6],work=[-.8,.8]))
        checked=checked_choice(c,candidate,reference,held,p)
        self.assertEqual(select([c],checked,p)[0][0]['action_id'],'rest')
        held=self.forecasts(c,dict(rest=0.,work=1.),dict(rest=[.6,.6],work=[.5,.5]))
        checked=checked_choice(c,candidate,reference,held,p)
        self.assertEqual(select([c],checked,p)[0][0]['action_id'],'work')

    def test_check_cannot_restore_a_reference_outside_strongest_principle_tier(self):
        c=scene();p=Policy()
        a=self.forecasts(c,dict(rest=0.,work=1.),dict(rest=[0.,0.],work=[1.,1.]))
        b=self.forecasts(c,dict(rest=1.,work=0.),dict(rest=[1.,1.],work=[0.,0.]))
        v=self.forecasts(c,dict(rest=0.,work=1.),dict(rest=[0.,0.],work=[1.,1.]))
        cc=copy.deepcopy(v.contexts[0]);cc['actions']=[a for a in cc['actions'] if a['id']=='work']
        v=replace(v,contexts=(cc,),roots={'work':('work',)},purpose={'work':1.})
        checked=checked_choice(c,a,b,v,p)
        self.assertEqual(select([c],checked,p)[0][0]['action_id'],'work')
        self.assertIn('tier',checked.audit['forecast_check']['reason'])

    def test_ledger_scores_only_when_delayed_target_is_observed(self):
        c=scene();ledger=ForecastLedger(c['scope']);running=Goal('goal','running',0.).record()
        row=dict(actions=['rest','work'],assessment=Goal('goal','scored',1.).record())
        ledger.commit(c,0,'rest',2,[row],running)
        self.assertEqual(ledger.observe(1,'rest',running),[])
        result=ledger.observe(2,'rest',Goal('goal','scored',1.).record(),terminal=True)[0]
        self.assertEqual(result['forecast_loss'],0.)
        self.assertEqual(result['matching_action_branches'],0)
        self.assertEqual(ledger.pending,[])

    def test_early_absorbing_result_and_failed_feedback_are_explicit(self):
        c=scene();ledger=ForecastLedger(c['scope']);running=Goal('goal','running',0.).record()
        row=dict(actions=['rest'],assessment=Goal('goal','failure',-1.).record())
        ledger.commit(c,0,'rest',4,[row],running);saved=ledger.record()
        with self.assertRaises(ValueError):ledger.observe(1,'rest',Goal('other','failure',-1.).record(),terminal=True)
        self.assertEqual(ledger.record(),saved)
        result=ledger.observe(1,'rest',row['assessment'],terminal=True)[0]
        self.assertEqual(result['resolved'],1);self.assertEqual(result['forecast_loss'],0.)


if __name__=='__main__':unittest.main()
