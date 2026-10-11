import copy,json,unittest
from dataclasses import replace
from reflex.core import Policy,digest
from reflex.examples import context,action,effect
from reflex.flow_rollout import rollout
from reflex.purpose_plan import Goal
from reflex.deliberation import select
from reflex.contingent_search import search_forecast
from reflex.observation_plan import ObservationPlan,View,PlanMemory
from tools.probe_observation_plans import Model,State


class ContingentSearchTests(unittest.TestCase):
    def setup_model(self,visible=True,horizon=3,reveal=1):
        m=Model(horizon,reveal,visible,.75)
        c=context('conditional',[action(k,effect()) for k in m.legal(State())],
                  needs={'physiology':0.,'safety':0.})
        def observe(s,memory):
            cc=copy.deepcopy(c);cc['state']=copy.deepcopy(memory);cc['tick']=s.tick
            cc['facts']={'observed':json.dumps(m.public(s))}
            cc['actions']=[action(k,effect(.2 if k=='b2' else -.05 if k in ('b0','b1') else 0)) for k in m.legal(s)]
            return cc
        def advance(s,k,seed):
            if s.tick==0:s=replace(s,world=seed%2)
            row=effect() if not s.tick else next(a['outcomes'][0] for a in observe(s,c['state'])['actions'] if a['id']==k)
            return m.step(s,k),row
        return m,c,observe,advance

    def forecast(self,visible=True,horizon=3,reveal=1,**extra):
        m,c,observe,advance=self.setup_model(visible,horizon,reveal)
        f=search_forecast(c,State(),observe=observe,advance=advance,terminal=m.terminal,
            assess=m.goal,horizon=horizon,seeds=(0,1),width=2,revisions=2,target=m.target,
            max_regret=0.,settlement_weight='absolute',**extra)
        return m,c,f

    def test_observed_branch_can_be_patched_without_changing_other_branch(self):
        m,c,f=self.forecast()
        d,v=select([c],f)
        self.assertEqual(d[0]['action_id'],'a1');self.assertEqual(v['selected_purpose'],1.)
        p=f.audit['plans'][v['selected_plan']]['proposal']
        traces=f.audit['contingent_search']['proposals'][p]['branches']
        self.assertEqual({t['actions'][1] for t in traces},{'b0','b1'})
        for t in traces:self.assertEqual(t['public_nodes'][1]['actor_state'],d[0]['next_state'])

    def test_blind_branches_share_policy_and_cannot_take_oracle_profit(self):
        m,c,f=self.forecast(False)
        self.assertEqual(select([c],f)[1]['selected_purpose'],.75)
        for p in f.audit['contingent_search']['proposals'].values():
            traces=p['branches']
            if p['root']=='a1':
                self.assertEqual(traces[0]['actions'],traces[1]['actions'])
                self.assertLessEqual(sum(t['assessment']['value'] for t in traces)/2,.5)

    def test_patches_can_reach_late_information_not_only_first_two_steps(self):
        _,c,f=self.forecast(True,8,6)
        self.assertEqual(select([c],f)[1]['selected_purpose'],1.)

    def test_public_control_api_has_no_world_or_seed_argument(self):
        m,c,observe,advance=self.setup_model();views=[]
        def control(view):
            self.assertIsInstance(view,View);views.append(view)
            self.assertEqual(set(view.__dataclass_fields__),{'context','allowed','prefix','node'})
            return None
        before=digest(c)
        rollout(c,State(),observe=observe,advance=advance,terminal=m.terminal,horizon=3,seeds=(0,1),public_controller=control)
        self.assertEqual(before,digest(c));self.assertGreater(len(views),0)

    def test_unknown_or_illegal_node_returns_to_persona_reflex(self):
        _,c,observe,advance=self.setup_model();v=View.make('',c)
        p=ObservationPlan.make(c,'goal','turns',{v.node:'not-currently-legal'})
        self.assertIsNone(p.choose(v));self.assertIsNone(ObservationPlan.make(c,'goal','turns').choose(v))
        other=copy.deepcopy(c);other['scope']['npc']='other'
        with self.assertRaises(ValueError):p.choose(View.make('',other))

    def test_node_does_not_depend_on_action_or_outcome_order(self):
        _,c,_,_=self.setup_model();other=copy.deepcopy(c);other['actions'].reverse()
        self.assertEqual(View.make('',c).node,View.make('',other).node)

    def test_program_restore_and_owned_history_commit(self):
        m,c,f=self.forecast();d,v=select([c],f);memory=PlanMemory(c,m.target)
        memory.remember(c,f,v['selected_plan'],d[0]['action_id'])
        program,prefix=memory.offer(c)
        self.assertEqual(ObservationPlan.restore(program.record()),program)
        self.assertEqual(len(prefix),64)
        with self.assertRaises(ValueError):memory.remember(c,f,v['selected_plan'],d[0]['action_id'])

    def test_validation_only_evaluates_frozen_programs(self):
        _,c,f=self.forecast(validation_seeds=(2,3))
        self.assertEqual(select([c],f)[1]['selected_purpose'],1.)
        self.assertEqual(f.audit['contingent_search']['validation']['seeds'],[2,3])
        with self.assertRaises(ValueError):self.forecast(validation_seeds=(1,2))

    def test_program_cannot_override_observed_progress_permission(self):
        m,c,observe,advance=self.setup_model()
        with self.assertRaises(ValueError):
            rollout(c,State(),observe=observe,advance=advance,terminal=m.terminal,horizon=3,
                    public_controller=lambda view:'absent')

    def test_failed_commit_is_transactional_and_checkpoint_restores(self):
        m,c,f=self.forecast();memory=PlanMemory(c,m.target);before=memory.record()
        d,v=select([c],f)
        with self.assertRaises(ValueError):memory.remember(c,f,v['selected_plan'],'a0')
        self.assertEqual(memory.record(),before)
        memory.remember(c,f,v['selected_plan'],d[0]['action_id'])
        self.assertEqual(PlanMemory.restore(c,memory.record()).record(),memory.record())

    def test_forecast_cannot_install_another_owners_or_purposes_program(self):
        from dataclasses import replace
        m,c,f=self.forecast();d,v=select([c],f)
        for field,value in (('owner','0'*64),('target','other-goal'),('unit','other-unit')):
            memory=PlanMemory(c,m.target);saved=memory.record();audit=copy.deepcopy(f.audit)
            proposal=audit['plans'][v['selected_plan']]['proposal']
            audit['contingent_search']['proposals'][proposal]['program'][field]=value
            with self.assertRaises(ValueError):memory.remember(c,replace(f,audit=audit),v['selected_plan'],d[0]['action_id'])
            self.assertEqual(memory.record(),saved)

    def test_serialized_program_requires_nonempty_owner_and_nodes(self):
        _,c,_,_=self.setup_model();p=ObservationPlan.make(c,'goal','turns')
        with self.assertRaises(ValueError):ObservationPlan('',p.target,p.unit)
        with self.assertRaises(ValueError):ObservationPlan(p.owner,p.target,p.unit,(('','a0'),))

    def test_large_legal_set_keeps_every_root_without_exceeding_numeric_bank(self):
        c=context('large',[action(f'a{i:03d}',effect()) for i in range(256)])
        def observe(s,memory):
            cc=copy.deepcopy(c);cc['tick']=s;cc['state']=memory;cc['actions']=[action('finish',effect())];return cc
        f=search_forecast(c,0,observe=observe,advance=lambda s,k,seed:(s+1,effect()),
            terminal=lambda s:s>=2,assess=lambda s:Goal('goal','scored' if s>=2 else 'running',0.).record(),
            horizon=2,target='goal',width=2,revisions=2)
        bank=f.audit['contingent_search']
        self.assertEqual(len(bank['all_viable_roots']),256)
        self.assertEqual({p['root'] for p in bank['proposals'].values()},{a['id'] for a in c['actions']})
        self.assertLessEqual(len(bank['proposals']),256)
        self.assertEqual(sum(bank['extra_slots'].values()),0)

    def test_expected_purpose_does_not_override_strongest_principle(self):
        m,c,observe,advance=self.setup_model()
        c['values']['benevolence']=1.;c['state']['mode']='principle'
        base_advance=advance
        def principled(s,k,seed):
            after,row=base_advance(s,k,seed)
            if k=='a0':row['values']['benevolence']=1.
            return after,row
        f=search_forecast(c,State(),observe=observe,advance=principled,terminal=m.terminal,
            assess=m.goal,horizon=3,seeds=(0,1),width=2,revisions=2,target=m.target,max_regret=0.)
        self.assertEqual(select([c],f)[0][0]['action_id'],'a0')

    def test_selected_program_survives_actual_reveal_with_executed_owner_state(self):
        m,c,f=self.forecast();decisions,vote=select([c],f)
        _,_,observe,advance=self.setup_model()
        for seed,key in ((0,'b0'),(1,'b1')):
            memory=PlanMemory(c,m.target)
            memory.remember(c,f,vote['selected_plan'],decisions[0]['action_id'])
            world,_=advance(State(),decisions[0]['action_id'],seed)
            cc=observe(world,decisions[0]['next_state']);program,prefix=memory.offer(cc)
            self.assertEqual(program.choose(View.make(prefix,cc)),key)

    def test_conditional_search_honors_shadow_progress_permissions(self):
        from reflex.progress import ProgressWatch,ProgressConfig,PurposeRequest,Activity
        from reflex.model_observer import ModelObserver,ModelFeedback
        m,c,observe,advance=self.setup_model();watch=ProgressWatch(c['scope'],ProgressConfig(grace=1));watch.stalls=2
        def purpose(s,cc,previous):
            return PurposeRequest(0.,0.,{a['id']:Activity('idle' if a['id']=='b0' else 'uncertain','public-test') for a in cc['actions']})
        from reflex.progress import PurposeFeedback
        observer=ModelObserver(c['scope'],purpose=purpose,feedback=lambda a,b,k:ModelFeedback(PurposeFeedback(0.,0.)),progress=watch)
        saved=watch.record()
        f=search_forecast(c,State(),observe=observe,advance=advance,terminal=m.terminal,
            assess=m.goal,horizon=3,seeds=(0,1),width=2,revisions=2,target=m.target,observer=observer,max_regret=0.)
        self.assertEqual(watch.record(),saved)
        for proposal in f.audit['contingent_search']['proposals'].values():
            for branch in proposal['branches']:
                for node in branch['public_nodes'][1:]:self.assertNotIn('b0',node['allowed'])
        self.assertEqual(select([c],f)[0][0]['action_id'],'a0')


if __name__=='__main__':unittest.main()
