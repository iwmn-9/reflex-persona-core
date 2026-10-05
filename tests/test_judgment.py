import copy
from concurrent.futures import ThreadPoolExecutor
import unittest
import numpy as np
from reflex.core import FEATURES, Policy, compile_batch, digest
from reflex.examples import action, context, effect
from reflex.judgment import Binding, OutcomeMemory, ReadControl, AdaptivePopulation, avoid_waste, choose
from reflex.judgment_experiment import Arena,Commons,Routes,play,snapshot,conflict_probe
from reflex.laboratory import profiles
from reflex.planning import vector


class JudgmentTests(unittest.TestCase):
    def fixture(self):
        c=context('test',[action('A',effect(.8)),action('B',effect(.3))],needs={})
        for n in c['needs']:c['needs'][n]=dict(supported=False,enabled=False,deficit=None)
        return c

    def bindings(self,c):return {a['id']:Binding(a['id'],'same',('objective',)) for a in c['actions']}

    def trial(self,m,c,outcome):
        bindings=self.bindings(c); prepared,_=m.prepare(c,bindings)
        d,_=choose(prepared,stochastic=False);ticket=m.commit(prepared,d,bindings)
        result=m.observe(ticket,vector(outcome)); c['tick']+=1
        return d,result

    def test_selected_bad_outcomes_change_means_not_personality(self):
        c=self.fixture();m=OutcomeMemory(c['scope']);original=copy.deepcopy(c)
        for _ in range(2): self.assertEqual(self.trial(m,c,effect(-.8))[0]['action_id'],'A')
        learned,stats=m.prepare(c,self.bindings(c))
        self.assertEqual(choose(learned,stochastic=False)[0]['action_id'],'B')
        self.assertEqual(learned['personality'],original['personality']);self.assertEqual(learned['values'],original['values'])
        self.assertEqual(stats['B']['observations'],0)
        self.assertEqual(len(m.entries),1)

    def test_single_loss_is_not_permanent_failure(self):
        c=self.fixture();m=OutcomeMemory(c['scope']);self.trial(m,c,effect(-.8))
        learned,stats=m.prepare(c,self.bindings(c))
        self.assertEqual(learned['actions'],c['actions']);self.assertEqual(stats['A']['empirical_weight'],0)
        self.assertFalse(any(a['known_failure'] for a in learned['actions']))
        # An unlearned future-value proxy is not a certificate of exactness.
        other=OutcomeMemory(c['scope']);bindings={a['id']:Binding(a['id'],'future-proxy',()) for a in c['actions']}
        self.assertEqual(other.prepare(c,bindings)[1]['A']['provenance'],'not_empirically_updated')

    def test_wrong_owner_ticket_duplicate_and_unselected_cannot_enter_memory(self):
        c=self.fixture();m=OutcomeMemory(c['scope']);bindings=self.bindings(c)
        other=copy.deepcopy(c);other['scope']['npc']='other'
        with self.assertRaises(ValueError):m.prepare(other,bindings)
        d,_=choose(c,stochastic=False);ticket=m.commit(c,d,bindings)
        with self.assertRaises(ValueError):m.observe('wrong',vector(effect(-.8)))
        with self.assertRaises(ValueError):m.commit(c,d,bindings)
        m.observe(ticket,vector(effect(-.8)))
        with self.assertRaises(ValueError):m.observe(ticket,vector(effect(-.8)))
        with self.assertRaises(ValueError):m.commit(c,d,bindings)
        fake=copy.deepcopy(d);fake['context_hash']='wrong'
        c['tick']+=1
        with self.assertRaises(ValueError):m.commit(c,fake,bindings)

    def test_exact_dimensions_are_not_overwritten_by_empirical_loss(self):
        c=self.fixture();c['actions'][0]['outcomes'][0]['cost']=.15
        m=OutcomeMemory(c['scope'])
        for _ in range(2):self.trial(m,c,effect(-.8,cost=.9))
        learned,_=m.prepare(c,self.bindings(c))
        self.assertTrue(all(o['cost']==.15 for o in learned['actions'][0]['outcomes']))
        self.assertAlmostEqual(sum(o['p'] for o in learned['actions'][0]['outcomes']),1)
        # The observed correlation of estimated vs exact branch effects is not
        # inferred. Deterministic cost stays exact, empirical joint features stay.
        compile_batch([learned])

    def test_memory_compression_preserves_joint_non_estimated_branches(self):
        fixed=np.arange(1,len(FEATURES))  # only objective is estimated
        def distribution(rows):
            result={}
            for row in rows:
                if row['p']:
                    key=tuple(vector(row)[fixed]);result[key]=result.get(key,0.)+row['p']
            return result
        for probabilities in ((.5,.5),(.000001,.399999,.6),tuple([.125]*8),(0.,.2,.3,.5)):
            with self.subTest(probabilities=probabilities):
                c=self.fixture();n=len(probabilities)
                prior=[effect(.8-i/(n-1),cost=i/(n-1),p=p,
                    values={'security':(-1.)**i,'benevolence':i/n},
                    style={'openness':.15 if i%2 else -.15}) for i,p in enumerate(probabilities)]
                c['actions']=[action('A',*prior)];m=OutcomeMemory(c['scope'])
                for observed in (.1,.2,.3,.4):self.trial(m,c,effect(observed))
                before=copy.deepcopy(c);saved=m.record()
                learned,stats=m.prepare(c,self.bindings(c));rows=learned['actions'][0]['outcomes']
                original=distribution(prior);actual=distribution(rows)
                self.assertEqual(set(actual),set(original))
                for key,p in original.items():self.assertAlmostEqual(actual[key],p)
                self.assertLessEqual(len(rows),8);self.assertAlmostEqual(sum(r['p'] for r in rows),1.)
                weight=stats['A']['empirical_weight']
                expected=sum(r['p']*vector(r) for r in prior)
                expected[0]=(1-weight)*expected[0]+weight*.25
                np.testing.assert_allclose(sum(r['p']*vector(r) for r in rows),expected,atol=1e-14)
                self.assertEqual(c,before);self.assertEqual(m.record(),saved)
                compile_batch([learned])

    def test_repeated_samples_do_not_force_lossy_estimated_compression(self):
        c=self.fixture();prior=[effect(.8,cost=0,p=.2),effect(-.8,cost=1,p=.4),effect(0,cost=.5,p=.4)]
        c['actions']=[action('A',*prior)];m=OutcomeMemory(c['scope'])
        for _ in range(4):self.trial(m,c,effect(0))
        learned,stats=m.prepare(c,self.bindings(c));rows=learned['actions'][0]['outcomes']
        # Fifteen mixture rows contain only five distinct full outcomes. All
        # five fit, so even the estimated objective lottery can remain exact.
        weight=stats['A']['empirical_weight'];expected={}
        for row in prior:
            for objective,p in ((row['objective'],row['p']*(1-weight)),(0.,row['p']*weight)):
                key=(objective,row['cost']);expected[key]=expected.get(key,0.)+p
        actual={(row['objective'],row['cost']):row['p'] for row in rows}
        self.assertEqual(len(rows),5);self.assertEqual(set(actual),set(expected))
        for key,p in expected.items():self.assertAlmostEqual(actual[key],p)

    def test_memory_is_bounded_and_conditions_do_not_leak(self):
        c=self.fixture();c['actions']=c['actions'][:1];m=OutcomeMemory(c['scope'],capacity=2)
        for tick in range(14):
            c['tick']=tick;bindings={'A':Binding('A','x'+str(tick//5),('objective',))}
            d,_=choose(c,stochastic=False);m.observe(m.commit(c,d,bindings),vector(effect(.5)))
        self.assertEqual(len(m.entries),2)
        self.assertTrue(all(len(e['samples'])<=4 and e['count']<=12 for e in m.entries.values()))
        fresh={'A':Binding('A','not-seen',('objective',))}
        self.assertEqual(m.prepare(c,fresh)[1]['A']['observations'],0)
        bad={'A':Binding('A','x2',('cost',))}
        with self.assertRaises(ValueError):m.prepare(c,bad)

    def test_out_of_support_shock_weakens_old_evidence_but_expected_lottery_does_not(self):
        c=self.fixture();c['actions']=c['actions'][:1];m=OutcomeMemory(c['scope'])
        for _ in range(4):self.trial(m,c,effect(.8))
        d,stats=self.trial(m,c,effect(-.8));self.assertTrue(stats['regime_reset']);self.assertEqual(stats['observations'],1)
        c=self.fixture();c['actions']=[action('A',effect(.8,p=.5),effect(-.8,p=.5))];m=OutcomeMemory(c['scope'])
        for _ in range(4):self.trial(m,c,effect(.8))
        self.assertFalse(self.trial(m,c,effect(-.8))[1]['regime_reset'])

    def test_guard_proven_waste_removed_but_estimated_equal_benefit_is_not_proof(self):
        c=context('waste',[action('cheap',effect(.4,cost=.1)),action('waste',effect(.4,cost=.11))])
        d,s=choose(c,{a['id']:FEATURES for a in c['actions']})
        self.assertEqual(s['waste_removed'],['waste']);self.assertEqual(d['action_id'],'cheap')
        _,s=choose(c,{a['id']:('cost',) for a in c['actions']})
        self.assertEqual(s['waste_removed'],[])
        c['actions'][1]['outcomes']=[effect(.5,p=.5,cost=.11),effect(.3,p=.5,cost=.11)]
        self.assertEqual(choose(c,{a['id']:FEATURES for a in c['actions']})[1]['waste_removed'],[])

    def test_guard_respects_familiarity_switch_reading_and_tradeoffs(self):
        c=context('waste',[action('A',effect(.4,cost=.1)),action('B',effect(.4,cost=.11))])
        c['state']['intent_action']='B';c['actions'][0]['switch_cost']=.1
        self.assertEqual(choose(c,{a['id']:FEATURES for a in c['actions']})[1]['waste_removed'],[])
        c['actions'][0]['switch_cost']=0;c['actions'][1]['outcomes'][0]['values']={'benevolence':.1}
        self.assertEqual(choose(c,{a['id']:FEATURES for a in c['actions']})[1]['waste_removed'],[])
        rows=conflict_probe();self.assertEqual(next(r['action'] for r in rows if r['profile']=='care'),'HELP')
        self.assertEqual(next(r['action'] for r in rows if r['profile']=='steady'),'SECURE')
        self.assertEqual(next(r['action'] for r in rows if r['profile']=='ego'),'CLAIM')

    def test_guard_batch_split_reverse_threads_and_input_immutability(self):
        cs=[snapshot(Commons(),p,0,0) for p in profiles()];b=compile_batch(cs)
        mask=np.ones(b.effects.shape[:2]+(len(FEATURES),),bool);old=b.legal.copy()
        g,removed=avoid_waste(b,mask);p=Policy();expected=p.decide(g).records(g)
        parts=[]
        for i in range(len(cs)):
            bb=b.take([i]);gg,_=avoid_waste(bb,mask[[i]]);parts+=p.decide(gg).records(gg)
        self.assertEqual(expected,parts);self.assertTrue(np.array_equal(old,b.legal))
        rev=b.take([3,2,1,0]);gg,_=avoid_waste(rev,mask[[3,2,1,0]])
        self.assertEqual(expected,list(reversed(p.decide(gg).records(gg))))
        with ThreadPoolExecutor(max_workers=4) as pool:
            actual=list(pool.map(lambda i:p.decide(avoid_waste(b.take([i]),mask[[i]])[0]).records(b.take([i]))[0],range(4)))
        self.assertEqual(expected,actual)

    def test_read_budget_protects_advantage_and_requires_available_model(self):
        c=context('reading',[action('A',effect(-.6),confidence=.7),action('B',effect(-.59),confidence=.7)])
        b=compile_batch([c]);d=Policy().decide(b);ctl=ReadControl(max_nodes=12)
        def request(available,ahead,threat):return ctl.request(b,d,np.array([available]),np.array([ahead]),np.array([threat]))
        self.assertTrue(request(True,False,False)['requested'][0])
        self.assertFalse(request(True,True,False)['requested'][0])
        self.assertTrue(request(True,True,True)['requested'][0])
        self.assertFalse(request(False,False,True)['requested'][0])
        self.assertEqual(request(True,False,True)['nodes'][0],12)

    def test_one_legal_root_does_not_read_to_break_ties(self):
        c=context('one',[action('A',effect(.4),confidence=.1)])
        b=compile_batch([c]);d=Policy().decide(b)
        req=ReadControl().request(b,d,np.array([True]),np.array([False]),np.array([False]))
        self.assertFalse(req['requested'][0])

    def test_realized_public_facts_only_and_hidden_regime_does_not_change_initial_choice(self):
        a=Arena();c=snapshot(a,profiles()[0],0,0)
        self.assertNotIn('regime',json_string(c));self.assertEqual(a.history,[])
        rival=c['facts']['public_moves'];self.assertEqual(rival,'[]')
        for world in (Arena,Commons,Routes):
            x=world();before=copy.deepcopy(x.__dict__);snap=snapshot(x,profiles()[0],0,0)
            self.assertEqual(before,x.__dict__)
            self.assertFalse(any('blocked' in key or 'schedule' in key for key in snap['facts']))

    def test_three_mechanics_reduce_repeated_losses_and_read_nodes_are_enforced(self):
        for world in (Arena,Commons,Routes):
            old=play(world,profiles()[0],0,'fixed')
            new=play(world,profiles()[0],0,'experience_read')
            self.assertLess(new['losses'],old['losses'],world.name)
            self.assertGreater(new['score'],old['score'],world.name)
            self.assertTrue(all(r['reading']['nodes']<=16 for r in new['trace']))
            for r in new['trace']:
                self.assertEqual(r['context']['personality'],new['trace'][0]['context']['personality'])
                self.assertTrue(next(a['legal'] for a in r['context']['actions'] if a['id']==r['decision']['action_id']))

    def test_same_seed_replays_actions_realized_results_and_state(self):
        for world in (Arena,Commons,Routes):
            first=play(world,profiles()[1],2,'experience_read',turns=24)
            second=play(world,profiles()[1],2,'experience_read',turns=24)
            self.assertEqual(first['trace'],second['trace'])

    def test_memory_checkpoint_replays_and_rejects_corrupt_or_foreign_state(self):
        import json
        c=self.fixture();m=OutcomeMemory(c['scope'])
        for _ in range(2):self.trial(m,c,effect(-.8))
        record=json.loads(json.dumps(m.record()))
        restored=OutcomeMemory.from_record(c['scope'],record)
        self.assertEqual(m.prepare(c,self.bindings(c)),restored.prepare(c,self.bindings(c)))
        self.assertEqual(m.record(),restored.record())
        corrupt=copy.deepcopy(record);corrupt['entries'][0]['samples'][0][0]=float('nan')
        with self.assertRaises(ValueError):OutcomeMemory.from_record(c['scope'],corrupt)
        foreign=copy.deepcopy(c['scope']);foreign['npc']='foreign'
        with self.assertRaises(ValueError):OutcomeMemory.from_record(foreign,record)
        learned,_=m.prepare(c,self.bindings(c));d,_=choose(learned,stochastic=False)
        m.commit(learned,d,self.bindings(c))
        with self.assertRaises(ValueError):m.record()

    def test_paid_information_acquisition_restarts_stalled_delivery(self):
        run=play(Routes,profiles()[0],0,'experience',turns=60)
        self.assertGreater(run['deliveries'],10)
        checks=[r for r in run['trace'] if r['decision']['action_id']=='CHECK']
        self.assertTrue(checks)
        for row in checks:
            self.assertEqual(row['realized']['objective'],-.03)
            self.assertEqual(row['context']['facts']['survey'],'None')
            self.assertIn(row['public_after']['surveyed_block'],('LEFT','RIGHT'))
        # The survey is paid and observed after a selected legal action, never
        # inferred by reading the driver's hidden blocked-lane variable.

    def test_numeric_adaptation_matches_snapshot_learning_and_batch_splits(self):
        cs=[self.fixture() for _ in range(4)]
        for i,c in enumerate(cs):c['scope']['npc']=f'actor-{i}'
        b=compile_batch(cs);mask=np.zeros(b.effects.shape[:2]+(len(FEATURES),),bool);mask[:,:,0]=True
        pop=AdaptivePopulation(cs,mask);split=[AdaptivePopulation([c],mask[[i]]) for i,c in enumerate(cs)]
        memories=[OutcomeMemory(c['scope']) for c in cs]
        for tick in range(12):
            numeric=pop.step(False); singles=[x.step(False) for x in split]
            self.assertEqual(pop.action_ids(numeric),[x.action_ids(d)[0] for x,d in zip(split,singles)])
            observed=[]
            for i,(c,m) in enumerate(zip(cs,memories)):
                c['tick']=tick;prepared,_=m.prepare(c,self.bindings(c));d,_=choose(prepared,stochastic=False)
                self.assertEqual(d['action_id'],pop.action_ids(numeric)[i])
                c['state']=d['next_state']
                value=effect(-.8 if i%2==0 and d['action_id']=='A' else .8)
                # Ticket must match the original selected snapshot, not its
                # subsequently updated persistent policy state.
                m.observe(m.commit(prepared,d,self.bindings(prepared)),vector(value));observed.append(vector(value))
            pop.observe(observed)
            for x,v in zip(split,observed):x.observe([v])
        self.assertTrue(np.array_equal(pop.count,np.concatenate([x.count for x in split])))

    def test_numeric_unknown_outcomes_learn_nothing_and_cannot_reuse_step(self):
        c=self.fixture();b=compile_batch([c]);mask=np.zeros(b.effects.shape[:2]+(len(FEATURES),),bool);mask[:,:,0]=True
        pop=AdaptivePopulation([c],mask);pop.step()
        with self.assertRaises(ValueError):pop.step()
        pop.observe([vector(effect(-.8))],np.array([False]))
        self.assertEqual(int(pop.count.sum()),0)
        with self.assertRaises(ValueError):pop.observe([vector(effect(-.8))])
        with self.assertRaises(ValueError):pop.step(effects=pop.batch.effects)


def json_string(value):
    import json
    return json.dumps(value)
