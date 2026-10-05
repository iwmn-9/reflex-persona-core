import copy
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock

import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from reflex import Policy, Population, compile_batch
from reflex.core import FEATURES, digest, MAX_ACTIONS
from reflex.examples import families, dataset, context, action, effect
from reflex.teachers import generate, validate, prompt, review, save_review, approved_examples
from reflex.training import fit, load
from reflex.experience import Experience
from reflex.opponent import forecast, as_outcomes
from reflex.adapters import ResourceAdapter, ArenaAdapter, profile


class ReflexTests(unittest.TestCase):
    def setUp(self): self.rows=families(); self.p=Policy()
    def case(self,name): return copy.deepcopy(next(r["context"] for r in self.rows if r["id"]==name))

    def test_semantic_seeds_and_parameter_variants(self):
        for r in dataset():
            with self.subTest(case=r["id"]): self.assertIn(self.p.choose(r["context"],False)["action_id"],r["preferred"])

    def test_batch_split_order_threads_match(self):
        cs=[r["context"] for r in self.rows]; b=compile_batch(cs)
        expected=self.p.decide(b).records(b)
        self.assertEqual(expected,[self.p.choose(c) for c in cs])
        reverse=self.p.decide(b.take(list(reversed(range(len(cs)))))).records(b.take(list(reversed(range(len(cs))))))
        self.assertEqual(expected,list(reversed(reverse)))
        split=self.p.decide(b.take(range(7))).records(b.take(range(7)))+self.p.decide(b.take(range(7,len(cs)))).records(b.take(range(7,len(cs))))
        self.assertEqual(expected,split)
        with ThreadPoolExecutor(max_workers=4) as pool: self.assertEqual(expected,list(pool.map(self.p.choose,cs)))

    def test_inputs_immutable_and_global_rng_unchanged(self):
        c=self.case("urgent_growth"); old=copy.deepcopy(c); b=compile_batch([c]); state=np.random.get_state()
        self.p.decide(b); self.assertEqual(c,old)
        self.assertFalse(b.effects.flags.writeable)
        self.assertTrue(np.array_equal(state[1],np.random.get_state()[1]))

    def test_candidate_order_invariant(self):
        c=self.case("compatible_values"); first=self.p.choose(c)
        c["actions"].reverse(); second=self.p.choose(c)
        self.assertEqual(first["action_id"],second["action_id"])
        self.assertNotEqual(first["context_hash"],second["context_hash"])

    def test_game_and_action_names_are_not_logic(self):
        c=self.case("urgent_growth"); result=self.p.choose(c,False)["action_id"]
        mapping={a["id"]:"z"+str(i) for i,a in enumerate(c["actions"])}
        for a in c["actions"]: a["id"]=mapping[a["id"]]
        c["scope"]["game"]="unseen_game"
        self.assertEqual(self.p.choose(c,False)["action_id"],mapping[result])

    def test_scope_randomness_isolation(self):
        c=self.case("urgent_growth"); first=compile_batch([c]).rng.copy()
        c["scope"]["npc"]="other"; self.assertFalse(np.array_equal(first,compile_batch([c]).rng))
        c["scope"]["npc"]="actor"; c["tick"]+=1; self.assertFalse(np.array_equal(first,compile_batch([c]).rng))

    def test_small_urgency_change_holds_primary_but_large_change_switches(self):
        c=self.case("urgent_growth"); c["needs"]["physiology"]["deficit"]=.8
        c["state"]["primary_need"]="physiology"
        self.assertEqual(self.p.choose(c)["next_state"]["primary_need"],"physiology")
        c["needs"]["physiology"]["deficit"]=.2
        self.assertEqual(self.p.choose(c)["next_state"]["primary_need"],"growth")

    def test_mode_personality_difference_and_equal_probability(self):
        c=self.case("principle_over_gain"); c["state"]["mode"]=None
        c["values"]["benevolence"]=.7; c["needs"]["physiology"]["deficit"]=.7
        counts=[]
        for con,neuro in ((.5,.5),(.9,.1),(.1,.9)):
            cs=[]
            for seed in range(600):
                v=copy.deepcopy(c); v["seed"]=seed; v["personality"].update(conscientiousness=con,neuroticism=neuro); cs.append(v)
            counts.append(self.p.decide(compile_batch(cs)).mode.mean())
        self.assertTrue(.43<counts[0]<.57,counts)
        self.assertGreater(counts[1],counts[0]+.15); self.assertLess(counts[2],counts[0]-.15)

    def test_urgent_change_can_break_old_mode(self):
        c=self.case("principle_over_gain"); c["needs"]["physiology"]["deficit"]=.95
        # No assertion of a single psychological truth; changing need bypasses hold.
        cs=[]
        for i in range(50): v=copy.deepcopy(c); v["seed"]=i; cs.append(v)
        self.assertIn(0,self.p.decide(compile_batch(cs)).mode)

    def test_mode_does_not_reroll_each_tick_when_need_is_unchanged_or_fulfilled(self):
        for deficit in (0,.7):
            c=self.case("principle_over_gain")
            for d in c["needs"].values(): d["deficit"]=deficit
            c["state"].update(primary_need=None,mode=None,mode_urgency=deficit)
            modes=[]
            for tick in range(30):
                c["tick"]=tick; d=self.p.choose(c); c["state"]=d["next_state"]; modes.append(c["state"]["mode"])
            self.assertEqual(len(set(modes)),1)

    def test_no_information_cannot_invent_reading(self):
        c=self.case("credible_threat"); c["opponent"]=None
        d=self.p.choose(c,False); self.assertFalse(d["reading_used"]); self.assertEqual(d["action_id"],"continue")

    def test_growth_not_a_blanket_risk_preference(self):
        c=self.case("growth_risk_high")
        for a in c["actions"]:
            for out in a["outcomes"]: out["needs"].pop("growth",None)
        self.assertEqual(self.p.choose(c,False)["action_id"],"safe")

    def test_no_materially_bad_random_choice(self):
        c=context("large_gap",[action("good",effect(.8)),action("bad",effect(-.8))])
        for seed in range(100):
            c["seed"]=seed; self.assertEqual(self.p.choose(c)["action_id"],"good")

    def test_near_ties_can_vary_and_replay(self):
        c=context("tie",[action("a",effect(.3)),action("b",effect(.3))]); choices=set()
        for seed in range(50):
            c["seed"]=seed; x=self.p.choose(c); self.assertEqual(x,self.p.choose(c)); choices.add(x["action_id"])
        self.assertEqual(choices,{"a","b"})

    def test_unsupported_disabled_and_fulfilled_distinct(self):
        c=self.case("urgent_growth"); hashes=[]
        for d in (dict(supported=False,enabled=False,deficit=None),dict(supported=True,enabled=False,deficit=None),dict(supported=True,enabled=True,deficit=0)):
            c["needs"]["growth"]=d; b=compile_batch([c]); hashes.append(b.hashes[0])
            self.assertEqual(b.needs[0,4],0)
        self.assertEqual(len(set(hashes)),3)

    def test_contract_rejects_bad_probabilities_nan_capacity(self):
        for mutate in (lambda c:c["actions"][0]["outcomes"][0].update(p=.4),
                       lambda c:c["personality"].update(openness=float("nan")),
                       lambda c:c["needs"]["growth"].update(supported=False),
                       lambda c:c.update(version="roleplayer-v2")):
            c=self.case("urgent_growth"); mutate(c)
            with self.assertRaises(ValueError): compile_batch([c])
        c=self.case("urgent_growth")
        c["actions"]=[dict(copy.deepcopy(c["actions"][0]),id=f"candidate-{i}") for i in range(MAX_ACTIONS+1)]
        with self.assertRaisesRegex(ValueError,"capacity"): compile_batch([c])

    def test_no_recovery_action_fails_instead_of_repeating_understood_failure(self):
        c=context("failure",[action("fail",effect(.9),failure=True)])
        with self.assertRaisesRegex(ValueError,"recovery"): self.p.choose(c)

    def test_experience_is_bounded_scoped_and_does_not_mutate_profile(self):
        c=self.case("uncertain_growth"); before=copy.deepcopy(c["personality"])
        c["actions"][1]["confidence"]=1
        self.assertEqual(self.p.choose(c,False)["action_id"],"unproven")
        e=Experience(c["scope"],2); f=np.zeros(len(FEATURES)); f[0]=-.6
        e.observe(c["scope"],"situation","unproven",f)
        estimate=e.estimate(c["scope"],"situation","unproven")
        c["actions"][1]["outcomes"][0]["objective"]=estimate["mean"][0]
        self.assertEqual(self.p.choose(c,False)["action_id"],"known"); self.assertEqual(before,c["personality"])
        estimate["mean"][0]=1; self.assertEqual(e.estimate(c["scope"],"situation","unproven")["mean"][0],-.6)
        with self.assertRaises(ValueError): e.observe(dict(game="other"),"x","x",f)
        e.observe(c["scope"],"x","x",f); e.observe(c["scope"],"y","y",f); self.assertIsNone(e.estimate(c["scope"],"situation","unproven"))

    def test_distinct_executable_game_adapters_and_hidden_world(self):
        p=profile(); p["deficits"]["physiology"]=.9; resource=ResourceAdapter()
        w=dict(time=1,food=1,recovered=0,skill=0,hidden=99); c=resource.observe(w,p)
        self.assertNotIn("PRACTICE",[a["id"] for a in c["actions"]]); self.assertEqual(self.p.choose(c,False)["action_id"],"REST")
        w2=copy.deepcopy(w); w2["hidden"]=500; self.assertEqual(c,resource.observe(w2,p))
        after=resource.transition(w,self.p.choose(c,False)); self.assertEqual(after["time"],0); self.assertEqual(w["time"],1)
        arena=ArenaAdapter(); w=dict(energy=0,score=0,secret_strategy="attack")
        c=arena.observe(w,profile()); d=self.p.choose(c,False); self.assertNotEqual(d["action_id"],"COUNTER")
        self.assertIsInstance(arena.transition(w,d,np.random.default_rng(42))["score"],float)

    def test_forecast_discount_budget_and_personality_scoring(self):
        f=np.zeros(len(FEATURES)); f[0]=.4; nextf=f.copy(); nextf[5]=.6
        tree=[dict(p=1,effects=f.tolist(),children=[dict(p=1,effects=nextf.tolist(),children=[])])]
        result=forecast(tree); self.assertEqual(result["nodes"],2)
        self.assertAlmostEqual(result["effects"][5],.8*.6/1.8)
        with self.assertRaisesRegex(ValueError,"budget"): forecast(tree,budget=1)
        tree[0]["p"]=.5
        with self.assertRaises(ValueError): forecast(tree)

    def test_future_outcomes_keep_risk_and_are_scored_by_personal_goal(self):
        zero=np.zeros(len(FEATURES)); growth=zero.copy(); growth[5]=.8
        kind=zero.copy(); kind[14]=.8  # benevolence feature
        def future(vector):
            return forecast([dict(p=1,effects=zero.tolist(),children=[dict(p=1,effects=vector.tolist(),children=[])])])
        acts=[action("growth_path",*as_outcomes(future(growth))),action("kind_path",*as_outcomes(future(kind)))]
        c=context("future",acts,needs={"growth":.85})
        self.assertEqual(self.p.choose(c,False)["action_id"],"growth_path")
        c=context("future",acts,values={"benevolence":.9},mode="principle"); c["state"]["primary_need"]="physiology"
        self.assertEqual(self.p.choose(c,False)["action_id"],"kind_path")
        loss=zero.copy(); loss[0]=-.7
        tree=[dict(p=.7,effects=growth.tolist(),children=[]),dict(p=.3,effects=loss.tolist(),children=[])]
        outcomes=as_outcomes(forecast(tree,depth=1))
        self.assertEqual(len(outcomes),2); self.assertAlmostEqual(outcomes[1]["p"],.3)

    def test_persistent_runtime_matches_validated_snapshots_across_ticks(self):
        cs=[copy.deepcopy(r["context"]) for r in self.rows]; pop=Population(cs)
        for tick in range(10):
            snapshots=[self.p.choose(c) for c in cs]; result=pop.step()
            self.assertEqual(pop.action_ids(result),[r["action_id"] for r in snapshots])
            for i,(c,r) in enumerate(zip(cs,snapshots)):
                self.assertEqual(pop.batch.mode[i],int(r["next_state"]["mode"]=="principle"))
                self.assertEqual(pop.batch.age[i],r["next_state"]["age"])
                c["state"]=r["next_state"]; c["tick"]+=1

    def test_persistent_runtime_partition_order_and_shared_policy_are_independent(self):
        cs=[copy.deepcopy(r["context"]) for r in self.rows]
        full=Population(cs,self.p); left=Population(cs[:7],self.p); right=Population(cs[7:],self.p); reverse=Population(cs[::-1],self.p)
        with ThreadPoolExecutor(max_workers=2) as pool:
            for _ in range(6):
                a,b=list(pool.map(lambda pop: pop.action_ids(pop.step()),(left,right)))
                expected=full.action_ids(full.step()); self.assertEqual(expected,a+b)
                self.assertEqual(expected,reverse.action_ids(reverse.step())[::-1])

    def test_persistent_runtime_numeric_update_changes_priority_not_personality(self):
        c=self.case("urgent_growth"); pop=Population([c]); traits=pop.batch.traits.copy(); needs=pop.batch.needs.copy()
        needs[0,0]=.95; needs[0,4]=.1; original=needs.copy()
        result=pop.step(needs=needs); self.assertEqual(pop.action_ids(result),["recover"])
        self.assertTrue(np.array_equal(traits,pop.batch.traits)); self.assertTrue(np.array_equal(needs,original))
        with self.assertRaises(ValueError): pop.step(probability=np.zeros_like(pop.batch.probability))
        with self.assertRaises(ValueError): pop.step(personality=np.zeros_like(traits))


class TeacherAndLearningTests(unittest.TestCase):
    def record(self):
        c=families()[0]["context"]
        answer=dict(version="reflex-v3",context_hash=digest(c),action_id="recover",target=None,evidence_ids=["options"],rationale="主目的の回復を満たす。")
        return dict(id="original",prompt=prompt(c),context=c,context_hash=digest(c),answer=json.dumps(answer),answer_complete=True,
                    repo="test-only-mock",revision="mock-revision",precision="mock",generation={"seed":42},raw_output="original mock text")

    def test_validation_scope_evidence_and_feasibility(self):
        r=self.record(); self.assertEqual(validate(r["answer"],r["context"])["action_id"],"recover")
        for key,value in (("context_hash","wrong"),("evidence_ids",["hidden"]),("action_id","absent"),("target","other")):
            a=json.loads(r["answer"]); a[key]=value
            with self.assertRaises(ValueError): validate(a,r["context"])

    def test_review_complete_not_quality_and_edit_revalidates(self):
        r=self.record(); original=copy.deepcopy(r)
        with self.assertRaises(ValueError): review(r,"accept","note")
        r["answer_complete"]=False
        with self.assertRaises(ValueError): review(r,"accept","note",True)
        self.assertTrue(review(r,"edit","note",True,r["answer"])["reconstructed"])
        a=json.loads(r["answer"]); a["action_id"]="invented"
        with self.assertRaises(ValueError): review(r,"edit","note",True,a)
        self.assertEqual(original["raw_output"],r["raw_output"])

    def test_review_export_provenance_tamper_and_rejection(self):
        with tempfile.TemporaryDirectory() as td:
            r=self.record(); save_review(td,r,"accept","相談済みというテスト入力",True)
            folder=Path(td)/"datasets/reflex_v3/reviews"
            self.assertEqual(len(approved_examples(folder)),1)
            path=folder/"original.json"; saved=json.loads(path.read_text(encoding="utf-8")); saved["original"]["revision"]="changed"
            path.write_text(json.dumps(saved),encoding="utf-8")
            with self.assertRaises(ValueError): approved_examples(folder)
            save_review(td,r,"reject","不採用テスト",True); self.assertEqual(approved_examples(folder),[])

    def test_teacher_generation_isolated_raw_saved_and_always_unloads(self):
        session=Mock(); session.answer.side_effect=[self.record(),RuntimeError("mock failure")]
        cases=families()[:2]
        with tempfile.TemporaryDirectory() as td:
            records,blind=generate(session,cases,["mock"],td)
            self.assertEqual(len(records),2); self.assertEqual(len(blind),2)
            self.assertNotIn("repo",blind[0]); self.assertTrue(session.unload.called)
            self.assertFalse(records[1]["answer_complete"])
            self.assertEqual(len(list((Path(td)/"results/reflex_v3").glob("*/*.json"))),3)
            self.assertNotEqual(session.answer.call_args_list[0].args[1],session.answer.call_args_list[1].args[1])

    def test_seed_smoke_training_and_default_rejection(self):
        rows=dataset(2)
        with tempfile.TemporaryDirectory() as td:
            path=Path(td)/"smoke.json"; report=fit(rows,path,True,30)
            self.assertEqual(report["artifact_type"],"pipeline_smoke")
            self.assertFalse(report["approved_for_default"])
            self.assertLess(report["last_loss"],report["initial_loss"])
            with self.assertRaises(ValueError): load(path)
            self.assertIsInstance(load(path,True),Policy)
            with self.assertRaises(ValueError): fit(rows,path,False)

    def test_family_split_leakage_rejected(self):
        rows=dataset(2); rows[1]["split"]="test"
        with tempfile.TemporaryDirectory() as td:
            with self.assertRaisesRegex(ValueError,"leakage"): fit(rows,Path(td)/"model.json",True)

    def test_learned_weights_cannot_bypass_feasibility(self):
        c=next(r["context"] for r in families() if r["id"]=="feasibility")
        self.assertEqual(Policy(np.full(8,.08)).choose(c,False)["action_id"],"feasible")
        with self.assertRaises(ValueError): Policy(np.ones(8))


if __name__=="__main__": unittest.main()
