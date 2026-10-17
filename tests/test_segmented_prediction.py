import copy,itertools,json,unittest
import numpy as np
from reflex.segmented_prediction import segment_prediction,persistent_prediction,SegmentedEventLearner,HAZARDS,SegmentMemory,SegmentEvidence


def enumerated(sequence,library,k):
    prior=np.array([.5/len(library)]*len(library)+[.5]) if library else np.array([1.])
    def segment(seq):
        counts=np.full(k,.5);unknown=1.
        for z in seq:unknown*=counts[z]/sum(counts);counts[z]+=1
        likelihood=np.array([np.prod([p[z] for z in seq]) for p in library]+[unknown])
        weight=prior*likelihood
        return sum(weight),(weight@np.array([*library,tuple(counts/sum(counts))]))/sum(weight)
    base=segment(())[1];total=0.;weighted=np.zeros(k)
    for h in HAZARDS:
        for cuts in itertools.product((0,1),repeat=max(0,len(sequence)-1)):
            mass=1.;parts=[];start=0
            for i,cut in enumerate(cuts,1):
                mass*=h if cut else 1-h
                if cut:parts.append(sequence[start:i]);start=i
            parts.append(sequence[start:])
            for part in parts:mass*=segment(part)[0]
            prediction=h*base+(1-h)*segment(parts[-1])[1]
            weighted+=mass*prediction;total+=mass
    return weighted/total


class SegmentedPredictionTests(unittest.TestCase):
    def test_conditional_hypotheses_use_each_observed_context(self):
        # A library hypothesis may change its response with public context.
        # Enumerate every change partition independently, keeping those past
        # conditional probabilities distinct from the current query's values.
        sequence=(0,2,1,2)
        libraries=(((.8,.1,.1),(.2,.6,.2)),((.1,.2,.7),(.5,.2,.3)),
                   ((.2,.7,.1),(.1,.3,.6)),((.4,.3,.3),(.1,.1,.8)))
        current=((.1,.8,.1),(.6,.1,.3));prior=np.array([.25,.25,.5])
        def segment(start,stop):
            counts=np.full(3,.5);likelihood=np.ones(3)
            for i in range(start,stop):
                z=sequence[i]
                likelihood[:2]*=[row[z] for row in libraries[i]]
                likelihood[2]*=counts[z]/sum(counts);counts[z]+=1
            weight=prior*likelihood
            prediction=weight@np.array([*current,tuple(counts/sum(counts))])/weight.sum()
            return weight.sum(),prediction
        weighted=np.zeros(3);total=0.;base=prior@np.array([*current,(1/3,)*3])
        for h in HAZARDS:
            for cuts in itertools.product((0,1),repeat=3):
                mass=1.;start=0
                for i,cut in enumerate(cuts,1):
                    mass*=h if cut else 1-h
                    if cut:mass*=segment(start,i)[0];start=i
                mass*=segment(start,4)[0]
                weighted+=mass*(h*base+(1-h)*segment(start,4)[1]);total+=mass
        p,_,_=segment_prediction(sequence,libraries,current,3)
        q,_=persistent_prediction(sequence,libraries,(),(),current,3)
        np.testing.assert_allclose(p,weighted/total,atol=1e-14)
        np.testing.assert_allclose(q,p,atol=1e-14)

    def test_persistent_uncertainty_is_not_independent_mean_sampling(self):
        # With no factual evidence and no library, integrating a Beta(1/2,1/2)
        # law gives P(a,a)=3/8, not the frozen marginal's 1/4.
        p,_=persistent_prediction((),(),(0,),((),),(),2)
        np.testing.assert_allclose(p,(.75,.25),atol=1e-14)
        self.assertAlmostEqual(.5*p[0],.375)
        # Independent enumeration of root segment partitions and the one
        # possible change before the first future event. No later reset.
        root=(0,0,1);future=(1,1);library=((.8,.2),(.1,.9));prior=np.array([.25,.25,.5])
        def segment(seq):
            counts=np.full(2,.5);unknown=1.
            for z in seq:unknown*=counts[z]/sum(counts);counts[z]+=1
            w=prior*np.array([np.prod([v[z] for z in seq]) for v in library]+[unknown])
            return w.sum(),(w@np.array([*library,tuple(counts/sum(counts))]))/w.sum()
        mass=0.;total=np.zeros(2)
        for h in HAZARDS:
            for cuts in itertools.product((0,1),repeat=len(root)-1):
                weight=1.;start=0
                for i,cut in enumerate(cuts,1):
                    weight*=h if cut else 1-h
                    if cut:weight*=segment(root[start:i])[0];start=i
                for probability,seq,extra in ((1-h,root[start:]+future,1.),(h,future,segment(root[start:])[0])):
                    evidence,q=segment(seq);v=weight*probability*extra*evidence;total+=v*q;mass+=v
        q,_=persistent_prediction(root,tuple(library for _ in root),future,tuple(library for _ in future),library,2)
        np.testing.assert_allclose(q,total/mass,atol=1e-14)

    def test_predictive_world_keeps_root_evidence_when_person_forgets(self):
        from reflex.event_reader import EventReader
        from reflex.examples import context,action,effect
        c=context('posterior',[action('a',effect())]);c['tick']=0
        reader=EventReader(c['scope'],(),lambda c:{},lambda *args:effect(),legal_for=lambda c:('a','b'),
            model_id='public',known_conditionals=True,learning='segmented')
        reader.learner=SegmentedEventLearner(owner=reader.owner,model_id='public',window=4)
        reader.memory=reader.learner.start()
        for z in ('a','b','a','a'):
            reader,_=reader.updated_event(c,dict(revealed_action=z),'observed');c['tick']+=1
        root=reader.record();world=reader.predictive_continuation();branch=copy.deepcopy(reader)
        for z in ('b','b','b'):
            branch,_=branch.updated_event(c,dict(revealed_action=z),'imagined');c['tick']+=1
        actual=world(branch,c)
        expected=enumerated((0,1,0,0,1,1,1),(),2)
        np.testing.assert_allclose([actual['a'],actual['b']],expected,atol=1e-14)
        self.assertNotAlmostEqual(actual['b'],branch.predict(c)['b'])
        self.assertEqual(reader.record(),root)
        self.assertEqual(len(branch.memory.history),4)

    def test_posterior_world_is_integrated_into_learning_rollouts(self):
        from tools.continuing_games import ContinuingGame
        from reflex.laboratory import PROFILES
        from reflex.closed_loop import owned_rollout
        from reflex.decision_loop import DecisionLoop
        from reflex.model_observer import resolve_feedback
        from reflex.core import digest
        for mode in ('posterior','persistent'):
            g=ContinuingGame('gambling',PROFILES[1],turns=3);w=g.start();own=g.actor(g.context(w),learning='segmented')
            before=own.record();rq=g.request(g.context(w));cap={}
            ctrl=owned_rollout(own,observe=g.context,events=g.events,terminal=g.terminal,
                assess=lambda w:g.goal(w).record(),record=lambda w:w,request_for=g.request,feedback=g.feedback,
                model_id=g.model_id,target=g.target,tail_steps=2,seeds=(701,),world_model=mode)
            ctrl.planner(own,rq,w,captured=cap)([own.preview(rq)[0]],[])
            self.assertEqual(own.record(),before)
            result=DecisionLoop.decide_batch([(own,rq)],False,ctrl.planner(own,rq,w))[0]
            key=result['decision']['action_id'];after,flow=g.transition(w,key,'2')
            resolve_feedback(own,result['ticket'],g.feedback(w,after,key,g.context(w),flow))
            expected=next(row['owner'] for row in cap['node']['children'][key] if row['public']==after)
            self.assertEqual(digest(own.record()),expected)
            with self.assertRaisesRegex(ValueError,'rebuild'):ctrl.planner(own,g.request(g.context(after)),after)

    def test_reader_future_person_and_restore_share_actual_update_semantics(self):
        from tools.continuing_games import ContinuingGame
        from reflex.laboratory import PROFILES
        from reflex.closed_loop import owned_rollout
        from reflex.decision_loop import DecisionLoop
        from reflex.core import digest
        from reflex.model_observer import resolve_feedback
        g=ContinuingGame('auction',PROFILES[1],turns=4);w=g.start();c=g.context(w)
        own=g.actor(c,learning='segmented');record=own.record();rq=g.request(c);cap={}
        ctrl=owned_rollout(own,observe=g.context,events=g.events,terminal=g.terminal,
            assess=lambda w:g.goal(w).record(),record=lambda w:w,request_for=g.request,
            feedback=g.feedback,model_id=g.model_id,target=g.target,tail_steps=3)
        effective,_=own.preview(rq);ctrl.planner(own,rq,w,captured=cap)([effective],[])
        self.assertEqual(record,own.record())
        restored=DecisionLoop.from_record(c,json.loads(json.dumps(record)),predictor=g.actor(c,learning='segmented').predictor)
        self.assertEqual(restored.record(),record)
        result=DecisionLoop.decide_batch([(own,rq)],False,ctrl.planner(own,rq,w))[0]
        key=result['decision']['action_id'];after,flow=g.transition(w,key,'2')
        resolve_feedback(own,result['ticket'],g.feedback(w,after,key,c,flow))
        expected=next(r['owner'] for r in cap['node']['children'][key] if r['public']==after)
        self.assertEqual(digest(own.record()),expected)
        self.assertEqual(own.predictor.observations,1)
        with self.assertRaises(ValueError):DecisionLoop.from_record(c,own.record(),predictor=g.actor(c).predictor)

    def test_matches_full_segment_and_model_enumeration(self):
        for library in ((),((.8,.1,.1),(.1,.8,.1))):
            for sequence in ((),(2,),(0,0,2),(0,2,1,2,2)):
                p,_,h=segment_prediction(sequence,tuple(library for _ in sequence),library,3)
                np.testing.assert_allclose(p,enumerated(sequence,library,3),atol=1e-14)
                self.assertAlmostEqual(sum(h),1.)

    def test_prefix_cache_preserves_results_across_threads(self):
        from concurrent.futures import ThreadPoolExecutor
        sequences=[tuple((i*j+j//3)%3 for j in range(1,65+i)) for i in range(8)]
        library=((.8,.1,.1),(.1,.8,.1))
        def call(seq):return segment_prediction(seq,tuple(library for _ in seq),library,3)
        expected=[call(s) for s in sequences]
        segment_prediction.cache_clear()
        from reflex.segmented_prediction import _PREFIXES,_PREFIX_LOCK
        with _PREFIX_LOCK:_PREFIXES.clear()
        with ThreadPoolExecutor(max_workers=4) as pool:actual=list(pool.map(call,sequences*3))
        for i,row in enumerate(actual):self.assertEqual(row,expected[i%len(sequences)])

    def test_checkpoint_owner_history_and_unknown_response(self):
        m=SegmentedEventLearner(owner='npc',model_id='rules',names=('a','b'))
        models={'a':dict(x=.8,y=.1,z=.1),'b':dict(x=.1,y=.8,z=.1)};s=m.start()
        for _ in range(40):s=m.observe(s,(),('x','y','z'),models,'z')
        self.assertGreater(m.predict(s,(),('x','y','z'),models)['z'],.9)
        record=json.loads(json.dumps(m.record(s)));self.assertEqual(m.restore(record),s)
        self.assertIs(copy.deepcopy(s),s)
        self.assertEqual(m.start().observations,0)
        with self.assertRaises(ValueError):SegmentedEventLearner(owner='other',model_id='rules',names=('a','b')).restore(record)
        bad=SegmentMemory(m.identity,(SegmentEvidence((),('x','y'),((.3,),(.5,.5)),'x'),),1)
        with self.assertRaises(ValueError):m.record(bad)

    def test_factor_order_and_symbols_do_not_encode_a_game(self):
        a=SegmentedEventLearner(owner='npc',model_id='r',factors=('x','y'))
        b=SegmentedEventLearner(owner='npc',model_id='s',factors=('b','a'))
        x=a.start();y=b.start();mapping={'a':'q','b':'z','c':'m'}
        for i in range(64):
            context=(str(i%2),str((i//2)%2));z='c' if i%2 else 'a'
            x=a.observe(x,context,('a','b','c'),{},z)
            y=b.observe(y,tuple(reversed(context)),('m','q','z'),{},mapping[z])
        p=a.predict(x,('1','0'),('a','b','c'),{});q=b.predict(y,('0','1'),('m','q','z'),{})
        self.assertGreater(p['c'],.8)
        for k,v in p.items():self.assertAlmostEqual(v,q[mapping[k]])

    def test_legal_strata_and_bounded_forgetting(self):
        m=SegmentedEventLearner(owner='a',model_id='b',window=8);s=m.start()
        for i in range(16):s=m.observe(s,(),('a',),{},'a')
        self.assertEqual(m.predict(s,(),('a','b'),{}),dict(a=.5,b=.5))
        self.assertEqual(len(s.history),8);self.assertEqual(s.observations,16)


if __name__=='__main__':unittest.main()
