import json,unittest
from functools import lru_cache
from reflex.laboratory import PROFILES
from tools.continuing_games import ContinuingGame
from tools.resource_reference import reference,account
from tools.run_owned_value import law


class ResourceReferenceTests(unittest.TestCase):
    def test_matches_actual_rules_exhaustively_with_liquidity_maturity_and_deadlines(self):
        for turns,start in ((3,0),(8,0),(10,7)):
            g=ContinuingGame('resources',PROFILES[0],turns)
            probability=lambda t:((.15,.25,.6),(.7,.2,.1),(.2,.5,.3))[t%3]
            value,choices=reference(turns,probability)
            @lru_cache(None)
            def brute(t,c,p,d):
                w=g.start();w.update(tick=t,cash=c,plant=p,investment=d)
                if g.terminal(w):return g.goal(w).value
                return max(q(w).values())
            def q(w):
                result={}
                for key in g.keys(w):
                    total=0.
                    for symbol,prob in enumerate(probability(w['tick'])):
                        after,_=g.transition(w,key,str(symbol))
                        total+=prob*brute(after['tick'],after['cash'],after['plant'],after['investment'])
                    result[key]=total
                return result
            states=[(8,0,0)] if start==0 else [(c,p,d) for c in (0,3,4,8) for p,d in ((0,0),(0,6),(0,1),(3,1),(4,0))]
            for cash,plant,construction in states:
                w=g.start();w.update(tick=start,cash=cash,plant=plant,investment=construction)
                expected=q(w);actual=choices(w)
                self.assertEqual(set(actual),set(g.keys(w)))
                self.assertAlmostEqual(value(w),max(expected.values()),places=12)
                for key in actual:self.assertAlmostEqual(actual[key],expected[key],places=12)

    def test_long_account_telescopes_and_residuals_have_zero_conditional_mean(self):
        turns=128;g=ContinuingGame('resources',PROFILES[1],turns);w=g.start();trace=[]
        value,choices=reference(turns,lambda t:tuple(law(t,turns,'irregular').values()))
        for tick in range(turns):
            keys=g.keys(w);key='invest' if 'invest' in keys and tick%2==0 else ('trade','crop:0','crop:1','crop:2')[tick%4]
            truth=law(tick,turns,'irregular');q=choices(w)
            for action in keys:
                expected=sum(p*value(g.transition(w,action,z)[0]) for z,p in truth.items())
                self.assertAlmostEqual(expected,q[action],places=12)
            after,_=g.transition(w,key,str((2*tick+1)%3))
            trace.append(dict(tick=tick,before=w,after=after,action=key,truth=truth));w=after
        row=dict(game='resources',profile='steady',turns=turns,regime='irregular',trace=trace,score=g.goal(w).value)
        saved=json.dumps(row,sort_keys=True);result=account(row)
        self.assertEqual(json.dumps(row,sort_keys=True),saved)
        self.assertAlmostEqual(result['initial_perfect_information']-result['purpose_opportunity_loss']+result['residual'],row['score'],places=12)
        self.assertGreater(result['purpose_opportunity_loss'],0.)
        self.assertAlmostEqual(value(w),g.goal(w).value,places=12)
        self.assertEqual(choices(w),{})

    def test_rejects_states_and_laws_outside_the_declared_benchmark(self):
        with self.assertRaises(ValueError):reference(0,lambda t:(1,0,0))
        for p in ((1,1,0),(float('nan'),0,0),(.5,.5)):
            with self.assertRaises(ValueError):reference(2,lambda t:p)
        value,_=reference(2,lambda t:(1,0,0));w=ContinuingGame('resources',PROFILES[0],2).start()
        for override in (dict(cash=8.5),dict(cash=7),dict(plant=4,investment=1),dict(hp=0),dict(investment=7)):
            with self.assertRaises(ValueError):value({**w,**override})


if __name__=='__main__':unittest.main()
