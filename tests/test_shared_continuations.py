import copy
import unittest

from reflex._continuation_portfolio import portfolio_planner
from reflex.core import digest
from reflex.decision_loop import DecisionLoop
from reflex.laboratory import PROFILES
from tools.continuing_games import ContinuingGame
from tools.run_owned_value import cumulative_basis


def forecast(game, scope):
    w=game.start();a=game.actor(game.context(w),learning='segmented')
    before=digest(a.record());r=game.request(game.context(w));cap={}
    p=portfolio_planner(a,r,w,observe=game.context,events=game.events,terminal=game.terminal,
        assess=lambda w:game.goal(w).record(),request_for=game.request,feedback=game.feedback,
        model_id=game.model_id,target=game.target,horizon=3,seeds=(701,),
        purpose_estimator='conditional',cumulative_preference=cumulative_basis(game,True),
        tail_recovery=2,recovery_cache=64,coherent_cumulative_mean=True,
        continuation_scope=scope,captured=cap)
    decision=DecisionLoop.decide_batch([(copy.deepcopy(a),r)],False,p)[0]
    assert digest(a.record())==before
    return decision,cap


class SharedContinuationTests(unittest.TestCase):
    def test_same_tail_bank_for_every_root_and_original_paths_retained(self):
        g=ContinuingGame('combat',PROFILES[2],turns=3)
        _,old=forecast(g,'root');_,new=forecast(g,'shared')
        f=new['forecast'];policies=f.audit['continuation_portfolio']['policies']
        old_keys={v['proposal']:k for k,v in old['forecast'].audit['plans'].items()}
        new_keys={v['proposal']:k for k,v in f.audit['plans'].items()}
        roots=sorted(g.keys(g.start()))
        for i,root in enumerate(roots):
            names=[name for name,r in f.roots.items() if r==(root,)]
            self.assertEqual({policies[f.audit['plans'][n]['proposal']]['preferred'] for n in names},{None,*roots})
            for mode in ('reflex','retry'):
                name=f'candidate-{i:03d}-{mode}'
                other=name+(f'-{i:03d}' if mode=='retry' else '')
                self.assertEqual(old['paths'][name],new['paths'][other])
                self.assertEqual(old['modeled'][name],new['modeled'][other])
                self.assertEqual(old['forecast'].purpose[old_keys[name]],f.purpose[new_keys[other]])
        for name,rows in new['modeled'].items():
            for row in rows:
                self.assertEqual(row['actions'][0],f.roots[new_keys[name]][0])
                self.assertTrue(row['terminal'])

    def test_forbidden_shared_mode_fails_without_mutating_owner(self):
        g=ContinuingGame('resources',PROFILES[0],turns=3)
        with self.assertRaises(ValueError):forecast(g,'unknown')


if __name__=='__main__':unittest.main()
