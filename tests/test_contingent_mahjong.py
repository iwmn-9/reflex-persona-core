import copy,importlib.util,unittest
from reflex.contingent_mahjong import HandModel,HandState
from reflex.mahjong_adapter import engine,candidates,PublicView
from reflex.laboratory import profiles
from reflex.core import digest


@unittest.skipUnless(importlib.util.find_spec('riichienv'),'optional closed-hand engine required')
class ContingentMahjongTests(unittest.TestCase):
    def initial(self):
        env=engine().RiichiEnv(game_mode='4p-red-single',seed=1200)
        obs=next(iter(env.reset().values()));view,rows=candidates(obs)
        return HandModel(profiles()[0],'test'),HandState(view,tuple(rows),0)

    def test_hypothetical_draw_cannot_mutate_actual_public_hand(self):
        m,s=self.initial();before=digest(m.context(s));key=next(r['id'] for r in s.rows if r['type']=='Discard')
        nxt,row=m.advance(s,key,401)
        self.assertEqual(digest(m.context(s)),before)
        self.assertEqual(len(nxt.view.hand),14);self.assertEqual(row['p'],1.)
        self.assertEqual(m.goal(nxt).status,'running')

    def test_two_own_decision_cutoff_is_explicit_proxy_not_kyoku_victory(self):
        m,s=self.initial()
        for seed in (401,402):
            key=next(r['id'] for r in s.rows if r['type']=='Discard');s,row=m.advance(s,key,seed)
        self.assertTrue(m.terminal(s));self.assertEqual(m.goal(s).status,'scored')
        self.assertEqual(m.goal(s).target,'closed-hand-shape-proxy')

    def test_complete_hypothetical_shape_is_bounded_without_legal_win_claim(self):
        from dataclasses import replace
        m,s=self.initial()
        # 123m 123p 123s 456s 77z, fourteen distinct physical tiles.
        hand=tuple(t*4+n for t,n in [(0,0),(1,0),(2,0),(9,0),(10,0),(11,0),
            (18,0),(19,0),(20,0),(21,0),(22,0),(23,0),(33,0),(33,1)])
        v=replace(s.view,hand=hand,discards=((),)*4,dora=())
        shape=replace(s,view=v,depth=2)
        self.assertEqual(m.goal(shape).value,1.)
        self.assertEqual(m.goal(shape).status,'scored');self.assertFalse(shape.finished)


if __name__=='__main__':unittest.main()
