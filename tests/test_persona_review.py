import copy
import json
from pathlib import Path
import unittest
from tools.persona_review import immediate, continuation, forecast_frame, unchanged_streak, render
from reflex.examples import context, action, effect
from reflex.core import digest


class PersonaReviewTests(unittest.TestCase):
    def test_legality_and_known_failure_survive_explanation(self):
        c=context('review',[action('impossible',effect(1),legal=False),
            action('broken',effect(1),failure=True),action('recover',effect(.1))])
        saved=digest(c);r=immediate(c)
        self.assertEqual(r['choice'],'recover')
        self.assertEqual([x['eligible'] for x in r['choices']],[False,False,True])
        self.assertEqual(digest(c),saved)

    def test_informative_loss_is_not_automatically_training_truth(self):
        c=context('share',[action('take',effect(.8,values={'benevolence':-1})),
            action('share',effect(.2,values={'benevolence':1}))],
            values={'benevolence':1},mode='principle')
        r=immediate(c);self.assertEqual(r['choice'],'share')
        self.assertNotIn('correct_action',r);self.assertNotIn('training_label',r)
        self.assertEqual(r['choices'][1]['outcomes'][0]['objective'],.2)

    def test_replanning_censoring_and_action_match_are_distinct(self):
        b=dict(actions=['a','b','c'],assessment={'status':'success','value':1},terminal=True)
        self.assertEqual(continuation(b,['a','x','c'])['first_difference'],2)
        self.assertEqual(continuation(b,['a'])['status'],'censored')
        r=continuation(b,['a','b','c'])
        self.assertEqual(r['status'],'actions_matched_states_unchecked')
        self.assertFalse(r['causal_error_assigned'])

    def test_observed_progress_is_not_a_wait_veto(self):
        steady=dict(level_before=.2,level_after=.2,readiness_before=.4,readiness_after=.4,completed=False)
        progress=dict(steady,readiness_after=.6)
        done=dict(steady,completed=True)
        fs=[{'progress':p} for p in (steady,steady,steady,progress,steady,None,steady,done)]
        self.assertEqual(unchanged_streak(fs),[1,2,3,0,1,None,1,0])
        self.assertNotIn('blocked',fs[2])

    def test_script_content_is_data_not_markup(self):
        data={'note':'</script><script>throw Error(1)</script>'}
        s=render(data,'<script type="application/json">__REVIEW_DATA__</script>')
        self.assertEqual(s.count('</script>'),1)
        self.assertEqual(json.loads(s[s.index('>')+1:s.rindex('</script>')]),data)

    def test_frame_separates_real_flow_from_modeled_settlement(self):
        t=dict(root='a',before={'stock':1},after={'stock':2},flow={'objective':.1},
            goal={'status':'running','value':.2},state={'mode':'need'},purpose=None,
            feedback={'purpose':None},progress={},persona_hash='owner',
            endpoints=[dict(actions=['a','b'],assessment={'status':'success','value':1},terminal=True)],
            deliberation={'selected_purpose':.8,'best_purpose':1})
        saved=copy.deepcopy(t);r=forecast_frame([t],0)
        self.assertAlmostEqual(r['model_purpose_gap'],.2)
        self.assertEqual(r['actual_goal']['value'],.2)
        self.assertEqual(r['branches'][0]['status'],'censored')
        self.assertEqual(r['verdict'],'pending_human_review');self.assertEqual(t,saved)
        t['deliberation']['selected_purpose']=1.2
        with self.assertRaises(ValueError):forecast_frame([t],0)

    def test_bundled_review_has_all_profiles_and_no_accepted_labels(self):
        root=Path(__file__).resolve().parents[1]
        fixture=root/'evidence/persona_review/data.json'
        if not fixture.is_file():fixture=root/'reflex_artifacts/persona_review/data.json'
        data=json.loads(fixture.read_text(encoding='utf-8'))
        expected=data.pop('digest');self.assertEqual(digest(data),expected)
        self.assertFalse(data['consent']['training_approved'])
        for c in data['cases']:
            self.assertEqual({t['profile'] for t in c['tracks']},{'growth','steady','care','ego'})
            for t in c['tracks']:
                for f in t['frames']:self.assertEqual(f['verdict'],'pending_human_review')
        self.assertEqual(data['proof']['laboratory']['snapshot_replay'],288)


if __name__=='__main__':unittest.main()
