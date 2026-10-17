import copy
import unittest
from tools.complete_owned_study import missing

class CompletionTests(unittest.TestCase):
    def setUp(self):
        self.registration=dict(jobs=[['auction',0,43,'outside',128,1,15,0]],arms=['receding_corrected','receding_recovery'])
        self.row=dict(game='auction',profile='growth',seed=43,regime='outside',turns=128,interval=1,tail_steps=15,offset=0,arm='receding_corrected')
    def test_only_unfinished_registered_execution_is_scheduled(self):
        self.assertEqual(missing(self.registration,[self.row]),[(self.registration['jobs'][0],'receding_recovery')])
        other=dict(self.row,arm='receding_recovery')
        self.assertEqual(missing(self.registration,[self.row,other]),[])
    def test_duplicate_or_wrong_configuration_cannot_complete_a_registration(self):
        with self.assertRaises(AssertionError):missing(self.registration,[self.row,self.row])
        for field,value in [('turns',64),('interval',7),('tail_steps',7),('offset',1),('profile','steady')]:
            with self.subTest(field=field),self.assertRaises(AssertionError):missing(self.registration,[dict(self.row,**{field:value})])
    def test_duplicate_registration_is_not_silently_deduplicated(self):
        reg=copy.deepcopy(self.registration);reg['jobs']*=2
        with self.assertRaises(AssertionError):missing(reg,[])

if __name__=='__main__':unittest.main()
