import unittest
from unittest.mock import patch
from tools.dispatch_registered_continuation import published_revision


class PublishedRevisionTests(unittest.TestCase):
    def test_waits_for_a_concurrent_push_before_any_dispatch(self):
        with patch('tools.dispatch_registered_continuation.command',side_effect=['b','a ref','b','b','b ref','b']) as read,patch('tools.dispatch_registered_continuation.time.sleep') as wait:
            self.assertEqual(published_revision('research/long-horizon'),'b')
            wait.assert_called_once_with(3)
            self.assertTrue(all(call.args[0]=='git' for call in read.call_args_list))

    def test_local_change_during_read_is_rechecked(self):
        with patch('tools.dispatch_registered_continuation.command',side_effect=['a','a ref','b','b','b ref','b']),patch('tools.dispatch_registered_continuation.time.sleep'):
            self.assertEqual(published_revision('research/long-horizon'),'b')

    def test_persistent_mismatch_stops_without_submission(self):
        with patch('tools.dispatch_registered_continuation.command',side_effect=['b','a ref','b']*5) as read,patch('tools.dispatch_registered_continuation.time.sleep') as wait:
            with self.assertRaisesRegex(RuntimeError,'nothing dispatched'):published_revision('research/long-horizon')
            self.assertEqual(wait.call_count,4)
            self.assertTrue(all(call.args[0]=='git' for call in read.call_args_list))


if __name__=='__main__':unittest.main()
