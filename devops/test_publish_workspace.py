"""Hermetic tests for partial publication recovery; no registry writes."""
import unittest
from unittest.mock import patch

import publish_workspace as publisher

RATE_LIMIT = "status 429 Too Many Requests: Please try again after Thu, 17 Sep 2026 16:32:58 GMT"


class PublishTests(unittest.TestCase):
    def test_delay_honors_server_date(self):
        self.assertEqual(publisher.retry_delay(RATE_LIMIT, 1789662770), 13)
        self.assertIsNone(publisher.retry_delay("status 403 Forbidden", 0))
        self.assertIsNone(publisher.retry_delay("status 429 Too Many Requests", 0))

    @patch.object(publisher.time, "sleep")
    @patch.object(publisher, "remaining", return_value=["second"])
    @patch.object(publisher, "upload", side_effect=[(101, RATE_LIMIT), (0, "")])
    def test_partial_success_is_removed_before_retry(self, upload, remaining, sleep):
        self.assertEqual(publisher.publish(["first", "second"], {"first": "1", "second": "1"}), 0)
        self.assertEqual(upload.call_args_list[1].args, (["second"],))
        remaining.assert_called_once()
        sleep.assert_called_once()

    @patch.object(publisher.time, "sleep")
    @patch.object(publisher, "upload", return_value=(101, "status 403 Forbidden"))
    def test_other_errors_are_fatal(self, upload, sleep):
        self.assertEqual(publisher.publish(["first"], {"first": "1"}), 101)
        upload.assert_called_once()
        sleep.assert_not_called()

    @patch.object(publisher.time, "sleep")
    @patch.object(publisher, "remaining", return_value=["first"])
    @patch.object(publisher, "upload", return_value=(101, RATE_LIMIT))
    def test_retries_are_bounded(self, upload, remaining, sleep):
        self.assertEqual(publisher.publish(["first"], {"first": "1"}), 101)
        self.assertEqual(upload.call_count, 2)
        self.assertEqual(sleep.call_count, 1)


if __name__ == "__main__":
    unittest.main()
