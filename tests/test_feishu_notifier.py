from __future__ import annotations

import json
import os
import unittest
from unittest import mock

from feishu_notifier import (
    FeishuNotificationConfig,
    FeishuNotifier,
)


class FakeResponse:
    def __init__(self, payload: dict) -> None:
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return False

    def read(self) -> bytes:
        return json.dumps(self.payload).encode("utf-8")


class FakeUrlOpen:
    def __init__(self, payloads: list[dict]) -> None:
        self.payloads = list(payloads)
        self.requests = []

    def __call__(self, request, *, timeout):
        self.requests.append((request, timeout))
        if not self.payloads:
            raise AssertionError("unexpected HTTP request")
        return FakeResponse(self.payloads.pop(0))


class FeishuNotifierTests(unittest.TestCase):
    def test_sends_text_and_reuses_tenant_token_without_leaking_secret(self) -> None:
        transport = FakeUrlOpen(
            [
                {
                    "code": 0,
                    "msg": "ok",
                    "tenant_access_token": "tenant-token",
                    "expire": 7200,
                },
                {"code": 0, "msg": "ok", "data": {"message_id": "m1"}},
                {"code": 0, "msg": "ok", "data": {"message_id": "m2"}},
            ]
        )
        config = FeishuNotificationConfig(
            app_id="cli_test",
            app_secret="secret-value",
            receive_id="oc_chat",
        )
        notifier = FeishuNotifier(config, urlopen=transport, monotonic=lambda: 10.0)

        self.assertEqual(notifier.send_text("进度 10%"), "m1")
        self.assertEqual(notifier.send_text("异常停止"), "m2")

        self.assertEqual(len(transport.requests), 3)
        token_request = transport.requests[0][0]
        token_body = json.loads(token_request.data.decode("utf-8"))
        self.assertEqual(token_body["app_secret"], "secret-value")
        message_request = transport.requests[1][0]
        self.assertIn("receive_id_type=chat_id", message_request.full_url)
        message_body = json.loads(message_request.data.decode("utf-8"))
        self.assertEqual(message_body["receive_id"], "oc_chat")
        self.assertEqual(
            json.loads(message_body["content"])["text"],
            "进度 10%",
        )
        self.assertEqual(
            message_request.headers["Authorization"], "Bearer tenant-token"
        )

    def test_lists_joined_chats_across_pages(self) -> None:
        transport = FakeUrlOpen(
            [
                {
                    "code": 0,
                    "tenant_access_token": "tenant-token",
                    "expire": 7200,
                },
                {
                    "code": 0,
                    "data": {
                        "items": [{"chat_id": "oc_1", "name": "采集通知"}],
                        "has_more": True,
                        "page_token": "next",
                    },
                },
                {
                    "code": 0,
                    "data": {
                        "items": [{"chat_id": "oc_2", "name": "备用群"}],
                        "has_more": False,
                    },
                },
            ]
        )
        config = FeishuNotificationConfig(
            app_id="cli_test",
            app_secret="secret-value",
            receive_id="",
        )
        chats = FeishuNotifier(config, urlopen=transport).list_joined_chats()

        self.assertEqual([chat["chat_id"] for chat in chats], ["oc_1", "oc_2"])
        self.assertIn("page_token=next", transport.requests[2][0].full_url)

    def test_partial_environment_configuration_is_rejected(self) -> None:
        with mock.patch.dict(
            os.environ,
            {"FEISHU_APP_ID": "cli_test"},
            clear=True,
        ):
            with self.assertRaisesRegex(ValueError, "App Secret"):
                FeishuNotificationConfig.from_env()


if __name__ == "__main__":
    unittest.main()
