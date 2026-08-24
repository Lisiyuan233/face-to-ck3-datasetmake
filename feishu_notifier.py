from __future__ import annotations

import argparse
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Any, Callable


FEISHU_API_ROOT = "https://open.feishu.cn/open-apis"
VALID_RECEIVE_ID_TYPES = {"chat_id", "open_id", "user_id", "union_id", "email"}
FEISHU_ENV_NAMES = (
    "FEISHU_APP_ID",
    "FEISHU_APP_SECRET",
    "FEISHU_RECEIVE_ID",
    "FEISHU_RECEIVE_ID_TYPE",
    "FEISHU_PROGRESS_EVERY",
    "FEISHU_PROGRESS_INTERVAL_SECONDS",
    "FEISHU_TIMEOUT_SECONDS",
)


class FeishuNotificationError(RuntimeError):
    pass


@dataclass(frozen=True)
class FeishuNotificationConfig:
    app_id: str
    app_secret: str
    receive_id: str
    receive_id_type: str = "chat_id"
    progress_every: int = 500
    progress_interval_seconds: float = 1800.0
    timeout_seconds: float = 10.0

    def validate(self, *, require_receiver: bool = True) -> None:
        if not self.app_id or not self.app_secret:
            raise ValueError("飞书 App ID 和 App Secret 不能为空")
        if require_receiver and not self.receive_id:
            raise ValueError("飞书消息接收 ID 不能为空")
        if self.receive_id_type not in VALID_RECEIVE_ID_TYPES:
            raise ValueError(
                "FEISHU_RECEIVE_ID_TYPE 必须是 "
                + "、".join(sorted(VALID_RECEIVE_ID_TYPES))
            )
        if self.progress_every < 1:
            raise ValueError("FEISHU_PROGRESS_EVERY 必须至少为 1")
        if self.progress_interval_seconds <= 0:
            raise ValueError("FEISHU_PROGRESS_INTERVAL_SECONDS 必须大于 0")
        if self.timeout_seconds <= 0:
            raise ValueError("FEISHU_TIMEOUT_SECONDS 必须大于 0")

    @classmethod
    def from_env(
        cls,
        *,
        require_receiver: bool = True,
        fallback_to_windows_user_environment: bool = True,
    ) -> FeishuNotificationConfig | None:
        values = {name: os.environ.get(name, "") for name in FEISHU_ENV_NAMES}
        if fallback_to_windows_user_environment:
            for name, value in _windows_user_environment().items():
                if not values.get(name):
                    values[name] = value
        app_id = values["FEISHU_APP_ID"].strip()
        app_secret = values["FEISHU_APP_SECRET"].strip()
        receive_id = values["FEISHU_RECEIVE_ID"].strip()
        if not app_id and not app_secret and not receive_id:
            return None
        config = cls(
            app_id=app_id,
            app_secret=app_secret,
            receive_id=receive_id,
            receive_id_type=(
                values.get("FEISHU_RECEIVE_ID_TYPE") or "chat_id"
            ).strip(),
            progress_every=_positive_int_env(
                "FEISHU_PROGRESS_EVERY", 500, values
            ),
            progress_interval_seconds=_positive_float_env(
                "FEISHU_PROGRESS_INTERVAL_SECONDS", 1800.0, values
            ),
            timeout_seconds=_positive_float_env(
                "FEISHU_TIMEOUT_SECONDS", 10.0, values
            ),
        )
        config.validate(require_receiver=require_receiver)
        return config


def _windows_user_environment() -> dict[str, str]:
    """Read current-user values even when a launcher has stale environment."""
    if os.name != "nt":
        return {}
    try:
        import winreg

        result = {}
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment") as key:
            for name in FEISHU_ENV_NAMES:
                try:
                    value, _value_type = winreg.QueryValueEx(key, name)
                except FileNotFoundError:
                    continue
                if isinstance(value, str):
                    result[name] = value
        return result
    except (ImportError, OSError):
        return {}


def _positive_int_env(
    name: str, default: int, environment: dict[str, str] | None = None
) -> int:
    source = os.environ if environment is None else environment
    raw = source.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        value = int(raw)
    except ValueError as error:
        raise ValueError(f"{name} 必须是正整数") from error
    if value < 1:
        raise ValueError(f"{name} 必须是正整数")
    return value


def _positive_float_env(
    name: str, default: float, environment: dict[str, str] | None = None
) -> float:
    source = os.environ if environment is None else environment
    raw = source.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        value = float(raw)
    except ValueError as error:
        raise ValueError(f"{name} 必须是正数") from error
    if value <= 0:
        raise ValueError(f"{name} 必须是正数")
    return value


class FeishuNotifier:
    def __init__(
        self,
        config: FeishuNotificationConfig,
        *,
        urlopen: Callable[..., Any] = urllib.request.urlopen,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        config.validate(require_receiver=False)
        self.config = config
        self.urlopen = urlopen
        self.monotonic = monotonic
        self._tenant_access_token: str | None = None
        self._token_expires_at = 0.0

    def _request_json(
        self,
        method: str,
        url: str,
        *,
        payload: dict[str, Any] | None = None,
        token: str | None = None,
    ) -> dict[str, Any]:
        data = None
        headers = {"Content-Type": "application/json; charset=utf-8"}
        if payload is not None:
            data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        if token:
            headers["Authorization"] = f"Bearer {token}"
        request = urllib.request.Request(
            url,
            data=data,
            headers=headers,
            method=method,
        )
        try:
            with self.urlopen(request, timeout=self.config.timeout_seconds) as response:
                raw = response.read()
        except urllib.error.HTTPError as error:
            try:
                detail = error.read().decode("utf-8", errors="replace")
            except Exception:
                detail = str(error)
            raise FeishuNotificationError(
                f"飞书 HTTP {error.code}: {detail[:500]}"
            ) from error
        except (OSError, urllib.error.URLError) as error:
            raise FeishuNotificationError(f"连接飞书失败: {error}") from error
        try:
            result = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise FeishuNotificationError("飞书返回了无效 JSON") from error
        if not isinstance(result, dict):
            raise FeishuNotificationError("飞书返回体不是 JSON 对象")
        return result

    @staticmethod
    def _ensure_success(result: dict[str, Any], operation: str) -> None:
        if result.get("code") != 0:
            raise FeishuNotificationError(
                f"{operation}失败: code={result.get('code')!r}, "
                f"msg={result.get('msg')!r}"
            )

    def _get_tenant_access_token(self, *, force_refresh: bool = False) -> str:
        now = self.monotonic()
        if (
            not force_refresh
            and self._tenant_access_token
            and now < self._token_expires_at
        ):
            return self._tenant_access_token
        result = self._request_json(
            "POST",
            f"{FEISHU_API_ROOT}/auth/v3/tenant_access_token/internal",
            payload={
                "app_id": self.config.app_id,
                "app_secret": self.config.app_secret,
            },
        )
        self._ensure_success(result, "获取 tenant_access_token")
        token = result.get("tenant_access_token")
        if not isinstance(token, str) or not token:
            raise FeishuNotificationError("飞书响应缺少 tenant_access_token")
        try:
            expires_in = int(result.get("expire", 7200))
        except (TypeError, ValueError):
            expires_in = 7200
        self._tenant_access_token = token
        self._token_expires_at = now + max(60, expires_in - 300)
        return token

    def send_text(self, text: str) -> str:
        self.config.validate(require_receiver=True)
        if not text.strip():
            raise ValueError("飞书通知内容不能为空")
        params = urllib.parse.urlencode(
            {"receive_id_type": self.config.receive_id_type}
        )
        url = f"{FEISHU_API_ROOT}/im/v1/messages?{params}"
        payload = {
            "receive_id": self.config.receive_id,
            "msg_type": "text",
            "content": json.dumps({"text": text}, ensure_ascii=False),
        }
        result = self._request_json(
            "POST",
            url,
            payload=payload,
            token=self._get_tenant_access_token(),
        )
        self._ensure_success(result, "发送飞书消息")
        message_id = result.get("data", {}).get("message_id", "")
        return str(message_id)

    def list_joined_chats(self) -> list[dict[str, Any]]:
        """List chats visible to the bot; useful for finding FEISHU_RECEIVE_ID."""
        token = self._get_tenant_access_token()
        page_token = ""
        chats: list[dict[str, Any]] = []
        while True:
            query = {"page_size": "100"}
            if page_token:
                query["page_token"] = page_token
            url = f"{FEISHU_API_ROOT}/im/v1/chats?{urllib.parse.urlencode(query)}"
            result = self._request_json("GET", url, token=token)
            self._ensure_success(result, "获取机器人所在群列表")
            data = result.get("data", {})
            items = data.get("items", [])
            if isinstance(items, list):
                chats.extend(item for item in items if isinstance(item, dict))
            if not data.get("has_more"):
                break
            page_token = str(data.get("page_token", ""))
            if not page_token:
                break
        return chats


def main() -> int:
    parser = argparse.ArgumentParser(description="Test or inspect Feishu bot notifications")
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--list-chats", action="store_true")
    action.add_argument("--test", action="store_true")
    args = parser.parse_args()
    config = FeishuNotificationConfig.from_env(
        require_receiver=bool(args.test)
    )
    if config is None:
        parser.error("请先设置 FEISHU_APP_ID 和 FEISHU_APP_SECRET")
    notifier = FeishuNotifier(config)
    if args.list_chats:
        chats = notifier.list_joined_chats()
        print(
            json.dumps(
                [
                    {
                        "chat_id": chat.get("chat_id"),
                        "name": chat.get("name"),
                        "chat_mode": chat.get("chat_mode"),
                    }
                    for chat in chats
                ],
                ensure_ascii=False,
                indent=2,
            )
        )
    else:
        message_id = notifier.send_text("Face to CK3 采集通知测试成功")
        print(f"消息发送成功: {message_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
