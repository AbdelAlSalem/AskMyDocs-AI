from unittest.mock import Mock

import httpx
import pytest

from app.core.config import DemoSettings
from app.services import demo_security
from app.services.demo_security import DemoAbuseGuard, DemoAccessError


def admit(guard, ip, token=None, **kwargs):
    with guard.authorize(ip, token, **kwargs):
        pass


def test_global_and_daily_quotas_use_rolling_windows():
    clock = Mock(return_value=0.0)
    settings = DemoSettings(demo_global_rate_limit=2, demo_daily_request_limit=3)
    guard = DemoAbuseGuard(settings, clock=clock)
    admit(guard, "192.0.2.1")
    admit(guard, "192.0.2.2")

    clock.return_value = 15.0
    with pytest.raises(DemoAccessError) as minute:
        admit(guard, "192.0.2.3")
    assert (minute.value.status_code, minute.value.code, minute.value.retry_after) == (429, "demo_rate_limit", 45)

    clock.return_value = 60.0
    admit(guard, "192.0.2.3")
    clock.return_value = 120.0
    with pytest.raises(DemoAccessError) as day:
        admit(guard, "192.0.2.4")
    assert (day.value.status_code, day.value.retry_after) == (429, 86280)

    clock.return_value = 86400.0
    admit(guard, "192.0.2.4")


def test_concurrency_reservation_is_released_when_generation_fails():
    guard = DemoAbuseGuard(DemoSettings(demo_max_concurrent_requests=1))

    with pytest.raises(RuntimeError, match="generation failed"):
        with guard.authorize("192.0.2.1"):
            with pytest.raises(DemoAccessError) as busy:
                admit(guard, "192.0.2.2")
            assert (busy.value.status_code, busy.value.retry_after) == (429, 3)
            raise RuntimeError("generation failed")

    admit(guard, "192.0.2.2")


def test_render_opt_in_accepts_one_valid_client_header():
    guard = DemoAbuseGuard(DemoSettings(demo_ip_rate_limit=1, demo_trust_render_client_ip=True))
    proxy = "192.0.2.1"
    admit(guard, proxy, render_client_hosts=["198.51.100.1"])
    admit(guard, proxy, render_client_hosts=["198.51.100.2"])
    with pytest.raises(DemoAccessError) as mapped:
        admit(guard, proxy, render_client_hosts=["::ffff:198.51.100.1"])
    assert mapped.value.status_code == 429

    admit(guard, proxy, render_client_hosts=["198.51.100.3", "198.51.100.4"])
    with pytest.raises(DemoAccessError) as malformed:
        admit(guard, proxy, render_client_hosts=["198.51.100.3, 198.51.100.4"])
    assert malformed.value.status_code == 429


@pytest.fixture
def verification(monkeypatch):
    request = httpx.Request("POST", demo_security.TURNSTILE_URL)
    post = Mock()
    monkeypatch.setattr(demo_security.httpx, "post", post)
    settings = DemoSettings(
        demo_turnstile_site_key="widget-public-key",
        demo_turnstile_secret_key="synthetic-widget-secret",
        demo_turnstile_hostname="demo.example.com",
        demo_daily_request_limit=2,
        demo_max_concurrent_requests=1,
    )

    def respond(body):
        post.return_value = httpx.Response(200, json=body, request=request)

    return DemoAbuseGuard(settings), post, respond


def test_turnstile_replay_cannot_consume_remaining_daily_quota(verification):
    guard, post, respond = verification
    accepted = {"success": True, "hostname": "demo.example.com", "action": "demo_chat"}
    respond(accepted)
    ip = "192.0.2.1"
    admit(guard, ip, "first-token")
    assert post.call_args.args == (demo_security.TURNSTILE_URL,)
    assert post.call_args.kwargs["data"] == {
        "secret": "synthetic-widget-secret", "response": "first-token", "remoteip": ip,
    }

    respond({"success": False, "error-codes": ["timeout-or-duplicate"]})
    with pytest.raises(DemoAccessError) as replay:
        admit(guard, ip, "first-token")
    assert (replay.value.status_code, replay.value.code) == (403, "demo_verification_required")

    respond(accepted)
    admit(guard, ip, "fresh-token")
    with pytest.raises(DemoAccessError) as day:
        admit(guard, ip, "third-token")
    assert day.value.status_code == 429
    assert post.call_count == 3


@pytest.mark.parametrize(("hostname", "action"), [
    ("unrelated.example.com", "demo_chat"),
    ("demo.example.com", "some_other_action"),
], ids=["incorrect-hostname", "incorrect-action"])
def test_turnstile_tokens_are_bound_to_frontend_and_action(verification, hostname, action):
    guard, _, respond = verification
    respond({"success": True, "hostname": hostname, "action": action})
    with pytest.raises(DemoAccessError) as denied:
        admit(guard, "192.0.2.1", "token")

    assert (denied.value.status_code, denied.value.code) == (403, "demo_verification_required")


def test_turnstile_network_failure_is_sanitized_and_fails_closed(verification):
    guard, post, _ = verification
    post.side_effect = httpx.ConnectTimeout("synthetic-widget-secret")

    with pytest.raises(DemoAccessError) as unavailable:
        admit(guard, "192.0.2.1", "token")

    assert (unavailable.value.status_code, unavailable.value.code) == (503, "demo_verification_unavailable")
    assert "synthetic-widget-secret" not in str(unavailable.value)
