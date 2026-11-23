from collections import deque
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from ipaddress import IPv6Address, ip_address
from math import ceil
from threading import Lock
from time import monotonic
from typing import Never

import httpx

from app.core.config import DemoSettings
from app.services.errors import ServiceError


TURNSTILE_URL = "https://challenges.cloudflare.com/turnstile/v0/siteverify"
TURNSTILE_MAX_TOKEN_CHARS = 2048
DAILY_WINDOW_SECONDS = 86400


class DemoAccessError(ServiceError):
    """Safe public demo access error, with an optional Retry-After delay."""

    def __init__(self, status_code: int, code: str, message: str, *, retry_after: int | None = None) -> None:
        super().__init__(status_code, code, message)
        self.retry_after = retry_after


def _normalize_ip(host: str | None) -> str | None:
    if not host or "%" in host:
        return None
    try:
        address = ip_address(host.strip())
    except ValueError:
        return None
    if isinstance(address, IPv6Address) and address.ipv4_mapped is not None:
        return str(address.ipv4_mapped)
    return str(address)


class DemoAbuseGuard:
    def __init__(self, settings: DemoSettings, *, clock: Callable[[], float] = monotonic) -> None:
        self._settings = settings
        self._clock = clock
        self._lock = Lock()
        self._recent: deque[tuple[float, str | None]] = deque()
        self._daily: deque[float] = deque()
        self._active = 0

    @contextmanager
    def authorize(
        self,
        client_host: str | None,
        turnstile_token: str | None = None,
        *,
        render_client_hosts: Sequence[str] = (),
    ) -> Iterator[None]:
        ip = _normalize_ip(client_host)
        if self._settings.demo_trust_render_client_ip and len(render_client_hosts) == 1:
            ip = _normalize_ip(render_client_hosts[0]) or ip

        if self._settings.turnstile_required:
            if not turnstile_token or len(turnstile_token) > TURNSTILE_MAX_TOKEN_CHARS:
                self._verification_rejected()

        self._reserve(ip)
        try:
            if self._settings.turnstile_required:
                self._verify_turnstile(turnstile_token or "", ip)
            with self._lock:
                now = self._clock()
                self._check_daily_limit(now)
                self._daily.append(now)
            yield
        finally:
            with self._lock:
                self._active -= 1

    def _reserve(self, ip: str | None) -> None:
        settings = self._settings
        window = settings.demo_rate_window_seconds
        with self._lock:
            now = self._clock()
            while self._recent and self._recent[0][0] <= now - window:
                self._recent.popleft()
            self._check_daily_limit(now)

            ip_times = [timestamp for timestamp, visitor_ip in self._recent if visitor_ip == ip]
            window_retry = 0
            if len(ip_times) >= settings.demo_ip_rate_limit:
                window_retry = ceil(ip_times[0] + window - now)
            if len(self._recent) >= settings.demo_global_rate_limit:
                window_retry = max(window_retry, ceil(self._recent[0][0] + window - now))
            if window_retry > 0 or self._active >= settings.demo_max_concurrent_requests:
                raise DemoAccessError(
                    429, "demo_rate_limit", "The public demo is busy. Wait briefly and try again.",
                    retry_after=window_retry or 3,
                )

            self._recent.append((now, ip))
            self._active += 1

    def _check_daily_limit(self, now: float) -> None:
        while self._daily and self._daily[0] <= now - DAILY_WINDOW_SECONDS:
            self._daily.popleft()
        if len(self._daily) >= self._settings.demo_daily_request_limit:
            wait = ceil(self._daily[0] + DAILY_WINDOW_SECONDS - now)
            raise DemoAccessError(
                429, "demo_rate_limit", "The public demo has reached its daily request limit. Try again later.",
                retry_after=max(1, wait),
            )

    def _verify_turnstile(self, token: str, ip: str | None) -> None:
        data = {"secret": self._settings.demo_turnstile_secret_key.get_secret_value(), "response": token}
        if ip is not None:
            data["remoteip"] = ip
        try:
            response = httpx.post(TURNSTILE_URL, data=data, timeout=5.0)
            response.raise_for_status()
            result = response.json()
        except (httpx.HTTPError, ValueError):
            raise self._verification_unavailable() from None

        if not isinstance(result, dict) or not isinstance(result.get("success"), bool):
            raise self._verification_unavailable()
        if result["success"] is False:
            errors = result.get("error-codes")
            if isinstance(errors, list) and any(
                item in ("internal-error", "missing-input-secret", "invalid-input-secret") for item in errors
            ):
                raise self._verification_unavailable()
            self._verification_rejected()

        hostname = result.get("hostname")
        expected_hostname = self._settings.demo_turnstile_hostname.lower().rstrip(".")
        if (
            not isinstance(hostname, str) or hostname.lower().rstrip(".") != expected_hostname
            or result.get("action") != self._settings.demo_turnstile_action
        ):
            self._verification_rejected()

    @staticmethod
    def _verification_rejected() -> Never:
        raise DemoAccessError(
            403, "demo_verification_required", "Complete the verification again and retry your question."
        )

    @staticmethod
    def _verification_unavailable() -> DemoAccessError:
        return DemoAccessError(
            503, "demo_verification_unavailable", "Verification is temporarily unavailable. Please try again shortly."
        )
