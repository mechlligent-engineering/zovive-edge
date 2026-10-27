"""Camera credentials never appear in configuration errors, the config
summary or logged URLs, whatever characters they contain.

configs/camera.yaml pastes ZOVIVE_CAMERA_USERNAME/PASSWORD into rtsp_url as
they are, and a raw '/', '?' or '#' cuts the URL's user:password part short.
Unhandled, urllib's error text quoted the start of the password, the
cut-short URL's "host" (really the username) was echoed, and a password such
as "123/..." even passed validation and was printed in full by the summary.

Every credential here is synthetic: two distinctive halves around the
character under test, so any fragment that shows up anywhere is unambiguous.
Leak checks ignore case because URL parsing lowercases the "host".
"""

from __future__ import annotations

import io
import os
import re
import shutil
import tempfile
import unittest
from collections.abc import Iterator
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from pathlib import Path
from unittest import mock
from urllib.parse import unquote, urlsplit

import paths
from capture.rtsp_reader import build_rtsp_url
from tests.helpers import TEST_CAMERA_ENV
from tools import camera_probe
from utils import config_loader
from utils.config_loader import (
    ConfigError,
    ConfigLoader,
    IssueKind,
    load,
    load_config,
    redact_url,
    rtsp_url_problem,
)

REPO_CONFIGS = Path(__file__).resolve().parent.parent / "configs"
CORE_FILES = ("node.yaml", "camera.yaml", "inference.yaml", "storage.yaml", "network.yaml")
HOST_VAR = "ZOVIVE_CAMERA_HOST"
USER_VAR, PASS_VAR = "ZOVIVE_CAMERA_USERNAME", "ZOVIVE_CAMERA_PASSWORD"
DEFAULT_CAMERA_HOST = "192.168.1.250"
STREAMS = ("sub_stream", "main_stream")

U1, U2 = "SynUserLeft", "SynUserRight"
P1, P2 = "SynPassLeft", "SynPassRight"
FRAGMENTS = (U1, U2, P1, P2)

# Characters that cut a URL's user:password part short: refused everywhere.
BREAKING = ("/", "?", "#")
# Characters that are percent-encoded on the way to the camera: accepted.
ENCODED = ("@", ":", "%", "%41", " ", "!$&'()*+,;=", "é")


def cases(chars: tuple[str, ...], usernames: bool = True) -> Iterator[tuple[str, str, str]]:
    """(label, username, password) with each character inside the password,
    then inside the username."""
    for c in chars:
        yield f"password with {c!r}", U1, f"{P1}{c}{P2}"
        if usernames:
            yield f"username with {c!r}", f"{U1}{c}{U2}", P1


BROKEN_CASES = (
    *cases(BREAKING),
    # Digits before the '/' parse as a port, so this used to pass validation.
    ("password '123/...'", U1, f"123/{P2}"),
    # '_' is not allowed in a hostname: the "host" error used to echo the username.
    ("username echoed as host", f"Syn_{U1}", f"123/{P2}"),
)
ENCODED_CASES = tuple(cases(ENCODED))

# Every way parse_camera() rejects camera.yaml's rtsp_url -> the start of its
# message. The credentials still come from the environment.
CREDENTIALS = "${ZOVIVE_CAMERA_USERNAME}:${ZOVIVE_CAMERA_PASSWORD}@"
URL_ERROR_PATHS = {
    "rtsp://" + CREDENTIALS + "[cam/live": "is not a valid URL",
    "http://" + CREDENTIALS + "cam.local/live": "must start with rtsp:// or rtsps://",
    "rtsp://" + CREDENTIALS + "cam.local/live?tag=a@b": "has an '@' after the host",
    "rtsp://" + CREDENTIALS + ":554/live": "has no host",
    "rtsp://" + CREDENTIALS + "cam.local:abc/live": "has a port that is not a number",
    "rtsp://" + CREDENTIALS + "cam.local:99999/live": "has a port that is not a number",
    "rtsp://" + CREDENTIALS + "cam_01:554/live": "host 'cam_01' is not a valid IPv4 address or hostname",
}


def leaked(text: str) -> list[str]:
    return [f for f in FRAGMENTS if f.lower() in text.lower()]


@contextmanager
def camera_env(user: str, password: str, host: str | None = None) -> Iterator[None]:
    """os.environ with these camera credentials and ZOVIVE_CAMERA_HOST set to
    `host` or removed when None (even if the developer exported it)."""
    config_loader.clear_cache()
    try:
        with mock.patch.dict(os.environ, {**TEST_CAMERA_ENV, USER_VAR: user, PASS_VAR: password}):
            os.environ.pop(HOST_VAR, None)
            if host is not None:
                os.environ[HOST_VAR] = host
            yield
    finally:
        config_loader.clear_cache()


def validate_command(config_dir: Path = REPO_CONFIGS) -> tuple[int, str]:
    """`python -m utils.config_loader`: exit code and everything it prints."""
    out, err = io.StringIO(), io.StringIO()
    with redirect_stdout(out), redirect_stderr(err):
        code = config_loader.main(["--config-dir", str(config_dir)])
    return code, out.getvalue() + err.getvalue()


def everything_shown(user: str, password: str) -> dict[str, str]:
    """Every text an operator or a log can see for these credentials."""
    shown = {}
    with camera_env(user, password):
        shown["validate command"] = validate_command()[1]
        try:
            shown["camera.yaml summary"] = str(load_config(REPO_CONFIGS, env=dict(os.environ)).summary())
        except ConfigError as exc:
            shown["camera.yaml error"] = str(exc)
        try:
            cam = load(paths.RTSP_CONFIG)["camera"]
        except ConfigError as exc:
            shown["rtsp_config.yaml error"] = str(exc)
        else:
            for key in STREAMS:
                url = build_rtsp_url(cam[key]["url"], cam["host"], cam["username"], cam["password"])
                shown[f"{key} redacted"] = redact_url(url)
                report = camera_probe.measure_stream(url, 0.0, opener=lambda _url: None)
                shown[f"{key} probe report"] = str(report)
    # Code that pastes the credentials in raw, as camera.yaml does.
    shown["raw url redacted"] = redact_url(f"rtsp://{user}:{password}@{DEFAULT_CAMERA_HOST}:554/video/live")
    return shown


class NoLeakTests(unittest.TestCase):
    def test_no_credential_fragment_is_ever_shown(self):
        for label, user, password in (*BROKEN_CASES, *ENCODED_CASES):
            for where, text in everything_shown(user, password).items():
                with self.subTest(case=label, where=where):
                    self.assertEqual(leaked(text), [], text)

    def test_errors_carry_no_urllib_text(self):
        for label, user, password in BROKEN_CASES:
            with self.subTest(case=label), camera_env(user, password):
                text = validate_command()[1]
                self.assertNotIn("could not be cast", text)
                self.assertNotIn("is not a valid URL (", text)

    def test_summary_masks_credentials_and_keeps_default_host(self):
        with camera_env(f"{U1}%{U2}", f"{P1}@{P2}"):
            summary = load_config(REPO_CONFIGS, env=dict(os.environ)).summary()
            code, printed = validate_command()
        stream = f"rtsp://***@{DEFAULT_CAMERA_HOST}:554/video/live?channel=1&subtype=1"
        self.assertEqual(summary["stream"], stream)
        self.assertEqual(code, 0, printed)
        self.assertIn(stream, printed)
        self.assertEqual(leaked(printed), [])


class ValidationErrorPathTests(unittest.TestCase):
    """Each rtsp_url error, end to end through load_config() and the
    validation command. The credentials hold accepted special characters,
    so the only problem is the one under test."""

    USER, PASSWORD = f"{U1}%{U2}", f"{P1}@{P2}"

    def test_each_error_path_rejects_without_leaking(self):
        camera_yaml = (REPO_CONFIGS / "camera.yaml").read_text(encoding="utf-8")
        for url, expected in URL_ERROR_PATHS.items():
            with self.subTest(expected=expected), tempfile.TemporaryDirectory() as tmp:
                config_dir = Path(tmp) / "configs"
                config_dir.mkdir()
                for name in CORE_FILES:
                    shutil.copy(REPO_CONFIGS / name, config_dir / name)
                edited, count = re.subn(r"(?m)^rtsp_url: .*$", f'rtsp_url: "{url}"', camera_yaml)
                self.assertEqual(count, 1)
                (config_dir / "camera.yaml").write_text(edited, encoding="utf-8")
                with camera_env(self.USER, self.PASSWORD):
                    with self.assertRaises(ConfigError) as ctx:
                        ConfigLoader(config_dir, base_dir=Path(tmp), env=dict(os.environ)).load()
                    code, printed = validate_command(config_dir)
                issues = ctx.exception.issues
                self.assertEqual(
                    [(i.source, i.key, i.kind) for i in issues],
                    [("camera.yaml", "rtsp_url", IssueKind.INVALID_VALUE)],
                )
                self.assertTrue(issues[0].message.startswith(expected), issues[0].message)
                self.assertEqual(code, 1, printed)
                for text in (str(ctx.exception), printed):
                    self.assertEqual(leaked(text), [], text)


class RejectionTests(unittest.TestCase):
    """A '/', '?' or '#' in a credential is refused by the validation command
    and by the service, so neither silently uses a cut-short URL."""

    def test_validate_command_refuses(self):
        for label, user, password in BROKEN_CASES:
            with self.subTest(case=label), camera_env(user, password):
                code, text = validate_command()
                self.assertEqual(code, 1, text)
                with self.assertRaises(ConfigError) as ctx:
                    load_config(REPO_CONFIGS, env=dict(os.environ))
                url_issues = [i for i in ctx.exception.issues if i.key == "rtsp_url"]
                self.assertEqual(
                    [(i.source, i.kind) for i in url_issues], [("camera.yaml", IssueKind.INVALID_VALUE)]
                )
                self.assertIn("user:password part is cut short", url_issues[0].message)

    def test_service_refuses_and_names_the_variable(self):
        for label, user, password in BROKEN_CASES:
            in_user = any(c in user for c in BREAKING)
            key, var = ("username", USER_VAR) if in_user else ("password", PASS_VAR)
            with self.subTest(case=label), camera_env(user, password):
                with self.assertRaises(ConfigError) as ctx:
                    load(paths.RTSP_CONFIG)
                issues = [(i.source, i.key, i.kind) for i in ctx.exception.issues]
                self.assertIn(("rtsp_config.yaml", f"camera.{key}", IssueKind.INVALID_VALUE), issues)
                self.assertIn(var, str(ctx.exception))

    def test_reported_together_with_a_bad_host(self):
        with camera_env(U1, f"{P1}/{P2}", host=""), self.assertRaises(ConfigError) as ctx:
            load(paths.RTSP_CONFIG)
        found = {(i.key, i.kind) for i in ctx.exception.issues}
        self.assertEqual(
            found, {("camera.host", IssueKind.INVALID_VALUE), ("camera.password", IssueKind.INVALID_VALUE)}
        )


class EncodedCharacterTests(unittest.TestCase):
    def test_accepted_and_reach_the_camera_intact(self):
        # A ':' in a username cannot work with FFmpeg even encoded (it splits
        # user:password at the first ':' after decoding), so it is not claimed here.
        no_colon = tuple(c for c in ENCODED if c != ":")
        for label, user, password in (*cases(ENCODED, usernames=False), *cases(no_colon)):
            with self.subTest(case=label), camera_env(user, password):
                self.assertEqual(validate_command()[0], 0)
                cam = load(paths.RTSP_CONFIG)["camera"]
                for key in STREAMS:
                    parts = urlsplit(
                        build_rtsp_url(cam[key]["url"], cam["host"], cam["username"], cam["password"])
                    )
                    self.assertEqual((parts.hostname, parts.port), (DEFAULT_CAMERA_HOST, 554))
                    self.assertEqual((unquote(parts.username), unquote(parts.password)), (user, password))

    def test_plain_credentials_build_the_same_url_as_before(self):
        template = "rtsp://{user}:{pass}@{host}:554/video/live?channel=1&subtype=1"
        user, password = "test-user", "AZaz09-._~"
        self.assertEqual(
            build_rtsp_url(template, DEFAULT_CAMERA_HOST, user, password),
            template.format(host=DEFAULT_CAMERA_HOST, user=user, **{"pass": password}),
        )


class MaskingTests(unittest.TestCase):
    def test_whole_user_password_part_masked(self):
        examples = {
            "rtsp://SynUser:SynPass@192.168.1.250:554/live": "rtsp://***@192.168.1.250:554/live",
            "rtsp://192.168.1.250:554/live": "rtsp://192.168.1.250:554/live",
            "rtsp://SynUser:Syn/Pass@192.168.1.250:554/live": "rtsp://***@192.168.1.250:554/live",
            "rtsp://SynUser:Syn#Pass@cam.local/live": "rtsp://***@cam.local/live",
            "rtsp://Syn@User:Syn@Pass@cam.local/live": "rtsp://***@cam.local/live",
            "rtsp:/SynUser:SynPass@cam.local/live": "***@cam.local/live",
            "SynUser:SynPass@cam.local": "***@cam.local",
        }
        for url, shown in examples.items():
            with self.subTest(url=url):
                self.assertEqual(redact_url(url), shown)
                self.assertEqual(config_loader._url_for_error(url), shown)

    def test_unparseable_url(self):
        self.assertEqual(redact_url("rtsp://SynUser:SynPass@[cam/live"), "<unparseable url>")

    def test_url_problems_quote_no_exception_text(self):
        examples = {
            "rtsp://cam.local:abc/live": "has a port that is not a number from 0 to 65535",
            "rtsp://cam.local:99999/live": "has a port that is not a number from 0 to 65535",
            "rtsp://SynUser:SynPass@[cam/live": "is not a valid URL",
            "rtsp://SynUser:Syn?Pass@cam.local/live": "has an '@' after the host",
            f"rtsp://SynUser:SynPass@{DEFAULT_CAMERA_HOST}:554/live": None,
        }
        for url, expected in examples.items():
            with self.subTest(url=url):
                problem = rtsp_url_problem(url)
                if expected is None:
                    self.assertIsNone(problem)
                else:
                    self.assertTrue(problem.startswith(expected), problem)
                    self.assertNotIn("SynPass", problem)


if __name__ == "__main__":
    unittest.main()
