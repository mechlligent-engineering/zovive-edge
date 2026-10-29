"""Lens control (optical zoom + focus) for the CP Plus motorized-lens IP
camera, plus a FakeLensCamera with the identical interface for tests
and for developing zoom logic without hardware.

This camera is NOT a PTZ camera: it has a fixed mount and a motorized
varifocal lens. The interface below therefore has no pan, tilt or
preset methods at all, so no caller can ever ask the camera to change
direction:

    zoom      -> ONVIF PTZ service  AbsoluteMove with ONLY a Zoom vector
                 (PanTilt is never included in any request)
    stop      -> ONVIF PTZ service  Stop(PanTilt=False, Zoom=True)
    autofocus -> ONVIF Imaging service  SetImagingSettings(Focus.AutoFocusMode=AUTO)
                 then Imaging.Stop, which on most Dahua-OEM firmware
                 (CP Plus included) re-runs the focus search after a zoom

Real ONVIF (`onvif-zeep`) talks SOAP to the camera's ONVIF service
(usually port 80 on CP Plus — confirm with the camera's web UI / ONVIF
Device Manager before first use). Import is deferred so this module
stays importable without `onvif-zeep` installed; only constructing
OnvifLensCamera requires it.

NOT YET VERIFIED against a real CP Plus unit — ONVIF zoom ranges and
autofocus behavior vary by firmware. Treat this as a starting point to
test against your actual camera. tools/camera_probe.py runs every
read-only query below against a real unit and reports what it supports.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Any, Protocol

import requests
from requests.auth import HTTPDigestAuth

from utils.config_loader import redact_url

log = logging.getLogger(__name__)


class LensCamera(Protocol):
    """Interface both OnvifLensCamera and FakeLensCamera implement, and
    what zoom_controller.py type-hints against."""

    def zoom_to(self, level: float) -> bool: ...
    def stop_zoom(self) -> None: ...
    def trigger_autofocus(self) -> bool: ...
    def get_zoom(self) -> float: ...


class CameraApiError(RuntimeError):
    pass


def _import_onvif():
    try:
        from onvif import ONVIFCamera  # type: ignore

        return ONVIFCamera
    except ImportError as exc:
        raise CameraApiError(
            "onvif-zeep is not installed. Install requirements-pi.txt, or use FakeLensCamera for development."
        ) from exc


ONVIF_TIMEOUT_SEC = 5


def _onvif_transport():
    """zeep transport with a bounded timeout. zeep's default has no
    operation timeout, so one unanswered SOAP call would block the
    pipeline thread until the supervisor declares it hung (60 s) and
    restarts the whole process. With this, the call fails after 5 s and
    is logged like any other ONVIF error."""
    from zeep.transports import Transport  # type: ignore

    return Transport(timeout=ONVIF_TIMEOUT_SEC, operation_timeout=ONVIF_TIMEOUT_SEC)


SNAPSHOT_TIMEOUT_SEC = 5.0
_JPEG_SOI = b"\xff\xd8"
# wstop:topic="true" marks a subscribable leaf in a GetEventProperties TopicSet.
_WSTOP_TOPIC_ATTR = "{http://docs.oasis-open.org/wsn/t-1}topic"


def _attr(obj: Any, *names: str, default: Any = None) -> Any:
    """Walk optional fields of a zeep response; a missing field anywhere
    on the path gives `default` instead of an AttributeError."""
    for name in names:
        if obj is None:
            return default
        obj = getattr(obj, name, None)
    return default if obj is None else obj


def _plain(value: Any) -> Any:
    """zeep values -> JSON-safe scalars (enums and decimals become str/float)."""
    if value is None or isinstance(value, bool | int | float | str):
        return value
    try:
        return float(value)
    except (TypeError, ValueError):
        return str(value)


def _describe_profile(profile: Any) -> dict:
    enc = _attr(profile, "VideoEncoderConfiguration")
    return {
        "token": _plain(_attr(profile, "token")),
        "name": _plain(_attr(profile, "Name")),
        "video_source_token": _plain(_attr(profile, "VideoSourceConfiguration", "SourceToken")),
        "encoding": _plain(_attr(enc, "Encoding")),
        "width": _plain(_attr(enc, "Resolution", "Width")),
        "height": _plain(_attr(enc, "Resolution", "Height")),
        "fps_limit": _plain(_attr(enc, "RateControl", "FrameRateLimit")),
        "bitrate_kbps": _plain(_attr(enc, "RateControl", "BitrateLimit")),
        "gov_length": _plain(_attr(enc, "H264", "GovLength")),
        "has_ptz": _attr(profile, "PTZConfiguration") is not None,
    }


def event_topic_paths(elements: Any) -> list[str]:
    """Flatten a GetEventProperties TopicSet (the XML elements zeep returns
    for its xsd:any content) into sorted 'RuleEngine/CellMotionDetector/Motion'
    style paths, one per subscribable topic. Namespace prefixes are dropped."""
    paths: set[str] = set()

    def walk(node: Any, prefix: str) -> None:
        tag = getattr(node, "tag", None)
        if not isinstance(tag, str):  # XML comments / processing instructions
            return
        name = tag.rsplit("}", 1)[-1]
        if name == "MessageDescription":
            return
        path = f"{prefix}/{name}" if prefix else name
        if str(node.get(_WSTOP_TOPIC_ATTR, "")).lower() == "true":
            paths.add(path)
        for child in node:
            walk(child, path)

    for element in elements or []:
        walk(element, "")
    return sorted(paths)


def fetch_snapshot(
    uri: str, username: str, password: str, timeout_sec: float = SNAPSHOT_TIMEOUT_SEC
) -> bytes | None:
    """GET a JPEG from the camera's ONVIF snapshot URI. Returns None on any
    failure (logged). Digest auth only: CP Plus (Dahua-OEM) firmware uses
    digest, and falling back to basic would send the camera password in
    clear text over the plain-http wireless bridge."""
    try:
        resp = requests.get(uri, auth=HTTPDigestAuth(username, password), timeout=timeout_sec)
    except requests.RequestException as exc:
        log.warning("snapshot fetch failed", extra={"uri": redact_url(uri), "error": str(exc)})
        return None
    if resp.status_code != 200 or not resp.content.startswith(_JPEG_SOI):
        log.warning(
            "snapshot fetch returned no JPEG",
            extra={"uri": redact_url(uri), "status": resp.status_code, "bytes": len(resp.content)},
        )
        return None
    return resp.content


class OnvifLensCamera:
    def __init__(self, host: str, port: int, username: str, password: str):
        ONVIFCamera = _import_onvif()
        self._username = username
        self._password = password
        self._cam = ONVIFCamera(host, port, username, password, transport=_onvif_transport())
        self._media = self._cam.create_media_service()
        self._ptz = self._cam.create_ptz_service()
        self._imaging = self._cam.create_imaging_service()
        self._events = None  # created on first get_event_topics()
        self._profiles = list(self._media.GetProfiles())
        profile = self._profiles[0]
        self._profile_token = profile.token
        self._video_source_token = profile.VideoSourceConfiguration.SourceToken
        self._ptz_config_token = _attr(profile, "PTZConfiguration", "token")
        self._snapshot_uris: dict[str, str] = {}
        self._lock = threading.Lock()

    def zoom_to(self, level: float) -> bool:
        """Absolute optical zoom. The request carries a Zoom vector only —
        never PanTilt — so the camera's direction cannot change."""
        with self._lock:
            try:
                req = self._ptz.create_type("AbsoluteMove")
                req.ProfileToken = self._profile_token
                req.Position = {"Zoom": {"x": level}}
                self._ptz.AbsoluteMove(req)
                return True
            except Exception:
                log.exception("AbsoluteMove (zoom only) failed", extra={"level": level})
                return False

    def stop_zoom(self) -> None:
        with self._lock:
            try:
                req = self._ptz.create_type("Stop")
                req.ProfileToken = self._profile_token
                req.PanTilt = False
                req.Zoom = True
                self._ptz.Stop(req)
            except Exception:
                log.exception("zoom Stop failed")

    def trigger_autofocus(self) -> bool:
        with self._lock:
            try:
                settings = self._imaging.GetImagingSettings({"VideoSourceToken": self._video_source_token})
                if settings.Focus is not None:
                    settings.Focus.AutoFocusMode = "AUTO"
                else:
                    settings.Focus = {"AutoFocusMode": "AUTO"}
                self._imaging.SetImagingSettings(
                    {
                        "VideoSourceToken": self._video_source_token,
                        "ImagingSettings": settings,
                        "ForcePersistence": False,
                    }
                )
                # Stopping any in-progress focus move makes the camera
                # re-evaluate focus under AUTO mode at the new zoom.
                self._imaging.Stop({"VideoSourceToken": self._video_source_token})
                return True
            except Exception:
                log.exception("autofocus trigger failed")
                return False

    def get_zoom(self) -> float:
        with self._lock:
            try:
                status = self._ptz.GetStatus({"ProfileToken": self._profile_token})
                return float(status.Position.Zoom.x)
            except Exception:
                log.exception("GetStatus (zoom) failed")
                return 0.0

    # -- read-only queries ------------------------------------------------------
    # None of these change camera state. tools/camera_probe.py uses them to
    # verify the unit; the mode, lens and snapshot code will use them too.
    # Each returns None on failure (logged), so a firmware that lacks an
    # optional ONVIF feature shows up as "not reported", not as a crash.

    def get_device_info(self) -> dict | None:
        with self._lock:
            try:
                info = self._cam.devicemgmt.GetDeviceInformation()
            except Exception:
                log.exception("GetDeviceInformation failed")
                return None
        keys = ("Manufacturer", "Model", "FirmwareVersion", "SerialNumber", "HardwareId")
        return {key: _plain(_attr(info, key)) for key in keys}

    def describe_media_profiles(self) -> list[dict]:
        """Every media profile the camera reported at connect time, with its
        video encoder settings (codec, resolution, fps limit, bitrate, GOP)."""
        return [_describe_profile(p) for p in self._profiles]

    def get_stream_uri(self, profile_token: str | None = None) -> str | None:
        token = profile_token or self._profile_token
        request = {
            "StreamSetup": {"Stream": "RTP-Unicast", "Transport": {"Protocol": "RTSP"}},
            "ProfileToken": token,
        }
        with self._lock:
            try:
                return str(self._media.GetStreamUri(request).Uri)
            except Exception:
                log.exception("GetStreamUri failed", extra={"profile_token": token})
                return None

    def get_snapshot_uri(self, profile_token: str | None = None) -> str | None:
        token = profile_token or self._profile_token
        with self._lock:
            cached = self._snapshot_uris.get(token)
            if cached is not None:
                return cached
            try:
                uri = str(self._media.GetSnapshotUri({"ProfileToken": token}).Uri)
            except Exception:
                log.exception("GetSnapshotUri failed", extra={"profile_token": token})
                return None
            self._snapshot_uris[token] = uri
            return uri

    def get_snapshot_jpeg(
        self, profile_token: str | None = None, timeout_sec: float = SNAPSHOT_TIMEOUT_SEC
    ) -> bytes | None:
        """One JPEG straight from the camera's encoder: no video decode on the Pi."""
        uri = self.get_snapshot_uri(profile_token)
        if uri is None:
            return None
        # Plain HTTP, no SOAP: fetched outside the lock so a slow snapshot
        # never delays a zoom or focus command.
        return fetch_snapshot(uri, self._username, self._password, timeout_sec)

    def get_move_status(self) -> str | None:
        """PTZ MoveStatus of the zoom axis ("IDLE" / "MOVING" / "UNKNOWN"),
        or None if the firmware doesn't report it."""
        with self._lock:
            try:
                status = self._ptz.GetStatus({"ProfileToken": self._profile_token})
            except Exception:
                log.exception("GetStatus (move status) failed")
                return None
        return _plain(_attr(status, "MoveStatus", "Zoom"))

    def get_focus_status(self) -> dict | None:
        """Imaging FocusStatus20 (position, move status, error), or None if
        the firmware doesn't report it."""
        with self._lock:
            try:
                status = self._imaging.GetStatus({"VideoSourceToken": self._video_source_token})
            except Exception:
                log.exception("Imaging GetStatus failed")
                return None
        focus = _attr(status, "FocusStatus20")
        if focus is None:
            return None
        return {
            "position": _plain(_attr(focus, "Position")),
            "move_status": _plain(_attr(focus, "MoveStatus")),
            "error": _plain(_attr(focus, "Error")),
        }

    def get_zoom_limits(self) -> tuple[float, float] | None:
        """(min, max) of the absolute zoom space, instead of assuming 0..1."""
        if self._ptz_config_token is None:
            log.warning("media profile has no PTZ configuration; zoom limits unknown")
            return None
        with self._lock:
            try:
                opts = self._ptz.GetConfigurationOptions({"ConfigurationToken": self._ptz_config_token})
            except Exception:
                log.exception("PTZ GetConfigurationOptions failed")
                return None
        spaces = _attr(opts, "Spaces", "AbsoluteZoomPositionSpace") or []
        if not spaces:
            return None
        x_range = _attr(spaces[0], "XRange")
        lo, hi = _attr(x_range, "Min"), _attr(x_range, "Max")
        if lo is None or hi is None:
            return None
        return float(lo), float(hi)

    def get_imaging_options(self) -> dict | None:
        """Supported autofocus / IR-cut modes and their current values."""
        source = {"VideoSourceToken": self._video_source_token}
        with self._lock:
            try:
                opts = self._imaging.GetOptions(source)
                settings = self._imaging.GetImagingSettings(source)
            except Exception:
                log.exception("Imaging GetOptions/GetImagingSettings failed")
                return None
        return {
            "autofocus_modes": [_plain(m) for m in _attr(opts, "Focus", "AutoFocusModes", default=[])],
            "ircut_modes": [_plain(m) for m in _attr(opts, "IrCutFilterModes", default=[])],
            "autofocus_current": _plain(_attr(settings, "Focus", "AutoFocusMode")),
            "ircut_current": _plain(_attr(settings, "IrCutFilter")),
        }

    def get_event_topics(self) -> list[str] | None:
        """Event topics the camera can publish (e.g. RuleEngine/CellMotionDetector/Motion)."""
        with self._lock:
            try:
                if self._events is None:
                    self._events = self._cam.create_events_service()
                props = self._events.GetEventProperties()
            except Exception:
                log.exception("GetEventProperties failed")
                return None
        return event_topic_paths(_attr(props, "TopicSet", "_value_1", default=[]))

    @classmethod
    def from_config(cls, cfg: dict) -> OnvifLensCamera:
        cam = cfg.get("camera", {})
        return cls(
            host=cam["host"],
            port=int(cam.get("onvif_port", 80)),
            username=cam["onvif_username"],
            password=cam["onvif_password"],
        )


class FakeLensCamera:
    """In-memory stand-in with the same interface, used by tests and by
    edge_main.py when configs/node_config.yaml runtime.environment != 'pi'.
    Zoom and focus are instantaneous; zoom_controller's own settle timers
    are what model the real lens travel time."""

    def __init__(self):
        self.zoom = 0.0
        self.focus_count = 0
        self.call_log: list[tuple[str, dict]] = []
        self._lock = threading.Lock()

    def zoom_to(self, level: float) -> bool:
        with self._lock:
            self.zoom = level
            self.call_log.append(("zoom_to", {"level": level, "ts": time.time()}))
            return True

    def stop_zoom(self) -> None:
        with self._lock:
            self.call_log.append(("stop_zoom", {"ts": time.time()}))

    def trigger_autofocus(self) -> bool:
        with self._lock:
            self.focus_count += 1
            self.call_log.append(("trigger_autofocus", {"ts": time.time()}))
            return True

    def get_zoom(self) -> float:
        with self._lock:
            return self.zoom
