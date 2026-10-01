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
test against your actual camera.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Protocol

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
            "onvif-zeep is not installed. Install requirements-pi.txt, or use "
            "FakeLensCamera for development."
        ) from exc


class OnvifLensCamera:
    def __init__(self, host: str, port: int, username: str, password: str):
        ONVIFCamera = _import_onvif()
        self._cam = ONVIFCamera(host, port, username, password)
        self._media = self._cam.create_media_service()
        self._ptz = self._cam.create_ptz_service()
        self._imaging = self._cam.create_imaging_service()
        profile = self._media.GetProfiles()[0]
        self._profile_token = profile.token
        self._video_source_token = profile.VideoSourceConfiguration.SourceToken
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

    @classmethod
    def from_config(cls, cfg: dict) -> OnvifLensCamera:
        cam = cfg.get("camera", {})
        return cls(
            host=cam["host"],
            port=int(cam.get("onvif_port", 80)),
            username=cam["username"],
            password=cam["password"],
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
