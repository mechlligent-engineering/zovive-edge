"""Minimal HTTP client for sending an event (alert) and a heartbeat to
the base station. This is a first pass covering what transfer_main.py
and health_main.py need to actually send something; the fuller
network/ module (chunked_uploader for large clips, priority_dispatcher,
ack_listener, sync_daemon, boot_registration, transfer_protocol.md) is
deferred — see the batch notes in the PR/commit message.

Store-and-forward is what makes this safe to be simple: db/outbox.py
already persisted the event before this is ever called, so a failed
POST here just means transfer_main.py's retry loop tries again later;
this client doesn't need its own queueing or retry state beyond
raising on failure.
"""

from __future__ import annotations

import logging

import requests

log = logging.getLogger(__name__)


class BaseStationError(RuntimeError):
    pass


class BaseStationClient:
    def __init__(self, base_url: str, api_key: str = "", timeout_sec: float = 10.0):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout_sec = timeout_sec

    def _headers(self) -> dict:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def send_event(self, event: dict, snapshot_bytes: bytes | None = None) -> None:
        """POST event metadata as JSON, plus the snapshot as a file if provided."""
        url = f"{self.base_url}/api/events"
        try:
            if snapshot_bytes is not None:
                files = {"snapshot": ("snapshot.jpg", snapshot_bytes, "image/jpeg")}
                resp = requests.post(
                    url, data={"payload": _to_json(event)}, files=files,
                    headers={"Authorization": self._headers().get("Authorization", "")},
                    timeout=self.timeout_sec,
                )
            else:
                resp = requests.post(url, json=event, headers=self._headers(), timeout=self.timeout_sec)
            resp.raise_for_status()
        except requests.RequestException as exc:
            raise BaseStationError(f"send_event failed: {exc}") from exc

    def send_video(self, event_id: str, video_bytes: bytes, timeout_sec: float | None = None) -> None:
        """POST the event video clip (pipeline/clip_extractor.py's MP4)
        for an event whose image alert was already sent by send_event().
        Separate endpoint and separate call from send_event() on purpose:
        the clip can finish (and need uploading) well after the image
        alert, on its own retry timeline — see db/outbox.py's
        video_status column and transfer_main.py's video-send loop,
        which is what actually calls this.
        """
        url = f"{self.base_url}/api/events/{event_id}/video"
        try:
            files = {"video": (f"{event_id}.mp4", video_bytes, "video/mp4")}
            resp = requests.post(
                url,
                files=files,
                headers={"Authorization": self._headers().get("Authorization", "")},
                timeout=timeout_sec if timeout_sec is not None else self.timeout_sec,
            )
            resp.raise_for_status()
        except requests.RequestException as exc:
            raise BaseStationError(f"send_video failed: {exc}") from exc

    def send_heartbeat(self, payload: dict) -> None:
        url = f"{self.base_url}/api/heartbeat"
        try:
            resp = requests.post(url, json=payload, headers=self._headers(), timeout=self.timeout_sec)
            resp.raise_for_status()
        except requests.RequestException as exc:
            raise BaseStationError(f"send_heartbeat failed: {exc}") from exc


def _to_json(obj: dict) -> str:
    import json

    return json.dumps(obj, default=str)


class NullBaseStationClient:
    """Used when configs/transfer_config.yaml has no base_station.url
    configured — logs instead of sending, so the rest of the pipeline
    (which doesn't depend on a base station existing) still runs end
    to end in dev."""

    def send_event(self, event: dict, snapshot_bytes: bytes | None = None) -> None:
        log.info("NullBaseStationClient: would send event", extra={"event_id": event.get("event_id")})

    def send_video(self, event_id: str, video_bytes: bytes, timeout_sec: float | None = None) -> None:
        log.info(
            "NullBaseStationClient: would send video",
            extra={"event_id": event_id, "video_bytes": len(video_bytes)},
        )

    def send_heartbeat(self, payload: dict) -> None:
        log.debug("NullBaseStationClient: would send heartbeat")
