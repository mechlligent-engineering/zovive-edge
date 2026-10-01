# Static IP plan (no DHCP)

| Unit | Role | IP |
|---|---|---|
| Base station | dashboard + socket registry | 192.168.1.10 |
| LAP-120 AP | wireless bridge AP (WDS) | 192.168.1.20 |
| LiteBeam (node 1) | wireless client (WDS) | 192.168.1.21 |
| Raspberry Pi 5 (node 1) | edge compute | 192.168.1.101 |
| CP Plus camera (node 1) | RTSP source | 192.168.1.201 |

Node N: Pi = 192.168.1.(100+N), camera = 192.168.1.(200+N), LiteBeam = 192.168.1.(20+N).
