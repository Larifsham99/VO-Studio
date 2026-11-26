"""python -m latif_studio  -> starts the home server and prints the phone link."""

from __future__ import annotations

import socket
import subprocess

import uvicorn

from .app import create_app
from .config import Settings


def _addresses() -> list[str]:
    ips: list[str] = []
    try:  # Tailscale address, reachable from the phone anywhere
        out = subprocess.run(["tailscale", "ip", "-4"], capture_output=True, text=True, timeout=3)
        ips += [line.strip() for line in out.stdout.splitlines() if line.strip()]
    except (OSError, subprocess.SubprocessError):
        pass
    try:  # home Wi-Fi address
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sk:
            sk.connect(("10.255.255.255", 1))
            ips.append(sk.getsockname()[0])
    except OSError:
        pass
    return list(dict.fromkeys(ips))


def main() -> None:
    s = Settings()
    app = create_app(s)
    app.state.engine.request_model(s.default_model)  # warm up the default model in the background
    print("\n=== Latif Voice AI ===")
    print("Buka pautan ini di Chrome pada fon Android anda (token disertakan):")
    for ip in _addresses() or ["<IP-PC-anda>"]:
        tag = " (Tailscale, dari mana-mana)" if ip.startswith("100.") else " (Wi-Fi rumah sahaja)"
        print(f"  http://{ip}:{s.port}/#token={s.token}{tag}")
    print("Jangan kongsi pautan ini. Token ada dalam", s.data_dir / "token.txt", "\n")
    uvicorn.run(app, host=s.host, port=s.port, log_level="info")


if __name__ == "__main__":
    main()
