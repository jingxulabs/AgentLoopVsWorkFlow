"""Serve the repo root so the dashboard can fetch results/*.json.

    python serve.py

Opening viewer/index.html as a file:// URL does not work - the browser blocks
the fetch of ../results/*.json as a cross-origin request.
"""

from __future__ import annotations

import http.server
import socketserver
import webbrowser
from functools import partial
from pathlib import Path

PORT = 8077
ROOT = Path(__file__).resolve().parent

if __name__ == "__main__":
    handler = partial(http.server.SimpleHTTPRequestHandler, directory=str(ROOT))
    socketserver.TCPServer.allow_reuse_address = True
    with socketserver.TCPServer(("127.0.0.1", PORT), handler) as httpd:
        url = f"http://localhost:{PORT}/viewer/"
        print(f"serving {ROOT} at {url}  (ctrl-c to stop)")
        webbrowser.open(url)
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nstopped")
