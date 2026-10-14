#!/usr/bin/env python3
"""Static server with COOP/COEP headers so onnxruntime-web can use
SharedArrayBuffer-backed WASM threads (~3-4x speedup vs single-thread).

    python serve.py [PORT]    # default 8000
"""

from __future__ import annotations

import http.server
import socketserver
import sys
from pathlib import Path


class CrossOriginIsolatedHandler(http.server.SimpleHTTPRequestHandler):
    """COOP/COEP + transparent br/gzip serving for pre-compressed files.

    If the client sends Accept-Encoding: br and a ``<file>.br`` exists next to
    the requested path, serve the compressed file with Content-Encoding: br.
    Same for ``<file>.gz`` + gzip. Lets us ship a 1-2 MB wire payload for a
    3-4 MB on-disk ONNX without runtime decompression in the page.
    """

    encodings = (("br", ".br"), ("gzip", ".gz"))

    def _pick_encoding(self, requested: Path) -> tuple[str, Path] | None:
        accept = self.headers.get("Accept-Encoding", "")
        for enc, suffix in self.encodings:
            if enc not in accept:
                continue
            candidate = requested.with_name(requested.name + suffix)
            if candidate.exists():
                return enc, candidate
        return None

    def send_head(self):
        path = Path(self.translate_path(self.path))
        if path.is_file():
            picked = self._pick_encoding(path)
            if picked is not None:
                enc, compressed = picked
                try:
                    f = compressed.open("rb")
                except OSError:
                    return super().send_head()
                self.send_response(200)
                # Preserve original mime type; tell client about encoding.
                ctype = self.guess_type(str(path))
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Encoding", enc)
                self.send_header("Content-Length", str(compressed.stat().st_size))
                self.send_header("Vary", "Accept-Encoding")
                self.end_headers()
                return f
        return super().send_head()

    def end_headers(self) -> None:
        self.send_header("Cross-Origin-Opener-Policy", "same-origin")
        self.send_header("Cross-Origin-Embedder-Policy", "require-corp")
        self.send_header("Cross-Origin-Resource-Policy", "same-origin")
        super().end_headers()


def main() -> int:
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
    root = Path(__file__).resolve().parent

    handler = lambda *a, **kw: CrossOriginIsolatedHandler(  # noqa: E731
        *a, directory=str(root), **kw
    )
    with socketserver.ThreadingTCPServer(("", port), handler) as httpd:
        httpd.allow_reuse_address = True
        print(f"serving {root} at http://localhost:{port}/")
        print("COOP/COEP enabled — WASM threads available")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nstopped")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
