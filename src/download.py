#!/usr/bin/env python3
"""Download the three MaleCNS v1.0 flat-connectome files (~1.1 GB total)."""
import os, sys, urllib.request
BASE = "https://storage.googleapis.com/flyem-male-cns/v1.0/connectome-data/flat-connectome"
FILES = [
    "body-annotations-male-cns-v1.0-minconf-0.5.feather",
    "body-neurotransmitters-male-cns-v1.0.feather",
    "connectome-weights-male-cns-v1.0-minconf-0.5.feather",
]

def _ssl_context():
    """Verified TLS context. python.org builds of Python on macOS ship without root certificates
    unless "Install Certificates.command" was run; use certifi's bundle when it is installed."""
    import ssl
    try:
        import certifi
        return ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        return ssl.create_default_context()

def main(dst="data/raw"):
    os.makedirs(dst, exist_ok=True)
    for f in FILES:
        p = os.path.join(dst, f)
        if os.path.exists(p):
            print("exists", p); continue
        print("downloading", f, flush=True)
        url = f"{BASE}/{f}"
        with urllib.request.urlopen(url, context=_ssl_context()) as r:
            expected = int(r.headers.get("Content-Length", "0"))
        urllib.request.urlretrieve(url, p + ".part")
        got = os.path.getsize(p + ".part")
        if expected and got != expected:
            os.remove(p + ".part")
            raise RuntimeError(f"{f}: downloaded {got} bytes, expected {expected}; re-run to retry")
        os.rename(p + ".part", p)
        print("done", f, os.path.getsize(p) // 2**20, "MB", flush=True)
if __name__ == "__main__":
    main(*sys.argv[1:])
