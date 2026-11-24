#!/usr/bin/env python3
"""Tiny child process for lifecycle tests; does not load any model."""

import argparse
import os
from http.server import BaseHTTPRequestHandler, HTTPServer

parser = argparse.ArgumentParser()
parser.add_argument("--host")
parser.add_argument("--port", type=int)
parser.add_argument("--model")
args, _ = parser.parse_known_args()
if args.model.endswith("fail.gguf"):
    raise SystemExit(1)


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        authenticated = self.headers.get("Authorization") == f"Bearer {os.environ['LLAMA_API_KEY']}"
        self.send_response(200 if authenticated else 401)
        self.end_headers()
        self.wfile.write(b'{"status":"ok"}')

    def log_message(self, *_args):
        pass


HTTPServer((args.host, args.port), Handler).serve_forever()
