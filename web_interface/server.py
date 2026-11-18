#!/usr/bin/env python3
"""
Lightweight HTTP Server for Stock Market ML CA1 & CA2 Web Dashboard
"""
import http.server
import socketserver
import os
import sys

PORT = 8000

class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=os.path.dirname(os.path.abspath(__file__)), **kwargs)

def main():
    with socketserver.TCPServer(("", PORT), Handler) as httpd:
        print("=" * 60)
        print(f"  Stock Market ML CA1 & CA2 Web Dashboard Running")
        print(f"  URL: http://localhost:{PORT}")
        print("  Press Ctrl+C to terminate")
        print("=" * 60)
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nServer stopped.")

if __name__ == '__main__':
    main()
