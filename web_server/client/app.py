# server.py
import http.server
import socketserver

PORT = 300

Handler = http.server.SimpleHTTPRequestHandler

# Create a TCPServer and serve files from the current directory
with socketserver.TCPServer(("", PORT), Handler) as httpd:
    print(f"Serving at http://localhost:{PORT}")
    httpd.serve_forever()
