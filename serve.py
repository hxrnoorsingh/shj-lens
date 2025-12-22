"""
Simple HTTP server for running the SHJ experiment locally
Usage: python serve.py
Then open: http://localhost:8000
"""

import http.server
import socketserver
import webbrowser
from pathlib import Path

PORT = 8000

class MyHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        # Add CORS headers for local development
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET')
        self.send_header('Cache-Control', 'no-store, no-cache, must-revalidate')
        super().end_headers()

if __name__ == '__main__':
    # Change to experiment directory
    experiment_dir = Path(__file__).parent / 'experiment'
    import os
    os.chdir(experiment_dir)
    
    Handler = MyHTTPRequestHandler
    
    print(f"{'='*60}")
    print(f"SHJ Category Learning Experiment Server")
    print(f"{'='*60}")
    print(f"\nServer starting on port {PORT}...")
    print(f"Serving files from: {experiment_dir}")
    print(f"\n→ Open your browser to: http://localhost:{PORT}")
    print(f"\nPress Ctrl+C to stop the server\n")
    print(f"{'='*60}\n")
    
    with socketserver.TCPServer(("", PORT), Handler) as httpd:
        # Optionally open browser automatically
        try:
            webbrowser.open(f'http://localhost:{PORT}')
        except:
            pass
        
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\n\nServer stopped.")
            print("Thanks for participating!")
