from flask import Flask, render_template, Response, jsonify
from flask_socketio import SocketIO, emit
from flask_cors import CORS
import cv2
import json
import os
from config import Config

class WebServer:
    def __init__(self, camera_manager):
        # Get absolute path to web directory
        current_dir = os.path.dirname(os.path.abspath(__file__))
        web_dir = os.path.join(os.path.dirname(current_dir), 'web')
        
        print(f"📁 Web directory: {web_dir}")
        print(f"📄 Looking for index.html at: {os.path.join(web_dir, 'index.html')}")
        
        # Check if web directory and files exist
        if not os.path.exists(web_dir):
            print(f"⚠️  Creating web directory: {web_dir}")
            os.makedirs(web_dir, exist_ok=True)
        
        if not os.path.exists(os.path.join(web_dir, 'index.html')):
            print(f"⚠️  index.html not found, creating minimal version...")
            self._create_minimal_html(web_dir)
        
        self.app = Flask(__name__, 
                        static_folder=web_dir,
                        template_folder=web_dir)
        CORS(self.app)
        
        # Use threading mode instead of eventlet (more compatible)
        self.socketio = SocketIO(self.app, cors_allowed_origins="*", async_mode='threading')
        
        self.camera_manager = camera_manager
        self.latest_result = None
        
        self._setup_routes()
        self._setup_socketio()
    
    def _create_minimal_html(self, web_dir):
        """Create a minimal HTML file for testing"""
        html_content = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AURA AI Glasses</title>
    <script src="https://cdn.socket.io/4.5.4/socket.io.min.js"></script>
    <style>
        body {
            margin: 0;
            padding: 0;
            font-family: Arial, sans-serif;
            background: #000;
            color: #fff;
        }
        #container {
            display: flex;
            flex-direction: column;
            align-items: center;
            padding: 20px;
        }
        #video-feed {
            max-width: 90vw;
            border: 2px solid #0f0;
            margin: 20px 0;
        }
        #status {
            background: rgba(0, 0, 0, 0.8);
            padding: 15px;
            border-radius: 8px;
            margin: 10px 0;
            min-width: 300px;
        }
        #answer {
            font-size: 24px;
            font-weight: bold;
            color: #0f0;
            margin: 20px 0;
            padding: 20px;
            background: rgba(0, 255, 0, 0.1);
            border-radius: 10px;
            min-height: 60px;
            display: none;
        }
        .info {
            color: #888;
            font-size: 14px;
        }
    </style>
</head>
<body>
    <div id="container">
        <h1>🤖 AURA AI Glasses</h1>
        
        <div id="status">
            <div id="status-text">Connecting...</div>
        </div>
        
        <img id="video-feed" src="/video_feed" alt="Camera Feed">
        
        <div id="answer"></div>
        
        <div class="info">
            <p>📝 Press SPACE on Jetson to record voice query</p>
            <p>🎥 Camera feed updates automatically</p>
        </div>
    </div>
    
    <script>
        const socket = io();
        
        socket.on('connect', () => {
            console.log('✅ Connected to server');
            document.getElementById('status-text').textContent = '✅ Connected';
            document.getElementById('status-text').style.color = '#0f0';
        });
        
        socket.on('disconnect', () => {
            console.log('❌ Disconnected from server');
            document.getElementById('status-text').textContent = '❌ Disconnected';
            document.getElementById('status-text').style.color = '#f00';
        });
        
        socket.on('status', (data) => {
            console.log('📡 Status:', data.message);
            document.getElementById('status-text').textContent = data.message;
        });
        
        socket.on('gemini_result', (result) => {
            console.log('📡 Result:', result);
            
            const answerDiv = document.getElementById('answer');
            answerDiv.textContent = result.answer;
            answerDiv.style.display = 'block';
            
            // Update status
            document.getElementById('status-text').innerHTML = 
                `Q: ${result.transcription}<br>Object: ${result.object}`;
            
            // Hide after 5 seconds
            setTimeout(() => {
                answerDiv.style.display = 'none';
            }, 5000);
        });
    </script>
</body>
</html>"""
        
        with open(os.path.join(web_dir, 'index.html'), 'w') as f:
            f.write(html_content)
        
        print(f"✅ Created minimal index.html")
        
    def _setup_routes(self):
        """Setup Flask routes"""
        
        @self.app.route('/')
        def index():
            return render_template('index.html')
        
        @self.app.route('/video_feed')
        def video_feed():
            return Response(
                self._generate_frames(),
                mimetype='multipart/x-mixed-replace; boundary=frame'
            )
        
        @self.app.route('/data')
        def get_data():
            if self.latest_result:
                return jsonify(self.latest_result)
            return jsonify({"status": "no_data"})
    
    def _generate_frames(self):
        """Generate JPEG frames for MJPEG stream"""
        while True:
            frame_left, frame_right, depth_map = self.camera_manager.get_frames()
            
            if frame_left is not None:
                # Encode frame as JPEG
                ret, buffer = cv2.imencode('.jpg', frame_left, 
                                          [cv2.IMWRITE_JPEG_QUALITY, 85])
                frame_bytes = buffer.tobytes()
                
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
    
    def _setup_socketio(self):
        """Setup WebSocket event handlers"""
        
        @self.socketio.on('connect')
        def handle_connect():
            print('🔌 Client connected')
            emit('status', {'message': 'Connected to AURA AI'})
        
        @self.socketio.on('disconnect')
        def handle_disconnect():
            print('🔌 Client disconnected')
    
    def broadcast_result(self, result):
        """Broadcast result to all connected clients"""
        self.latest_result = result
        self.socketio.emit('gemini_result', result)
        print(f"📡 Broadcast result: {result['answer']}")
    
    def run(self):
        """Run the Flask server"""
        print(f"🌐 Starting web server on http://{Config.SERVER_HOST}:{Config.SERVER_PORT}")
        self.socketio.run(
            self.app,
            host=Config.SERVER_HOST,
            port=Config.SERVER_PORT,
            debug=False,
            use_reloader=False
        )
