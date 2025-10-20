from flask import Flask, render_template, Response, jsonify, request
from flask_socketio import SocketIO, emit
from flask_cors import CORS
import cv2
import json
import os
import logging
from config import Config

logger = logging.getLogger(__name__)

class WebServer:
    def __init__(self, camera_manager):
        # Get absolute path to web directory
        current_dir = os.path.dirname(os.path.abspath(__file__))
        web_dir = os.path.join(os.path.dirname(current_dir), 'web')
        
        logger.info(f"📁 Web directory: {web_dir}")
        logger.info(f"📄 Looking for index.html at: {os.path.join(web_dir, 'index.html')}")
        
        # Check if web directory and files exist
        if not os.path.exists(web_dir):
            logger.warning(f"⚠️  Creating web directory: {web_dir}")
            os.makedirs(web_dir, exist_ok=True)
        
        if not os.path.exists(os.path.join(web_dir, 'index.html')):
            logger.warning(f"⚠️  index.html not found, creating minimal version...")
            self._create_minimal_html(web_dir)
        
        self.app = Flask(__name__, 
                        static_folder=web_dir,
                        template_folder=web_dir)
        
        # Important: Set secret key for Flask sessions (required for SocketIO)
        self.app.config['SECRET_KEY'] = 'aura-ai-glasses-secret-key-2024'
        
        # Enable CORS
        CORS(self.app, resources={r"/*": {"origins": "*"}})
        
        # Initialize SocketIO with proper configuration for Jetson
        logger.info("Initializing SocketIO...")
        self.socketio = SocketIO(
            self.app,
            cors_allowed_origins="*",
            async_mode='threading',
            logger=True,
            engineio_logger=True,
            ping_timeout=60,
            ping_interval=25,
            manage_session=False
        )
        
        self.camera_manager = camera_manager
        self.latest_result = None
        
        logger.info("Setting up Flask routes...")
        self._setup_routes()
        
        logger.info("Setting up WebSocket handlers...")
        self._setup_socketio()
        
        logger.info("✅ Web server initialized")
    
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
        #location-info {
            background: rgba(0,255,0,0.2);
            padding: 10px;
            margin: 10px 0;
            border-radius: 5px;
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
        
        <div id="location-info">
            <strong>Object:</strong> <span id="object-name"></span><br>
            <strong>Location:</strong> <span id="object-location"></span>
        </div>
        
        <div class="info">
            <p>📝 Press 't' on Jetson to test (test mode)</p>
            <p>🎥 Camera feed updates automatically</p>
        </div>
    </div>
    
    <script>
        console.log('Starting Socket.IO initialization...');
        
        // Initialize Socket.IO
        const socket = io(window.location.origin, {
            path: '/socket.io',
            transports: ['polling', 'websocket'],
            upgrade: true,
            reconnection: true,
            reconnectionDelay: 1000,
            reconnectionAttempts: 10,
            timeout: 20000
        });
        
        console.log('Socket.IO object created');
        
        socket.on('connect', () => {
            console.log('✅ Connected to server!');
            console.log('Socket ID:', socket.id);
            console.log('Transport:', socket.io.engine.transport.name);
            
            document.getElementById('status-text').textContent = '✅ Connected';
            document.getElementById('status-text').style.color = '#0f0';
        });
        
        socket.on('connect_error', (error) => {
            console.error('❌ Connection error:', error);
            document.getElementById('status-text').textContent = '❌ Connection Error: ' + error.message;
            document.getElementById('status-text').style.color = '#f00';
        });
        
        socket.on('disconnect', (reason) => {
            console.log('❌ Disconnected:', reason);
            document.getElementById('status-text').textContent = '❌ Disconnected: ' + reason;
            document.getElementById('status-text').style.color = '#f00';
        });
        
        socket.on('status', (data) => {
            console.log('📡 Status:', data);
            document.getElementById('status-text').textContent = data.message;
        });
        
        socket.on('gemini_result', (result) => {
            console.log('📡 Gemini Result:', result);
            
            // Show answer
            const answerDiv = document.getElementById('answer');
            answerDiv.textContent = result.answer;
            answerDiv.style.display = 'block';
            
            // Show location info
            document.getElementById('object-name').textContent = result.object;
            document.getElementById('object-location').textContent = result.location || result.position.description;
            document.getElementById('location-info').style.display = 'block';
            
            // Update status
            document.getElementById('status-text').innerHTML = 
                `Q: ${result.transcription}`;
            
            // Hide after 10 seconds
            setTimeout(() => {
                answerDiv.style.display = 'none';
                document.getElementById('location-info').style.display = 'none';
            }, 10000);
        });
        
        // Debug: Log all events
        socket.onAny((eventName, ...args) => {
            console.log('Event received:', eventName, args);
        });
    </script>
</body>
</html>"""
        
        with open(os.path.join(web_dir, 'index.html'), 'w') as f:
            f.write(html_content)
        
        logger.info(f"✅ Created minimal index.html")
    
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
            logger.info('🔌 Client connected via SocketIO')
            try:
                client_id = request.sid
                logger.info(f'   Client SID: {client_id}')
            except:
                logger.info('   Could not get client SID')
            
            emit('status', {'message': 'Connected to AURA AI'})
            logger.info('   Sent initial status message')
        
        @self.socketio.on('disconnect')
        def handle_disconnect():
            logger.info('🔌 Client disconnected from SocketIO')
        
        @self.socketio.on('ping')
        def handle_ping(data):
            logger.info(f'📡 Received ping: {data}')
            emit('pong', {'data': data})
    
    def broadcast_result(self, result):
        """Broadcast result to all connected clients"""
        logger.info(f"📡 Broadcasting result to web clients...")
        logger.info(f"   Answer: {result['answer']}")
        logger.info(f"   Object: {result['object']}")
        logger.info(f"   Location: {result.get('location', 'N/A')}")
        
        try:
            self.latest_result = result
            
            # Broadcast to all connected clients
            # Note: use 'room' parameter, not 'broadcast' for Flask-SocketIO
            self.socketio.emit('gemini_result', result)
            
            logger.info("✅ Broadcast sent successfully")
        except Exception as e:
            logger.error(f"❌ Broadcast error: {e}")
            import traceback
            logger.error(traceback.format_exc())
    
    def run(self):
        """Run the Flask server"""
        logger.info(f"🌐 Starting web server on http://{Config.SERVER_HOST}:{Config.SERVER_PORT}")
        
        try:
            self.socketio.run(
                self.app,
                host=Config.SERVER_HOST,
                port=Config.SERVER_PORT,
                debug=False,
                use_reloader=False,
                allow_unsafe_werkzeug=True  # Suppress development server warning
            )
        except Exception as e:
            logger.error(f"❌ Web server error: {e}")
            import traceback
            logger.error(traceback.format_exc())
