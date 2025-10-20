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
        self.show_overlay = True  # Flag to control overlay
        
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
    
    def _draw_3d_text_overlay(self, frame, result):
        """Draw 3D text overlay on frame for web stream"""
        if not result or not self.show_overlay:
            return frame
        
        # Get frame dimensions
        h, w = frame.shape[:2]
        
        # Map location to coordinates
        location_map = {
            'top-left': (int(w * 0.15), int(h * 0.15)),
            'top-center': (int(w * 0.5), int(h * 0.15)),
            'top-right': (int(w * 0.85), int(h * 0.15)),
            'center-left': (int(w * 0.15), int(h * 0.5)),
            'center': (int(w * 0.5), int(h * 0.5)),
            'center-right': (int(w * 0.85), int(h * 0.5)),
            'bottom-left': (int(w * 0.15), int(h * 0.85)),
            'bottom-center': (int(w * 0.5), int(h * 0.85)),
            'bottom-right': (int(w * 0.85), int(h * 0.85)),
        }
        
        location = result.get('location', 'center')
        x, y = location_map.get(location, (int(w * 0.5), int(h * 0.5)))
        
        # Get answer text
        answer = result['answer']
        object_name = result['object']
        
        # Word wrap
        max_chars = 20  # Smaller for web view
        words = answer.split()
        lines = []
        current_line = ""
        
        for word in words:
            if len(current_line) + len(word) + 1 <= max_chars:
                current_line += word + " "
            else:
                if current_line:
                    lines.append(current_line.strip())
                current_line = word + " "
        
        if current_line:
            lines.append(current_line.strip())
        
        # Text settings
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        thickness = 1
        
        max_width = 0
        total_height = 0
        line_heights = []
        
        for line in lines:
            (tw, th), baseline = cv2.getTextSize(line, font, font_scale, thickness)
            max_width = max(max_width, tw)
            line_heights.append(th + baseline)
            total_height += th + baseline + 3
        
        # Add object name
        obj_text = f"[{object_name}]"
        (obj_w, obj_h), obj_baseline = cv2.getTextSize(obj_text, font, 0.4, 1)
        max_width = max(max_width, obj_w)
        total_height += obj_h + obj_baseline + 8
        
        # Draw background box
        padding = 10
        box_x1 = x - max_width // 2 - padding
        box_y1 = y - total_height // 2 - padding
        box_x2 = x + max_width // 2 + padding
        box_y2 = y + total_height // 2 + padding
        
        # Clamp to frame
        box_x1 = max(3, box_x1)
        box_y1 = max(3, box_y1)
        box_x2 = min(w - 3, box_x2)
        box_y2 = min(h - 3, box_y2)
        
        # Semi-transparent background
        overlay = frame.copy()
        cv2.rectangle(overlay, (box_x1, box_y1), (box_x2, box_y2), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # Border
        cv2.rectangle(frame, (box_x1, box_y1), (box_x2, box_y2), (0, 255, 0), 1)
        
        # Draw text
        current_y = y - total_height // 2 + 8
        
        for i, line in enumerate(lines):
            # Shadow
            cv2.putText(frame, line, (x - max_width // 2 + 1, current_y + 1),
                       font, font_scale, (0, 0, 0), thickness + 1, cv2.LINE_AA)
            # Main text
            cv2.putText(frame, line, (x - max_width // 2, current_y),
                       font, font_scale, (0, 255, 0), thickness, cv2.LINE_AA)
            
            current_y += line_heights[i] + 3
        
        # Object name
        current_y += 3
        cv2.putText(frame, obj_text, (x - obj_w // 2, current_y),
                   font, 0.4, (0, 200, 255), 1, cv2.LINE_AA)
        
        # Location dot
        cv2.circle(frame, (x, y), 3, (0, 255, 0), -1)
        cv2.circle(frame, (x, y), 5, (255, 255, 255), 1)
        
        return frame
    
    def _generate_frames(self):
        """Generate JPEG frames for MJPEG stream"""
        while True:
            frame_left, frame_right, depth_map = self.camera_manager.get_frames()
            
            if frame_left is not None:
                # Add overlay if we have results
                if self.latest_result:
                    frame_left = self._draw_3d_text_overlay(frame_left, self.latest_result)
                
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
