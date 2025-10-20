from flask import Flask, render_template, Response, jsonify
from flask_socketio import SocketIO, emit
from flask_cors import CORS
import cv2
import json
import base64
import numpy as np
from config import Config

class WebServer:
    def __init__(self, camera_manager):
        self.app = Flask(__name__, 
                        static_folder='../../web',
                        template_folder='../../web')
        CORS(self.app)
        
        # Use threading mode instead of eventlet (more compatible)
        self.socketio = SocketIO(self.app, cors_allowed_origins="*", async_mode='threading')
        
        self.camera_manager = camera_manager
        self.latest_result = None
        
        self._setup_routes()
        self._setup_socketio()
        
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
