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
        """Create HTML with Three.js 3D text rendering"""
        html_content = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AURA AI Glasses</title>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
    <script src="https://cdn.socket.io/4.5.4/socket.io.min.js"></script>
    <style>
        body {
            margin: 0;
            padding: 0;
            font-family: Arial, sans-serif;
            background: #000;
            color: #fff;
            overflow: hidden;
        }
        #container {
            position: relative;
            width: 100vw;
            height: 100vh;
        }
        #video-feed {
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            object-fit: cover;
        }
        #three-canvas {
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            pointer-events: none;
        }
        #status {
            position: absolute;
            top: 20px;
            left: 20px;
            background: rgba(0, 0, 0, 0.8);
            padding: 15px;
            border-radius: 8px;
            z-index: 10;
        }
        #info {
            position: absolute;
            bottom: 20px;
            left: 20px;
            background: rgba(0, 0, 0, 0.8);
            padding: 10px;
            border-radius: 5px;
            font-size: 12px;
            color: #888;
            z-index: 10;
        }
    </style>
</head>
<body>
    <div id="container">
        <img id="video-feed" src="/video_feed" alt="Camera Feed">
        <canvas id="three-canvas"></canvas>
        
        <div id="status">
            <div id="status-text">Connecting...</div>
        </div>
        
        <div id="info">
            <p>📝 Press 't' on Jetson to test</p>
            <p>🎥 3D text powered by Three.js</p>
        </div>
    </div>
    
    <script>
        // Three.js setup
        let scene, camera, renderer;
        let textMesh = null;
        
        function initThree() {
            scene = new THREE.Scene();
            
            const canvas = document.getElementById('three-canvas');
            camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);
            camera.position.z = 5;
            
            renderer = new THREE.WebGLRenderer({ canvas: canvas, alpha: true, antialias: true });
            renderer.setSize(window.innerWidth, window.innerHeight);
            renderer.setClearColor(0x000000, 0);
            
            // Lighting for 3D effect
            const ambientLight = new THREE.AmbientLight(0xffffff, 0.6);
            scene.add(ambientLight);
            
            const directionalLight = new THREE.DirectionalLight(0xffffff, 0.8);
            directionalLight.position.set(5, 5, 5);
            scene.add(directionalLight);
            
            animate();
            console.log('✅ Three.js initialized');
        }
        
        function animate() {
            requestAnimationFrame(animate);
            
            // Subtle rotation for 3D effect
            if (textMesh) {
                textMesh.rotation.y = Math.sin(Date.now() * 0.0005) * 0.1;
            }
            
            renderer.render(scene, camera);
        }
        
        function create3DText(result) {
            console.log('Creating 3D text from result:', result);
            
            // Remove old text
            if (textMesh) {
                scene.remove(textMesh);
                textMesh.geometry.dispose();
                textMesh.material.dispose();
            }
            
            const text = result.answer;
            const label = result.object;
            const location = result.location || 'center';
            
            // Create text canvas with higher resolution
            const canvas = document.createElement('canvas');
            const context = canvas.getContext('2d');
            canvas.width = 2048;
            canvas.height = 1024;
            
            // Clear canvas
            context.fillStyle = 'rgba(0, 0, 0, 0)';
            context.fillRect(0, 0, canvas.width, canvas.height);
            
            // Word wrap
            const maxWidth = 1800;
            const words = text.split(' ');
            const lines = [];
            let currentLine = words[0];
            
            context.font = 'bold 120px Arial';
            for (let i = 1; i < words.length; i++) {
                const testLine = currentLine + ' ' + words[i];
                const metrics = context.measureText(testLine);
                if (metrics.width > maxWidth) {
                    lines.push(currentLine);
                    currentLine = words[i];
                } else {
                    currentLine = testLine;
                }
            }
            lines.push(currentLine);
            
            // Draw depth layers for 3D effect
            const layerCount = 8;
            let yPos = 400;
            const lineHeight = 140;
            
            for (let line of lines) {
                // Depth layers
                for (let i = layerCount; i > 0; i--) {
                    context.fillStyle = `rgba(80, 80, 80, ${0.4 * (layerCount - i) / layerCount})`;
                    context.font = 'bold 120px Arial';
                    context.textAlign = 'center';
                    context.textBaseline = 'middle';
                    context.fillText(line, 1024 - i * 3, yPos - i * 3);
                }
                
                // Main text with glow
                context.shadowColor = 'rgba(0, 255, 0, 0.8)';
                context.shadowBlur = 20;
                context.shadowOffsetX = 0;
                context.shadowOffsetY = 0;
                context.fillStyle = '#00ff00';
                context.font = 'bold 120px Arial';
                context.fillText(line, 1024, yPos);
                
                yPos += lineHeight;
            }
            
            // Label
            context.shadowBlur = 10;
            context.fillStyle = '#00ccff';
            context.font = 'bold 60px Arial';
            context.fillText(`[${label}]`, 1024, yPos + 40);
            
            // Create texture
            const texture = new THREE.CanvasTexture(canvas);
            
            // Create material with proper 3D properties
            const material = new THREE.MeshPhongMaterial({
                map: texture,
                transparent: true,
                side: THREE.DoubleSide,
                shininess: 60,
                specular: 0x555555,
                emissive: 0x002200
            });
            
            // Create geometry with actual depth
            const geometry = new THREE.BoxGeometry(8, 4, 0.5);
            
            // Create mesh
            textMesh = new THREE.Mesh(geometry, material);
            
            // Position based on location
            const locationMap = {
                'top-left': [-4, 2.5, -3],
                'top-center': [0, 2.5, -3],
                'top-right': [4, 2.5, -3],
                'center-left': [-4, 0, -3],
                'center': [0, 0, -3],
                'center-right': [4, 0, -3],
                'bottom-left': [-4, -2.5, -3],
                'bottom-center': [0, -2.5, -3],
                'bottom-right': [4, -2.5, -3]
            };
            
            const pos = locationMap[location] || [0, 0, -3];
            const depth = result.position && result.position.z ? result.position.z : 0.5;
            const zPos = pos[2] - (depth * 3);
            
            textMesh.position.set(pos[0], pos[1], zPos);
            
            scene.add(textMesh);
            
            console.log(`📝 3D text created at location: ${location}, z: ${zPos}`);
        }
        
        // Socket.IO setup
        const socket = io(window.location.origin, {
            path: '/socket.io',
            transports: ['polling', 'websocket'],
            upgrade: true,
            reconnection: true
        });
        
        socket.on('connect', () => {
            console.log('✅ Connected to server');
            document.getElementById('status-text').textContent = '✅ Connected';
            document.getElementById('status-text').style.color = '#0f0';
        });
        
        socket.on('disconnect', (reason) => {
            console.log('❌ Disconnected:', reason);
            document.getElementById('status-text').textContent = '❌ Disconnected';
            document.getElementById('status-text').style.color = '#f00';
        });
        
        socket.on('status', (data) => {
            console.log('📡 Status:', data.message);
            document.getElementById('status-text').textContent = data.message;
        });
        
        socket.on('gemini_result', (result) => {
            console.log('📡 Gemini Result received:', result);
            
            // Update status
            document.getElementById('status-text').innerHTML = 
                `Q: ${result.transcription}<br>A: ${result.answer}<br>Location: ${result.location}`;
            
            // Create 3D text
            try {
                create3DText(result);
                console.log('✅ 3D text created successfully');
            } catch (error) {
                console.error('❌ Error creating 3D text:', error);
            }
            
            // Auto-hide after 10 seconds
            setTimeout(() => {
                if (textMesh) {
                    scene.remove(textMesh);
                    textMesh.geometry.dispose();
                    textMesh.material.dispose();
                    textMesh = null;
                    console.log('🗑️ 3D text removed');
                }
            }, 10000);
        });
        
        // Debug: Log all events
        socket.onAny((eventName, ...args) => {
            console.log('📨 Socket event:', eventName, args);
        });
        
        // Handle window resize
        window.addEventListener('resize', () => {
            camera.aspect = window.innerWidth / window.innerHeight;
            camera.updateProjectionMatrix();
            renderer.setSize(window.innerWidth, window.innerHeight);
        });
        
        // Initialize
        window.addEventListener('load', () => {
            initThree();
            console.log('🚀 AURA AI Glasses loaded');
        });
    </script>
</body>
</html>"""
        
        with open(os.path.join(web_dir, 'index.html'), 'w') as f:
            f.write(html_content)
        
        logger.info(f"✅ Created index.html with Three.js")
    
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
        """Generate JPEG frames - NO OVERLAY"""
        while True:
            frame_left, frame_right, depth_map = self.camera_manager.get_frames()
            
            if frame_left is not None:
                # NO OVERLAY - just send raw frame
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
