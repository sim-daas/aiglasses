from flask import Flask, Response, jsonify, send_from_directory, request, render_template
from flask_socketio import SocketIO, emit
from flask_cors import CORS
import cv2
import logging
import json
import os
import time  # <-- ADD THIS MISSING IMPORT
from config import Config
from text_3d_renderer import Text3DRenderer

logger = logging.getLogger(__name__)

class WebServer:
    def __init__(self, camera_manager, tape_measure=None):
        """Initialize web server with optional tape measure for depth data"""
        self.camera_manager = camera_manager
        self.tape_measure = tape_measure
        self.latest_result = None  # <-- ADD THIS (was missing)
        
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
        
        # Initialize 3D text renderer for web feed
        logger.info("Initializing 3D text renderer for web feed...")
        self.text_renderer = Text3DRenderer()
        
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
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
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
            z-index: 1;
        }
        #three-canvas {
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            pointer-events: none;
            z-index: 2;
        }
        #status {
            position: absolute;
            top: 20px;
            right: 20px;
            background: rgba(0, 0, 0, 0.9);
            padding: 15px 20px;
            border-radius: 8px;
            z-index: 100;
            font-size: 14px;
            max-width: 400px;
            border: 1px solid #0f0;
        }
        #info {
            position: absolute;
            bottom: 20px;
            left: 20px;
            background: rgba(0, 0, 0, 0.9);
            padding: 10px 15px;
            border-radius: 5px;
            font-size: 12px;
            color: #888;
            z-index: 100;
            border: 1px solid #333;
        }
        #debug {
            position: absolute;
            top: 20px;
            left: 20px;
            background: rgba(0, 0, 0, 0.9);
            padding: 10px 15px;
            border-radius: 5px;
            font-size: 11px;
            font-family: monospace;
            color: #0f0;
            z-index: 100;
            max-width: 300px;
            border: 1px solid #0f0;
        }
    </style>
</head>
<body>
    <div id="container">
        <img id="video-feed" src="/video_feed" alt="Camera Feed">
        <canvas id="three-canvas"></canvas>
        
        <div id="debug">
            <div>Three.js Status: <span id="threejs-status">Initializing...</span></div>
            <div>Last Event: <span id="last-event">None</span></div>
        </div>
        
        <div id="status">
            <div id="status-text">Connecting...</div>
        </div>
        
        <div id="info">
            <p>📝 Press 't' on Jetson to test</p>
            <p>🎥 3D text powered by Three.js</p>
        </div>
    </div>
    
    <script>
        console.log('🚀 Starting AURA AI Glasses web interface...');
        
        // Three.js setup
        let scene, camera, renderer;
        let textMesh = null;
        
        function updateDebug(status, event) {
            document.getElementById('threejs-status').textContent = status;
            document.getElementById('last-event').textContent = event;
        }
        
        function initThree() {
            console.log('🎨 Initializing Three.js...');
            
            try {
                scene = new THREE.Scene();
                
                const canvas = document.getElementById('three-canvas');
                camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);
                camera.position.z = 5;
                
                renderer = new THREE.WebGLRenderer({ 
                    canvas: canvas, 
                    alpha: true, 
                    antialias: true 
                });
                renderer.setSize(window.innerWidth, window.innerHeight);
                renderer.setClearColor(0x000000, 0);
                
                // Lighting
                const ambientLight = new THREE.AmbientLight(0xffffff, 0.6);
                scene.add(ambientLight);
                
                const directionalLight = new THREE.DirectionalLight(0xffffff, 0.8);
                directionalLight.position.set(5, 5, 5);
                scene.add(directionalLight);
                
                // Start animation loop
                animate();
                
                console.log('✅ Three.js initialized successfully');
                updateDebug('Ready', 'Initialized');
            } catch (error) {
                console.error('❌ Three.js initialization failed:', error);
                updateDebug('Failed', error.message);
            }
        }
        
        function animate() {
            requestAnimationFrame(animate);
            
            // Subtle rotation for 3D effect
            if (textMesh) {
                textMesh.rotation.y = Math.sin(Date.now() * 0.0005) * 0.1;
                textMesh.rotation.x = Math.sin(Date.now() * 0.0003) * 0.05;
            }
            
            renderer.render(scene, camera);
        }
        
        function create3DText(result) {
            console.log('🎨 Creating 3D text from result:', result);
            updateDebug('Creating text...', JSON.stringify(result.object));
            
            try {
                // Remove old text
                if (textMesh) {
                    scene.remove(textMesh);
                    textMesh.geometry.dispose();
                    textMesh.material.dispose();
                    console.log('🗑️ Removed old text mesh');
                }
                
                const text = result.answer || 'No answer';
                const label = result.object || 'unknown';
                const location = result.location || 'center';
                
                console.log(`Creating text: "${text}" at ${location}`);
                
                // Create text canvas
                const canvas = document.createElement('canvas');
                const context = canvas.getContext('2d');
                canvas.width = 2048;
                canvas.height = 1024;
                
                // Clear canvas
                context.clearRect(0, 0, canvas.width, canvas.height);
                
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
                
                console.log(`Text wrapped into ${lines.length} lines`);
                
                // Draw depth layers for 3D effect
                const layerCount = 8;
                let yPos = 350;
                const lineHeight = 140;
                
                for (let line of lines) {
                    // Depth layers
                    for (let i = layerCount; i > 0; i--) {
                        context.fillStyle = `rgba(80, 80, 80, ${0.5 * (layerCount - i) / layerCount})`;
                        context.font = 'bold 120px Arial';
                        context.textAlign = 'center';
                        context.textBaseline = 'middle';
                        context.fillText(line, 1024 - i * 3, yPos - i * 3);
                    }
                    
                    // Main text with glow
                    context.shadowColor = 'rgba(0, 255, 0, 1)';
                    context.shadowBlur = 25;
                    context.shadowOffsetX = 0;
                    context.shadowOffsetY = 0;
                    
                    // Outline
                    context.strokeStyle = '#000000';
                    context.lineWidth = 8;
                    context.strokeText(line, 1024, yPos);
                    
                    // Fill
                    context.fillStyle = '#00ff00';
                    context.fillText(line, 1024, yPos);
                    
                    yPos += lineHeight;
                }
                
                // Label
                yPos += 20;
                context.shadowBlur = 15;
                context.strokeStyle = '#000000';
                context.lineWidth = 6;
                context.fillStyle = '#00ccff';
                context.font = 'bold 70px Arial';
                const labelText = `[${label}]`;
                context.strokeText(labelText, 1024, yPos);
                context.fillText(labelText, 1024, yPos);
                
                console.log('Canvas text drawn');
                
                // Create texture
                const texture = new THREE.CanvasTexture(canvas);
                texture.needsUpdate = true;
                
                // Create material
                const material = new THREE.MeshPhongMaterial({
                    map: texture,
                    transparent: true,
                    opacity: 0.95,
                    side: THREE.DoubleSide,
                    shininess: 60,
                    specular: 0x444444,
                    emissive: 0x003300
                });
                
                // Create geometry with depth
                const geometry = new THREE.BoxGeometry(10, 5, 0.6);
                
                // Create mesh
                textMesh = new THREE.Mesh(geometry, material);
                
                // Position based on location
                const locationMap = {
                    'top-left': [-5, 3, -4],
                    'top-center': [0, 3, -4],
                    'top-right': [5, 3, -4],
                    'center-left': [-5, 0, -4],
                    'center': [0, 0, -4],
                    'center-right': [5, 0, -4],
                    'bottom-left': [-5, -3, -4],
                    'bottom-center': [0, -3, -4],
                    'bottom-right': [5, -3, -4]
                };
                
                const pos = locationMap[location] || [0, 0, -4];
                const depth = result.position && result.position.z ? result.position.z : 0.5;
                const zPos = pos[2] - (depth * 2);
                
                textMesh.position.set(pos[0], pos[1], zPos);
                textMesh.scale.set(1, 1, 1);
                
                scene.add(textMesh);
                
                console.log(`✅ 3D text added to scene at position: [${pos[0]}, ${pos[1]}, ${zPos}]`);
                updateDebug('Text visible!', `${location} @ z:${zPos.toFixed(2)}`);
                
            } catch (error) {
                console.error('❌ Error creating 3D text:', error);
                updateDebug('Error!', error.message);
            }
        }
        
        // Socket.IO setup
        console.log('🔌 Connecting to Socket.IO...');
        const socket = io(window.location.origin, {
            path: '/socket.io',
            transports: ['polling', 'websocket'],
            upgrade: true,
            reconnection: true
        });
        
        socket.on('connect', () => {
            console.log('✅ Socket.IO connected');
            document.getElementById('status-text').textContent = '✅ Connected';
            document.getElementById('status-text').style.color = '#0f0';
            updateDebug('Connected', 'Socket.IO active');
        });
        
        socket.on('disconnect', (reason) => {
            console.log('❌ Socket.IO disconnected:', reason);
            document.getElementById('status-text').textContent = '❌ Disconnected: ' + reason;
            document.getElementById('status-text').style.color = '#f00';
            updateDebug('Disconnected', reason);
        });
        
        socket.on('status', (data) => {
            console.log('📡 Status message:', data.message);
            document.getElementById('status-text').textContent = data.message;
        });
        
        socket.on('gemini_result', (result) => {
            console.log('📡 ========== GEMINI RESULT RECEIVED ==========');
            console.log('Full result object:', result);
            console.log('Answer:', result.answer);
            console.log('Object:', result.object);
            console.log('Location:', result.location);
            console.log('Position:', result.position);
            console.log('==============================================');
            
            // Update status
            document.getElementById('status-text').innerHTML = 
                `<strong>Q:</strong> ${result.transcription}<br>` +
                `<strong>A:</strong> ${result.answer}<br>` +
                `<strong>Object:</strong> ${result.object}<br>` +
                `<strong>Location:</strong> ${result.location}`;
            
            // Create 3D text
            try {
                create3DText(result);
                console.log('✅ 3D text creation completed');
            } catch (error) {
                console.error('❌ Fatal error creating 3D text:', error);
                console.error('Stack trace:', error.stack);
            }
            
            // Auto-hide after 15 seconds
            setTimeout(() => {
                if (textMesh) {
                    scene.remove(textMesh);
                    textMesh.geometry.dispose();
                    textMesh.material.dispose();
                    textMesh = null;
                    console.log('🗑️ 3D text removed after timeout');
                    updateDebug('Ready', 'Text cleared');
                }
            }, 15000);
        });
        
        // Handle window resize
        window.addEventListener('resize', () => {
            camera.aspect = window.innerWidth / window.innerHeight;
            camera.updateProjectionMatrix();
            renderer.setSize(window.innerWidth, window.innerHeight);
            console.log('📐 Window resized');
        });
        
        // Initialize on load
        window.addEventListener('load', () => {
            console.log('📄 Page loaded, initializing...');
            initThree();
        });
        
        console.log('✅ Script loaded and ready');
    </script>
</body>
</html>"""
        
        with open(os.path.join(web_dir, 'index.html'), 'w') as f:
            f.write(html_content)
        
        logger.info(f"✅ Created index.html with Three.js (all inline)")
    
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
        
        @self.app.route('/depth')
        def depth():
            """Endpoint for depth grid data"""
            if self.tape_measure:
                depth_data = self.tape_measure.get_depth_grid()
                if depth_data:
                    return jsonify(depth_data)
            return jsonify({'error': 'Depth data not available'}), 404
    
    def _map_location_to_position(self, location, frame_width, frame_height):
        """Map Gemini's text location to pixel coordinates"""
        location_map = {
            'top-left': (int(frame_width * 0.20), int(frame_height * 0.20)),
            'top-center': (int(frame_width * 0.50), int(frame_height * 0.20)),
            'top-right': (int(frame_width * 0.80), int(frame_height * 0.20)),
            'center-left': (int(frame_width * 0.20), int(frame_height * 0.50)),
            'center': (int(frame_width * 0.50), int(frame_height * 0.50)),
            'center-right': (int(frame_width * 0.80), int(frame_height * 0.50)),
            'bottom-left': (int(frame_width * 0.20), int(frame_height * 0.80)),
            'bottom-center': (int(frame_width * 0.50), int(frame_height * 0.80)),
            'bottom-right': (int(frame_width * 0.80), int(frame_height * 0.80)),
        }
        return location_map.get(location.lower(), (int(frame_width * 0.5), int(frame_height * 0.5)))
    
    def _generate_frames(self):
        """Generate JPEG frames with gesture keyboard overlay"""
        while True:
            # Try to get processed frame first (has keyboard overlay)
            frame = self.camera_manager.get_processed_frame()
            
            # Fallback to raw left frame if no processed frame
            if frame is None:
                frame_left, _, _ = self.camera_manager.get_frames()
                frame = frame_left
            
            if frame is not None:
                # Encode frame as JPEG
                ret, buffer = cv2.imencode('.jpg', frame, 
                                          [cv2.IMWRITE_JPEG_QUALITY, 85])
                frame_bytes = buffer.tobytes();
                
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            
            time.sleep(0.033)  # ~30 FPS
    
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
