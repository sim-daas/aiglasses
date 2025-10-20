// WebSocket connection
const socket = io();

// Three.js setup
let scene, camera, renderer;
let textMesh = null;

function initThree() {
    // Scene
    scene = new THREE.Scene();

    // Camera
    const canvas = document.getElementById('three-canvas');
    camera = new THREE.PerspectiveCamera(
        75,
        window.innerWidth / window.innerHeight,
        0.1,
        1000
    );
    camera.position.z = 5;

    // Renderer
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

    console.log('✅ Three.js initialized');
}

function animate() {
    requestAnimationFrame(animate);

    // Rotate text slightly for 3D effect
    if (textMesh) {
        textMesh.rotation.y = Math.sin(Date.now() * 0.001) * 0.1;
    }

    renderer.render(scene, camera);
}

function create3DText(text, position) {
    // Remove old text
    if (textMesh) {
        scene.remove(textMesh);
        textMesh.geometry.dispose();
        textMesh.material.dispose();
    }

    // Create text geometry using TextGeometry would require font loading
    // For simplicity, use a simple 3D plane with texture
    const canvas = document.createElement('canvas');
    const context = canvas.getContext('2d');
    canvas.width = 512;
    canvas.height = 256;

    // Draw text on canvas
    context.fillStyle = '#ffffff';
    context.font = 'bold 48px Arial';
    context.textAlign = 'center';
    context.textBaseline = 'middle';

    // Add shadow
    context.shadowColor = 'rgba(0, 0, 0, 0.8)';
    context.shadowBlur = 10;
    context.shadowOffsetX = 5;
    context.shadowOffsetY = 5;

    // Wrap text if needed
    const maxWidth = 480;
    const words = text.split(' ');
    let line = '';
    let y = 128;
    const lineHeight = 60;

    for (let word of words) {
        const testLine = line + word + ' ';
        const metrics = context.measureText(testLine);

        if (metrics.width > maxWidth && line !== '') {
            context.fillText(line, 256, y);
            line = word + ' ';
            y += lineHeight;
        } else {
            line = testLine;
        }
    }
    context.fillText(line, 256, y);

    // Create texture from canvas
    const texture = new THREE.CanvasTexture(canvas);

    // Create material
    const material = new THREE.MeshBasicMaterial({
        map: texture,
        transparent: true,
        side: THREE.DoubleSide
    });

    // Create geometry
    const geometry = new THREE.PlaneGeometry(4, 2);

    // Create mesh
    textMesh = new THREE.Mesh(geometry, material);

    // Position based on normalized coordinates and depth
    const x = (position.x - 0.5) * 10;  // Map 0-1 to -5 to 5
    const y = -(position.y - 0.5) * 7.5; // Map 0-1 to 3.75 to -3.75 (inverted Y)
    const z = -position.z * 5;  // Map depth to Z position

    textMesh.position.set(x, y, z);

    scene.add(textMesh);

    console.log(`📝 3D text created at (${x.toFixed(2)}, ${y.toFixed(2)}, ${z.toFixed(2)})`);
}

// WebSocket event handlers
socket.on('connect', () => {
    console.log('🔌 Connected to server');
    document.getElementById('status-text').textContent = 'Connected';
    document.getElementById('status-text').style.color = '#0f0';
});

socket.on('disconnect', () => {
    console.log('🔌 Disconnected from server');
    document.getElementById('status-text').textContent = 'Disconnected';
    document.getElementById('status-text').style.color = '#f00';
});

socket.on('status', (data) => {
    console.log('📡 Status:', data.message);
    document.getElementById('status-text').textContent = data.message;
});

socket.on('gemini_result', (result) => {
    console.log('📡 Received result:', result);

    // Update query text
    const queryEl = document.getElementById('query-text');
    queryEl.textContent = `Q: ${result.transcription}`;
    queryEl.style.display = 'block';

    // Update answer text (fallback 2D display)
    const answerEl = document.getElementById('answer-text');
    answerEl.textContent = result.answer;
    answerEl.style.display = 'block';

    // Create 3D text at detected position with proper depth
    if (result.position) {
        const position = {
            x: result.position.x,
            y: result.position.y,
            z: result.position.z || 0.5  // Use stereo depth if available
        };
        
        console.log(`Creating 3D text at normalized position: (${position.x.toFixed(3)}, ${position.y.toFixed(3)}, ${position.z.toFixed(3)})`);
        
        create3DText(result.answer, position);
    }

    // Hide after 5 seconds
    setTimeout(() => {
        queryEl.style.display = 'none';
        answerEl.style.display = 'none';

        if (textMesh) {
            scene.remove(textMesh);
            textMesh = null;
        }
    }, 5000);
});

// Handle window resize
window.addEventListener('resize', () => {
    camera.aspect = window.innerWidth / window.innerHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(window.innerWidth, window.innerHeight);
});

// Initialize on page load
window.addEventListener('load', () => {
    initThree();
    console.log('🚀 AURA AI Glasses web interface loaded');
});
