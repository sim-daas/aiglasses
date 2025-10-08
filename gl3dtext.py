import cv2
import numpy as np
import moderngl
from PIL import Image, ImageDraw, ImageFont
from pyrr import Matrix44
import glfw

# ---------- Create a window for moderngl ------------
if not glfw.init():
    raise RuntimeError("GLFW init failed")

width, height = 640, 480
glfw.window_hint(glfw.VISIBLE, glfw.FALSE)        # hidden window
window = glfw.create_window(width, height, "offscreen", None, None)
glfw.make_context_current(window)

ctx = moderngl.create_context()
fbo = ctx.simple_framebuffer((width, height))
fbo.use()

# ---------- Build a text texture --------------------
def make_text_texture(text="HELLO 3D", size=128):
    # Create a transparent RGBA image
    img = Image.new("RGBA", (512, 128), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", size)
    w, h = draw.textsize(text, font=font)
    draw.text(((512 - w) // 2, (128 - h) // 2), text, font=font, fill=(255, 255, 0, 255))
    return ctx.texture(img.size, 4, img.tobytes())

text_tex = make_text_texture("AI-Glasses")
text_tex.build_mipmaps()
text_tex.use()

# ---------- Simple quad geometry --------------------
quad = ctx.buffer(np.array([
    # x,    y,   z,   u,  v
    -1.0, -0.2, 0.0, 0.0, 0.0,
     1.0, -0.2, 0.0, 1.0, 0.0,
    -1.0,  0.2, 0.0, 0.0, 1.0,
     1.0,  0.2, 0.0, 1.0, 1.0,
], dtype='f4').tobytes())

indices = ctx.buffer(np.array([0, 1, 2, 2, 1, 3], dtype='i4').tobytes())

prog = ctx.program(
    vertex_shader='''
    #version 330
    in vec3 in_pos;
    in vec2 in_uv;
    uniform mat4 mvp;
    out vec2 uv;
    void main() {
        gl_Position = mvp * vec4(in_pos, 1.0);
        uv = in_uv;
    }
    ''',
    fragment_shader='''
    #version 330
    in vec2 uv;
    uniform sampler2D tex;
    out vec4 fragColor;
    void main() {
        vec4 c = texture(tex, uv);
        fragColor = c;
    }
    '''
)

vao = ctx.vertex_array(
    prog,
    [(quad, '3f 2f', 'in_pos', 'in_uv')],
    indices
)

# ---------- Camera + Projection ---------------------
proj = Matrix44.perspective_projection(45.0, width / height, 0.1, 100.0)
view = Matrix44.look_at(
    eye=[0.0, 0.0, 3.0],    # camera 3 units away
    target=[0.0, 0.0, 0.0],
    up=[0.0, 1.0, 0.0]
)

# ---------- OpenCV Webcam ---------------------------
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Cannot open webcam")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # ---- Render 3-D text in moderngl FBO -------------
    fbo.clear(0.0, 0.0, 0.0, 0.0)
    model = Matrix44.from_translation([0.0, 0.0, 0.0]) @ Matrix44.from_scale([0.8, 0.8, 1.0])
    prog['mvp'].write((proj @ view @ model).astype('f4').tobytes())
    text_tex.use()
    vao.render()

    # ---- Read back FBO as image ---------------------
    overlay = np.frombuffer(fbo.read(components=4, alignment=1), dtype=np.uint8)
    overlay = overlay.reshape((height, width, 4))
    overlay_bgr = cv2.cvtColor(overlay, cv2.COLOR_RGBA2BGRA)

    # ---- Composite overlay onto webcam frame --------
    alpha = overlay_bgr[:, :, 3] / 255.0
    for c in range(3):
        frame[:, :, c] = (1 - alpha) * frame[:, :, c] + alpha * overlay_bgr[:, :, c]

    cv2.imshow("AR Text Overlay", frame)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC to exit
        break

cap.release()
cv2.destroyAllWindows()
glfw.terminate()
