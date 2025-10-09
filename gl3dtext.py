import cv2
import moderngl
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import math

# ---------- 1. Setup webcam ----------
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Could not open webcam")

w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# ---------- 2. Create ModernGL context (offscreen) ----------
ctx = moderngl.create_standalone_context()

# Framebuffer to render text into
fbo = ctx.simple_framebuffer((w, h))
fbo.use()
fbo.clear(0.0, 0.0, 0.0, 0.0)

# ---------- 3. Create a 2D texture with PIL text ----------
font = ImageFont.truetype("DejaVuSans-Bold.ttf", 128)  # adjust path
text = "HELLO 3D"

# Make a simple 3D-looking text image (with shadow and gradient)
txt_img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
draw = ImageDraw.Draw(txt_img)

bbox = draw.textbbox((0, 0), text, font=font)
text_w, text_h = bbox[2] - bbox[0], bbox[3] - bbox[1]
x = (w - text_w) // 2
y = (h - text_h) // 2

# Shadow layer
for offset in range(8):
    draw.text((x + offset, y + offset), text, font=font, fill=(30, 30, 30, 180))

# Main text with bright color
draw.text((x, y), text, font=font, fill=(255, 215, 0, 255))

# Upload as texture
text_tex = ctx.texture((w, h), 4, txt_img.transpose(Image.FLIP_TOP_BOTTOM).tobytes())
text_tex.build_mipmaps()
text_tex.use()

# ---------- 4. Shaders (perspective + rotation) ----------
prog = ctx.program(
    vertex_shader="""
        #version 330
        uniform mat4 Mvp;
        in vec3 in_vert;
        in vec2 in_tex;
        out vec2 v_tex;
        void main() {
            gl_Position = Mvp * vec4(in_vert, 1.0);
            v_tex = in_tex;
        }
    """,
    fragment_shader="""
        #version 330
        in vec2 v_tex;
        out vec4 fragColor;
        uniform sampler2D Texture;
        void main() {
            fragColor = texture(Texture, v_tex);
        }
    """,
)

quad_vertices = np.array([
    #  x, y, z,   u, v
    -1, -0.5, 0, 0, 0,
     1, -0.5, 0, 1, 0,
     1,  0.5, 0, 1, 1,
    -1,  0.5, 0, 0, 1,
], dtype='f4')

indices = np.array([0, 1, 2, 2, 3, 0], dtype='i4')

vbo = ctx.buffer(quad_vertices.tobytes())
ibo = ctx.buffer(indices.tobytes())
vao = ctx.simple_vertex_array(prog, vbo, 'in_vert', 'in_tex', index_buffer=ibo)

# ---------- 5. Projection + View ----------
def perspective(fov, aspect, near, far):
    f = 1.0 / math.tan(fov / 2)
    return np.array([
        [f / aspect, 0, 0, 0],
        [0, f, 0, 0],
        [0, 0, (far + near) / (near - far), -1],
        [0, 0, (2 * far * near) / (near - far), 0],
    ], dtype='f4')

def rotation_y(angle):
    c, s = math.cos(angle), math.sin(angle)
    return np.array([
        [c, 0, s, 0],
        [0, 1, 0, 0],
        [-s, 0, c, 0],
        [0, 0, 0, 1],
    ], dtype='f4')

def translate(z):
    return np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, z],
        [0, 0, 0, 1],
    ], dtype='f4')

proj = perspective(math.radians(45), w / h, 0.1, 10.0)
view = translate(-2.0)

# ---------- 6. Main loop ----------
angle = 0.0
while True:
    ret, frame = cap.read()
    if not ret:
        break

    angle += 0.02  # slowly rotate
    model = rotation_y(angle)
    mvp = proj @ view @ model
    prog['Mvp'].write(mvp.tobytes())

    fbo.use()
    fbo.clear(0.0, 0.0, 0.0, 0.0)
    vao.render()

    data = fbo.read(components=4, alignment=1)
    text_img = np.frombuffer(data, dtype=np.uint8).reshape(h, w, 4)
    text_img = np.flipud(text_img)  # fix upside-down output

    frame_bgra = cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA)

    alpha = text_img[:, :, 3:4] / 255.0
    blended = (frame_bgra * (1 - alpha) + text_img[:, :, :4] * alpha).astype(np.uint8)

    cv2.imshow("3D-Looking Text Overlay", blended)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC to quit
        break

cap.release()
cv2.destroyAllWindows()
