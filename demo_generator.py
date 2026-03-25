"""
演示GIF生成脚本 - 独立实现，不依赖 main.py / cube.py
用于生成三角形和立方体旋转的演示动画
"""
import taichi as ti
import numpy as np
from PIL import Image

ti.init(arch=ti.cpu, log_level=ti.CRITICAL)

WIDTH, HEIGHT = 700, 700
PI = 3.14159265358979323846

# ---------- 三角形数据 ----------
tri_vertices = ti.Vector.field(3, dtype=ti.f32, shape=3)
tri_colors   = ti.Vector.field(3, dtype=ti.f32, shape=3)
tri_pixels   = ti.Vector.field(3, dtype=ti.f32, shape=(WIDTH, HEIGHT))

tri_vertices[0] = ti.Vector([2.0,  0.0, -2.0])
tri_vertices[1] = ti.Vector([0.0,  2.0, -2.0])
tri_vertices[2] = ti.Vector([-2.0, 0.0, -2.0])
tri_colors[0] = ti.Vector([1.0, 0.0, 0.0])
tri_colors[1] = ti.Vector([0.0, 1.0, 0.0])
tri_colors[2] = ti.Vector([0.0, 0.0, 1.0])

# ---------- 立方体数据 ----------
cube_vertices    = ti.Vector.field(3, dtype=ti.f32, shape=8)
cube_edges       = ti.Vector.field(2, dtype=ti.i32, shape=12)
cube_edge_colors = ti.Vector.field(3, dtype=ti.f32, shape=12)
cube_pixels      = ti.Vector.field(3, dtype=ti.f32, shape=(WIDTH, HEIGHT))

cube_vertices[0] = ti.Vector([-1.0, -1.0, -1.0])
cube_vertices[1] = ti.Vector([ 1.0, -1.0, -1.0])
cube_vertices[2] = ti.Vector([ 1.0,  1.0, -1.0])
cube_vertices[3] = ti.Vector([-1.0,  1.0, -1.0])
cube_vertices[4] = ti.Vector([-1.0, -1.0,  1.0])
cube_vertices[5] = ti.Vector([ 1.0, -1.0,  1.0])
cube_vertices[6] = ti.Vector([ 1.0,  1.0,  1.0])
cube_vertices[7] = ti.Vector([-1.0,  1.0,  1.0])

for _i, _e in enumerate([(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7)]):
    cube_edges[_i] = ti.Vector([_e[0], _e[1]])

_back  = [[0.3,0.3,1.0],[0.4,0.4,1.0],[0.5,0.5,1.0],[0.6,0.6,1.0]]
_front = [[1.0,0.3,0.3],[1.0,0.4,0.4],[1.0,0.5,0.5],[1.0,0.6,0.6]]
_conn  = [[0.3,1.0,0.3],[0.4,1.0,0.4],[0.5,1.0,0.5],[0.6,1.0,0.6]]
for _i, _c in enumerate(_back + _front + _conn):
    cube_edge_colors[_i] = ti.Vector(_c)


# ================================================================
#  共用 Taichi 函数
# ================================================================

@ti.func
def build_model_z(angle: ti.f32) -> ti.math.mat4:
    r = angle * PI / 180.0
    c, s = ti.cos(r), ti.sin(r)
    return ti.math.mat4(
        [ c,  s, 0.0, 0.0],
        [-s,  c, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    )

@ti.func
def build_model_xyz(ax: ti.f32, ay: ti.f32, az: ti.f32) -> ti.math.mat4:
    rx, ry, rz = ax*PI/180.0, ay*PI/180.0, az*PI/180.0
    cx, sx = ti.cos(rx), ti.sin(rx)
    cy, sy = ti.cos(ry), ti.sin(ry)
    cz, sz = ti.cos(rz), ti.sin(rz)
    rot_x = ti.math.mat4(
        [1.0, 0.0, 0.0, 0.0], [0.0, cx, sx, 0.0],
        [0.0, -sx, cx, 0.0],  [0.0, 0.0, 0.0, 1.0])
    rot_y = ti.math.mat4(
        [cy, 0.0, -sy, 0.0], [0.0, 1.0, 0.0, 0.0],
        [sy, 0.0,  cy, 0.0], [0.0, 0.0, 0.0, 1.0])
    rot_z = ti.math.mat4(
        [ cz,  sz, 0.0, 0.0], [-sz, cz, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0])
    return rot_z @ rot_y @ rot_x

@ti.func
def build_view(ex: ti.f32, ey: ti.f32, ez: ti.f32) -> ti.math.mat4:
    return ti.math.mat4(
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [-ex, -ey, -ez, 1.0],
    )

@ti.func
def build_proj(fov_deg: ti.f32, aspect: ti.f32, zn: ti.f32, zf: ti.f32) -> ti.math.mat4:
    fov = fov_deg * PI / 180.0
    n, f = -zn, -zf
    t = ti.tan(fov / 2.0) * ti.abs(n)
    b, r, l = -t, aspect * t, -aspect * t
    persp = ti.math.mat4(
        [n, 0.0, 0.0, 0.0],
        [0.0, n, 0.0, 0.0],
        [0.0, 0.0, n+f, 1.0],
        [0.0, 0.0, -n*f, 0.0],
    )
    scale = ti.math.mat4(
        [2.0/(r-l), 0.0, 0.0, 0.0],
        [0.0, 2.0/(t-b), 0.0, 0.0],
        [0.0, 0.0, 2.0/(n-f), 0.0],
        [0.0, 0.0, 0.0, 1.0],
    )
    trans = ti.math.mat4(
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [-(r+l)/2.0, -(t+b)/2.0, -(n+f)/2.0, 1.0],
    )
    return scale @ trans @ persp

@ti.func
def draw_line_colored(x0: ti.i32, y0: ti.i32, x1: ti.i32, y1: ti.i32,
                      c0: ti.math.vec3, c1: ti.math.vec3,
                      buf: ti.template()):
    dx = ti.abs(x1 - x0)
    dy = ti.abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx - dy
    x, y = x0, y0
    total = ti.max(dx, dy)
    step = 0
    while True:
        t_val = ti.cast(step, ti.f32) / ti.cast(total, ti.f32) if total > 0 else 0.0
        color = c0 * (1.0 - t_val) + c1 * t_val
        if 0 <= x < WIDTH and 0 <= y < HEIGHT:
            buf[x, y] = color
        if x == x1 and y == y1:
            break
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x += sx
        if e2 < dx:
            err += dx
            y += sy
        step += 1

@ti.func
def draw_line_solid(x0: ti.i32, y0: ti.i32, x1: ti.i32, y1: ti.i32,
                    color: ti.math.vec3, buf: ti.template()):
    draw_line_colored(x0, y0, x1, y1, color, color, buf)


# NDC 中间缓冲（全局 field，避免 kernel 内部数据竞争）
tri_ndc  = ti.Vector.field(2, dtype=ti.f32, shape=3)
cube_ndc = ti.Vector.field(2, dtype=ti.f32, shape=8)

# ================================================================
#  三角形渲染：拆成变换 + 绘制两个 kernel
# ================================================================

@ti.kernel
def xform_triangle(angle: ti.f32):
    mvp = build_proj(45.0, 1.0, 0.1, 50.0) @ build_view(0.0, 0.0, 5.0) @ build_model_z(angle)
    for i in range(3):
        v = mvp @ ti.math.vec4(tri_vertices[i][0], tri_vertices[i][1], tri_vertices[i][2], 1.0)
        if v[3] != 0.0:
            v /= v[3]
        tri_ndc[i] = ti.math.vec2(v[0], v[1])

@ti.kernel
def draw_triangle():
    for i in range(3):
        j = (i + 1) % 3
        x0 = int((tri_ndc[i][0] + 1.0) * WIDTH  / 2.0)
        y0 = int((tri_ndc[i][1] + 1.0) * HEIGHT / 2.0)
        x1 = int((tri_ndc[j][0] + 1.0) * WIDTH  / 2.0)
        y1 = int((tri_ndc[j][1] + 1.0) * HEIGHT / 2.0)
        draw_line_colored(x0, y0, x1, y1, tri_colors[i], tri_colors[j], tri_pixels)

def render_triangle(angle: float):
    xform_triangle(angle)
    draw_triangle()


# ================================================================
#  立方体渲染：拆成变换 + 绘制两个 kernel
# ================================================================

@ti.kernel
def xform_cube(ax: ti.f32, ay: ti.f32, az: ti.f32):
    mvp = build_proj(45.0, 1.0, 0.1, 50.0) @ build_view(0.0, 0.0, 5.0) @ build_model_xyz(ax, ay, az)
    for i in range(8):
        v = mvp @ ti.math.vec4(cube_vertices[i][0], cube_vertices[i][1], cube_vertices[i][2], 1.0)
        if v[3] != 0.0:
            v /= v[3]
        cube_ndc[i] = ti.math.vec2(v[0], v[1])

@ti.kernel
def draw_cube():
    for i in range(12):
        s = cube_edges[i][0]
        e = cube_edges[i][1]
        x0 = int((cube_ndc[s][0] + 1.0) * WIDTH  / 2.0)
        y0 = int((cube_ndc[s][1] + 1.0) * HEIGHT / 2.0)
        x1 = int((cube_ndc[e][0] + 1.0) * WIDTH  / 2.0)
        y1 = int((cube_ndc[e][1] + 1.0) * HEIGHT / 2.0)
        draw_line_solid(x0, y0, x1, y1, cube_edge_colors[i], cube_pixels)

def render_cube(ax: float, ay: float, az: float):
    xform_cube(ax, ay, az)
    draw_cube()


# ================================================================
#  帧抓取 + GIF 保存
# ================================================================

def grab(buf) -> Image.Image:
    arr = (np.clip(buf.to_numpy(), 0.0, 1.0) * 255).astype(np.uint8)
    return Image.fromarray(arr, 'RGB')


def save_gif(frames, path, duration=60):
    frames[0].save(path, save_all=True, append_images=frames[1:],
                   duration=duration, loop=0, optimize=False)


def generate_triangle_demo():
    print("正在生成三角形演示动画...")
    frames = []
    n_frames = 72  # 每帧 5°，旋转两圈
    for frame in range(n_frames):
        angle = frame * (360.0 / n_frames) * 2
        tri_pixels.fill(0.0)
        render_triangle(angle)
        frames.append(grab(tri_pixels))
        if (frame + 1) % 18 == 0:
            print(f"  {frame + 1}/{n_frames} 帧")
    save_gif(frames, 'triangle_demo.gif', duration=55)
    print("✓ triangle_demo.gif 已保存")


def generate_cube_demo():
    print("正在生成立方体演示动画...")
    frames = []
    n_frames = 90
    for frame in range(n_frames):
        ax = frame * 1.8
        ay = frame * 2.4
        az = frame * 0.9
        cube_pixels.fill(0.0)
        render_cube(ax, ay, az)
        frames.append(grab(cube_pixels))
        if (frame + 1) % 30 == 0:
            print(f"  {frame + 1}/{n_frames} 帧")
    save_gif(frames, 'cube_demo.gif', duration=55)
    print("✓ cube_demo.gif 已保存")


if __name__ == '__main__':
    print("=" * 50)
    print("MVP 变换演示动画生成器")
    print("=" * 50)
    generate_triangle_demo()
    generate_cube_demo()
    print("=" * 50)
    print("所有演示动画生成完成！")
    print("=" * 50)
