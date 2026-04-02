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
    return ti.Matrix([
        [ c, -s, 0.0, 0.0],
        [ s,  c, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ])

@ti.func
def build_model_xyz(ax: ti.f32, ay: ti.f32, az: ti.f32) -> ti.math.mat4:
    rx, ry, rz = ax*PI/180.0, ay*PI/180.0, az*PI/180.0
    cx, sx = ti.cos(rx), ti.sin(rx)
    cy, sy = ti.cos(ry), ti.sin(ry)
    cz, sz = ti.cos(rz), ti.sin(rz)
    rot_x = ti.Matrix([
        [1.0,  0.0,   0.0,  0.0],
        [0.0,  cx,   -sx,   0.0],
        [0.0,  sx,    cx,   0.0],
        [0.0,  0.0,   0.0,  1.0],
    ])
    rot_y = ti.Matrix([
        [ cy,  0.0,  sy,   0.0],
        [ 0.0, 1.0,  0.0,  0.0],
        [-sy,  0.0,  cy,   0.0],
        [ 0.0, 0.0,  0.0,  1.0],
    ])
    rot_z = ti.Matrix([
        [cz, -sz, 0.0, 0.0],
        [sz,  cz, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ])
    return rot_z @ rot_y @ rot_x

@ti.func
def build_view(ex: ti.f32, ey: ti.f32, ez: ti.f32) -> ti.math.mat4:
    return ti.Matrix([
        [1.0, 0.0, 0.0, -ex],
        [0.0, 1.0, 0.0, -ey],
        [0.0, 0.0, 1.0, -ez],
        [0.0, 0.0, 0.0, 1.0],
    ])

@ti.func
def build_proj(fov_deg: ti.f32, aspect: ti.f32, zn: ti.f32, zf: ti.f32) -> ti.math.mat4:
    fov = fov_deg * PI / 180.0
    n, f = -zn, -zf
    t = ti.tan(fov / 2.0) * ti.abs(n)
    b, r, l = -t, aspect * t, -aspect * t
    persp = ti.Matrix([
        [n,   0.0, 0.0,    0.0   ],
        [0.0, n,   0.0,    0.0   ],
        [0.0, 0.0, n+f,   -n*f   ],
        [0.0, 0.0, 1.0,    0.0   ],
    ])
    scale = ti.Matrix([
        [2.0/(r-l), 0.0,       0.0,       0.0],
        [0.0,       2.0/(t-b), 0.0,       0.0],
        [0.0,       0.0,       2.0/(n-f), 0.0],
        [0.0,       0.0,       0.0,       1.0],
    ])
    trans = ti.Matrix([
        [1.0, 0.0, 0.0, -(r+l)/2.0],
        [0.0, 1.0, 0.0, -(t+b)/2.0],
        [0.0, 0.0, 1.0, -(n+f)/2.0],
        [0.0, 0.0, 0.0, 1.0       ],
    ])
    return scale @ trans @ persp

@ti.func
def euler_to_quat(ax: ti.f32, ay: ti.f32, az: ti.f32) -> ti.math.vec4:
    hx = ax * PI / 180.0 * 0.5
    hy = ay * PI / 180.0 * 0.5
    hz = az * PI / 180.0 * 0.5
    cx, sx = ti.cos(hx), ti.sin(hx)
    cy, sy = ti.cos(hy), ti.sin(hy)
    cz, sz = ti.cos(hz), ti.sin(hz)
    w = cx*cy*cz + sx*sy*sz
    x = sx*cy*cz - cx*sy*sz
    y = cx*sy*cz + sx*cy*sz
    z = cx*cy*sz - sx*sy*cz
    return ti.math.vec4(x, y, z, w)

@ti.func
def quat_slerp(qa: ti.math.vec4, qb: ti.math.vec4, t: ti.f32) -> ti.math.vec4:
    dot = qa[0]*qb[0] + qa[1]*qb[1] + qa[2]*qb[2] + qa[3]*qb[3]
    qb2 = qb
    if dot < 0.0:
        qb2 = -qb
        dot = -dot
    result = ti.math.vec4(0.0, 0.0, 0.0, 1.0)
    if dot > 0.9995:
        result = qa + t * (qb2 - qa)
        length = ti.sqrt(result[0]**2 + result[1]**2 + result[2]**2 + result[3]**2)
        result = result / length
    else:
        theta_0 = ti.acos(dot)
        theta = theta_0 * t
        sin_theta   = ti.sin(theta)
        sin_theta_0 = ti.sin(theta_0)
        s0 = ti.cos(theta) - dot * sin_theta / sin_theta_0
        s1 = sin_theta / sin_theta_0
        result = s0 * qa + s1 * qb2
    return result

@ti.func
def quat_to_mat4(q: ti.math.vec4) -> ti.math.mat4:
    x, y, z, w = q[0], q[1], q[2], q[3]
    return ti.Matrix([
        [1.0 - 2*(y*y + z*z),       2*(x*y - z*w),       2*(x*z + y*w), 0.0],
        [      2*(x*y + z*w), 1.0 - 2*(x*x + z*z),       2*(y*z - x*w), 0.0],
        [      2*(x*z - y*w),       2*(y*z + x*w), 1.0 - 2*(x*x + y*y), 0.0],
        [0.0,                 0.0,                 0.0,                  1.0],
    ])

@ti.func
def set_pixel_thick(x: ti.i32, y: ti.i32, color: ti.math.vec3, buf: ti.template()):
    # 2x2 thick pixel
    for dx in ti.static(range(2)):
        for dy in ti.static(range(2)):
            px, py = x + dx, y + dy
            if 0 <= px < WIDTH and 0 <= py < HEIGHT:
                buf[px, py] = color

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
        set_pixel_thick(x, y, color, buf)
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
tri_ndc   = ti.Vector.field(2, dtype=ti.f32, shape=3)
cube_ndc  = ti.Vector.field(2, dtype=ti.f32, shape=8)
interp_ndc = ti.Vector.field(2, dtype=ti.f32, shape=8)

# 插值演示独立像素缓冲
interp_pixels = ti.Vector.field(3, dtype=ti.f32, shape=(WIDTH, HEIGHT))

# ---------- interp_cube 多立方体渲染所需的额外 fields ----------
N_GHOSTS_GEN = 5
MAX_SLOTS_GEN = N_GHOSTS_GEN + 3   # ghostsA + B + anim

# 缩小的立方体顶点（避免距离太远相互遗漏）
IC_SCALE = 0.55
ic_verts = ti.Vector.field(3, dtype=ti.f32, shape=8)
for _ii, _vv in enumerate([(-1,-1,-1),(1,-1,-1),(1,1,-1),(-1,1,-1),
                            (-1,-1, 1),(1,-1, 1),(1,1, 1),(-1,1, 1)]):
    ic_verts[_ii] = ti.Vector([_vv[0]*IC_SCALE, _vv[1]*IC_SCALE, _vv[2]*IC_SCALE])

# slot-based NDC 缓冲（MAX_SLOTS_GEN 个立方体）
ic_ndc   = ti.Vector.field(2, dtype=ti.f32, shape=(MAX_SLOTS_GEN, 8))
ic_color = ti.Vector.field(3, dtype=ti.f32, shape=MAX_SLOTS_GEN)
ic_pixels = ti.Vector.field(3, dtype=ti.f32, shape=(WIDTH, HEIGHT))

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
#  插值渲染：SLERP 在两个姿态之间过渡（旧）
# ================================================================

@ti.kernel
def xform_cube_interp(t: ti.f32,
                      ax_a: ti.f32, ay_a: ti.f32, az_a: ti.f32,
                      ax_b: ti.f32, ay_b: ti.f32, az_b: ti.f32):
    qa = euler_to_quat(ax_a, ay_a, az_a)
    qb = euler_to_quat(ax_b, ay_b, az_b)
    q  = quat_slerp(qa, qb, t)
    model = quat_to_mat4(q)
    mvp = build_proj(45.0, 1.0, 0.1, 50.0) @ build_view(0.0, 0.0, 5.0) @ model
    for i in range(8):
        v = mvp @ ti.math.vec4(cube_vertices[i][0], cube_vertices[i][1], cube_vertices[i][2], 1.0)
        if v[3] != 0.0:
            v /= v[3]
        interp_ndc[i] = ti.math.vec2(v[0], v[1])

@ti.kernel
def draw_cube_interp():
    for i in range(12):
        s = cube_edges[i][0]
        e = cube_edges[i][1]
        x0 = int((interp_ndc[s][0] + 1.0) * WIDTH  / 2.0)
        y0 = int((interp_ndc[s][1] + 1.0) * HEIGHT / 2.0)
        x1 = int((interp_ndc[e][0] + 1.0) * WIDTH  / 2.0)
        y1 = int((interp_ndc[e][1] + 1.0) * HEIGHT / 2.0)
        draw_line_solid(x0, y0, x1, y1, cube_edge_colors[i], interp_pixels)

def render_cube_interp(t: float, pose_a, pose_b):
    xform_cube_interp(t, pose_a[0], pose_a[1], pose_a[2],
                         pose_b[0], pose_b[1], pose_b[2])
    draw_cube_interp()


# ================================================================
#  新式插值渲染： slot式，支持同时画多个立方体
# ================================================================

@ti.func
def build_translate_gen(tx: ti.f32, ty: ti.f32, tz: ti.f32) -> ti.math.mat4:
    return ti.Matrix([
        [1.0, 0.0, 0.0, tx],
        [0.0, 1.0, 0.0, ty],
        [0.0, 0.0, 1.0, tz],
        [0.0, 0.0, 0.0, 1.0],
    ])


@ti.kernel
def ic_xform_slot(slot: ti.i32,
                  qx: ti.f32, qy: ti.f32, qz: ti.f32, qw: ti.f32,
                  tx: ti.f32, ty: ti.f32, tz: ti.f32):
    q     = ti.math.vec4(qx, qy, qz, qw)
    rot   = quat_to_mat4(q)
    trans = build_translate_gen(tx, ty, tz)
    model = trans @ rot
    view  = build_view(0.0, 0.0, 6.0)
    proj  = build_proj(45.0, 1.0, 0.1, 50.0)
    mvp   = proj @ view @ model
    for i in range(8):
        v = mvp @ ti.math.vec4(ic_verts[i][0], ic_verts[i][1], ic_verts[i][2], 1.0)
        if v[3] != 0.0:
            v /= v[3]
        ic_ndc[slot, i] = ti.math.vec2(v[0], v[1])


@ti.kernel
def ic_draw_slot(slot: ti.i32):
    color = ic_color[slot]
    for i in range(12):
        s = cube_edges[i][0]
        e = cube_edges[i][1]
        x0 = int((ic_ndc[slot, s][0] + 1.0) * WIDTH  / 2.0)
        y0 = int((ic_ndc[slot, s][1] + 1.0) * HEIGHT / 2.0)
        x1 = int((ic_ndc[slot, e][0] + 1.0) * WIDTH  / 2.0)
        y1 = int((ic_ndc[slot, e][1] + 1.0) * HEIGHT / 2.0)
        draw_line_solid(x0, y0, x1, y1, color, ic_pixels)


def np_euler_to_quat_gen(ax, ay, az):
    hx, hy, hz = np.radians(ax/2), np.radians(ay/2), np.radians(az/2)
    cx, sx = np.cos(hx), np.sin(hx)
    cy, sy = np.cos(hy), np.sin(hy)
    cz, sz = np.cos(hz), np.sin(hz)
    w = cx*cy*cz + sx*sy*sz
    x = sx*cy*cz - cx*sy*sz
    y = cx*sy*cz + sx*cy*sz
    z = cx*cy*sz - sx*sy*cz
    return np.array([x, y, z, w], dtype=np.float32)


def np_quat_slerp_gen(qa, qb, t):
    dot = float(np.dot(qa, qb))
    if dot < 0.0:
        qb = -qb
        dot = -dot
    if dot > 0.9995:
        r = qa + t * (qb - qa)
        return r / np.linalg.norm(r)
    theta0    = np.arccos(dot)
    theta     = theta0 * t
    sin_theta  = np.sin(theta)
    sin_theta0 = np.sin(theta0)
    s0 = np.cos(theta) - dot * sin_theta / sin_theta0
    s1 = sin_theta / sin_theta0
    return s0 * qa + s1 * qb


def ic_render_slot(slot, q, tx, ty, tz, color):
    ic_xform_slot(slot,
                  float(q[0]), float(q[1]), float(q[2]), float(q[3]),
                  float(tx), float(ty), float(tz))
    ic_color[slot] = ti.Vector(list(color))
    ic_draw_slot(slot)


# ================================================================
#  帧抓取 + GIF 保存
# ================================================================

def grab(buf) -> Image.Image:
    # buf shape is (W, H, 3); PIL needs (H, W, 3) — transpose axes 0 and 1
    arr = (np.clip(buf.to_numpy(), 0.0, 1.0) * 255).astype(np.uint8)
    arr = np.transpose(arr, (1, 0, 2))  # (W,H,3) -> (H,W,3)
    arr = np.flipud(arr)                # flip Y: Taichi y=0 is bottom, PIL y=0 is top
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


def generate_interp_demo():
    """
    生成旋转插值演示动画（新版）：
    - 左侧蓝色立方体：姿态 A
    - 右侧青色立方体：姿态 B
    - 中间灰色幽灵帧：SLERP 路径上均匀采样
    - 金色动画立方体：沿插值路径往复运动
    """
    print("正在生成旋转插值演示动画 (双姿态 + 插值幽灵帧)...")

    POSE_A_ROT   = (0.0,   0.0,  0.0)
    POSE_A_TRANS = (-1.8,  0.0,  0.0)
    POSE_B_ROT   = (25.0, 45.0,  0.0)
    POSE_B_TRANS = ( 1.8,  0.0,  0.0)

    COLOR_A     = (0.3, 0.5, 1.0)
    COLOR_B     = (0.2, 0.9, 0.9)
    COLOR_GHOST = (0.7, 0.7, 0.7)
    COLOR_ANIM  = (1.0, 0.85, 0.2)

    n_half   = 60
    n_frames = n_half * 2
    frames   = []

    qa_rot = np_euler_to_quat_gen(*POSE_A_ROT)
    qb_rot = np_euler_to_quat_gen(*POSE_B_ROT)

    for frame in range(n_frames):
        t = frame / n_half if frame < n_half else 1.0 - (frame - n_half) / n_half
        ic_pixels.fill(0.0)

        # 幽灵帧（先画，在端点下方）
        for gi in range(N_GHOSTS_GEN):
            t_g = (gi + 1) / (N_GHOSTS_GEN + 1)
            q_g = np_quat_slerp_gen(qa_rot, qb_rot, t_g)
            tx_g = POSE_A_TRANS[0] + t_g * (POSE_B_TRANS[0] - POSE_A_TRANS[0])
            ty_g = POSE_A_TRANS[1] + t_g * (POSE_B_TRANS[1] - POSE_A_TRANS[1])
            tz_g = POSE_A_TRANS[2] + t_g * (POSE_B_TRANS[2] - POSE_A_TRANS[2])
            ic_render_slot(2 + gi, q_g, tx_g, ty_g, tz_g, COLOR_GHOST)

        # 姿态 A（蓝色）
        ic_render_slot(0, qa_rot, POSE_A_TRANS[0], POSE_A_TRANS[1], POSE_A_TRANS[2], COLOR_A)
        # 姿态 B（青色）
        ic_render_slot(1, qb_rot, POSE_B_TRANS[0], POSE_B_TRANS[1], POSE_B_TRANS[2], COLOR_B)

        # 动画立方体（金色）
        q_anim = np_quat_slerp_gen(qa_rot, qb_rot, t)
        tx_a = POSE_A_TRANS[0] + t * (POSE_B_TRANS[0] - POSE_A_TRANS[0])
        ty_a = POSE_A_TRANS[1] + t * (POSE_B_TRANS[1] - POSE_A_TRANS[1])
        tz_a = POSE_A_TRANS[2] + t * (POSE_B_TRANS[2] - POSE_A_TRANS[2])
        ic_render_slot(2 + N_GHOSTS_GEN, q_anim, tx_a, ty_a, tz_a, COLOR_ANIM)

        frames.append(grab(ic_pixels))
        if (frame + 1) % 30 == 0:
            print(f"  {frame + 1}/{n_frames} 帧  (t={t:.2f})")

    save_gif(frames, 'interp_demo.gif', duration=55)
    print("✓ interp_demo.gif 已保存")


if __name__ == '__main__':
    print("=" * 50)
    print("MVP 变换演示动画生成器")
    print("=" * 50)
    generate_triangle_demo()
    generate_cube_demo()
    generate_interp_demo()
    print("=" * 50)
    print("所有演示动画生成完成！")
    print("=" * 50)
