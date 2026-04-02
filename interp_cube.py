"""
interp_cube.py
==============
展示旋转插值（SLERP）的可视化程序：
- 左侧蓝色立方体：姿态 A（初始位置）
- 右侧青色立方体：姿态 B（目标位置）
- 中间半透明幽灵帧：SLERP 插值路径上均匀采样的中间姿态
- 一个高亮立方体沿插值路径自动动画
按 ESC 退出，按空格键暂停/继续动画。
"""
import taichi as ti
import numpy as np

ti.init(arch=ti.cpu)

WIDTH, HEIGHT = 700, 700
PI = 3.14159265358979323846

# ---- 姿态定义 ----
# 姿态 A：左侧，平移 (-1.8, 0, 0)，无旋转
POSE_A_ROT   = (0.0,   0.0,  0.0)
POSE_A_TRANS = (-1.8,  0.0,  0.0)

# 姿态 B：右侧，平移 (+1.8, 0, 0)，绕 Y 轴旋转 45°，绕 X 轴旋转 25°
POSE_B_ROT   = (25.0, 45.0,  0.0)
POSE_B_TRANS = ( 1.8,  0.0,  0.0)

# 插值幽灵帧数量（不含端点）
N_GHOSTS = 5
# 动画立方体总帧数（A→B→A 往复）
ANIM_HALF = 120

# ---- 立方体拓扑 ----
CUBE_VERTS_NP = np.array([
    [-1,-1,-1],[1,-1,-1],[1,1,-1],[-1,1,-1],
    [-1,-1, 1],[1,-1, 1],[1,1, 1],[-1,1, 1],
], dtype=np.float32) * 0.55   # 缩小一点，避免重叠

CUBE_EDGES_NP = np.array([
    [0,1],[1,2],[2,3],[3,0],   # 后面
    [4,5],[5,6],[6,7],[7,4],   # 前面
    [0,4],[1,5],[2,6],[3,7],   # 连接
], dtype=np.int32)

N_VERTS = 8
N_EDGES = 12

# ---- 颜色定义 ----
COLOR_A     = (0.3, 0.5, 1.0)   # 蓝色：姿态 A
COLOR_B     = (0.2, 0.9, 0.9)   # 青色：姿态 B
COLOR_GHOST = (0.75, 0.75, 0.75) # 灰色：幽灵中间帧
COLOR_ANIM  = (1.0, 0.85, 0.2)  # 金色：动画立方体

# ---- Taichi Fields ----
cube_verts = ti.Vector.field(3, dtype=ti.f32, shape=N_VERTS)
cube_edges = ti.Vector.field(2, dtype=ti.i32, shape=N_EDGES)
pixels     = ti.Vector.field(3, dtype=ti.f32, shape=(WIDTH, HEIGHT))

# NDC 缓冲：每帧最多需要渲染 (N_GHOSTS + 3) 个立方体
# 用 shape=(max_cubes, N_VERTS) 统一存储
MAX_CUBES = N_GHOSTS + 3   # ghost + A + B + anim
ndc_buf   = ti.Vector.field(2, dtype=ti.f32, shape=(MAX_CUBES, N_VERTS))
color_buf = ti.Vector.field(3, dtype=ti.f32, shape=MAX_CUBES)

# 初始化几何数据
for i in range(N_VERTS):
    cube_verts[i] = ti.Vector(CUBE_VERTS_NP[i].tolist())
for i in range(N_EDGES):
    cube_edges[i] = ti.Vector(CUBE_EDGES_NP[i].tolist())


# ================================================================
#  数学工具函数
# ================================================================

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
        ln = ti.sqrt(result[0]**2 + result[1]**2 + result[2]**2 + result[3]**2)
        result = result / ln
    else:
        theta0    = ti.acos(dot)
        theta     = theta0 * t
        sin_theta  = ti.sin(theta)
        sin_theta0 = ti.sin(theta0)
        s0 = ti.cos(theta) - dot * sin_theta / sin_theta0
        s1 = sin_theta / sin_theta0
        result = s0 * qa + s1 * qb2
    return result


@ti.func
def quat_to_mat4(q: ti.math.vec4) -> ti.math.mat4:
    x, y, z, w = q[0], q[1], q[2], q[3]
    return ti.Matrix([
        [1.0-2*(y*y+z*z),   2*(x*y-z*w),   2*(x*z+y*w), 0.0],
        [  2*(x*y+z*w), 1.0-2*(x*x+z*z),   2*(y*z-x*w), 0.0],
        [  2*(x*z-y*w),   2*(y*z+x*w), 1.0-2*(x*x+y*y), 0.0],
        [0.0,            0.0,            0.0,             1.0],
    ])


@ti.func
def build_translate(tx: ti.f32, ty: ti.f32, tz: ti.f32) -> ti.math.mat4:
    return ti.Matrix([
        [1.0, 0.0, 0.0, tx],
        [0.0, 1.0, 0.0, ty],
        [0.0, 0.0, 1.0, tz],
        [0.0, 0.0, 0.0, 1.0],
    ])


@ti.func
def build_view(ez: ti.f32) -> ti.math.mat4:
    return ti.Matrix([
        [1.0, 0.0, 0.0,  0.0],
        [0.0, 1.0, 0.0,  0.0],
        [0.0, 0.0, 1.0, -ez],
        [0.0, 0.0, 0.0,  1.0],
    ])


@ti.func
def build_proj(fov_deg: ti.f32, aspect: ti.f32, zn: ti.f32, zf: ti.f32) -> ti.math.mat4:
    fov = fov_deg * PI / 180.0
    n, f = -zn, -zf
    t_  = ti.tan(fov / 2.0) * ti.abs(n)
    b_, r_, l_ = -t_, aspect * t_, -aspect * t_
    persp = ti.Matrix([
        [n,   0.0, 0.0,         0.0],
        [0.0, n,   0.0,         0.0],
        [0.0, 0.0, n+f,        -n*f],
        [0.0, 0.0, 1.0,         0.0],
    ])
    scale = ti.Matrix([
        [2.0/(r_-l_), 0.0,          0.0,          0.0],
        [0.0,         2.0/(t_-b_),  0.0,          0.0],
        [0.0,         0.0,          2.0/(n-f),    0.0],
        [0.0,         0.0,          0.0,          1.0],
    ])
    trans = ti.Matrix([
        [1.0, 0.0, 0.0, -(r_+l_)/2.0],
        [0.0, 1.0, 0.0, -(t_+b_)/2.0],
        [0.0, 0.0, 1.0, -(n+f)/2.0  ],
        [0.0, 0.0, 0.0,  1.0        ],
    ])
    return scale @ trans @ persp


# ================================================================
#  变换 kernel：将一组立方体写入 ndc_buf
#  cube_idx : 写入 ndc_buf 的行索引
#  rot_q    : 旋转四元数
#  tx,ty,tz : 平移
# ================================================================

@ti.kernel
def xform_cube_to_slot(cube_idx: ti.i32,
                       qx: ti.f32, qy: ti.f32, qz: ti.f32, qw: ti.f32,
                       tx: ti.f32, ty: ti.f32, tz: ti.f32):
    q     = ti.math.vec4(qx, qy, qz, qw)
    rot   = quat_to_mat4(q)
    trans = build_translate(tx, ty, tz)
    model = trans @ rot
    view  = build_view(6.0)
    proj  = build_proj(45.0, 1.0, 0.1, 50.0)
    mvp   = proj @ view @ model
    for i in range(N_VERTS):
        v = mvp @ ti.math.vec4(cube_verts[i][0], cube_verts[i][1], cube_verts[i][2], 1.0)
        if v[3] != 0.0:
            v /= v[3]
        ndc_buf[cube_idx, i] = ti.math.vec2(v[0], v[1])


# ================================================================
#  绘制 kernel：将 ndc_buf[cube_idx] 的边画入 pixels
# ================================================================

@ti.func
def draw_line(x0: ti.i32, y0: ti.i32, x1: ti.i32, y1: ti.i32,
              color: ti.math.vec3, thick: ti.i32):
    dx = ti.abs(x1 - x0)
    dy = ti.abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx - dy
    x, y = x0, y0
    while True:
        for ddx in range(-thick, thick + 1):
            for ddy in range(-thick, thick + 1):
                px, py = x + ddx, y + ddy
                if 0 <= px < WIDTH and 0 <= py < HEIGHT:
                    pixels[px, py] = color
        if x == x1 and y == y1:
            break
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x += sx
        if e2 < dx:
            err += dx
            y += sy


@ti.kernel
def draw_slot(cube_idx: ti.i32, thick: ti.i32):
    color = color_buf[cube_idx]
    for i in range(N_EDGES):
        s = cube_edges[i][0]
        e = cube_edges[i][1]
        x0 = int((ndc_buf[cube_idx, s][0] + 1.0) * WIDTH  / 2.0)
        y0 = int((ndc_buf[cube_idx, s][1] + 1.0) * HEIGHT / 2.0)
        x1 = int((ndc_buf[cube_idx, e][0] + 1.0) * WIDTH  / 2.0)
        y1 = int((ndc_buf[cube_idx, e][1] + 1.0) * HEIGHT / 2.0)
        draw_line(x0, y0, x1, y1, color, thick)


# ================================================================
#  Python 辅助：计算插值参数并驱动 kernel
# ================================================================

def np_euler_to_quat(ax, ay, az):
    hx, hy, hz = np.radians(ax/2), np.radians(ay/2), np.radians(az/2)
    cx, sx = np.cos(hx), np.sin(hx)
    cy, sy = np.cos(hy), np.sin(hy)
    cz, sz = np.cos(hz), np.sin(hz)
    w = cx*cy*cz + sx*sy*sz
    x = sx*cy*cz - cx*sy*sz
    y = cx*sy*cz + sx*cy*sz
    z = cx*cy*sz - sx*sy*cz
    return np.array([x, y, z, w], dtype=np.float32)


def np_quat_slerp(qa, qb, t):
    dot = np.dot(qa, qb)
    if dot < 0.0:
        qb = -qb
        dot = -dot
    if dot > 0.9995:
        result = qa + t * (qb - qa)
        return result / np.linalg.norm(result)
    theta0    = np.arccos(dot)
    theta     = theta0 * t
    sin_theta  = np.sin(theta)
    sin_theta0 = np.sin(theta0)
    s0 = np.cos(theta) - dot * sin_theta / sin_theta0
    s1 = sin_theta / sin_theta0
    return s0 * qa + s1 * qb


def render_cube_slot(slot_idx, qa, tx_a, ty_a, tz_a, qa_arr=None):
    """将四元数+平移写入 ndc_buf[slot_idx]"""
    q = qa_arr if qa_arr is not None else qa
    xform_cube_to_slot(slot_idx,
                       float(q[0]), float(q[1]), float(q[2]), float(q[3]),
                       float(tx_a), float(ty_a), float(tz_a))


def set_slot_color(slot_idx, color):
    color_buf[slot_idx] = ti.Vector(list(color))


def render_frame(anim_t: float):
    """
    渲染一帧：
    slot 0        = 姿态 A（蓝色，厚线）
    slot 1        = 姿态 B（青色，厚线）
    slot 2..N+1   = 幽灵插值帧（灰色，细线）
    slot N+2      = 动画立方体（金色，中线）
    """
    pixels.fill(0.0)

    qa_rot = np_euler_to_quat(*POSE_A_ROT)
    qb_rot = np_euler_to_quat(*POSE_B_ROT)

    # ---- 幽灵帧（先画，避免遮挡端点）----
    for gi in range(N_GHOSTS):
        t_g = (gi + 1) / (N_GHOSTS + 1)   # 均匀采样，不含端点
        q_g = np_quat_slerp(qa_rot, qb_rot, t_g)
        # 位置在 A、B 之间线性插值
        tx_g = POSE_A_TRANS[0] + t_g * (POSE_B_TRANS[0] - POSE_A_TRANS[0])
        ty_g = POSE_A_TRANS[1] + t_g * (POSE_B_TRANS[1] - POSE_A_TRANS[1])
        tz_g = POSE_A_TRANS[2] + t_g * (POSE_B_TRANS[2] - POSE_A_TRANS[2])
        slot = 2 + gi
        render_cube_slot(slot, None, tx_g, ty_g, tz_g, qa_arr=q_g)
        set_slot_color(slot, COLOR_GHOST)
        draw_slot(slot, 0)

    # ---- 姿态 A（蓝色）----
    render_cube_slot(0, None, POSE_A_TRANS[0], POSE_A_TRANS[1], POSE_A_TRANS[2],
                     qa_arr=qa_rot)
    set_slot_color(0, COLOR_A)
    draw_slot(0, 1)

    # ---- 姿态 B（青色）----
    render_cube_slot(1, None, POSE_B_TRANS[0], POSE_B_TRANS[1], POSE_B_TRANS[2],
                     qa_arr=qb_rot)
    set_slot_color(1, COLOR_B)
    draw_slot(1, 1)

    # ---- 动画立方体（金色，沿插值路径运动）----
    q_anim = np_quat_slerp(qa_rot, qb_rot, anim_t)
    tx_anim = POSE_A_TRANS[0] + anim_t * (POSE_B_TRANS[0] - POSE_A_TRANS[0])
    ty_anim = POSE_A_TRANS[1] + anim_t * (POSE_B_TRANS[1] - POSE_A_TRANS[1])
    tz_anim = POSE_A_TRANS[2] + anim_t * (POSE_B_TRANS[2] - POSE_A_TRANS[2])
    anim_slot = 2 + N_GHOSTS
    render_cube_slot(anim_slot, None, tx_anim, ty_anim, tz_anim, qa_arr=q_anim)
    set_slot_color(anim_slot, COLOR_ANIM)
    draw_slot(anim_slot, 1)


# ================================================================
#  主循环
# ================================================================

def main():
    gui = ti.GUI('Rotation Interpolation (SLERP)', res=(WIDTH, HEIGHT))

    anim_frame = 0
    paused     = False
    anim_dir   = 1   # +1 or -1

    print("控制说明:")
    print("  空格键: 暂停/继续动画")
    print("  ESC键:  退出程序")
    print(f"  姿态A (蓝色): 旋转{POSE_A_ROT}, 位置{POSE_A_TRANS}")
    print(f"  姿态B (青色): 旋转{POSE_B_ROT}, 位置{POSE_B_TRANS}")
    print(f"  幽灵插值帧 (灰色): {N_GHOSTS} 帧均匀采样")
    print("  金色立方体: 沿 SLERP 路径往复运动")

    while gui.running:
        for e in gui.get_events(ti.GUI.PRESS):
            if e.key == ti.GUI.ESCAPE:
                gui.running = False
            elif e.key == ti.GUI.SPACE:
                paused = not paused
                print(f"{'暂停' if paused else '继续'}")

        if not paused:
            anim_frame += anim_dir
            if anim_frame >= ANIM_HALF:
                anim_frame = ANIM_HALF
                anim_dir = -1
            elif anim_frame <= 0:
                anim_frame = 0
                anim_dir = 1

        anim_t = anim_frame / ANIM_HALF

        render_frame(anim_t)

        gui.set_image(pixels)
        gui.text(f"SLERP 插值  t = {anim_t:.2f}",
                 pos=(0.02, 0.97), font_size=18, color=0xFFFFFF)
        gui.text(f"蓝=姿态A  青=姿态B  灰=插值路径  金=动画",
                 pos=(0.02, 0.93), font_size=16, color=0xCCCCCC)
        gui.show()


if __name__ == '__main__':
    main()
