import taichi as ti
import numpy as np
import math

ti.init(arch=ti.cpu)

# 窗口尺寸
width, height = 700, 700

# 定义立方体的8个顶点(中心在原点,边长为2)
vertices = ti.Vector.field(3, dtype=ti.f32, shape=8)
vertices[0] = ti.Vector([-1.0, -1.0, -1.0])  # 后左下
vertices[1] = ti.Vector([1.0, -1.0, -1.0])   # 后右下
vertices[2] = ti.Vector([1.0, 1.0, -1.0])    # 后右上
vertices[3] = ti.Vector([-1.0, 1.0, -1.0])   # 后左上
vertices[4] = ti.Vector([-1.0, -1.0, 1.0])   # 前左下
vertices[5] = ti.Vector([1.0, -1.0, 1.0])    # 前右下
vertices[6] = ti.Vector([1.0, 1.0, 1.0])     # 前右上
vertices[7] = ti.Vector([-1.0, 1.0, 1.0])    # 前左上

# 定义立方体的12条边(每条边用两个顶点索引表示)
edges = ti.Vector.field(2, dtype=ti.i32, shape=12)
# 后面的4条边
edges[0] = ti.Vector([0, 1])
edges[1] = ti.Vector([1, 2])
edges[2] = ti.Vector([2, 3])
edges[3] = ti.Vector([3, 0])
# 前面的4条边
edges[4] = ti.Vector([4, 5])
edges[5] = ti.Vector([5, 6])
edges[6] = ti.Vector([6, 7])
edges[7] = ti.Vector([7, 4])
# 连接前后面的4条边
edges[8] = ti.Vector([0, 4])
edges[9] = ti.Vector([1, 5])
edges[10] = ti.Vector([2, 6])
edges[11] = ti.Vector([3, 7])

# 边的颜色
edge_colors = ti.Vector.field(3, dtype=ti.f32, shape=12)
# 后面 - 蓝色系
edge_colors[0] = ti.Vector([0.3, 0.3, 1.0])
edge_colors[1] = ti.Vector([0.4, 0.4, 1.0])
edge_colors[2] = ti.Vector([0.5, 0.5, 1.0])
edge_colors[3] = ti.Vector([0.6, 0.6, 1.0])
# 前面 - 红色系
edge_colors[4] = ti.Vector([1.0, 0.3, 0.3])
edge_colors[5] = ti.Vector([1.0, 0.4, 0.4])
edge_colors[6] = ti.Vector([1.0, 0.5, 0.5])
edge_colors[7] = ti.Vector([1.0, 0.6, 0.6])
# 连接边 - 绿色系
edge_colors[8] = ti.Vector([0.3, 1.0, 0.3])
edge_colors[9] = ti.Vector([0.4, 1.0, 0.4])
edge_colors[10] = ti.Vector([0.5, 1.0, 0.5])
edge_colors[11] = ti.Vector([0.6, 1.0, 0.6])

# NDC 中间缓冲（全局 field，不能在 kernel 内部声明 field）
ndc = ti.Vector.field(2, dtype=ti.f32, shape=8)

# ---- 插值模式：两个姿态的欧拉角 (角度制) ----
# 姿态 A: 正视方向
POSE_A = (0.0, 0.0, 0.0)
# 姿态 B: 绕 X 45°, 绕 Y 60°, 绕 Z 30°
POSE_B = (45.0, 60.0, 30.0)


@ti.func
def get_model_matrix(angle_x: ti.f32, angle_y: ti.f32, angle_z: ti.f32) -> ti.math.mat4:
    """
    返回组合旋转的模型变换矩阵(绕X、Y、Z轴)
    
    参数:
        angle_x: 绕X轴旋转角度(角度制)
        angle_y: 绕Y轴旋转角度(角度制)
        angle_z: 绕Z轴旋转角度(角度制)
    
    返回:
        4x4 齐次坐标变换矩阵
    """
    # 将角度转换为弧度
    rad_x = angle_x * 3.14159265358979323846 / 180.0
    rad_y = angle_y * 3.14159265358979323846 / 180.0
    rad_z = angle_z * 3.14159265358979323846 / 180.0
    
    cos_x, sin_x = ti.cos(rad_x), ti.sin(rad_x)
    cos_y, sin_y = ti.cos(rad_y), ti.sin(rad_y)
    cos_z, sin_z = ti.cos(rad_z), ti.sin(rad_z)
    
    # 绕X轴旋转
    rot_x = ti.Matrix([
        [1.0,   0.0,    0.0,   0.0],
        [0.0,   cos_x, -sin_x, 0.0],
        [0.0,   sin_x,  cos_x, 0.0],
        [0.0,   0.0,    0.0,   1.0]
    ])
    
    # 绕Y轴旋转
    rot_y = ti.Matrix([
        [ cos_y, 0.0, sin_y, 0.0],
        [ 0.0,   1.0, 0.0,   0.0],
        [-sin_y, 0.0, cos_y, 0.0],
        [ 0.0,   0.0, 0.0,   1.0]
    ])
    
    # 绕Z轴旋转
    rot_z = ti.Matrix([
        [cos_z, -sin_z, 0.0, 0.0],
        [sin_z,  cos_z, 0.0, 0.0],
        [0.0,    0.0,   1.0, 0.0],
        [0.0,    0.0,   0.0, 1.0]
    ])
    
    # 组合旋转: Z * Y * X
    return rot_z @ rot_y @ rot_x


@ti.func
def get_view_matrix(eye_pos: ti.math.vec3) -> ti.math.mat4:
    """
    返回视图变换矩阵,将相机平移至原点
    
    参数:
        eye_pos: 相机位置(三维向量)
    
    返回:
        4x4 视图变换矩阵
    """
    return ti.Matrix([
        [1.0, 0.0, 0.0, -eye_pos[0]],
        [0.0, 1.0, 0.0, -eye_pos[1]],
        [0.0, 0.0, 1.0, -eye_pos[2]],
        [0.0, 0.0, 0.0, 1.0]
    ])


@ti.func
def get_projection_matrix(eye_fov: ti.f32, aspect_ratio: ti.f32, zNear: ti.f32, zFar: ti.f32) -> ti.math.mat4:
    """
    返回透视投影矩阵
    
    参数:
        eye_fov: 视场角(Y轴方向,角度制)
        aspect_ratio: 屏幕长宽比
        zNear: 近截面距离(正值)
        zFar: 远截面距离(正值)
    
    返回:
        4x4 透视投影矩阵
    """
    fov_rad = eye_fov * 3.14159265358979323846 / 180.0
    
    n = -zNear
    f = -zFar
    
    t = ti.tan(fov_rad / 2.0) * ti.abs(n)
    b = -t
    r = aspect_ratio * t
    l = -r
    
    persp_to_ortho = ti.Matrix([
        [n,   0.0, 0.0,    0.0],
        [0.0, n,   0.0,    0.0],
        [0.0, 0.0, n + f, -n * f],
        [0.0, 0.0, 1.0,    0.0]
    ])
    
    ortho_scale = ti.Matrix([
        [2.0 / (r - l), 0.0,           0.0,           0.0],
        [0.0,           2.0 / (t - b), 0.0,           0.0],
        [0.0,           0.0,           2.0 / (n - f), 0.0],
        [0.0,           0.0,           0.0,           1.0]
    ])
    
    ortho_translate = ti.Matrix([
        [1.0, 0.0, 0.0, -(r + l) / 2.0],
        [0.0, 1.0, 0.0, -(t + b) / 2.0],
        [0.0, 0.0, 1.0, -(n + f) / 2.0],
        [0.0, 0.0, 0.0, 1.0]
    ])
    
    ortho = ortho_scale @ ortho_translate
    
    return ortho @ persp_to_ortho


@ti.kernel
def xform_cube(angle_x: ti.f32, angle_y: ti.f32, angle_z: ti.f32):
    """
    第一步：对立方体顶点执行 MVP 变换，结果写入全局 ndc field
    """
    eye_pos = ti.math.vec3(0.0, 0.0, 5.0)
    model = get_model_matrix(angle_x, angle_y, angle_z)
    view = get_view_matrix(eye_pos)
    projection = get_projection_matrix(45.0, 1.0, 0.1, 50.0)
    mvp = projection @ view @ model

    for i in range(8):
        v = ti.math.vec4(vertices[i][0], vertices[i][1], vertices[i][2], 1.0)
        v_t = mvp @ v
        if v_t[3] != 0.0:
            v_t /= v_t[3]
        ndc[i] = ti.math.vec2(v_t[0], v_t[1])


@ti.func
def euler_to_quat(ax: ti.f32, ay: ti.f32, az: ti.f32) -> ti.math.vec4:
    """
    将欧拉角(弧度)转换为四元数 (x, y, z, w)
    旋转顺序: Z -> Y -> X
    """
    PI = 3.14159265358979323846
    hx, hy, hz = ax * PI / 180.0 * 0.5, ay * PI / 180.0 * 0.5, az * PI / 180.0 * 0.5
    cx, sx = ti.cos(hx), ti.sin(hx)
    cy, sy = ti.cos(hy), ti.sin(hy)
    cz, sz = ti.cos(hz), ti.sin(hz)
    # ZYX 组合四元数
    w = cx*cy*cz + sx*sy*sz
    x = sx*cy*cz - cx*sy*sz
    y = cx*sy*cz + sx*cy*sz
    z = cx*cy*sz - sx*sy*cz
    return ti.math.vec4(x, y, z, w)


@ti.func
def quat_slerp(qa: ti.math.vec4, qb: ti.math.vec4, t: ti.f32) -> ti.math.vec4:
    """
    球面线性插值 SLERP
    """
    dot = qa[0]*qb[0] + qa[1]*qb[1] + qa[2]*qb[2] + qa[3]*qb[3]
    # 保证最短路径
    qb2 = qb
    if dot < 0.0:
        qb2 = -qb
        dot = -dot
    result = ti.math.vec4(0.0, 0.0, 0.0, 1.0)
    if dot > 0.9995:
        # 接近平行时退化为线性插值
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
    """
    四元数转 4x4 旋转矩阵
    """
    x, y, z, w = q[0], q[1], q[2], q[3]
    return ti.Matrix([
        [1.0 - 2*(y*y + z*z),       2*(x*y - z*w),       2*(x*z + y*w), 0.0],
        [      2*(x*y + z*w), 1.0 - 2*(x*x + z*z),       2*(y*z - x*w), 0.0],
        [      2*(x*z - y*w),       2*(y*z + x*w), 1.0 - 2*(x*x + y*y), 0.0],
        [0.0,                 0.0,                 0.0,                  1.0]
    ])


@ti.kernel
def xform_cube_interp(t: ti.f32,
                      ax_a: ti.f32, ay_a: ti.f32, az_a: ti.f32,
                      ax_b: ti.f32, ay_b: ti.f32, az_b: ti.f32):
    """
    插值模式：在姿态 A 和 B 之间用 SLERP 插值，t∈[0,1]
    """
    qa = euler_to_quat(ax_a, ay_a, az_a)
    qb = euler_to_quat(ax_b, ay_b, az_b)
    q  = quat_slerp(qa, qb, t)
    model = quat_to_mat4(q)

    eye_pos = ti.math.vec3(0.0, 0.0, 5.0)
    view       = get_view_matrix(eye_pos)
    projection = get_projection_matrix(45.0, 1.0, 0.1, 50.0)
    mvp = projection @ view @ model

    for i in range(8):
        v   = ti.math.vec4(vertices[i][0], vertices[i][1], vertices[i][2], 1.0)
        v_t = mvp @ v
        if v_t[3] != 0.0:
            v_t /= v_t[3]
        ndc[i] = ti.math.vec2(v_t[0], v_t[1])


@ti.kernel
def draw_cube(pixels: ti.template()):
    """
    第二步：读取 ndc field，将 12 条边用 Bresenham 算法绘制到像素缓冲
    """
    for i in range(12):
        s = edges[i][0]
        e = edges[i][1]
        x0 = int((ndc[s][0] + 1.0) * width  / 2.0)
        y0 = int((ndc[s][1] + 1.0) * height / 2.0)
        x1 = int((ndc[e][0] + 1.0) * width  / 2.0)
        y1 = int((ndc[e][1] + 1.0) * height / 2.0)
        draw_line(x0, y0, x1, y1, edge_colors[i], pixels)


@ti.func
def draw_line(x0: ti.i32, y0: ti.i32, x1: ti.i32, y1: ti.i32, 
              color: ti.math.vec3, pixels: ti.template()):
    """
    使用Bresenham算法绘制线段
    """
    dx = ti.abs(x1 - x0)
    dy = ti.abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx - dy
    
    x, y = x0, y0
    
    while True:
        if 0 <= x < width and 0 <= y < height:
            pixels[x, y] = color
        
        if x == x1 and y == y1:
            break
        
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x += sx
        if e2 < dx:
            err += dx
            y += sy


def main():
    """主函数"""
    gui = ti.GUI('MVP Transformation - Cube', res=(width, height))
    
    pixels = ti.Vector.field(3, dtype=ti.f32, shape=(width, height))
    
    angle_x = 0.0
    angle_y = 0.0
    angle_z = 0.0
    
    auto_rotate = True

    # ---- 插值模式状态 ----
    interp_mode  = False   # True: SLERP 插值模式; False: 自由旋转模式
    interp_t     = 0.0     # 插值参数 [0, 1]
    interp_dir   = 1.0     # 当前插值方向 (+1 或 -1)
    interp_speed = 0.008   # 每帧步进量
    
    print("控制说明:")
    print("  W/S键: 绕X轴旋转")
    print("  A/D键: 绕Y轴旋转")
    print("  Q/E键: 绕Z轴旋转")
    print("  空格键: 切换自动旋转")
    print("  I键:   切换插值模式 (在姿态A/B之间 SLERP 过渡)")
    print("  ESC键: 退出程序")
    print(f"  姿态A: X={POSE_A[0]}° Y={POSE_A[1]}° Z={POSE_A[2]}°")
    print(f"  姿态B: X={POSE_B[0]}° Y={POSE_B[1]}° Z={POSE_B[2]}°")
    
    while gui.running:
        for e in gui.get_events(ti.GUI.PRESS):
            if e.key == ti.GUI.ESCAPE:
                gui.running = False
            elif e.key == ti.GUI.SPACE:
                if not interp_mode:
                    auto_rotate = not auto_rotate
                    print(f"自动旋转: {'开启' if auto_rotate else '关闭'}")
            elif e.key == 'i':
                interp_mode = not interp_mode
                if interp_mode:
                    interp_t   = 0.0
                    interp_dir = 1.0
                    print("切换到插值模式：SLERP 姿态A→B→A 往复过渡")
                else:
                    print("切换到自由旋转模式")
        
        pixels.fill(0.0)

        if interp_mode:
            # ---- 插值模式：SLERP ping-pong ----
            interp_t += interp_dir * interp_speed
            if interp_t >= 1.0:
                interp_t   = 1.0
                interp_dir = -1.0
            elif interp_t <= 0.0:
                interp_t   = 0.0
                interp_dir = 1.0
            xform_cube_interp(
                interp_t,
                POSE_A[0], POSE_A[1], POSE_A[2],
                POSE_B[0], POSE_B[1], POSE_B[2]
            )
            # 在屏幕上显示当前插值进度
            gui.text(f"插值模式  t = {interp_t:.2f}  (I 键切换)",
                     pos=(0.02, 0.97), font_size=18, color=0xFFFFFF)
            gui.text(f"姿态A: ({POSE_A[0]:.0f},{POSE_A[1]:.0f},{POSE_A[2]:.0f})  →  "
                     f"姿态B: ({POSE_B[0]:.0f},{POSE_B[1]:.0f},{POSE_B[2]:.0f})",
                     pos=(0.02, 0.93), font_size=16, color=0xCCCCCC)
        else:
            # ---- 自由旋转模式 ----
            if gui.is_pressed('w'):
                angle_x += 1.0
            if gui.is_pressed('s'):
                angle_x -= 1.0
            if gui.is_pressed('a'):
                angle_y += 1.0
            if gui.is_pressed('d'):
                angle_y -= 1.0
            if gui.is_pressed('q'):
                angle_z += 1.0
            if gui.is_pressed('e'):
                angle_z -= 1.0
            if auto_rotate:
                angle_x += 0.5
                angle_y += 0.7
                angle_z += 0.3
            xform_cube(angle_x, angle_y, angle_z)
            gui.text("自由旋转模式  (I 键切换插值模式)",
                     pos=(0.02, 0.97), font_size=18, color=0xFFFFFF)

        draw_cube(pixels)
        gui.set_image(pixels)
        gui.show()


if __name__ == '__main__':
    main()
