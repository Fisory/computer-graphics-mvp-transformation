import taichi as ti
import numpy as np

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
    rot_x = ti.math.mat4(
        [1.0, 0.0, 0.0, 0.0],
        [0.0, cos_x, sin_x, 0.0],
        [0.0, -sin_x, cos_x, 0.0],
        [0.0, 0.0, 0.0, 1.0]
    )
    
    # 绕Y轴旋转
    rot_y = ti.math.mat4(
        [cos_y, 0.0, -sin_y, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [sin_y, 0.0, cos_y, 0.0],
        [0.0, 0.0, 0.0, 1.0]
    )
    
    # 绕Z轴旋转
    rot_z = ti.math.mat4(
        [cos_z, sin_z, 0.0, 0.0],
        [-sin_z, cos_z, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0]
    )
    
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
    return ti.math.mat4(
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [-eye_pos[0], -eye_pos[1], -eye_pos[2], 1.0]
    )


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
    
    persp_to_ortho = ti.math.mat4(
        [n, 0.0, 0.0, 0.0],
        [0.0, n, 0.0, 0.0],
        [0.0, 0.0, n + f, 1.0],
        [0.0, 0.0, -n * f, 0.0]
    )
    
    ortho_scale = ti.math.mat4(
        [2.0 / (r - l), 0.0, 0.0, 0.0],
        [0.0, 2.0 / (t - b), 0.0, 0.0],
        [0.0, 0.0, 2.0 / (n - f), 0.0],
        [0.0, 0.0, 0.0, 1.0]
    )
    
    ortho_translate = ti.math.mat4(
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [-(r + l) / 2.0, -(t + b) / 2.0, -(n + f) / 2.0, 1.0]
    )
    
    ortho = ortho_scale @ ortho_translate
    
    return ortho @ persp_to_ortho


@ti.kernel
def transform_and_draw(angle_x: ti.f32, angle_y: ti.f32, angle_z: ti.f32, pixels: ti.template()):
    """
    对立方体顶点进行MVP变换并绘制到屏幕
    """
    eye_pos = ti.math.vec3(0.0, 0.0, 5.0)
    
    model = get_model_matrix(angle_x, angle_y, angle_z)
    view = get_view_matrix(eye_pos)
    projection = get_projection_matrix(45.0, 1.0, 0.1, 50.0)
    
    mvp = projection @ view @ model
    
    # 变换后的顶点
    transformed = ti.Vector.field(3, dtype=ti.f32, shape=8)
    
    # 对每个顶点进行变换
    for i in range(8):
        v = ti.math.vec4(vertices[i][0], vertices[i][1], vertices[i][2], 1.0)
        v_transformed = mvp @ v
        
        if v_transformed[3] != 0.0:
            v_transformed /= v_transformed[3]
        
        transformed[i] = ti.math.vec3(v_transformed[0], v_transformed[1], v_transformed[2])
    
    # 绘制立方体的12条边
    for i in range(12):
        start_idx = edges[i][0]
        end_idx = edges[i][1]
        
        # NDC坐标转换到屏幕坐标
        x0 = int((transformed[start_idx][0] + 1.0) * width / 2.0)
        y0 = int((transformed[start_idx][1] + 1.0) * height / 2.0)
        x1 = int((transformed[end_idx][0] + 1.0) * width / 2.0)
        y1 = int((transformed[end_idx][1] + 1.0) * height / 2.0)
        
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
    
    print("控制说明:")
    print("  W/S键: 绕X轴旋转")
    print("  A/D键: 绕Y轴旋转")
    print("  Q/E键: 绕Z轴旋转")
    print("  空格键: 切换自动旋转")
    print("  ESC键: 退出程序")
    
    while gui.running:
        for e in gui.get_events(ti.GUI.PRESS):
            if e.key == ti.GUI.ESCAPE:
                gui.running = False
            elif e.key == ti.GUI.SPACE:
                auto_rotate = not auto_rotate
                print(f"自动旋转: {'开启' if auto_rotate else '关闭'}")
        
        # 手动控制
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
        
        # 自动旋转
        if auto_rotate:
            angle_x += 0.5
            angle_y += 0.7
            angle_z += 0.3
        
        pixels.fill(0.0)
        
        transform_and_draw(angle_x, angle_y, angle_z, pixels)
        
        gui.set_image(pixels)
        gui.show()


if __name__ == '__main__':
    main()
