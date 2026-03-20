import taichi as ti
import numpy as np

ti.init(arch=ti.cpu)

# 窗口尺寸
width, height = 700, 700

# 定义三角形的三个顶点(三维空间)
vertices = ti.Vector.field(3, dtype=ti.f32, shape=3)
vertices[0] = ti.Vector([2.0, 0.0, -2.0])
vertices[1] = ti.Vector([0.0, 2.0, -2.0])
vertices[2] = ti.Vector([-2.0, 0.0, -2.0])

# 顶点颜色
colors = ti.Vector.field(3, dtype=ti.f32, shape=3)
colors[0] = ti.Vector([1.0, 0.0, 0.0])  # 红色
colors[1] = ti.Vector([0.0, 1.0, 0.0])  # 绿色
colors[2] = ti.Vector([0.0, 0.0, 1.0])  # 蓝色


@ti.func
def get_model_matrix(angle: ti.f32) -> ti.math.mat4:
    """
    返回绕Z轴旋转的模型变换矩阵
    
    参数:
        angle: 旋转角度(角度制)
    
    返回:
        4x4 齐次坐标变换矩阵
    """
    # 将角度转换为弧度
    radian = angle * 3.14159265358979323846 / 180.0
    
    cos_a = ti.cos(radian)
    sin_a = ti.sin(radian)
    
    # 绕Z轴旋转矩阵
    # | cos  -sin  0  0 |
    # | sin   cos  0  0 |
    # |  0     0   1  0 |
    # |  0     0   0  1 |
    return ti.math.mat4(
        [cos_a, sin_a, 0.0, 0.0],
        [-sin_a, cos_a, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0]
    )


@ti.func
def get_view_matrix(eye_pos: ti.math.vec3) -> ti.math.mat4:
    """
    返回视图变换矩阵,将相机平移至原点
    
    参数:
        eye_pos: 相机位置(三维向量)
    
    返回:
        4x4 视图变换矩阵
    """
    # 视图矩阵是将相机移动到原点的变换
    # 即将世界坐标系平移 -eye_pos
    # | 1  0  0  -x |
    # | 0  1  0  -y |
    # | 0  0  1  -z |
    # | 0  0  0   1 |
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
    # 将角度转换为弧度
    fov_rad = eye_fov * 3.14159265358979323846 / 180.0
    
    # 注意:按照右手坐标系,相机看向-Z方向
    # 近截面和远截面的实际坐标值为负
    n = -zNear
    f = -zFar
    
    # 计算视锥体边界
    t = ti.tan(fov_rad / 2.0) * ti.abs(n)  # top
    b = -t  # bottom
    r = aspect_ratio * t  # right
    l = -r  # left
    
    # 透视到正交矩阵 M_persp->ortho
    # 将透视平截头体挤压为正交长方体
    # | n  0  0  0 |
    # | 0  n  0  0 |
    # | 0  0 n+f -nf|
    # | 0  0  1  0 |
    persp_to_ortho = ti.math.mat4(
        [n, 0.0, 0.0, 0.0],
        [0.0, n, 0.0, 0.0],
        [0.0, 0.0, n + f, 1.0],
        [0.0, 0.0, -n * f, 0.0]
    )
    
    # 正交投影矩阵 M_ortho
    # 先平移到原点,再缩放到[-1,1]^3
    # 缩放矩阵
    # | 2/(r-l)    0        0      0 |
    # |   0     2/(t-b)     0      0 |
    # |   0        0     2/(n-f)   0 |
    # |   0        0        0      1 |
    ortho_scale = ti.math.mat4(
        [2.0 / (r - l), 0.0, 0.0, 0.0],
        [0.0, 2.0 / (t - b), 0.0, 0.0],
        [0.0, 0.0, 2.0 / (n - f), 0.0],
        [0.0, 0.0, 0.0, 1.0]
    )
    
    # 平移矩阵
    # | 1  0  0  -(r+l)/2 |
    # | 0  1  0  -(t+b)/2 |
    # | 0  0  1  -(n+f)/2 |
    # | 0  0  0      1    |
    ortho_translate = ti.math.mat4(
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [-(r + l) / 2.0, -(t + b) / 2.0, -(n + f) / 2.0, 1.0]
    )
    
    # M_ortho = 缩放 @ 平移
    ortho = ortho_scale @ ortho_translate
    
    # 最终投影矩阵 = M_ortho @ M_persp->ortho
    return ortho @ persp_to_ortho


@ti.kernel
def transform_and_draw(angle: ti.f32, pixels: ti.template()):
    """
    对顶点进行MVP变换并绘制到屏幕
    """
    # 相机参数
    eye_pos = ti.math.vec3(0.0, 0.0, 5.0)
    
    # 获取MVP矩阵
    model = get_model_matrix(angle)
    view = get_view_matrix(eye_pos)
    projection = get_projection_matrix(45.0, 1.0, 0.1, 50.0)
    
    # MVP = Projection @ View @ Model
    mvp = projection @ view @ model
    
    # 变换后的顶点(NDC坐标)
    transformed = ti.Vector.field(3, dtype=ti.f32, shape=3)
    
    # 对每个顶点进行变换
    for i in range(3):
        # 转换为齐次坐标
        v = ti.math.vec4(vertices[i][0], vertices[i][1], vertices[i][2], 1.0)
        
        # MVP变换
        v_transformed = mvp @ v
        
        # 透视除法
        if v_transformed[3] != 0.0:
            v_transformed /= v_transformed[3]
        
        # 存储NDC坐标
        transformed[i] = ti.math.vec3(v_transformed[0], v_transformed[1], v_transformed[2])
    
    # 绘制三角形的三条边
    for i in range(3):
        # 起点和终点索引
        start_idx = i
        end_idx = (i + 1) % 3
        
        # NDC坐标转换到屏幕坐标
        # NDC范围[-1,1] -> 屏幕坐标[0, width/height]
        x0 = int((transformed[start_idx][0] + 1.0) * width / 2.0)
        y0 = int((transformed[start_idx][1] + 1.0) * height / 2.0)
        x1 = int((transformed[end_idx][0] + 1.0) * width / 2.0)
        y1 = int((transformed[end_idx][1] + 1.0) * height / 2.0)
        
        # 使用Bresenham算法绘制线段
        draw_line(x0, y0, x1, y1, colors[start_idx], colors[end_idx], pixels)


@ti.func
def draw_line(x0: ti.i32, y0: ti.i32, x1: ti.i32, y1: ti.i32, 
              color0: ti.math.vec3, color1: ti.math.vec3, pixels: ti.template()):
    """
    使用Bresenham算法绘制线段,支持颜色插值
    """
    dx = ti.abs(x1 - x0)
    dy = ti.abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx - dy
    
    x, y = x0, y0
    total_steps = ti.max(dx, dy)
    step = 0
    
    while True:
        # 颜色插值
        if total_steps > 0:
            t = ti.cast(step, ti.f32) / ti.cast(total_steps, ti.f32)
        else:
            t = 0.0
        color = color0 * (1.0 - t) + color1 * t
        
        # 绘制像素
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
        
        step += 1


def main():
    """主函数"""
    # 创建GUI窗口
    gui = ti.GUI('MVP Transformation - Triangle', res=(width, height))
    
    # 像素缓冲区
    pixels = ti.Vector.field(3, dtype=ti.f32, shape=(width, height))
    
    # 旋转角度
    angle = 0.0
    
    print("控制说明:")
    print("  A键: 逆时针旋转")
    print("  D键: 顺时针旋转")
    print("  ESC键: 退出程序")
    
    while gui.running:
        # 处理键盘输入
        for e in gui.get_events(ti.GUI.PRESS):
            if e.key == ti.GUI.ESCAPE:
                gui.running = False
        
        if gui.is_pressed('a'):
            angle += 1.0
        if gui.is_pressed('d'):
            angle -= 1.0
        
        # 清空画布
        pixels.fill(0.0)
        
        # 变换并绘制
        transform_and_draw(angle, pixels)
        
        # 显示
        gui.set_image(pixels)
        gui.show()


if __name__ == '__main__':
    main()
