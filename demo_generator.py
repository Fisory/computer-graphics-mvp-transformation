"""
演示GIF生成脚本
用于生成三角形和立方体旋转的演示动画
"""
import taichi as ti
import numpy as np
from PIL import Image
import os

ti.init(arch=ti.cpu)

width, height = 700, 700


def generate_triangle_demo():
    """生成三角形旋转演示"""
    print("正在生成三角形演示动画...")
    
    # 导入主程序的函数
    from main import (vertices, colors, get_model_matrix, get_view_matrix, 
                      get_projection_matrix, draw_line)
    
    pixels = ti.Vector.field(3, dtype=ti.f32, shape=(width, height))
    frames = []
    
    # 生成60帧动画(旋转一圈)
    for frame in range(60):
        angle = frame * 6.0  # 每帧旋转6度
        
        pixels.fill(0.0)
        
        # MVP变换
        eye_pos = ti.math.vec3(0.0, 0.0, 5.0)
        model = get_model_matrix(angle)
        view = get_view_matrix(eye_pos)
        projection = get_projection_matrix(45.0, 1.0, 0.1, 50.0)
        mvp = projection @ view @ model
        
        # 变换顶点
        transformed = []
        for i in range(3):
            v = ti.math.vec4(vertices[i][0], vertices[i][1], vertices[i][2], 1.0)
            v_transformed = mvp @ v
            if v_transformed[3] != 0.0:
                v_transformed /= v_transformed[3]
            transformed.append(v_transformed)
        
        # 绘制边
        for i in range(3):
            start_idx = i
            end_idx = (i + 1) % 3
            
            x0 = int((transformed[start_idx][0] + 1.0) * width / 2.0)
            y0 = int((transformed[start_idx][1] + 1.0) * height / 2.0)
            x1 = int((transformed[end_idx][0] + 1.0) * width / 2.0)
            y1 = int((transformed[end_idx][1] + 1.0) * height / 2.0)
            
            draw_line(x0, y0, x1, y1, colors[start_idx], colors[end_idx], pixels)
        
        # 转换为PIL图像
        img_array = pixels.to_numpy()
        img_array = (img_array * 255).astype(np.uint8)
        img = Image.fromarray(img_array, 'RGB')
        frames.append(img)
        
        print(f"  帧 {frame + 1}/60 完成")
    
    # 保存GIF
    frames[0].save(
        'triangle_demo.gif',
        save_all=True,
        append_images=frames[1:],
        duration=50,
        loop=0
    )
    print("✓ 三角形演示已保存为 triangle_demo.gif")


def generate_cube_demo():
    """生成立方体旋转演示"""
    print("\n正在生成立方体演示动画...")
    
    from cube import (vertices, edges, edge_colors, get_model_matrix, 
                      get_view_matrix, get_projection_matrix, draw_line)
    
    pixels = ti.Vector.field(3, dtype=ti.f32, shape=(width, height))
    frames = []
    
    # 生成120帧动画
    for frame in range(120):
        angle_x = frame * 1.5
        angle_y = frame * 2.1
        angle_z = frame * 0.9
        
        pixels.fill(0.0)
        
        # MVP变换
        eye_pos = ti.math.vec3(0.0, 0.0, 5.0)
        model = get_model_matrix(angle_x, angle_y, angle_z)
        view = get_view_matrix(eye_pos)
        projection = get_projection_matrix(45.0, 1.0, 0.1, 50.0)
        mvp = projection @ view @ model
        
        # 变换顶点
        transformed = []
        for i in range(8):
            v = ti.math.vec4(vertices[i][0], vertices[i][1], vertices[i][2], 1.0)
            v_transformed = mvp @ v
            if v_transformed[3] != 0.0:
                v_transformed /= v_transformed[3]
            transformed.append(v_transformed)
        
        # 绘制边
        for i in range(12):
            start_idx = edges[i][0]
            end_idx = edges[i][1]
            
            x0 = int((transformed[start_idx][0] + 1.0) * width / 2.0)
            y0 = int((transformed[start_idx][1] + 1.0) * height / 2.0)
            x1 = int((transformed[end_idx][0] + 1.0) * width / 2.0)
            y1 = int((transformed[end_idx][1] + 1.0) * height / 2.0)
            
            draw_line(x0, y0, x1, y1, edge_colors[i], pixels)
        
        # 转换为PIL图像
        img_array = pixels.to_numpy()
        img_array = (img_array * 255).astype(np.uint8)
        img = Image.fromarray(img_array, 'RGB')
        frames.append(img)
        
        if (frame + 1) % 20 == 0:
            print(f"  帧 {frame + 1}/120 完成")
    
    # 保存GIF
    frames[0].save(
        'cube_demo.gif',
        save_all=True,
        append_images=frames[1:],
        duration=50,
        loop=0
    )
    print("✓ 立方体演示已保存为 cube_demo.gif")


if __name__ == '__main__':
    print("=" * 50)
    print("MVP变换演示动画生成器")
    print("=" * 50)
    
    generate_triangle_demo()
    generate_cube_demo()
    
    print("\n" + "=" * 50)
    print("所有演示动画生成完成!")
    print("=" * 50)
