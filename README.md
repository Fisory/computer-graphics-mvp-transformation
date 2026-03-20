# 计算机图形学实验二：MVP 矩阵变换

这个项目实现了三维图形学中的 MVP（模型-视图-投影）矩阵变换，使用 Taichi 图形库将三维物体渲染到二维屏幕上。

## 项目简介

MVP 变换是计算机图形学的核心概念，它将三维空间中的物体坐标转换为屏幕上的二维像素。这个过程包括三个关键步骤：

- **模型变换（Model）**：在物体自身的坐标系中进行旋转、缩放等操作
- **视图变换（View）**：将世界坐标转换到相机坐标系
- **投影变换（Projection）**：将三维场景投影到二维平面，产生透视效果

本项目包含两个实现：

1. **基础任务**：绘制一个可旋转的彩色三角形
2. **进阶任务**：绘制一个支持多轴旋转的立方体

## 演示效果

### 三角形旋转

![三角形演示](triangle_demo.gif)

按 A/D 键可以控制三角形绕 Z 轴旋转。三角形的三个顶点分别用红、绿、蓝三色标记，边缘颜色会自然过渡。

### 立方体旋转

![立方体演示](cube_demo.gif)

立方体支持绕 X、Y、Z 三个轴同时旋转，展现出真实的三维透视效果。不同的边用不同颜色标记：后面是蓝色系，前面是红色系，连接边是绿色系。

## 环境配置

### 依赖安装

项目使用 `uv` 进行依赖管理。首先安装 uv：

```bash
pip install uv
```

然后创建虚拟环境并安装依赖：

```bash
uv venv
uv pip install -r requirements.txt
```

或者使用传统方式：

```bash
pip install -r requirements.txt
```

### 系统要求

- Python 3.8 或更高版本
- 支持的操作系统：Windows、macOS、Linux

## 运行程序

### 运行三角形演示

```bash
python main.py
```

**控制说明：**

- `A` 键：逆时针旋转
- `D` 键：顺时针旋转
- `ESC` 键：退出程序

### 运行立方体演示

```bash
python cube.py
```

**控制说明：**

- `W/S` 键：绕 X 轴旋转
- `A/D` 键：绕 Y 轴旋转
- `Q/E` 键：绕 Z 轴旋转
- `空格` 键：切换自动旋转模式
- `ESC` 键：退出程序

### 生成演示动画

如果想生成 GIF 动画文件：

```bash
python demo_generator.py
```

这会在当前目录生成 `triangle_demo.gif` 和 `cube_demo.gif` 两个文件。

## 核心实现

### 1. 模型变换矩阵

模型变换负责物体的旋转。对于绕 Z 轴旋转角度 θ 的变换矩阵：

```text
| cos(θ)  -sin(θ)   0   0 |
| sin(θ)   cos(θ)   0   0 |
|   0        0      1   0 |
|   0        0      0   1 |
```

立方体版本还支持绕 X 轴和 Y 轴的旋转，通过矩阵相乘组合多个旋转。

### 2. 视图变换矩阵

视图变换将相机移动到原点。如果相机位置在 (x, y, z)，变换矩阵为：

```text
| 1  0  0  -x |
| 0  1  0  -y |
| 0  0  1  -z |
| 0  0  0   1 |
```

### 3. 投影变换矩阵

投影变换最复杂，分两步完成：

#### 第一步：透视到正交变换

将视锥体（frustum）挤压成长方体：

```text
| n  0   0    0  |
| 0  n   0    0  |
| 0  0  n+f   1  |
| 0  0  -nf   0  |
```

其中 n 和 f 分别是近平面和远平面的 z 坐标（负值）。

#### 第二步：正交投影

先平移到原点，再缩放到标准立方体 [-1, 1]³：

```text
M_ortho = Scale × Translate
```

视锥体的边界通过视场角（FOV）和近平面距离计算：

```text
t = tan(FOV/2) × |n|
b = -t
r = aspect_ratio × t
l = -r
```

### 4. 透视除法

经过 MVP 变换后，顶点坐标变成齐次坐标 (x, y, z, w)。需要将 x、y、z 都除以 w，才能得到标准设备坐标（NDC）：

```python
if v_transformed[3] != 0.0:
    v_transformed /= v_transformed[3]
```

### 5. 屏幕映射

最后将 NDC 坐标（范围 [-1, 1]）映射到屏幕坐标（范围 [0, width/height]）：

```python
screen_x = (ndc_x + 1.0) * width / 2.0
screen_y = (ndc_y + 1.0) * height / 2.0
```

## 代码结构

```text
实验二/
├── main.py              # 三角形实现（基础任务）
├── cube.py              # 立方体实现（进阶任务）
├── demo_generator.py    # GIF 动画生成脚本
├── requirements.txt     # Python 依赖
├── .gitignore          # Git 忽略配置
└── README.md           # 项目文档
```

### 主要函数说明

**`get_model_matrix(angle)`**

- 参数：旋转角度（角度制）
- 返回：4×4 模型变换矩阵
- 功能：生成绕 Z 轴旋转的变换矩阵

**`get_view_matrix(eye_pos)`**

- 参数：相机位置（三维向量）
- 返回：4×4 视图变换矩阵
- 功能：将相机平移到原点

**`get_projection_matrix(eye_fov, aspect_ratio, zNear, zFar)`**

- 参数：视场角、长宽比、近平面距离、远平面距离
- 返回：4×4 透视投影矩阵
- 功能：生成透视投影变换

**`transform_and_draw(angle, pixels)`**

- 参数：旋转角度、像素缓冲区
- 功能：执行 MVP 变换并绘制到屏幕

**`draw_line(x0, y0, x1, y1, color, pixels)`**

- 参数：起点、终点、颜色、像素缓冲区
- 功能：使用 Bresenham 算法绘制线段

## 技术要点

### 角度与弧度转换

Python 的三角函数使用弧度制，需要转换：

```python
radian = angle * π / 180.0
```

### 右手坐标系

相机看向 -Z 方向，因此：

- 近平面：n = -zNear
- 远平面：f = -zFar

### 矩阵乘法顺序

使用列向量，变换从右向左执行：

```python
MVP = Projection @ View @ Model
v_transformed = MVP @ v
```

### Bresenham 算法

用于高效绘制直线，避免浮点运算。算法通过整数误差累积来决定下一个像素位置。

## 实验收获

通过这个实验，我掌握了：

1. **MVP 变换的数学原理**：理解了如何用矩阵表示三维空间中的几何变换
2. **透视投影的实现**：学会了如何将三维场景投影到二维平面，产生近大远小的透视效果
3. **齐次坐标的应用**：明白了为什么需要 4×4 矩阵和透视除法
4. **图形渲染管线**：体验了从模型坐标到屏幕坐标的完整流程
5. **Taichi 编程**：熟悉了使用 Taichi 进行高性能图形计算

## 常见问题

**Q: 为什么三角形/立方体不显示？**

A: 检查相机位置是否合理。默认相机在 (0, 0, 5)，物体在 z = -2 附近。如果相机太近或太远，物体可能在视锥体外。

**Q: 旋转方向为什么和预期相反？**

A: 这取决于坐标系的定义。右手坐标系中，绕 Z 轴逆时针旋转对应正角度。如果觉得反了，可以调整角度的符号。

**Q: 如何调整视场角？**

A: 修改 `get_projection_matrix` 的第一个参数。默认是 45 度。更大的角度会产生更强的透视效果（广角），更小的角度接近正交投影（长焦）。

**Q: 能否添加更多物体？**

A: 可以。只需定义新的顶点数组和边数组，在绘制循环中对它们执行相同的 MVP 变换即可。

## 参考资料

- [GAMES101: 现代计算机图形学入门](https://sites.cs.ucsb.edu/~lingqi/teaching/games101.html)
- [Taichi 官方文档](https://docs.taichi-lang.org/)
- [Learn OpenGL - Coordinate Systems](https://learnopengl.com/Getting-started/Coordinate-Systems)

## 许可证

本项目仅用于教学目的。

## 作者

北京师范大学 计算机图形学课程实验

---

**注意**：运行程序前确保已安装所有依赖。如遇到问题，请检查 Python 版本和依赖包版本是否符合要求。
