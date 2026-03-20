# Git 仓库设置指南

本文档说明如何将项目推送到 GitHub。

## 初始化 Git 仓库

在项目目录下执行以下命令：

```bash
# 初始化 Git 仓库
git init

# 添加所有文件到暂存区
git add .

# 创建初始提交
git commit -m "Initial commit: MVP transformation implementation"
```

## 创建 GitHub 仓库

1. 访问 [GitHub](https://github.com)
2. 点击右上角的 "+" 按钮，选择 "New repository"
3. 填写仓库信息：
   - **Repository name**: `computer-graphics-mvp-transformation`（或其他名称）
   - **Description**: `计算机图形学实验二：MVP矩阵变换实现`
   - **Public/Private**: 根据需要选择
   - **不要**勾选 "Initialize this repository with a README"（我们已经有了）
4. 点击 "Create repository"

## 推送到 GitHub

创建仓库后，GitHub 会显示推送命令。执行以下命令：

```bash
# 添加远程仓库（替换 YOUR_USERNAME 为你的 GitHub 用户名）
git remote add origin https://github.com/YOUR_USERNAME/computer-graphics-mvp-transformation.git

# 推送到 GitHub
git branch -M main
git push -u origin main
```

## 生成演示 GIF（可选）

如果想在 README 中显示演示动画，需要先生成 GIF 文件：

```bash
# 安装依赖
pip install -r requirements.txt

# 生成演示动画
python demo_generator.py
```

这会生成 `triangle_demo.gif` 和 `cube_demo.gif` 两个文件。

然后将它们添加到 Git：

```bash
# 临时修改 .gitignore 以允许提交 GIF
# 或者直接强制添加
git add -f triangle_demo.gif cube_demo.gif

# 提交
git commit -m "Add demo GIF animations"

# 推送
git push
```

## 验证

推送成功后，访问你的 GitHub 仓库页面，应该能看到：

- ✅ 所有源代码文件
- ✅ README.md 显示完整的项目文档
- ✅ 演示 GIF 正常显示（如果已上传）

## 常见问题

**Q: 推送时要求输入用户名和密码？**

A: GitHub 已不再支持密码认证。你需要：
1. 生成 Personal Access Token（个人访问令牌）
2. 或者配置 SSH 密钥

详见：https://docs.github.com/zh/authentication

**Q: 如何更新远程仓库？**

A: 修改文件后执行：

```bash
git add .
git commit -m "描述你的修改"
git push
```

**Q: 如何克隆仓库到其他电脑？**

A: 使用以下命令：

```bash
git clone https://github.com/YOUR_USERNAME/computer-graphics-mvp-transformation.git
cd computer-graphics-mvp-transformation
pip install -r requirements.txt
python main.py
```

## 提交清单

在提交作业前，确保：

- [ ] 代码可以正常运行
- [ ] README.md 文档完整
- [ ] 包含演示 GIF（如果要求）
- [ ] 所有文件已推送到 GitHub
- [ ] 仓库链接可以正常访问
- [ ] 仓库设置为 Public（如果要求）

---

完成后，将 GitHub 仓库链接提交给老师即可。
