# 发布 Weex SDK 到 PyPI

本文档说明如何将 `weex-sdk` 包发布到 PyPI（Python Package Index）。

## 前置准备

### 1. 注册 PyPI 账户

- **TestPyPI** (测试环境): https://test.pypi.org/account/register/
- **PyPI** (正式环境): https://pypi.org/account/register/

### 2. 安装发布工具

```bash
pip install build twine
```

### 3. 配置 API Token（推荐）

在 PyPI 账户设置中创建 API Token，比使用密码更安全：

1. 登录 PyPI
2. 进入 Account settings → API tokens
3. 创建新的 API token
4. 保存 token（只显示一次）

创建 `~/.pypirc` 文件（Linux/Mac）或 `%USERPROFILE%\.pypirc`（Windows）：

```ini
[pypi]
username = __token__
password = pypi-你的API-token

[testpypi]
username = __token__
password = pypi-你的TestPyPI-API-token
```

## 发布步骤

### 步骤 1: 更新版本号

在以下文件中更新版本号：

- `setup.py`: `version="1.0.0"`
- `pyproject.toml`: `version = "1.0.0"`
- `weex_sdk/__init__.py`: `__version__ = "1.0.0"`

遵循 [语义化版本](https://semver.org/)：
- **MAJOR**: 不兼容的 API 变更
- **MINOR**: 向后兼容的功能新增
- **PATCH**: 向后兼容的问题修复

### 步骤 2: 检查代码质量

```bash
# 代码格式化
black weex_sdk/

# 类型检查
mypy weex_sdk/

# 代码检查
ruff check weex_sdk/
```

### 步骤 3: 清理旧构建文件

```bash
# 删除旧的构建文件
rm -rf build/ dist/ *.egg-info/
```

### 步骤 4: 构建分发包

```bash
# 使用现代构建工具（推荐）
python -m build

# 或者使用旧方法
python setup.py sdist bdist_wheel
```

构建完成后会生成：
- `dist/weex-sdk-1.0.0.tar.gz` (源码包)
- `dist/weex_sdk-1.0.0-py3-none-any.whl` (wheel 包)

### 步骤 5: 检查构建的包

```bash
# 检查包是否有问题
twine check dist/*
```

### 步骤 6: 上传到 TestPyPI（测试）

首次发布建议先上传到 TestPyPI 进行测试：

```bash
# 上传到 TestPyPI
twine upload --repository testpypi dist/*

# 如果配置了 .pypirc，可以简化为：
twine upload --repository testpypi dist/*
```

### 步骤 7: 测试安装

从 TestPyPI 安装测试：

```bash
# 创建新的虚拟环境
python -m venv test_env
source test_env/bin/activate  # Linux/Mac
# 或
test_env\Scripts\activate  # Windows

# 从 TestPyPI 安装
pip install --index-url https://test.pypi.org/simple/ weex-sdk

# 测试导入
python -c "from weex_sdk import WeexClient; print('Success!')"
```

### 步骤 8: 上传到正式 PyPI

测试通过后，上传到正式 PyPI：

```bash
# 上传到正式 PyPI
twine upload dist/*

# 或明确指定
twine upload --repository pypi dist/*
```

### 步骤 9: 验证发布

```bash
# 等待几分钟后，从 PyPI 安装
pip install weex-sdk

# 验证安装
python -c "import weex_sdk; print(weex_sdk.__version__)"
```

## 完整发布脚本

创建 `publish.sh` (Linux/Mac) 或 `publish.bat` (Windows):

```bash
#!/bin/bash
# publish.sh

set -e  # 遇到错误立即退出

echo "🧹 清理旧构建文件..."
rm -rf build/ dist/ *.egg-info/

echo "📦 构建分发包..."
python -m build

echo "✅ 检查包..."
twine check dist/*

echo "📤 上传到 TestPyPI..."
read -p "是否上传到 TestPyPI? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    twine upload --repository testpypi dist/*
fi

echo "📤 上传到正式 PyPI..."
read -p "是否上传到正式 PyPI? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    twine upload dist/*
fi

echo "✅ 发布完成！"
```

使用：

```bash
chmod +x publish.sh
./publish.sh
```

## 版本更新流程

### 小版本更新（1.0.0 → 1.0.1）

```bash
# 1. 更新版本号（在 setup.py, pyproject.toml, __init__.py）
# 2. 提交更改
git add .
git commit -m "Bump version to 1.0.1"
git tag v1.0.1
git push origin main --tags

# 3. 构建和发布
python -m build
twine upload dist/*
```

### 使用 bump2version 自动化版本管理

```bash
pip install bump2version
```

创建 `.bumpversion.cfg`:

```ini
[bumpversion]
current_version = 1.0.0
commit = True
tag = True

[bumpversion:file:setup.py]
search = version="{current_version}"
replace = version="{new_version}"

[bumpversion:file:pyproject.toml]
search = version = "{current_version}"
replace = version = "{new_version}"

[bumpversion:file:weex_sdk/__init__.py]
search = __version__ = "{current_version}"
replace = __version__ = "{new_version}"
```

使用：

```bash
# 更新 patch 版本 (1.0.0 → 1.0.1)
bump2version patch

# 更新 minor 版本 (1.0.0 → 1.1.0)
bump2version minor

# 更新 major 版本 (1.0.0 → 2.0.0)
bump2version major
```

## 常见问题

### 1. 包名已被占用

如果 `weex-sdk` 已被占用，需要：
- 选择其他名称（如 `weex-api-python`, `pyweex`）
- 更新 `setup.py` 和 `pyproject.toml` 中的 `name`

### 2. 上传失败：文件已存在

如果版本号已存在，需要：
- 更新版本号
- 或删除 PyPI 上的旧版本（如果有权限）

### 3. 认证失败

检查：
- API token 是否正确
- `.pypirc` 文件格式是否正确
- 网络连接是否正常

### 4. 依赖问题

确保 `requirements.txt` 中的所有依赖都正确列出，并且版本兼容。

## GitHub Actions 自动化发布

创建 `.github/workflows/publish.yml`:

```yaml
name: Publish to PyPI

on:
  release:
    types: [created]

jobs:
  publish:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.8'
      
      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install build twine
      
      - name: Build package
        run: python -m build
      
      - name: Check package
        run: twine check dist/*
      
      - name: Publish to PyPI
        env:
          TWINE_USERNAME: __token__
          TWINE_PASSWORD: ${{ secrets.PYPI_API_TOKEN }}
        run: twine upload dist/*
```

在 GitHub Secrets 中添加 `PYPI_API_TOKEN`。

## 发布检查清单

- [ ] 更新版本号（setup.py, pyproject.toml, __init__.py）
- [ ] 更新 CHANGELOG.md（如果有）
- [ ] 运行测试（如果有）
- [ ] 代码格式化（black）
- [ ] 类型检查（mypy）
- [ ] 构建包（python -m build）
- [ ] 检查包（twine check）
- [ ] 先上传到 TestPyPI 测试
- [ ] 从 TestPyPI 安装测试
- [ ] 上传到正式 PyPI
- [ ] 验证安装
- [ ] 创建 Git tag
- [ ] 更新文档

## 参考资源

- [PyPI 官方文档](https://packaging.python.org/en/latest/guides/distributing-packages-using-setuptools/)
- [Twine 文档](https://twine.readthedocs.io/)
- [Python 打包指南](https://packaging.python.org/)
