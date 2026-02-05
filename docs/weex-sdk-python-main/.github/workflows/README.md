# GitHub Actions Workflows

本项目包含以下自动化工作流：

## 1. Test and Lint (`test.yml`)

**触发时机：**
- Push 到 `main` 或 `develop` 分支
- Pull Request 到 `main` 或 `develop` 分支
- 手动触发

**功能：**
- 在多个 Python 版本（3.8-3.12）上运行测试
- 代码格式化检查（black）
- 代码质量检查（ruff）
- 类型检查（mypy）
- 构建和验证包

## 2. Publish to PyPI (`publish.yml`)

**触发时机：**
- 创建 GitHub Release 时自动发布到 PyPI
- 手动触发（可选择发布到 TestPyPI 或 PyPI）

**功能：**
- 构建 Python 包（source distribution 和 wheel）
- 检查包质量
- 发布到 TestPyPI（测试环境）
- 发布到 PyPI（正式环境）

**设置步骤：**

1. **获取 PyPI API Token：**
   - 登录 https://pypi.org
   - 进入 Account settings → API tokens
   - 创建新的 API token（scope: 整个账户或特定项目）
   - 复制 token（格式：`pypi-...`）

2. **获取 TestPyPI API Token：**
   - 登录 https://test.pypi.org
   - 重复上述步骤获取 TestPyPI token

3. **在 GitHub 仓库中添加 Secrets：**
   - 进入仓库 Settings → Secrets and variables → Actions
   - 添加以下 secrets：
     - `PYPI_API_TOKEN`: PyPI 的 API token（**必需**）
     - `TESTPYPI_API_TOKEN`: TestPyPI 的 API token（**可选**，仅当需要发布到 TestPyPI 时配置）

## 3. Verify Release Tag (`release.yml`)

**触发时机：**
- 推送版本标签时（格式：`v*.*.*`，如 `v1.0.1`）

**功能：**
- 验证标签版本与代码中的版本号是否一致
- 构建并检查包

## 使用方法

### 方法 1: 通过 GitHub Release 发布（推荐）

1. **更新版本号：**
   ```bash
   # 在以下文件中更新版本号：
   # - pyproject.toml
   # - setup.py
   # - weex_sdk/__init__.py
   ```

2. **提交并推送：**
   ```bash
   git add .
   git commit -m "Bump version to 1.0.2"
   git push origin main
   ```

3. **创建版本标签：**
   ```bash
   git tag v1.0.2
   git push origin v1.0.2
   ```

4. **创建 GitHub Release：**
   - 在 GitHub 仓库页面点击 "Releases" → "Create a new release"
   - 选择刚创建的标签 `v1.0.2`
   - 填写发布说明
   - 点击 "Publish release"
   - **发布创建后会自动触发 `publish.yml` 工作流，将包发布到 PyPI**

### 方法 2: 手动触发发布

1. 进入 GitHub 仓库的 "Actions" 标签页
2. 选择 "Publish to PyPI" 工作流
3. 点击 "Run workflow"
4. 选择发布选项：
   - `publish_to_testpypi`: 发布到 TestPyPI（测试）
   - `publish_to_pypi`: 发布到 PyPI（正式）
5. 点击 "Run workflow"

### 方法 3: 仅发布到 TestPyPI 进行测试

**注意：** 需要先配置 `TESTPYPI_API_TOKEN` secret。

1. 手动触发 `publish.yml` 工作流
2. 勾选 `publish_to_testpypi`，取消勾选 `publish_to_pypi`
3. 运行工作流
4. 测试安装：
   ```bash
   pip install --index-url https://test.pypi.org/simple/ weex-sdk
   ```

**如果未配置 `TESTPYPI_API_TOKEN`：**
- 工作流会跳过 TestPyPI 发布并显示提示信息
- 不会导致工作流失败

## 注意事项

1. **版本号一致性：** 确保 `pyproject.toml`、`setup.py` 和 `weex_sdk/__init__.py` 中的版本号一致
2. **API Token 安全：** 不要将 API token 提交到代码仓库，只通过 GitHub Secrets 管理
3. **必需 vs 可选：** 
   - `PYPI_API_TOKEN` 是**必需的**，用于发布到正式 PyPI
   - `TESTPYPI_API_TOKEN` 是**可选的**，仅当需要发布到 TestPyPI 时配置
4. **测试优先：** 如果配置了 TestPyPI token，建议先发布到 TestPyPI 测试，确认无误后再发布到正式 PyPI
5. **版本标签格式：** 使用语义化版本（SemVer），标签格式为 `v1.0.0`（带 `v` 前缀）

## 故障排除

### 发布失败：认证错误
- 检查 GitHub Secrets 中的 `PYPI_API_TOKEN` 是否正确
- 确认 token 未过期
- 确认 token 有正确的权限

### 发布失败：版本已存在
- 更新版本号后重试
- 确认版本号遵循语义化版本规范

### 版本验证失败
- 检查所有文件中的版本号是否一致
- 确认标签格式正确（`v1.0.1`）
