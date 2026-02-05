#!/bin/bash
# 发布脚本 - 将 weex-sdk 发布到 PyPI

set -e  # 遇到错误立即退出

echo "🚀 Weex SDK 发布脚本"
echo "===================="
echo ""

# 询问是否更新版本号
if [ -f "bump_version.py" ]; then
    echo "📦 版本号管理"
    echo "   当前版本号可以从以下文件查看:"
    echo "   - pyproject.toml"
    echo "   - setup.py"
    echo "   - weex_sdk/__init__.py"
    echo ""
    read -p "是否要更新版本号? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo ""
        echo "请选择版本更新类型:"
        echo "  1) patch  - 补丁版本 (1.0.1 -> 1.0.2)"
        echo "  2) minor   - 次版本 (1.0.1 -> 1.1.0)"
        echo "  3) major   - 主版本 (1.0.1 -> 2.0.0)"
        echo "  4) custom  - 自定义版本号"
        echo ""
        read -p "请选择 (1-4): " -n 1 -r version_choice
        echo
        echo ""
        
        case $version_choice in
            1)
                python bump_version.py patch
                ;;
            2)
                python bump_version.py minor
                ;;
            3)
                python bump_version.py major
                ;;
            4)
                read -p "请输入新版本号 (格式: x.y.z): " custom_version
                python bump_version.py "$custom_version"
                ;;
            *)
                echo "⏭️  跳过版本号更新"
                ;;
        esac
        echo ""
    else
        echo "⏭️  跳过版本号更新"
        echo ""
    fi
fi

# 检查必要的工具
if ! command -v python &> /dev/null; then
    echo "❌ 错误: 未找到 python 命令"
    exit 1
fi

if ! python -c "import build" 2>/dev/null; then
    echo "📦 安装 build 工具..."
    pip install build twine
fi

# 清理旧构建文件
echo "🧹 清理旧构建文件..."
rm -rf build/ dist/ *.egg-info/
echo "✅ 清理完成"
echo ""

# 构建分发包
echo "📦 构建分发包..."
python -m build
echo "✅ 构建完成"
echo ""

# 检查包
echo "✅ 检查包..."
twine check dist/*
echo "✅ 检查通过"
echo ""

# 显示构建的文件
echo "📁 构建的文件:"
ls -lh dist/
echo ""

# 询问是否上传到 TestPyPI
read -p "📤 是否上传到 TestPyPI 进行测试? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "📤 上传到 TestPyPI..."
    twine upload --repository testpypi dist/*
    echo "✅ 已上传到 TestPyPI"
    echo ""
    echo "💡 测试安装命令:"
    echo "   pip install --index-url https://test.pypi.org/simple/ weex-sdk"
    echo ""
fi

# 询问是否上传到正式 PyPI
read -p "📤 是否上传到正式 PyPI? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "📤 上传到正式 PyPI..."
    twine upload dist/*
    echo "✅ 已上传到 PyPI"
    echo ""
    echo "🎉 发布完成！"
    echo ""
    echo "💡 安装命令:"
    echo "   pip install weex-sdk"
    echo ""
else
    echo "⏭️  跳过上传到正式 PyPI"
fi

echo "✅ 完成！"
