#!/usr/bin/env python3
"""自动更新版本号脚本

用法:
    python bump_version.py patch   # 1.0.1 -> 1.0.2
    python bump_version.py minor   # 1.0.1 -> 1.1.0
    python bump_version.py major   # 1.0.1 -> 2.0.0
    python bump_version.py 1.2.3   # 直接指定版本号
"""

import re
import sys
from pathlib import Path
from typing import Tuple


def parse_version(version_str: str) -> Tuple[int, int, int]:
    """解析版本号字符串为 (major, minor, patch)"""
    match = re.match(r"^(\d+)\.(\d+)\.(\d+)$", version_str)
    if not match:
        raise ValueError(f"无效的版本号格式: {version_str}")
    return tuple(int(x) for x in match.groups())


def format_version(major: int, minor: int, patch: int) -> str:
    """格式化版本号为字符串"""
    return f"{major}.{minor}.{patch}"


def read_current_version() -> str:
    """从 pyproject.toml 读取当前版本号"""
    pyproject_path = Path("pyproject.toml")
    if not pyproject_path.exists():
        raise FileNotFoundError("未找到 pyproject.toml 文件")
    
    content = pyproject_path.read_text(encoding="utf-8")
    match = re.search(r'version\s*=\s*"([^"]+)"', content)
    if not match:
        raise ValueError("无法从 pyproject.toml 中读取版本号")
    return match.group(1)


def update_version_in_file(file_path: Path, old_version: str, new_version: str) -> bool:
    """更新文件中的版本号"""
    if not file_path.exists():
        print(f"⚠️  警告: 文件不存在，跳过: {file_path}")
        return False
    
    content = file_path.read_text(encoding="utf-8")
    original_content = content
    
    # 根据不同文件类型使用不同的替换模式
    if file_path.name == "pyproject.toml":
        # pyproject.toml: version = "1.0.1"
        pattern = rf'version\s*=\s*"{re.escape(old_version)}"'
        replacement = f'version = "{new_version}"'
        content = re.sub(pattern, replacement, content)
    
    elif file_path.name == "setup.py":
        # setup.py: version="1.0.1"
        pattern = rf'version\s*=\s*"{re.escape(old_version)}"'
        replacement = f'version="{new_version}"'
        content = re.sub(pattern, replacement, content)
    
    elif file_path.name == "__init__.py":
        # __init__.py: __version__ = "1.0.1"
        pattern = rf'__version__\s*=\s*"{re.escape(old_version)}"'
        replacement = f'__version__ = "{new_version}"'
        content = re.sub(pattern, replacement, content)
    
    if content != original_content:
        file_path.write_text(content, encoding="utf-8")
        return True
    return False


def bump_version(current_version: str, bump_type: str) -> str:
    """根据 bump_type 计算新版本号"""
    major, minor, patch = parse_version(current_version)
    
    if bump_type == "patch":
        patch += 1
    elif bump_type == "minor":
        minor += 1
        patch = 0
    elif bump_type == "major":
        major += 1
        minor = 0
        patch = 0
    else:
        # 假设是直接指定的版本号
        return bump_type
    
    return format_version(major, minor, patch)


def main():
    """主函数"""
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    
    bump_type = sys.argv[1].lower()
    
    # 如果直接指定了版本号（格式为 x.y.z）
    if re.match(r"^\d+\.\d+\.\d+$", bump_type):
        new_version = bump_type
        current_version = read_current_version()
    else:
        if bump_type not in ["patch", "minor", "major"]:
            print(f"❌ 错误: 无效的 bump 类型 '{bump_type}'")
            print("   支持的类型: patch, minor, major, 或直接指定版本号 (如 1.2.3)")
            sys.exit(1)
        
        current_version = read_current_version()
        new_version = bump_version(current_version, bump_type)
    
    print(f"📦 当前版本: {current_version}")
    print(f"🚀 新版本: {new_version}")
    print()
    
    # 需要更新的文件
    files_to_update = [
        Path("pyproject.toml"),
        Path("setup.py"),
        Path("weex_sdk/__init__.py"),
    ]
    
    updated_files = []
    for file_path in files_to_update:
        if update_version_in_file(file_path, current_version, new_version):
            updated_files.append(file_path)
            print(f"✅ 已更新: {file_path}")
        else:
            print(f"⏭️  跳过: {file_path}")
    
    if updated_files:
        print()
        print(f"✨ 版本号已从 {current_version} 更新到 {new_version}")
        print()
        print("📝 更新的文件:")
        for file_path in updated_files:
            print(f"   - {file_path}")
    else:
        print("⚠️  没有文件被更新")
        sys.exit(1)


if __name__ == "__main__":
    main()
