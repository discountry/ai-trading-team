#!/bin/bash
# AI Trading Bot 部署脚本
# 用法: ./deploy/install.sh

set -e

echo "=== AI Trading Bot 部署 ==="

# 检查是否为 root
if [ "$EUID" -ne 0 ]; then
    echo "请使用 sudo 运行此脚本"
    exit 1
fi

# 变量
APP_DIR="/home/ubuntu/ai-trading-team"
SERVICE_NAME="ai-trading"
USER="ubuntu"

# 1. 创建日志目录
echo "[1/5] 创建日志目录..."
mkdir -p "$APP_DIR/logs"
chown -R $USER:$USER "$APP_DIR/logs"

# 2. 复制 .env 文件 (如果不存在)
if [ ! -f "$APP_DIR/.env" ]; then
    echo "[2/5] 请先配置 .env 文件!"
    echo "  cp $APP_DIR/env.example $APP_DIR/.env"
    echo "  然后编辑 .env 填入 API 密钥"
    exit 1
else
    echo "[2/5] .env 文件已存在"
fi

# 3. 安装依赖
echo "[3/5] 安装 Python 依赖..."
cd "$APP_DIR"
sudo -u $USER /home/ubuntu/.local/bin/uv sync --all-extras

# 4. 安装 systemd 服务
echo "[4/5] 安装 systemd 服务..."
cp "$APP_DIR/deploy/ai-trading.service" /etc/systemd/system/
systemctl daemon-reload
systemctl enable $SERVICE_NAME

# 5. 启动服务
echo "[5/5] 启动服务..."
systemctl start $SERVICE_NAME

echo ""
echo "=== 部署完成 ==="
echo ""
echo "常用命令:"
echo "  查看状态:   sudo systemctl status $SERVICE_NAME"
echo "  查看日志:   sudo journalctl -u $SERVICE_NAME -f"
echo "  停止服务:   sudo systemctl stop $SERVICE_NAME"
echo "  重启服务:   sudo systemctl restart $SERVICE_NAME"
echo "  查看输出:   tail -f $APP_DIR/logs/service.log"
echo ""
