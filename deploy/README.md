# Ubuntu 服务器部署指南

## 快速部署

### 1. 上传代码到服务器

```bash
# 本地执行
rsync -avz --exclude '.git' --exclude '__pycache__' --exclude '.venv' \
  ./ ubuntu@your-server:/home/ubuntu/ai-trading-team/
```

### 2. 服务器上安装 uv

```bash
# SSH 到服务器
ssh ubuntu@your-server

# 安装 uv
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.bashrc
```

### 3. 配置环境变量

```bash
cd /home/ubuntu/ai-trading-team
cp env.example .env
nano .env  # 编辑填入 API 密钥
```

### 4. 运行部署脚本

```bash
sudo ./deploy/install.sh
```

## 手动部署

如果不想用脚本，可以手动执行：

```bash
# 1. 安装依赖
cd /home/ubuntu/ai-trading-team
uv sync --all-extras

# 2. 复制服务文件
sudo cp deploy/ai-trading.service /etc/systemd/system/

# 3. 修改服务文件中的路径和用户名（如需要）
sudo nano /etc/systemd/system/ai-trading.service

# 4. 启用并启动服务
sudo systemctl daemon-reload
sudo systemctl enable ai-trading
sudo systemctl start ai-trading
```

## 常用命令

```bash
# 查看服务状态
sudo systemctl status ai-trading

# 查看实时日志
sudo journalctl -u ai-trading -f

# 查看最近 100 行日志
sudo journalctl -u ai-trading -n 100

# 重启服务
sudo systemctl restart ai-trading

# 停止服务
sudo systemctl stop ai-trading

# 查看应用日志
tail -f /home/ubuntu/ai-trading-team/logs/service.log
tail -f /home/ubuntu/ai-trading-team/logs/service-error.log
```

## 使用 Screen/Tmux（替代方案）

如果不想用 systemd，可以用 screen：

```bash
# 安装 screen
sudo apt install screen

# 创建新会话
screen -S trading

# 运行程序
cd /home/ubuntu/ai-trading-team
uv run python main.py

# 分离会话 (Ctrl+A, D)

# 重新连接
screen -r trading
```

## 日志轮转

创建日志轮转配置：

```bash
sudo tee /etc/logrotate.d/ai-trading << EOF
/home/ubuntu/ai-trading-team/logs/*.log {
    daily
    rotate 7
    compress
    delaycompress
    missingok
    notifempty
    create 644 ubuntu ubuntu
}
EOF
```

## 监控建议

1. **设置告警**：使用 Telegram 通知关键事件
2. **监控内存**：`htop` 或 `watch free -h`
3. **监控进程**：`watch systemctl status ai-trading`
