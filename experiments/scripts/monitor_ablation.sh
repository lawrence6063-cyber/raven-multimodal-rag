#!/bin/bash
# 消融实验进度监控脚本

LOG_FILE=~/Desktop/new_career/resume/ai项目/MODULAR-RAG-MCP-SERVER/ablation_run.log
PID_FILE=~/Desktop/new_career/resume/ai项目/MODULAR-RAG-MCP-SERVER/ablation.pid

echo "=== RAG Ablation 实验监控 ==="
echo ""

# 检查进程状态
if [ -f "$PID_FILE" ]; then
    PID=$(cat "$PID_FILE")
    if ps -p $PID > /dev/null 2>&1; then
        echo "✅ 进程运行中 (PID: $PID)"
    else
        echo "❌ 进程已结束 (PID: $PID)"
    fi
else
    echo "⚠️  未找到 PID 文件"
fi

echo ""
echo "=== 已完成的配置 ==="
grep "HR=" "$LOG_FILE" 2>/dev/null | tail -5 || echo "暂无输出"

echo ""
echo "=== 当前正在跑 ==="
grep "Running:" "$LOG_FILE" 2>/dev/null | tail -1 || echo "尚未开始"

echo ""
echo "=== 最新 10 行输出 ==="
tail -10 "$LOG_FILE" 2>/dev/null || echo "日志文件不存在"

echo ""
echo "---"
echo "持续监控: tail -f $LOG_FILE"
echo "查看汇总: tail -20 $LOG_FILE"
