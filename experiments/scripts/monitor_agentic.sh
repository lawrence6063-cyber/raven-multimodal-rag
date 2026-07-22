#!/bin/bash
# Agentic multihop 监控脚本

LOG=~/Desktop/new_career/resume/ai项目/MODULAR-RAG-MCP-SERVER/eval_agentic_multihop.log
PID_FILE=~/Desktop/new_career/resume/ai项目/MODULAR-RAG-MCP-SERVER/agentic_multihop.pid

echo "=== Agentic Multihop 进度监控 ==="
echo ""

if [ -f "$PID_FILE" ]; then
    PID=$(cat "$PID_FILE")
    if ps -p $PID > /dev/null 2>&1; then
        echo "✅ 进程运行中 (PID: $PID)"
    else
        echo "✅ 进程已完成 (PID: $PID)"
    fi
else
    echo "⚠️  未找到 PID"
fi

echo ""
echo "=== 进度 (共 8 条 multihop query) ==="
grep -E "^\[" "$LOG" 2>/dev/null | tail -3 || echo "尚未开始"

echo ""
echo "=== 最新输出 ==="
tail -10 "$LOG" 2>/dev/null || echo "日志文件不存在"

echo ""
echo "---"
echo "实时监控: tail -f $LOG"
echo "查看结果: cat ~/Desktop/new_career/resume/ai项目/MODULAR-RAG-MCP-SERVER/eval_agentic_multihop.json"
