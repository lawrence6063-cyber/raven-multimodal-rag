#!/bin/bash
# 监控消融实验进度

cd /Users/alaindong/Desktop/new_career/resume/ai项目/MODULAR-RAG-MCP-SERVER

echo "=== 消融实验监控 ==="
echo "开始时间: $(date)"
echo ""

# 检查进程
if ps aux | grep -E "(run_ablation|evaluate.py)" | grep -v grep > /dev/null; then
    echo "✅ 实验正在运行"
    echo ""
    echo "当前进程："
    ps aux | grep -E "(run_ablation|evaluate.py)" | grep -v grep
else
    echo "❌ 实验已停止"
fi

echo ""
echo "=== 已完成的配置 ==="
if [ -f "ablation_result.json" ]; then
    echo "ablation_result.json 已生成"
    python3 -c "
import json
try:
    with open('ablation_result.json') as f:
        content = f.read().strip()
        # 尝试解析 JSON
        if content.endswith(']'):
            data = json.loads(content)
            print(f'已完成 {len(data)} 个配置:')
            for d in data:
                print(f\"  - {d['pipeline']}: HR={d['hit_rate']:.4f}, MRR={d['mrr']:.4f}, RC={d['recall_completeness']:.4f}\")
        else:
            print('JSON 还未完成（部分写入）')
            print(f'当前文件大小: {len(content)} bytes')
except Exception as e:
    print(f'读取失败: {e}')
"
else
    echo "ablation_result.json 还未生成"
fi

echo ""
echo "=== 最新日志 ==="
if [ -f "logs/evaluation.log" ]; then
    echo "最后 20 行日志："
    tail -20 logs/evaluation.log
fi

echo ""
echo "监控时间: $(date)"
