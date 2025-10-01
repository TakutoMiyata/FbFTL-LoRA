#!/bin/bash
# Check the status of running hyperparameter sweep

echo "==================================="
echo "Hyperparameter Sweep Status"
echo "==================================="

# Check for PID files
PARALLEL_PID_FILE="logs/sweep/sweep_parallel.pid"
SEQUENTIAL_PID_FILE="logs/sweep/sweep_sequential.pid"

check_sweep() {
    local mode=$1
    local pid_file=$2
    
    if [ -f "$pid_file" ]; then
        PID=$(cat "$pid_file")
        if ps -p $PID > /dev/null 2>&1; then
            echo "✅ $mode sweep is RUNNING (PID: $PID)"
            
            # Get process info
            echo "   Process info:"
            ps -p $PID -o pid,etime,pcpu,pmem,cmd | tail -n 1 | sed 's/^/   /'
            
            # Find latest log file
            if [ "$mode" = "Parallel" ]; then
                LOG_FILE=$(ls -t logs/sweep/sweep_parallel_*.log 2>/dev/null | head -1)
            else
                LOG_FILE=$(ls -t logs/sweep/sweep_sequential_*.log 2>/dev/null | head -1)
            fi
            
            if [ -n "$LOG_FILE" ]; then
                echo "   Latest log: $LOG_FILE"
                
                # Try to extract progress information
                echo "   Recent activity:"
                tail -n 5 "$LOG_FILE" | sed 's/^/   /'
            fi
            
            return 0
        else
            echo "❌ $mode sweep is NOT running (stale PID: $PID)"
            return 1
        fi
    else
        echo "ℹ️  No $mode sweep PID file found"
        return 2
    fi
}

echo ""
check_sweep "Parallel" "$PARALLEL_PID_FILE"
echo ""
check_sweep "Sequential" "$SEQUENTIAL_PID_FILE"

echo ""
echo "==================================="
echo "Recent Sweep Logs:"
echo "==================================="
if [ -d "logs/sweep" ]; then
    ls -lht logs/sweep/*.log 2>/dev/null | head -5 | awk '{print $9, "("$5")"}'
else
    echo "No sweep logs found"
fi

echo ""
echo "==================================="
echo "Completed Experiments:"
echo "==================================="
# Count completed experiments in sweep directories
SWEEP_DIRS=$(find experiments/hyperparameter_sweep* -maxdepth 1 -mindepth 1 -type d 2>/dev/null | sort -r | head -3)
if [ -n "$SWEEP_DIRS" ]; then
    for dir in $SWEEP_DIRS; do
        if [ -d "$dir" ]; then
            TOTAL=$(find "$dir" -maxdepth 1 -mindepth 1 -type d | wc -l)
            COMPLETED=$(find "$dir" -name "final_results_*.json" | wc -l)
            echo "$(basename $dir): $COMPLETED/$TOTAL experiments completed"
        fi
    done
else
    echo "No sweep experiments found"
fi

echo ""
echo "==================================="
echo "GPU Usage:"
echo "==================================="
nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits | \
    awk -F', ' '{printf "GPU %s (%s): %s%% util, %sMB / %sMB\n", $1, $2, $3, $4, $5}'

echo ""
