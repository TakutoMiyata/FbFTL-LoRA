#!/bin/bash
# Force stop all hyperparameter sweep related processes
# Use this when stop_sweep.sh doesn't work

echo "========================================"
echo "Force Stop All Sweep Processes"
echo "========================================"

# Find all sweep-related processes
echo "Searching for sweep processes..."

# Find main sweep scripts
MAIN_SWEEPS=$(pgrep -f "run_hyperparameter_sweep" | tr '\n' ' ')

# Find quickstart processes started by sweeps (in hyperparameter directories)
CHILD_PROCESSES=$(pgrep -f "quickstart_resnet.py.*experiments/hyperparameter" | tr '\n' ' ')

echo ""
if [ -z "$MAIN_SWEEPS" ] && [ -z "$CHILD_PROCESSES" ]; then
    echo "✅ No sweep processes found"
    exit 0
fi

echo "Found processes:"
if [ -n "$MAIN_SWEEPS" ]; then
    echo "  Main sweep processes: $MAIN_SWEEPS"
    ps -p $MAIN_SWEEPS -o pid,etime,%cpu,%mem,cmd 2>/dev/null | head -10
fi

if [ -n "$CHILD_PROCESSES" ]; then
    echo "  Child processes: $CHILD_PROCESSES"
    ps -p $CHILD_PROCESSES -o pid,etime,%cpu,%mem,cmd 2>/dev/null | head -10
fi

echo ""
read -p "Stop all these processes? (yes/no): " confirm

if [ "$confirm" != "yes" ] && [ "$confirm" != "y" ]; then
    echo "Cancelled."
    exit 0
fi

STOPPED_COUNT=0

# Stop main sweep processes first
if [ -n "$MAIN_SWEEPS" ]; then
    echo ""
    echo "Stopping main sweep processes..."
    for PID in $MAIN_SWEEPS; do
        if ps -p $PID > /dev/null 2>&1; then
            echo "  Stopping PID $PID..."
            kill $PID 2>/dev/null
            sleep 1
            if ps -p $PID > /dev/null 2>&1; then
                kill -9 $PID 2>/dev/null
            fi
            STOPPED_COUNT=$((STOPPED_COUNT + 1))
        fi
    done
fi

# Stop child processes
if [ -n "$CHILD_PROCESSES" ]; then
    echo ""
    echo "Stopping child processes..."
    for PID in $CHILD_PROCESSES; do
        if ps -p $PID > /dev/null 2>&1; then
            echo "  Stopping PID $PID..."
            kill $PID 2>/dev/null
            sleep 1
            if ps -p $PID > /dev/null 2>&1; then
                kill -9 $PID 2>/dev/null
            fi
            STOPPED_COUNT=$((STOPPED_COUNT + 1))
        fi
    done
fi

# Clean up PID files
echo ""
echo "Cleaning up PID files..."
rm -f logs/sweep/sweep_parallel.pid
rm -f logs/sweep/sweep_sequential.pid

echo ""
echo "========================================"
echo "✅ Stopped $STOPPED_COUNT processes"
echo "========================================"

# Verify
sleep 2
REMAINING=$(pgrep -f "run_hyperparameter_sweep|quickstart_resnet.py.*hyperparameter" | wc -l)
if [ $REMAINING -gt 0 ]; then
    echo "⚠️  Warning: $REMAINING processes still running"
    echo "Run 'ps aux | grep hyperparameter' to check"
else
    echo "✅ All processes stopped successfully"
fi
