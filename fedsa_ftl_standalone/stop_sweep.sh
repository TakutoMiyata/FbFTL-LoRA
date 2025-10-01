#!/bin/bash
# Stop running hyperparameter sweep

echo "Checking for running sweeps..."

PARALLEL_PID_FILE="logs/sweep/sweep_parallel.pid"
SEQUENTIAL_PID_FILE="logs/sweep/sweep_sequential.pid"

STOPPED=0

# Function to stop a sweep
stop_sweep() {
    local mode=$1
    local pid_file=$2
    
    if [ -f "$pid_file" ]; then
        PID=$(cat "$pid_file")
        if ps -p $PID > /dev/null 2>&1; then
            echo "Stopping $mode sweep (PID: $PID)..."
            kill $PID
            
            # Wait for graceful shutdown
            for i in {1..10}; do
                if ! ps -p $PID > /dev/null 2>&1; then
                    echo "✅ $mode sweep stopped successfully"
                    rm "$pid_file"
                    STOPPED=$((STOPPED + 1))
                    return 0
                fi
                sleep 1
            done
            
            # Force kill if still running
            if ps -p $PID > /dev/null 2>&1; then
                echo "⚠️  Forcing $mode sweep to stop..."
                kill -9 $PID
                rm "$pid_file"
                echo "✅ $mode sweep force stopped"
                STOPPED=$((STOPPED + 1))
            fi
        else
            echo "ℹ️  $mode sweep is not running (cleaning stale PID file)"
            rm "$pid_file"
        fi
    else
        echo "ℹ️  No $mode sweep PID file found"
    fi
}

# Stop both types of sweeps
stop_sweep "Parallel" "$PARALLEL_PID_FILE"
stop_sweep "Sequential" "$SEQUENTIAL_PID_FILE"

if [ $STOPPED -eq 0 ]; then
    echo ""
    echo "ℹ️  No running sweeps found"
else
    echo ""
    echo "🛑 Stopped $STOPPED sweep(s)"
fi
