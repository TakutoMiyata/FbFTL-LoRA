#!/bin/bash
# Stop running hyperparameter sweep

echo "Checking for running sweeps..."

PARALLEL_PID_FILE="logs/sweep/sweep_parallel.pid"
SEQUENTIAL_PID_FILE="logs/sweep/sweep_sequential.pid"

STOPPED=0

# Function to stop a sweep and all its children
stop_sweep() {
    local mode=$1
    local pid_file=$2
    
    if [ -f "$pid_file" ]; then
        PID=$(cat "$pid_file")
        if ps -p $PID > /dev/null 2>&1; then
            echo "Stopping $mode sweep (PID: $PID) and child processes..."
            
            # Find all child processes
            CHILD_PIDS=$(pgrep -P $PID)
            
            # Kill parent process
            kill $PID
            
            # Wait for graceful shutdown
            for i in {1..10}; do
                if ! ps -p $PID > /dev/null 2>&1; then
                    echo "✅ $mode sweep parent process stopped"
                    break
                fi
                sleep 1
            done
            
            # Force kill parent if still running
            if ps -p $PID > /dev/null 2>&1; then
                echo "⚠️  Forcing $mode sweep parent to stop..."
                kill -9 $PID
            fi
            
            # Kill child processes
            if [ -n "$CHILD_PIDS" ]; then
                echo "Stopping child processes: $CHILD_PIDS"
                for CHILD_PID in $CHILD_PIDS; do
                    if ps -p $CHILD_PID > /dev/null 2>&1; then
                        kill $CHILD_PID 2>/dev/null
                        sleep 1
                        # Force kill if still running
                        if ps -p $CHILD_PID > /dev/null 2>&1; then
                            kill -9 $CHILD_PID 2>/dev/null
                        fi
                    fi
                done
                echo "✅ Child processes stopped"
            fi
            
            rm "$pid_file"
            STOPPED=$((STOPPED + 1))
        else
            echo "ℹ️  $mode sweep is not running (cleaning stale PID file)"
            rm "$pid_file"
        fi
    else
        echo "ℹ️  No $mode sweep PID file found"
        
        # Try to find sweep processes without PID file
        echo "Searching for orphaned sweep processes..."
        SWEEP_PIDS=$(pgrep -f "run_hyperparameter_sweep")
        if [ -n "$SWEEP_PIDS" ]; then
            echo "Found orphaned sweep processes: $SWEEP_PIDS"
            read -p "Stop these processes? (yes/no): " confirm
            if [ "$confirm" = "yes" ] || [ "$confirm" = "y" ]; then
                for SWEEP_PID in $SWEEP_PIDS; do
                    echo "Stopping PID $SWEEP_PID..."
                    # Find children of this sweep process
                    CHILDREN=$(pgrep -P $SWEEP_PID)
                    
                    # Kill parent
                    kill $SWEEP_PID 2>/dev/null
                    sleep 1
                    if ps -p $SWEEP_PID > /dev/null 2>&1; then
                        kill -9 $SWEEP_PID 2>/dev/null
                    fi
                    
                    # Kill children
                    if [ -n "$CHILDREN" ]; then
                        for CHILD in $CHILDREN; do
                            kill $CHILD 2>/dev/null
                            sleep 1
                            if ps -p $CHILD > /dev/null 2>&1; then
                                kill -9 $CHILD 2>/dev/null
                            fi
                        done
                    fi
                done
                echo "✅ Orphaned processes stopped"
                STOPPED=$((STOPPED + 1))
            fi
        fi
    fi
}

# Stop both types of sweeps
stop_sweep "Parallel" "$PARALLEL_PID_FILE"
stop_sweep "Sequential" "$SEQUENTIAL_PID_FILE"

echo ""
if [ $STOPPED -eq 0 ]; then
    echo "ℹ️  No running sweeps found"
else
    echo "🛑 Stopped $STOPPED sweep(s)"
fi

# Verify all processes are stopped
echo ""
echo "Verifying remaining processes..."
REMAINING=$(pgrep -f "run_hyperparameter_sweep|quickstart_resnet.py.*hyperparameter" | wc -l)
if [ $REMAINING -gt 0 ]; then
    echo "⚠️  Warning: $REMAINING related processes still running"
    echo "Run 'ps aux | grep hyperparameter' to check"
    echo "Use 'pkill -f hyperparameter_sweep' to force stop all"
else
    echo "✅ All sweep processes stopped"
fi
