#!/bin/bash
# Nohup wrapper script for hyperparameter sweep
# Usage: ./run_sweep_nohup.sh [sequential|parallel]

# Default to sequential mode
MODE=${1:-sequential}

# Create logs directory
mkdir -p logs/sweep

# Generate timestamp for log file
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

if [ "$MODE" = "parallel" ]; then
    echo "Starting PARALLEL hyperparameter sweep in background..."
    echo "Log file: logs/sweep/sweep_parallel_${TIMESTAMP}.log"
    nohup python -u run_hyperparameter_sweep_parallel.py --yes > logs/sweep/sweep_parallel_${TIMESTAMP}.log 2>&1 &
    SWEEP_PID=$!
    echo "Sweep started with PID: $SWEEP_PID"
    echo $SWEEP_PID > logs/sweep/sweep_parallel.pid
    echo "To monitor: tail -f logs/sweep/sweep_parallel_${TIMESTAMP}.log"
    echo "To check status: ps -p $SWEEP_PID"
    echo "To stop: kill $SWEEP_PID"
elif [ "$MODE" = "sequential" ]; then
    echo "Starting SEQUENTIAL hyperparameter sweep in background..."
    echo "Log file: logs/sweep/sweep_sequential_${TIMESTAMP}.log"
    nohup python -u run_hyperparameter_sweep.py --yes > logs/sweep/sweep_sequential_${TIMESTAMP}.log 2>&1 &
    SWEEP_PID=$!
    echo "Sweep started with PID: $SWEEP_PID"
    echo $SWEEP_PID > logs/sweep/sweep_sequential.pid
    echo "To monitor: tail -f logs/sweep/sweep_sequential_${TIMESTAMP}.log"
    echo "To check status: ps -p $SWEEP_PID"
    echo "To stop: kill $SWEEP_PID"
else
    echo "Error: Invalid mode '$MODE'"
    echo "Usage: $0 [sequential|parallel]"
    exit 1
fi

# Wait a moment and check if process is running
sleep 2
if ps -p $SWEEP_PID > /dev/null; then
    echo "✅ Sweep is running successfully!"
else
    echo "❌ Sweep failed to start. Check the log file for errors."
    exit 1
fi
