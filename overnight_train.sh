#!/bin/bash
echo "=========================================="
echo "   OVERNIGHT TRAINING PIPELINE"
echo "   Started: $(date)"
echo "=========================================="

echo ""
echo "Step 1/3: Optuna Hyperparameter Optimization"
echo "-------------------------------------------"
python src/tune.py

echo ""
echo "Step 2/3: Training with Optimized Parameters"
echo "-------------------------------------------"
python src/train.py

echo ""
echo "Step 3/3: Final Evaluation"
echo "-------------------------------------------"
python src/eval.py

echo ""
echo "=========================================="
echo "   PIPELINE COMPLETE!"
echo "   Finished: $(date)"
echo "=========================================="
echo ""
echo "Results:"
echo "  - Model: ./checkpoints/brain2hand-hybrid.ckpt"
echo "  - Confusion Matrix: ./eval_confusion_matrix.png"
echo "  - Hyperparameters: ./best_params.json"
