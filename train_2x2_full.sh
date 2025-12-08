#!/bin/bash
# Script para entrenar el cubo 2x2 hasta maestría (~95%+ en todos los niveles)

echo "=========================================="
echo "Entrenamiento COMPLETO del cubo 2x2"
echo "=========================================="
echo ""
echo "Parámetros:"
echo "  - 10M timesteps"
echo "  - 512 entornos paralelos"
echo "  - Success threshold: 0.99 (99%)"
echo "  - Min steps por nivel: 50,000"
echo ""
echo "Tiempo estimado: ~2-3 horas en CPU"
echo "                 ~30-40 min en GPU"
echo ""
echo "Presiona Ctrl+C para cancelar..."
sleep 3

python3 train_2x2.py \
    --total-timesteps 10_000_000 \
    --num-envs 512 \
    --success-threshold 0.99 \
    --min-steps-per-level 50000 \
    --curriculum-window 200 \
    --save-path rubik_2x2_master.pt \
    --device cpu

echo ""
echo "=========================================="
echo "Entrenamiento completado"
echo "Modelo guardado en: rubik_2x2_master.pt"
echo "=========================================="
echo ""
echo "Para evaluar el modelo:"
echo "  python3 visualize_2x2.py --model rubik_2x2_master.pt --batch-test --num-tests 1000"
