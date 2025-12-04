# RL Cubo de Rubik

Entorno de Reinforcement Learning para resolver un cubo de Rubik 3x3 usando PufferLib.

## Instalacion

```bash
pip install -r requirements.txt
```

## Uso

### Entrenar el agente

```bash
python train.py
```

### Probar el entorno

```bash
python -c "from rubik_env import RubiksCubeEnv; env = RubiksCubeEnv(); print(env)"
```

## Estructura del proyecto

- `rubik_cube.py` - Simulador del cubo de Rubik 3x3
- `rubik_env.py` - Entorno PufferEnv para RL
- `train.py` - Script de entrenamiento
- `rubik.ini` - Configuracion de hiperparametros

## Espacio de observacion

El cubo de Rubik 3x3 tiene 6 caras con 9 stickers cada una (54 total).
Cada sticker puede ser uno de 6 colores, representado como one-hot encoding.
Observacion: (54, 6) = 324 valores

## Espacio de acciones

12 movimientos posibles:
- F, F', B, B' (Front, Back)
- U, U', D, D' (Up, Down)
- L, L', R, R' (Left, Right)

## Recompensa

- +100 por resolver el cubo
- Recompensa intermedia basada en el numero de stickers correctos
- Penalizacion pequena por cada paso para fomentar soluciones cortas
