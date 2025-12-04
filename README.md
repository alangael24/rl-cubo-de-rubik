# RL Cubo de Rubik

Entorno de Reinforcement Learning para resolver un cubo de Rubik 3x3 usando PufferLib.

Implementa recomendaciones de papers academicos:
- **DeepCubeA** (McAleer et al., 2019)
- **Autodidactic Iteration (ADI)**

## Instalacion

```bash
pip install -r requirements.txt
```

## Uso

### Entrenar con Curriculum Learning (recomendado)

```bash
# Entrenamiento basico con curriculum
python train.py

# Con parametros personalizados
python train.py --max-scramble 10 --success-threshold 0.8 --total-timesteps 500000
```

### Con PufferLib (alto rendimiento)

```bash
pip install pufferlib
python train.py --use-pufferlib
```

### Probar el entorno

```bash
python rubik_env.py
python rubik_cube.py
```

## Estructura del proyecto

| Archivo | Descripcion |
|---------|-------------|
| `rubik_cube.py` | Simulador del cubo de Rubik 3x3 |
| `rubik_env.py` | Entorno PufferEnv para RL |
| `train.py` | Script de entrenamiento con curriculum |
| `rubik.ini` | Configuracion de hiperparametros |

## Caracteristicas academicas

### 1. Curriculum Learning (ADI)

En lugar de intentar resolver cubos mezclados con 20 movimientos desde el inicio
(casi imposible), el entrenamiento progresa gradualmente:

1. Empezar con cubos de **1 scramble move**
2. Cuando `success_rate > 80%`, aumentar a **2 scramble moves**
3. Continuar hasta **20 scramble moves** (God's number)

```bash
python train.py --start-scramble 1 --max-scramble 20 --success-threshold 0.8
```

### 2. Recompensa Sparse (recomendado)

Los papers advierten que contar stickers correctos crea **minimos locales**.
Por defecto usamos recompensa sparse:

- `+1.0` solo cuando el cubo esta **completamente resuelto**
- `-0.01` penalizacion por cada paso

```bash
# Sparse (default, recomendado)
python train.py --reward-mode sparse

# Dense (puede causar minimos locales)
python train.py --reward-mode dense
```

### 3. One-Hot Encoding

El estado se representa como one-hot encoding (54 stickers x 6 colores = 324 valores).
Esto es el estandar validado por la literatura.

## Espacio de observacion

- **Shape**: `(324,)` - One-hot encoding aplanado
- **Contenido**: 54 stickers x 6 colores posibles
- **Tipo**: `float32` en rango `[0, 1]`

## Espacio de acciones

12 movimientos discretos:

| Indice | Movimiento | Descripcion |
|--------|------------|-------------|
| 0 | F | Front clockwise |
| 1 | F' | Front counter-clockwise |
| 2 | B | Back clockwise |
| 3 | B' | Back counter-clockwise |
| 4 | U | Up clockwise |
| 5 | U' | Up counter-clockwise |
| 6 | D | Down clockwise |
| 7 | D' | Down counter-clockwise |
| 8 | L | Left clockwise |
| 9 | L' | Left counter-clockwise |
| 10 | R | Right clockwise |
| 11 | R' | Right counter-clockwise |

## Parametros del curriculum

| Parametro | Default | Descripcion |
|-----------|---------|-------------|
| `--start-scramble` | 1 | Scramble inicial |
| `--max-scramble` | 20 | Scramble maximo (God's number) |
| `--success-threshold` | 0.8 | Tasa de exito para avanzar |
| `--eval-window` | 100 | Episodios para evaluar |
| `--steps-per-level` | 50000 | Min pasos por nivel |

## Referencias

- McAleer, S., et al. "Solving the Rubik's Cube with Deep Reinforcement Learning and Search." arXiv:1805.07470 (2019)
- Silver, D., et al. "Mastering the game of Go with deep neural networks and tree search." Nature (2016)
