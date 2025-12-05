"""
Visualizador del Cubo 2x2 - Muestra cómo el agente resuelve el cubo
"""

import argparse
import time
import torch
import numpy as np

# Colores ANSI para terminal
COLORS = {
    0: '\033[97;47m',  # U - Blanco (fondo blanco)
    1: '\033[93;43m',  # D - Amarillo
    2: '\033[92;42m',  # F - Verde
    3: '\033[94;44m',  # B - Azul
    4: '\033[91;41m',  # L - Rojo (naranja)
    5: '\033[95;45m',  # R - Magenta (rojo)
}
RESET = '\033[0m'
FACE_NAMES = ['U', 'D', 'F', 'B', 'L', 'R']
FACE_COLORS = ['Blanco', 'Amarillo', 'Verde', 'Azul', 'Naranja', 'Rojo']
MOVE_NAMES = ['R', "R'", 'R2', 'U', "U'", 'U2', 'F', "F'", 'F2']

import rubik2x2_c

def obs_to_state(obs):
    """Convierte observación one-hot a estado del cubo (colores)."""
    state = np.zeros((6, 4), dtype=int)
    obs_np = obs.cpu().numpy() if torch.is_tensor(obs) else obs

    for sticker_idx in range(24):
        face = sticker_idx // 4
        pos = sticker_idx % 4
        # Encuentra qué color está activo en el one-hot
        start = sticker_idx * 6
        for color in range(6):
            if obs_np[start + color] > 0.5:
                state[face][pos] = color
                break
    return state

def print_cube(state, move_history=None):
    """Imprime el cubo 2x2 en formato de cruz desplegada."""
    print("\033[2J\033[H")  # Limpiar pantalla

    def colored_cell(color):
        return f"{COLORS[color]}  {RESET}"

    # Formato:
    #       U
    #     L F R B
    #       D

    print("\n" + "="*40)
    if move_history:
        print(f"Movimientos: {' '.join(move_history[-10:])}")
        print(f"Total: {len(move_history)} movimientos")
    print("="*40 + "\n")

    # Cara U (arriba)
    print("        " + colored_cell(state[0][0]) + colored_cell(state[0][1]))
    print("        " + colored_cell(state[0][2]) + colored_cell(state[0][3]))
    print()

    # Caras L, F, R, B (en una línea)
    # L
    line1 = colored_cell(state[4][0]) + colored_cell(state[4][1])
    line2 = colored_cell(state[4][2]) + colored_cell(state[4][3])
    # F
    line1 += colored_cell(state[2][0]) + colored_cell(state[2][1])
    line2 += colored_cell(state[2][2]) + colored_cell(state[2][3])
    # R
    line1 += colored_cell(state[5][0]) + colored_cell(state[5][1])
    line2 += colored_cell(state[5][2]) + colored_cell(state[5][3])
    # B
    line1 += colored_cell(state[3][0]) + colored_cell(state[3][1])
    line2 += colored_cell(state[3][2]) + colored_cell(state[3][3])

    print(line1)
    print(line2)
    print()

    # Cara D (abajo)
    print("        " + colored_cell(state[1][0]) + colored_cell(state[1][1]))
    print("        " + colored_cell(state[1][2]) + colored_cell(state[1][3]))
    print()

    # Leyenda
    print("Caras: ", end="")
    for i, (name, color) in enumerate(zip(FACE_NAMES, FACE_COLORS)):
        print(f"{COLORS[i]} {name} {RESET}={color}", end="  ")
    print("\n")

def is_solved(state):
    """Verifica si el cubo está resuelto."""
    for face in range(6):
        if not all(state[face] == state[face][0]):
            return False
    return True

def load_policy(model_path, device='cpu'):
    """Carga el modelo entrenado."""
    from train_2x2 import Policy2x2

    checkpoint = torch.load(model_path, map_location=device)
    args = checkpoint.get('args', {})

    policy = Policy2x2(
        obs_size=144,
        action_size=9,
        hidden_size=args.get('hidden_size', 256),
        num_blocks=args.get('num_blocks', 2),
    )

    # Cargar pesos (manejar torch.compile)
    state_dict = checkpoint['policy_state_dict']
    # Remover prefijo '_orig_mod.' si existe (de torch.compile)
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('_orig_mod.'):
            new_state_dict[k[10:]] = v
        else:
            new_state_dict[k] = v

    policy.load_state_dict(new_state_dict)
    policy.eval()
    return policy, checkpoint.get('scramble_level', 7)

def solve_cube(policy, scramble_moves=7, delay=0.5, device='cpu'):
    """Resuelve un cubo scrambleado y muestra el proceso."""

    # Crear entorno
    env = rubik2x2_c.Rubik2x2Env(
        scramble_moves=scramble_moves,
        max_steps=50,
        seed=int(time.time()) % 10000
    )

    obs, _ = env.reset()
    obs_tensor = torch.FloatTensor(obs).to(device)

    state = obs_to_state(obs_tensor)
    move_history = []

    print_cube(state, move_history)
    print(f"Cubo scrambleado con {scramble_moves} movimientos")
    print("Presiona Enter para iniciar...")
    input()

    max_steps = 50
    for step in range(max_steps):
        # Obtener acción del modelo
        with torch.no_grad():
            logits, _ = policy(obs_tensor.unsqueeze(0))
            action = logits.argmax(dim=-1).item()

        move_history.append(MOVE_NAMES[action])

        # Simular el movimiento localmente para mostrar el estado ANTES del auto-reset
        # (El entorno hace auto-reset después de resolver, así que obs ya sería el nuevo cubo)
        state = obs_to_state(obs_tensor)
        # Aplicar movimiento manualmente al estado para visualización
        # Usamos un env temporal solo para aplicar el movimiento
        temp_env = rubik2x2_c.Rubik2x2Env(scramble_moves=1, max_steps=50, seed=1)
        temp_env.reset()
        # Copiar estado actual al temp_env no es posible directamente,
        # así que ejecutamos el step y confiamos en la lógica

        # Ejecutar acción (esto puede causar auto-reset si resuelve)
        obs, reward, terminal, truncation, _ = env.step(action)

        if terminal:
            # El cubo fue resuelto! Mostrar estado resuelto (todas las caras de un color)
            solved_state = np.zeros((6, 4), dtype=int)
            for face in range(6):
                solved_state[face] = face  # Cada cara tiene su color
            print_cube(solved_state, move_history)
            print(f"RESUELTO en {len(move_history)} movimientos!")
            print(f"Secuencia: {' '.join(move_history)}")
            return True, len(move_history)

        # No resuelto, actualizar estado normalmente
        obs_tensor = torch.FloatTensor(obs).to(device)
        state = obs_to_state(obs_tensor)
        print_cube(state, move_history)

        if truncation:
            print("Tiempo agotado - no resolvió")
            return False, len(move_history)

        time.sleep(delay)

    print("Máximo de pasos alcanzado")
    return False, len(move_history)

def batch_test(policy, scramble_moves=7, num_tests=100, device='cpu'):
    """Prueba el modelo en muchos cubos y reporta estadísticas."""

    env = rubik2x2_c.Rubik2x2BatchEnv(
        num_envs=num_tests,
        scramble_moves=scramble_moves,
        max_steps=50,
        seed=42
    )

    obs, _ = env.reset()
    obs_tensor = torch.FloatTensor(obs).to(device)

    solved = np.zeros(num_tests, dtype=bool)
    steps_to_solve = np.zeros(num_tests)

    for step in range(50):
        with torch.no_grad():
            logits, _ = policy(obs_tensor)
            actions = logits.argmax(dim=-1).cpu().numpy().astype(np.int32)

        obs, rewards, terminals, truncations, _ = env.step(actions)
        obs_tensor = torch.FloatTensor(obs).to(device)

        # Registrar resueltos
        for i in range(num_tests):
            if terminals[i] and not solved[i]:
                solved[i] = True
                steps_to_solve[i] = step + 1

    solve_rate = solved.mean()
    avg_steps = steps_to_solve[solved].mean() if solved.any() else 0

    print(f"\n{'='*50}")
    print(f"TEST: {num_tests} cubos con {scramble_moves} scrambles")
    print(f"{'='*50}")
    print(f"Resueltos: {solved.sum()}/{num_tests} ({solve_rate:.1%})")
    if solved.any():
        print(f"Pasos promedio: {avg_steps:.1f}")
        print(f"Pasos mínimo: {steps_to_solve[solved].min():.0f}")
        print(f"Pasos máximo: {steps_to_solve[solved].max():.0f}")
    print(f"{'='*50}\n")

    return solve_rate, avg_steps

def main():
    parser = argparse.ArgumentParser(description="Visualizar solución del cubo 2x2")
    parser.add_argument("--model", type=str, default="rubik_2x2.pt", help="Ruta al modelo")
    parser.add_argument("--scramble", type=int, default=None, help="Movimientos de scramble")
    parser.add_argument("--delay", type=float, default=0.3, help="Delay entre movimientos (segundos)")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--batch-test", action="store_true", help="Ejecutar test en batch")
    parser.add_argument("--num-tests", type=int, default=100, help="Número de tests en batch")
    parser.add_argument("--interactive", action="store_true", help="Modo interactivo (múltiples cubos)")

    args = parser.parse_args()

    print("Cargando modelo...")
    policy, trained_level = load_policy(args.model, args.device)
    print(f"Modelo cargado (entrenado hasta nivel {trained_level})")

    scramble = args.scramble if args.scramble else trained_level

    if args.batch_test:
        # Test en batch para diferentes niveles
        print("\nProbando diferentes niveles de scramble:\n")
        for level in range(1, 12):
            batch_test(policy, scramble_moves=level, num_tests=args.num_tests, device=args.device)
    elif args.interactive:
        # Modo interactivo
        while True:
            print(f"\nNivel de scramble actual: {scramble}")
            print("Opciones: [Enter]=resolver, [n]=nuevo nivel, [q]=salir")
            choice = input("> ").strip().lower()

            if choice == 'q':
                break
            elif choice == 'n':
                try:
                    scramble = int(input("Nuevo nivel (1-11): "))
                except:
                    pass
            else:
                solve_cube(policy, scramble_moves=scramble, delay=args.delay, device=args.device)
    else:
        # Resolver un cubo
        solve_cube(policy, scramble_moves=scramble, delay=args.delay, device=args.device)

if __name__ == "__main__":
    main()
