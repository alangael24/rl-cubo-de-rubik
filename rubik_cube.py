"""
Simulador de Cubo de Rubik 3x3

Representacion del cubo:
- 6 caras: U (Up/Arriba), D (Down/Abajo), F (Front/Frente),
           B (Back/Atras), L (Left/Izquierda), R (Right/Derecha)
- Cada cara tiene 9 stickers numerados 0-8:
    0 1 2
    3 4 5
    6 7 8
- Colores: 0=Blanco(U), 1=Amarillo(D), 2=Verde(F),
           3=Azul(B), 4=Naranja(L), 5=Rojo(R)
"""

import numpy as np
from typing import Optional


class RubiksCube:
    # Indices de las caras
    U, D, F, B, L, R = 0, 1, 2, 3, 4, 5

    # Nombres de movimientos
    MOVE_NAMES = ["F", "F'", "B", "B'", "U", "U'", "D", "D'", "L", "L'", "R", "R'"]

    def __init__(self):
        """Inicializa un cubo resuelto."""
        # Cada cara tiene 9 stickers con el color de esa cara
        self.state = np.zeros((6, 9), dtype=np.int8)
        self.reset()

    def reset(self):
        """Resetea el cubo al estado resuelto."""
        for face in range(6):
            self.state[face, :] = face
        return self

    def copy(self) -> 'RubiksCube':
        """Crea una copia del cubo."""
        new_cube = RubiksCube()
        new_cube.state = self.state.copy()
        return new_cube

    def _rotate_face_cw(self, face: int):
        """Rota una cara 90 grados en sentido horario."""
        # Indices: 0 1 2    6 3 0
        #          3 4 5 -> 7 4 1
        #          6 7 8    8 5 2
        f = self.state[face].copy()
        self.state[face, 0] = f[6]
        self.state[face, 1] = f[3]
        self.state[face, 2] = f[0]
        self.state[face, 3] = f[7]
        # 4 se queda igual (centro)
        self.state[face, 5] = f[1]
        self.state[face, 6] = f[8]
        self.state[face, 7] = f[5]
        self.state[face, 8] = f[2]

    def _rotate_face_ccw(self, face: int):
        """Rota una cara 90 grados en sentido antihorario."""
        # Indices: 0 1 2    2 5 8
        #          3 4 5 -> 1 4 7
        #          6 7 8    0 3 6
        f = self.state[face].copy()
        self.state[face, 0] = f[2]
        self.state[face, 1] = f[5]
        self.state[face, 2] = f[8]
        self.state[face, 3] = f[1]
        # 4 se queda igual (centro)
        self.state[face, 5] = f[7]
        self.state[face, 6] = f[0]
        self.state[face, 7] = f[3]
        self.state[face, 8] = f[6]

    def move_F(self):
        """Movimiento F: Rotar cara frontal en sentido horario."""
        self._rotate_face_cw(self.F)
        # Guardar fila inferior de U
        temp = self.state[self.U, 6:9].copy()
        # U[6,7,8] <- L[8,5,2]
        self.state[self.U, 6] = self.state[self.L, 8]
        self.state[self.U, 7] = self.state[self.L, 5]
        self.state[self.U, 8] = self.state[self.L, 2]
        # L[2,5,8] <- D[0,1,2]
        self.state[self.L, 2] = self.state[self.D, 0]
        self.state[self.L, 5] = self.state[self.D, 1]
        self.state[self.L, 8] = self.state[self.D, 2]
        # D[0,1,2] <- R[6,3,0]
        self.state[self.D, 0] = self.state[self.R, 6]
        self.state[self.D, 1] = self.state[self.R, 3]
        self.state[self.D, 2] = self.state[self.R, 0]
        # R[0,3,6] <- temp
        self.state[self.R, 0] = temp[0]
        self.state[self.R, 3] = temp[1]
        self.state[self.R, 6] = temp[2]
        return self

    def move_F_prime(self):
        """Movimiento F': Rotar cara frontal en sentido antihorario."""
        self._rotate_face_ccw(self.F)
        temp = self.state[self.U, 6:9].copy()
        # U[6,7,8] <- R[0,3,6]
        self.state[self.U, 6] = self.state[self.R, 0]
        self.state[self.U, 7] = self.state[self.R, 3]
        self.state[self.U, 8] = self.state[self.R, 6]
        # R[0,3,6] <- D[2,1,0]
        self.state[self.R, 0] = self.state[self.D, 2]
        self.state[self.R, 3] = self.state[self.D, 1]
        self.state[self.R, 6] = self.state[self.D, 0]
        # D[0,1,2] <- L[2,5,8]
        self.state[self.D, 0] = self.state[self.L, 2]
        self.state[self.D, 1] = self.state[self.L, 5]
        self.state[self.D, 2] = self.state[self.L, 8]
        # L[2,5,8] <- temp[2,1,0]
        self.state[self.L, 2] = temp[2]
        self.state[self.L, 5] = temp[1]
        self.state[self.L, 8] = temp[0]
        return self

    def move_B(self):
        """Movimiento B: Rotar cara trasera en sentido horario."""
        self._rotate_face_cw(self.B)
        temp = self.state[self.U, 0:3].copy()
        # U[0,1,2] <- R[2,5,8]
        self.state[self.U, 0] = self.state[self.R, 2]
        self.state[self.U, 1] = self.state[self.R, 5]
        self.state[self.U, 2] = self.state[self.R, 8]
        # R[2,5,8] <- D[8,7,6]
        self.state[self.R, 2] = self.state[self.D, 8]
        self.state[self.R, 5] = self.state[self.D, 7]
        self.state[self.R, 8] = self.state[self.D, 6]
        # D[6,7,8] <- L[0,3,6]
        self.state[self.D, 6] = self.state[self.L, 0]
        self.state[self.D, 7] = self.state[self.L, 3]
        self.state[self.D, 8] = self.state[self.L, 6]
        # L[0,3,6] <- temp[2,1,0]
        self.state[self.L, 0] = temp[2]
        self.state[self.L, 3] = temp[1]
        self.state[self.L, 6] = temp[0]
        return self

    def move_B_prime(self):
        """Movimiento B': Rotar cara trasera en sentido antihorario."""
        self._rotate_face_ccw(self.B)
        temp = self.state[self.U, 0:3].copy()
        # U[0,1,2] <- L[6,3,0]
        self.state[self.U, 0] = self.state[self.L, 6]
        self.state[self.U, 1] = self.state[self.L, 3]
        self.state[self.U, 2] = self.state[self.L, 0]
        # L[0,3,6] <- D[6,7,8]
        self.state[self.L, 0] = self.state[self.D, 6]
        self.state[self.L, 3] = self.state[self.D, 7]
        self.state[self.L, 6] = self.state[self.D, 8]
        # D[6,7,8] <- R[8,5,2]
        self.state[self.D, 6] = self.state[self.R, 8]
        self.state[self.D, 7] = self.state[self.R, 5]
        self.state[self.D, 8] = self.state[self.R, 2]
        # R[2,5,8] <- temp
        self.state[self.R, 2] = temp[0]
        self.state[self.R, 5] = temp[1]
        self.state[self.R, 8] = temp[2]
        return self

    def move_U(self):
        """Movimiento U: Rotar cara superior en sentido horario."""
        self._rotate_face_cw(self.U)
        temp = self.state[self.F, 0:3].copy()
        # F[0,1,2] <- R[0,1,2]
        self.state[self.F, 0:3] = self.state[self.R, 0:3]
        # R[0,1,2] <- B[0,1,2]
        self.state[self.R, 0:3] = self.state[self.B, 0:3]
        # B[0,1,2] <- L[0,1,2]
        self.state[self.B, 0:3] = self.state[self.L, 0:3]
        # L[0,1,2] <- temp
        self.state[self.L, 0:3] = temp
        return self

    def move_U_prime(self):
        """Movimiento U': Rotar cara superior en sentido antihorario."""
        self._rotate_face_ccw(self.U)
        temp = self.state[self.F, 0:3].copy()
        # F[0,1,2] <- L[0,1,2]
        self.state[self.F, 0:3] = self.state[self.L, 0:3]
        # L[0,1,2] <- B[0,1,2]
        self.state[self.L, 0:3] = self.state[self.B, 0:3]
        # B[0,1,2] <- R[0,1,2]
        self.state[self.B, 0:3] = self.state[self.R, 0:3]
        # R[0,1,2] <- temp
        self.state[self.R, 0:3] = temp
        return self

    def move_D(self):
        """Movimiento D: Rotar cara inferior en sentido horario."""
        self._rotate_face_cw(self.D)
        temp = self.state[self.F, 6:9].copy()
        # F[6,7,8] <- L[6,7,8]
        self.state[self.F, 6:9] = self.state[self.L, 6:9]
        # L[6,7,8] <- B[6,7,8]
        self.state[self.L, 6:9] = self.state[self.B, 6:9]
        # B[6,7,8] <- R[6,7,8]
        self.state[self.B, 6:9] = self.state[self.R, 6:9]
        # R[6,7,8] <- temp
        self.state[self.R, 6:9] = temp
        return self

    def move_D_prime(self):
        """Movimiento D': Rotar cara inferior en sentido antihorario."""
        self._rotate_face_ccw(self.D)
        temp = self.state[self.F, 6:9].copy()
        # F[6,7,8] <- R[6,7,8]
        self.state[self.F, 6:9] = self.state[self.R, 6:9]
        # R[6,7,8] <- B[6,7,8]
        self.state[self.R, 6:9] = self.state[self.B, 6:9]
        # B[6,7,8] <- L[6,7,8]
        self.state[self.B, 6:9] = self.state[self.L, 6:9]
        # L[6,7,8] <- temp
        self.state[self.L, 6:9] = temp
        return self

    def move_L(self):
        """Movimiento L: Rotar cara izquierda en sentido horario."""
        self._rotate_face_cw(self.L)
        temp = np.array([self.state[self.U, 0], self.state[self.U, 3], self.state[self.U, 6]])
        # U[0,3,6] <- B[8,5,2]
        self.state[self.U, 0] = self.state[self.B, 8]
        self.state[self.U, 3] = self.state[self.B, 5]
        self.state[self.U, 6] = self.state[self.B, 2]
        # B[2,5,8] <- D[6,3,0]
        self.state[self.B, 2] = self.state[self.D, 6]
        self.state[self.B, 5] = self.state[self.D, 3]
        self.state[self.B, 8] = self.state[self.D, 0]
        # D[0,3,6] <- F[0,3,6]
        self.state[self.D, 0] = self.state[self.F, 0]
        self.state[self.D, 3] = self.state[self.F, 3]
        self.state[self.D, 6] = self.state[self.F, 6]
        # F[0,3,6] <- temp
        self.state[self.F, 0] = temp[0]
        self.state[self.F, 3] = temp[1]
        self.state[self.F, 6] = temp[2]
        return self

    def move_L_prime(self):
        """Movimiento L': Rotar cara izquierda en sentido antihorario."""
        self._rotate_face_ccw(self.L)
        temp = np.array([self.state[self.U, 0], self.state[self.U, 3], self.state[self.U, 6]])
        # U[0,3,6] <- F[0,3,6]
        self.state[self.U, 0] = self.state[self.F, 0]
        self.state[self.U, 3] = self.state[self.F, 3]
        self.state[self.U, 6] = self.state[self.F, 6]
        # F[0,3,6] <- D[0,3,6]
        self.state[self.F, 0] = self.state[self.D, 0]
        self.state[self.F, 3] = self.state[self.D, 3]
        self.state[self.F, 6] = self.state[self.D, 6]
        # D[0,3,6] <- B[8,5,2]
        self.state[self.D, 0] = self.state[self.B, 8]
        self.state[self.D, 3] = self.state[self.B, 5]
        self.state[self.D, 6] = self.state[self.B, 2]
        # B[2,5,8] <- temp[2,1,0]
        self.state[self.B, 2] = temp[2]
        self.state[self.B, 5] = temp[1]
        self.state[self.B, 8] = temp[0]
        return self

    def move_R(self):
        """Movimiento R: Rotar cara derecha en sentido horario."""
        self._rotate_face_cw(self.R)
        temp = np.array([self.state[self.U, 2], self.state[self.U, 5], self.state[self.U, 8]])
        # U[2,5,8] <- F[2,5,8]
        self.state[self.U, 2] = self.state[self.F, 2]
        self.state[self.U, 5] = self.state[self.F, 5]
        self.state[self.U, 8] = self.state[self.F, 8]
        # F[2,5,8] <- D[2,5,8]
        self.state[self.F, 2] = self.state[self.D, 2]
        self.state[self.F, 5] = self.state[self.D, 5]
        self.state[self.F, 8] = self.state[self.D, 8]
        # D[2,5,8] <- B[6,3,0]
        self.state[self.D, 2] = self.state[self.B, 6]
        self.state[self.D, 5] = self.state[self.B, 3]
        self.state[self.D, 8] = self.state[self.B, 0]
        # B[0,3,6] <- temp[2,1,0]
        self.state[self.B, 0] = temp[2]
        self.state[self.B, 3] = temp[1]
        self.state[self.B, 6] = temp[0]
        return self

    def move_R_prime(self):
        """Movimiento R': Rotar cara derecha en sentido antihorario."""
        self._rotate_face_ccw(self.R)
        temp = np.array([self.state[self.U, 2], self.state[self.U, 5], self.state[self.U, 8]])
        # U[2,5,8] <- B[6,3,0]
        self.state[self.U, 2] = self.state[self.B, 6]
        self.state[self.U, 5] = self.state[self.B, 3]
        self.state[self.U, 8] = self.state[self.B, 0]
        # B[0,3,6] <- D[8,5,2]
        self.state[self.B, 0] = self.state[self.D, 8]
        self.state[self.B, 3] = self.state[self.D, 5]
        self.state[self.B, 6] = self.state[self.D, 2]
        # D[2,5,8] <- F[2,5,8]
        self.state[self.D, 2] = self.state[self.F, 2]
        self.state[self.D, 5] = self.state[self.F, 5]
        self.state[self.D, 8] = self.state[self.F, 8]
        # F[2,5,8] <- temp
        self.state[self.F, 2] = temp[0]
        self.state[self.F, 5] = temp[1]
        self.state[self.F, 8] = temp[2]
        return self

    def apply_move(self, move: int):
        """Aplica un movimiento por indice (0-11)."""
        moves = [
            self.move_F, self.move_F_prime,
            self.move_B, self.move_B_prime,
            self.move_U, self.move_U_prime,
            self.move_D, self.move_D_prime,
            self.move_L, self.move_L_prime,
            self.move_R, self.move_R_prime,
        ]
        moves[move]()
        return self

    def scramble(self, num_moves: int = 20, rng: Optional[np.random.Generator] = None):
        """Mezcla el cubo con movimientos aleatorios."""
        if rng is None:
            rng = np.random.default_rng()

        moves = rng.integers(0, 12, size=num_moves)
        for move in moves:
            self.apply_move(move)
        return self

    def is_solved(self) -> bool:
        """Verifica si el cubo esta resuelto."""
        for face in range(6):
            if not np.all(self.state[face] == face):
                return False
        return True

    def count_correct_stickers(self) -> int:
        """Cuenta cuantos stickers estan en su posicion correcta."""
        count = 0
        for face in range(6):
            count += np.sum(self.state[face] == face)
        return count

    def get_flat_state(self) -> np.ndarray:
        """Retorna el estado como un array plano de 54 elementos."""
        return self.state.flatten()

    def get_one_hot_state(self) -> np.ndarray:
        """Retorna el estado como one-hot encoding (54, 6)."""
        flat = self.get_flat_state()
        one_hot = np.zeros((54, 6), dtype=np.float32)
        one_hot[np.arange(54), flat] = 1.0
        return one_hot

    def __str__(self) -> str:
        """Representacion visual del cubo."""
        color_chars = ['W', 'Y', 'G', 'B', 'O', 'R']  # White, Yellow, Green, Blue, Orange, Red

        def face_str(face_idx):
            f = self.state[face_idx]
            return [
                f"{color_chars[f[0]]} {color_chars[f[1]]} {color_chars[f[2]]}",
                f"{color_chars[f[3]]} {color_chars[f[4]]} {color_chars[f[5]]}",
                f"{color_chars[f[6]]} {color_chars[f[7]]} {color_chars[f[8]]}",
            ]

        u = face_str(self.U)
        d = face_str(self.D)
        f = face_str(self.F)
        b = face_str(self.B)
        l = face_str(self.L)
        r = face_str(self.R)

        lines = [
            "        " + u[0],
            "        " + u[1],
            "        " + u[2],
            l[0] + " " + f[0] + " " + r[0] + " " + b[0],
            l[1] + " " + f[1] + " " + r[1] + " " + b[1],
            l[2] + " " + f[2] + " " + r[2] + " " + b[2],
            "        " + d[0],
            "        " + d[1],
            "        " + d[2],
        ]
        return "\n".join(lines)


if __name__ == "__main__":
    # Test del cubo
    cube = RubiksCube()
    print("Cubo resuelto:")
    print(cube)
    print(f"Resuelto: {cube.is_solved()}")
    print(f"Stickers correctos: {cube.count_correct_stickers()}/54")

    print("\nMezclando con 5 movimientos...")
    cube.scramble(5)
    print(cube)
    print(f"Resuelto: {cube.is_solved()}")
    print(f"Stickers correctos: {cube.count_correct_stickers()}/54")
