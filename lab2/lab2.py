from dataclasses import dataclass, field

import numpy as np


# Часть 1: Определение линейного кода и основные операции
@dataclass
class LinearCode:
    k: int  # Длина информационного слова
    n: int  # Длина кодового слова
    g: np.ndarray = field(init=False)
    h: np.ndarray = field(init=False)
    syndrome_table: dict[tuple[int, ...], np.ndarray] = field(init=False)

    def __post_init__(self):
        self.g = self._generate_matrix_g()
        self.h = self._generate_matrix_h()
        self.syndrome_table = self._generate_syndrome_table()

    def _generate_matrix_g(self) -> np.ndarray:
        """Создает порождающую матрицу G для кода (7, 4, 3)."""
        i_k = np.identity(self.k, dtype=int)
        x = np.array([[1, 0, 1],
                      [1, 1, 0],
                      [0, 1, 1],
                      [1, 1, 1]])  # Пример для d=3
        g = np.hstack((i_k, x))
        return g

    def _generate_matrix_h(self) -> np.ndarray:
        """Создает проверочную матрицу H на основе G."""
        x = self.g[:, self.k:]
        i_n_k = np.identity(self.n - self.k, dtype=int)
        h = np.hstack((x.T, i_n_k))
        return h

    def _generate_syndrome_table(self) -> dict[tuple[int, ...], np.ndarray]:
        """Создает таблицу синдромов для одноразрядных и двукратных ошибок."""
        n = self.h.shape[1]
        syndromes = {}

        # Для одноразрядных ошибок
        for i in range(n):
            error = np.zeros(n, dtype=int)
            error[i] = 1
            syndrome = tuple(self.calculate_syndrome(error))
            syndromes[syndrome] = error

        # Для двукратных ошибок
        for i in range(n):
            for j in range(i + 1, n):
                error = np.zeros(n, dtype=int)
                error[i] = 1
                error[j] = 1
                syndrome = tuple(self.calculate_syndrome(error))
                syndromes[syndrome] = error

        return syndromes

    def calculate_syndrome(self, received_word: np.ndarray) -> np.ndarray:
        """Вычисляет синдром для полученного кодового слова."""
        return np.dot(received_word, self.h.T) % 2


@dataclass
class Encoder:
    linear_code: LinearCode

    def encode(self, message: np.ndarray) -> np.ndarray:
        """Кодирует сообщение с использованием порождающей матрицы G."""
        if len(message) != self.linear_code.k:
            raise ValueError(f"Длина сообщения должна быть {self.linear_code.k}.")
        return np.dot(message, self.linear_code.g) % 2


@dataclass
class Decoder:
    linear_code: LinearCode

    @staticmethod
    def introduce_error(codeword: np.ndarray, error_positions: tuple[int, ...]) -> np.ndarray:
        """Вносит ошибки в кодовое слово на указанных позициях."""
        codeword_with_error = codeword.copy()
        for pos in error_positions:
            if pos < 0 or pos >= len(codeword):
                raise ValueError(f"Позиция ошибки {pos} вне диапазона кодового слова.")
            codeword_with_error[pos] ^= 1  # Инвертируем бит
        return codeword_with_error

    def correct_errors(self, received_word: np.ndarray) -> np.ndarray:
        """Исправляет ошибки с использованием таблицы синдромов."""
        syndrome = tuple(self.linear_code.calculate_syndrome(received_word))
        error = self.linear_code.syndrome_table.get(syndrome)
        if error is not None:
            corrected_word = (received_word + error) % 2
            return corrected_word
        return received_word  # Если синдром не найден, ошибка не исправляется


def main():
    # Инициализация кода с параметрами (k=4, n=7)
    linear_code = LinearCode(k=4, n=7)
    encoder = Encoder(linear_code)
    decoder = Decoder(linear_code)

    # Пример сообщения для кодирования
    message = np.array([1, 0, 1, 1])
    print("Оригинальное сообщение:", message)

    # Кодирование сообщения
    codeword = encoder.encode(message)
    print("Закодированное кодовое слово:", codeword)

    # Внесение одноразрядной ошибки
    received_word = decoder.introduce_error(codeword, (2,))
    print("Кодовое слово с одноразрядной ошибкой:", received_word)

    # Исправление одноразрядной ошибки
    corrected_word = decoder.correct_errors(received_word)
    print("Исправленное кодовое слово:", corrected_word)

    # Проверка результата
    if np.array_equal(codeword, corrected_word):
        print("Ошибка успешно исправлена!")
    else:
        print("Ошибка не исправлена.")

    # Внесение двукратной ошибки
    received_word = decoder.introduce_error(codeword, (1, 3))
    print("Кодовое слово с двукратной ошибкой:", received_word)

    # Исправление двукратной ошибки
    corrected_word = decoder.correct_errors(received_word)
    print("Исправленное кодовое слово после двукратной ошибки:", corrected_word)

    # Проверка результата
    if np.array_equal(codeword, corrected_word):
        print("Двукратная ошибка успешно исправлена!")
    else:
        print("Двукратная ошибка не исправлена.")


if __name__ == "__main__":
    main()
