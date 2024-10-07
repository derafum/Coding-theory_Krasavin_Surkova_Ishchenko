import numpy as np

# Генерация порождающей матрицы G (7, 4, 3)
def generate_G():
    I = np.eye(4, dtype=int)  # Единичная матрица размером 4x4
    X = np.array([[1, 1, 0], [1, 0, 1], [0, 1, 1], [1, 1, 1]])  # Дополнительная матрица X
    return np.hstack((I, X))  # Объединяем матрицы для получения G = [I | X]

G = generate_G()
print("\nПорождающая матрица G:")
print(G)

# Генерация проверочной матрицы H
def generate_H():
    X = np.array([[1, 1, 0], [1, 0, 1], [0, 1, 1], [1, 1, 1]])
    I = np.eye(3, dtype=int)
    return np.hstack((X.T, I))  # Формируем матрицу H = [X^T | I]

H = generate_H()
print("\nПроверочная матрица H:")
print(H)

# Генерация синдромов для однократных ошибок
def generate_syndromes(H):
    syndromes = {}
    for i in range(H.shape[1]):
        error_vector = np.zeros(H.shape[1], dtype=int)
        error_vector[i] = 1
        syndrome = np.dot(H, error_vector) % 2
        syndromes[tuple(syndrome)] = error_vector
    return syndromes

syndromes = generate_syndromes(H)
print("\nТаблица синдромов для однократных ошибок:")
for syndrome, error in syndromes.items():
    print(f"Синдром {syndrome}: Ошибка {error}")

# Создание кодового слова
def generate_codeword(data, G):
    return np.dot(data, G) % 2

# Внесение ошибки в кодовое слово
def introduce_error(codeword, position):
    codeword[position] ^= 1
    return codeword

# Вычисление синдрома
def calculate_syndrome(received_word, H):
    return np.dot(H, received_word) % 2

# Исправление ошибки с использованием синдрома
def correct_error(received_word, syndrome, syndromes):
    error_vector = syndromes.get(tuple(syndrome))
    if error_vector is not None:
        return (received_word + error_vector) % 2
    return received_word

# Пример использования
if __name__ == "__main__":
    data_word = np.array([1, 0, 1, 1])
    codeword = generate_codeword(data_word, G)
    print("\nКодовое слово:", codeword)

    error_position = 2
    received_word = introduce_error(codeword.copy(), error_position)
    print("Кодовое слово с ошибкой:", received_word)

    syndrome = calculate_syndrome(received_word, H)
    print("Синдром:", syndrome)

    corrected_word = correct_error(received_word, syndrome, syndromes)
    print("Исправленное слово:", corrected_word)

    # Пример внесения двукратной ошибки
    def introduce_double_error(codeword, positions):
        for position in positions:
            codeword[position] ^= 1
        return codeword

    error_positions = [2, 5]
    received_word = introduce_double_error(codeword.copy(), error_positions)
    print("\nКодовое слово с двукратной ошибкой:", received_word)

    syndrome = calculate_syndrome(received_word, H)
    print("Синдром:", syndrome)

    corrected_word = correct_error(received_word, syndrome, syndromes)
    print("Попытка исправления:", corrected_word)
    print("Полученное слово отличается от исходного:", not np.array_equal(corrected_word, codeword))
