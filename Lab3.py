import random

def identity_matrix(size):
    return [[1 if i == j else 0 for j in range(size)] for i in range(size)]

def combine_horizontally(matrix_a, matrix_b):
    return [a_row + b_row for a_row, b_row in zip(matrix_a, matrix_b)]

def combine_vertically(matrix_a, matrix_b):
    return matrix_a + matrix_b

def generate_combinations(n, k):
    result = []
    vector = [0] * (n - k)
    while len(result) < k:
        for i in reversed(range(len(vector))):
            if vector[i] == 0:
                vector[i] = 1
                result.append(vector[:])
                break
            else:
                vector[i] = 0
    return result

def create_hamming_matrix(r):
    n = 2 ** r - 1
    k = n - r
    identity = identity_matrix(k)
    combinations = generate_combinations(n, k)
    return combine_horizontally(identity, combinations)

def create_parity_check_matrix(r):
    n = 2 ** r - 1
    k = n - r
    identity = identity_matrix(n - k)
    combinations = generate_combinations(n, k)
    return combine_vertically(combinations, identity)

def multiply_vector_by_matrix(vector, matrix):
    return [sum(v * m for v, m in zip(vector, col)) % 2 for col in zip(*matrix)]

def create_syndrome_dict(h_matrix):
    syndrome_dict = {}
    for i in range(len(h_matrix[0])):
        error = [0] * len(h_matrix[0])
        error[i] = 1
        syndrome = multiply_vector_by_matrix(error, h_matrix)
        syndrome_dict[tuple(syndrome)] = error
    return syndrome_dict

def generate_extended_hamming_matrix(r):
    base_matrix = create_hamming_matrix(r)
    for row in base_matrix:
        row.append(sum(row) % 2)
    return base_matrix

def generate_extended_parity_matrix(r):
    parity_matrix = create_parity_check_matrix(r)
    new_row = [0] * len(parity_matrix[0])
    parity_matrix.append(new_row)
    for row in parity_matrix:
        row.append(1)
    return parity_matrix

def create_error_vector(size, error_count):
    vector = [0] * size
    indices = random.sample(range(size), error_count)
    for idx in indices:
        vector[idx] = 1
    return vector

def fix_errors(h_matrix, received):
    syndrome = multiply_vector_by_matrix(received, h_matrix)
    syndrome_dict = create_syndrome_dict(h_matrix)
    if tuple(syndrome) in syndrome_dict:
        error = syndrome_dict[tuple(syndrome)]
        corrected = [(bit + err) % 2 for bit, err in zip(received, error)]
        return corrected, syndrome
    return received, syndrome

def analyze_hamming_code(r, extended=False):
    if extended:
        generator_matrix = generate_extended_hamming_matrix(r)
        parity_matrix = generate_extended_parity_matrix(r)
        max_errors = 4
        print("\nАнализ расширенного кода Хэмминга")
    else:
        generator_matrix = create_hamming_matrix(r)
        parity_matrix = create_parity_check_matrix(r)
        max_errors = 3
        print("\nАнализ стандартного кода Хэмминга")

    print("\nПорождающая матрица G:")
    for row in generator_matrix:
        print(row)

    print("\nПроверочная матрица H:")
    for row in parity_matrix:
        print(row)

    codewords = [list(col) for col in zip(*generator_matrix)]
    chosen_codeword = random.choice(codewords)
    print("\nСгенерированное кодовое слово:")
    print(chosen_codeword)

    # Проверка на ошибки (от 1 до max_errors)
    for error_count in range(1, max_errors + 1):
        if error_count > len(chosen_codeword):
            break
        print(f"\nПроверка для {error_count} ошибок:")

        error_vector = create_error_vector(len(chosen_codeword), error_count)
        print(f"Вектор ошибок: {error_vector}")

        received_word = [(bit + err) % 2 for bit, err in zip(chosen_codeword, error_vector)]
        print(f"Полученное слово с ошибками: {received_word}")

        corrected_word, syndrome = fix_errors(parity_matrix, received_word)
        print(f"Синдром: {syndrome}")
        print(f"Исправленное слово: {corrected_word}")

        final_syndrome = multiply_vector_by_matrix(corrected_word, parity_matrix)
        print(f"Синдром после коррекции (должен быть [0,...,0]): {final_syndrome}")

# Примеры вызова функций для анализа кодов Хэмминга
analyze_hamming_code(2)
analyze_hamming_code(3)
analyze_hamming_code(4)

# Анализ расширенного кода Хэмминга
analyze_hamming_code(2, extended=True)
analyze_hamming_code(3, extended=True)
analyze_hamming_code(4, extended=True)
