import numpy as np

from itertools import combinations

def ref(matrix):
    """Преобразуем матрицу в формат numpy для удобной работы"""
    mat = np.array(matrix)
    n_rows, n_cols = mat.shape
    lead = 0
    for r in range(n_rows):
        if lead >= n_cols:
            return mat
        i = r
        while mat[i, lead] == 0:
            i += 1
            if i == n_rows:
                i = r
                lead += 1
                if lead == n_cols:
                    return mat
        # Меняем строки местами, если нужно
        mat[[i, r]] = mat[[r, i]]

        # Обрабатываем все строки ниже текущей
        for i in range(r + 1, n_rows):
            if mat[i, lead] != 0:
                mat[i] = (mat[i] + mat[r]) % 2
        lead += 1
    return mat



def rref(matrix):
    """Приводим матрицу к приведённому ступенчатому виду"""
    matrix = ref(matrix)
    n_rows, n_cols = matrix.shape

    # Пройдёмся по строкам сверху вниз
    for r in range(n_rows - 1, -1, -1):
        # Находим ведущий элемент в строке
        lead = np.argmax(matrix[r] != 0)
        if matrix[r, lead] != 0:
            # Обнуляем все элементы выше ведущего
            for i in range(r - 1, -1, -1):
                if matrix[i, lead] != 0:
                    matrix[i] = (matrix[i] + matrix[r]) % 2
    while not any(matrix[n_rows - 1]):
        matrix = matrix[:-1, :]
        n_rows -= 1
    return matrix



def find_lead_columns(matrix):
    """Cоздаем сокращенную матрицу"""
    lead_columns = []
    for r in range(len(matrix)):
        row = matrix[r]
        for i, val in enumerate(row):
            if val == 1:
                lead_columns.append(i)
                break
    return lead_columns



def remove_lead_columns(matrix, lead_columns):
    """Удаление ведущих столбцов"""
    mat = np.array(matrix)
    reduced_matrix = np.delete(mat, lead_columns, axis=1)
    return reduced_matrix


def form_H_matrix(X, lead_columns, n_cols):
    """Формирование матрицы H"""
    n_rows = np.shape(X)[1]

    H = np.zeros((n_cols, n_rows), dtype=int)
    I = np.eye(6, dtype=int)

    H[lead_columns, :] = X
    not_lead = [i for i in range(n_cols) if i not in lead_columns]
    H[not_lead, :] = I

    return H


def LinearCode(mat):
    """Основная функция для выполнения всех шагов"""
    G_star = rref(mat)

    print("G* (RREF матрица) =")
    print(G_star)


    lead_columns = find_lead_columns(G_star)
    print(f"lead = {lead_columns}")

    X = remove_lead_columns(G_star, lead_columns)
    print("Сокращённая матрица X =")
    print(X)
    n_cols = np.shape(mat)[1]
    H = form_H_matrix(X, lead_columns, n_cols)
    print("Проверочная матрица H =")
    print(H)

    return H




def generate_codewords_from_combinations(G):
    """Нахождения все кодовые слов из порождающей матрицы"""
    rows = G.shape[0]
    codewords = set()

    # Перебираем все возможные комбинации строк матрицы G
    for r in range(1, rows + 1):
        for comb in combinations(range(rows), r):
            # Суммируем строки и добавляем результат в множество
            codeword = np.bitwise_xor.reduce(G[list(comb)], axis=0)
            codewords.add(tuple(codeword))

    # Добавляем в множество нулевой вектор
    codewords.add(tuple(np.zeros(G.shape[1], dtype=int)))

    return np.array(list(codewords))

def generate_codewords_binary_multiplication(G):
    """Функция для умножения всех двоичных слов длины k на G"""
    k = G.shape[0]
    n = G.shape[1]
    codewords = []
    for i in range(2**k):
        binary_word = np.array(list(np.binary_repr(i, k)), dtype=int)
        codeword = np.dot(binary_word, G) % 2
        codewords.append(codeword)

    return np.array(codewords)

def check_codeword(codeword, H):
    """Проверка кодового слова с помощью проверочной матрицы H"""
    return np.dot(codeword, H) % 2

def calculate_code_distance(codewords):
    """Вычисление кодового расстояния"""
    min_distance = float('inf')

    # Считаем количество ненулевых элементов для всех попарных разностей кодовых слов
    for i in range(len(codewords)):
        for j in range(i + 1, len(codewords)):
            distance = np.sum(np.bitwise_xor(codewords[i], codewords[j]))
            if distance > 0:
                min_distance = min(min_distance, distance)

    return min_distance

def LinearCodeWithErrors(mat):
    """Основная функция для выполнения всех шагов"""
    # Выполнение шагов, как и ранее
    G_star = rref(mat)
    lead_columns = find_lead_columns(G_star)
    X = remove_lead_columns(G_star, lead_columns)
    n_cols = np.shape(mat)[1]
    H = form_H_matrix(X, lead_columns, n_cols)

    print("G* (RREF матрица) =")
    print(G_star)
    print(f"lead = {lead_columns}")
    print("Сокращённая матрица X =")
    print(X)
    print("Проверочная матрица H =")
    print(H)

    codewords_1 = generate_codewords_from_combinations(G_star)
    print("Все кодовые слова (способ 1):")
    print(codewords_1)

    codewords_2 = generate_codewords_binary_multiplication(G_star)
    print("Все кодовые слова (способ 2):")
    print(codewords_2)

    assert set(map(tuple, codewords_1)) == set(map(tuple, codewords_2)), "Наборы кодовых слов не совпадают!"
    for codeword in codewords_1:
        result = check_codeword(codeword, H)
        assert np.all(result == 0), f"Ошибка: кодовое слово {codeword} не прошло проверку матрицей H"

    print("Все кодовые слова прошли проверку матрицей H.")
    d = calculate_code_distance(codewords_1)
    t = 0
    if t == 0:
        t = 1
    else:
        t = (d - 1) // 2
    print(f"Кодовое расстояние d = {d}")
    print(f"Кратность обнаруживаемой ошибки t = {t}")

    e1 = np.zeros(n_cols, dtype=int)
    e1[2] = 1  # Внесение ошибки в один бит
    v = codewords_1[4]
    print(f"e1 = {e1}")
    print(f"v = {v}")
    v_e1 = (v + e1) % 2
    print(f"v + e1 = {v_e1}")
    print(f"(v + e1)@H = {check_codeword(v_e1, H)} - error")

    e2 = np.zeros(n_cols, dtype=int)
    e2[6] = 1
    e2[9] = 1  # Внесение ошибки в два бита
    print(f"e2 = {e2}")
    v_e2 = (v + e2) % 2
    print(f"v + e2 = {v_e2}")
    print(f"(v + e2)@H = {check_codeword(v_e2, H)} - no error")

    return H

if __name__ == "__main__":
    matrix = ([[1, 0, 1, 1, 0, 0, 0, 1, 0, 0, 1],
           [0, 0, 0, 1, 1, 1, 0, 1, 0, 1, 0],
           [0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1],
           [1, 0, 1, 0, 1, 1, 1, 0, 0, 0, 1],
           [0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 0],
           [1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0]])
    result = LinearCodeWithErrors(matrix)

