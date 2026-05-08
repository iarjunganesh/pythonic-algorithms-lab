from algorithms.cpu.backtracking.n_queens import solve_n_queens


def test_n_queens_1():
    solutions = solve_n_queens(1)
    assert solutions == [[0]]


def test_n_queens_2_no_solution():
    assert solve_n_queens(2) == []


def test_n_queens_3_no_solution():
    assert solve_n_queens(3) == []


def test_n_queens_4_count():
    solutions = solve_n_queens(4)
    assert len(solutions) == 2


def test_n_queens_4_valid():
    solutions = solve_n_queens(4)
    for board in solutions:
        _assert_valid(board)


def test_n_queens_8_count():
    solutions = solve_n_queens(8)
    assert len(solutions) == 92


def test_n_queens_8_all_valid():
    for board in solve_n_queens(8):
        _assert_valid(board)


def _assert_valid(board):
    """Assert that a board (list of column positions per row) is a valid N-Queens solution."""
    n = len(board)
    assert len(set(board)) == n, "duplicate column"
    diag1 = [board[r] - r for r in range(n)]
    diag2 = [board[r] + r for r in range(n)]
    assert len(set(diag1)) == n, "diagonal conflict"
    assert len(set(diag2)) == n, "anti-diagonal conflict"


if __name__ == "__main__":
    test_n_queens_1()
    test_n_queens_2_no_solution()
    test_n_queens_3_no_solution()
    test_n_queens_4_count()
    test_n_queens_4_valid()
    test_n_queens_8_count()
    test_n_queens_8_all_valid()
    print("Backtracking tests passed")
