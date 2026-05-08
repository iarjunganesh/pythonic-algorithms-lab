def solve_n_queens(n):
    solutions = []
    cols = set()
    diag = set()
    anti = set()
    board = [-1] * n

    def backtrack(r):
        if r == n:
            solutions.append(board.copy())
            return
        for c in range(n):
            if c in cols or (r - c) in diag or (r + c) in anti:
                continue
            cols.add(c)
            diag.add(r - c)
            anti.add(r + c)
            board[r] = c
            backtrack(r + 1)
            cols.remove(c)
            diag.remove(r - c)
            anti.remove(r + c)

    backtrack(0)
    return solutions
