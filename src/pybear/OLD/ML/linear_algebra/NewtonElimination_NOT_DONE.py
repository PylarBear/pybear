from copy import deepcopy
import numpy as n, pandas as p


class NewtonElimination:
    def __init__(self, MATRIX):
        self.MATRIX_WIP = deepcopy(MATRIX)

    def run(self):
        for col_idx in range(len(self.MATRIX_WIP[0])-1):
            for row_idx in range(len(self.MATRIX_WIP)-1, col_idx, -1):
                if self.MATRIX_WIP[col_idx][col_idx] == 0: continue
                factor = self.MATRIX_WIP[row_idx][col_idx] / self.MATRIX_WIP[col_idx][col_idx]
                self.MATRIX_WIP[row_idx] = n.subtract(self.MATRIX_WIP[row_idx],  n.multiply(factor, self.MATRIX_WIP[col_idx]))

        return self.MATRIX_WIP








if __name__ == '__main__':
    p.set_option('max_colwidth', 6)
    TEST_MATRIX = n.random.randint(0,10,[10,10])
    print('\nTEST MATRIX:')
    print(p.DataFrame(data=TEST_MATRIX, columns=[str(_) for _ in range(len(TEST_MATRIX[0]))]))
    print()
    TEST = NewtonElimination(TEST_MATRIX).run()
    print(f'NEWTON ELIMINATION:')
    print(p.DataFrame(data=TEST, columns=[str(_) for _ in range(len(TEST[0]))]))
















