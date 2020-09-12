from datetime import datetime
from collections import namedtuple
from enum import Enum
import sys

cell_info = namedtuple('cell_info', ['x_pos', 'y_pos', 'data'])
NUMBER = set(range(1, 10))


class Mode(Enum):
    INTERACTIVE = 1
    CMDLINE = 2


class Cell:
    '''contains cell information'''

    def __init__(self, value=0, modifiable=True):
        self.value = value
        self.modifiable = modifiable

    def __str__(self):
        return f'{self.value}'


class Sudoku:
    '''provide solution for sudoku puzzle'''

    def __init__(self, show_steps=False):
        self.show_steps = show_steps
        rows = [[0]] * 9
        cells = [row * 9 for row in rows]
        self.squares = [[] for i in range(0, 9)]
        for r in range(0, 9):
            for c in range(0, 9):
                sqr_pos = (r // 3) * 3 + c // 3
                cells[r][c] = Cell()
                self.squares[sqr_pos].append(cells[r][c])
        self.board = cells

    def show(self):
        for i in range(0, 9):
            for j in range(0, 9):
                print(self.board[i][j], end=' ')
            print()

    def load_data(self, mode=Mode.INTERACTIVE, data=[]):
        if mode == Mode.INTERACTIVE:
            print('Enter value for each cell. Press enterkey for no data')
            for i in range(1, 10):
                for j in range(1, 10):
                    try:
                        self.board[i - 1][j - 1].value = int(input(f'Enter value for {i} X {j}:'))
                        self.board[i - 1][j - 1].modifiable = False
                    except:
                        pass
        else:
            for cell_info in data:
                if cell_info is None:
                    continue
                self.board[cell_info.x_pos - 1][cell_info.y_pos - 1].value = int(cell_info.data)
                self.board[cell_info.x_pos - 1][cell_info.y_pos - 1].modifiable = False

    def _possible_numbers(self, x_pos, y_pos):

        if self.board[x_pos][y_pos].value != 0:
            return list()

        num_in_col = set(self.board[i][y_pos].value for i in range(0, 9))
        num_in_row = set(self.board[x_pos][i].value for i in range(0, 9))

        sqr_pos = (x_pos // 3) * 3 + y_pos // 3
        num_in_sqr = set(cell.value for cell in self.squares[sqr_pos])

        used_num = num_in_sqr.union(num_in_row).union(num_in_col)
        unused_num = list(NUMBER - used_num)
        unused_num.sort()

        return unused_num

    def _solved(self):
        for sq in self.squares:
            for cell in sq:
                if cell.value == 0:
                    return False

        return True

    def solve(self, row_start=0, col_start=0):
        '''
        solve sudoku
        '''
        if self.show_steps:
            print('*' * 10)
            self.show()
        if self._solved():
            return True

        for row in range(row_start, 9):
            for col in range(col_start, 9):
                cell = self.board[row][col]
                if cell.value == 0:
                    unused_num = self._possible_numbers(row, col)
                    if not unused_num:
                        return False
                    for val in unused_num:
                        cell.value = val
                        # cell.modifiable = False
                        if self.solve(row_start=0 if row == 8 else row, col_start=0 if col == 8 else col):
                            return True
                        cell.value = 0
                        # cell.modifiable = True
                    else:
                        if not self._solved():
                            return False
            else:
                col_start = 0

        return False

    def get_solution(self):
        '''
        return sudoku solution
        '''
        solution = [
            (self.board[i][j].value, self.board[i][j].modifiable)
            for i in range(0, 9) for j in range(0, 9)
            ]
        return solution


def get_cell_info():
    '''
    get data interactively
    :return:
    '''
    print('Enter q for quit')
    cells = list()
    while True:
        try:
            x_pos = input('Enter X POS: ')
            if x_pos == 'q':
                break
            y_pos = input('Enter Y POS: ')
            if y_pos == 'q':
                break
            data = input('Enter data: ')
            if data == 'q':
                break
            cell = cell_info(int(x_pos), int(y_pos), int(data))
        except Exception:
            print('Incorrect input. Please enter number(0-9)')
        else:
            cells.append(cell)

    return cells


def read_data_from_file(filepath):
    '''
    read data from file
    :param filepath: filepath
    :return: list of cell info
    '''
    cells = list()
    with open(filepath, 'r') as infile:
        for line in infile:
            x_pos, y_pos, data = line.split(',')
            cells.append(cell_info(int(x_pos), int(y_pos), int(data)))

    return cells


if __name__ == '__main__':

    sudoku = Sudoku(show_steps=True)
    FILE_PATH = 'dataInputFile'
    sudoku.load_data(mode=Mode.CMDLINE, data=read_data_from_file(FILE_PATH))
    print('Given Sudoku')
    sudoku.show()

    while True:
        choice = input('Start Solving(Y/N)?').lower()
        if choice == 'y':
            break
        if choice == 'n':
            sys.exit()

    starttime = datetime.now()
    print(f'Starting: {starttime}')

    if sudoku.solve():
        print("Solution ---->")
        sudoku.show()
    else:
        print('No solution found')

    endtime = datetime.now()
    print(f'Ending: {endtime}')
    print(f'Total Time :{endtime - starttime}')
