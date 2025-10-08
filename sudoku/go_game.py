class GoGame:
    def __init__(self, board_size=19):
        self.board_size = board_size
        self.board = [['.' for _ in range(board_size)] for _ in range(board_size)]
        self.black_stones = 0
        self.white_stones = 0

    def place_stone(self, x, y, color):
        if self.board[x][y] == '.':
            self.board[x][y] = color
            if color == 'B':
                self.black_stones += 1
            else:
                self.white_stones += 1
        else:
            raise ValueError('Position already occupied')

    def display_board(self):
        for row in self.board:
            print(' '.join(row))

# Example usage:
if __name__ == '__main__':
    game = GoGame()
    game.place_stone(0, 0, 'B')
    game.place_stone(0, 1, 'W')
    game.display_board()  # Display the board after placing stones
    print(f'Black stones: {game.black_stones}, White stones: {game.white_stones}')