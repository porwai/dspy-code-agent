class Chess:
    def __init__(self):
        self.board = self.create_board()
        self.current_turn = 'White'
        self.pieces = self.initialize_pieces()

    def create_board(self):
        board = [[' ' for _ in range(8)] for _ in range(8)]
        return board

    def initialize_pieces(self):
        pieces = {'White': [], 'Black': []}
        pieces['White'] = ['R', 'N', 'B', 'Q', 'K', 'B', 'N', 'R']
        pieces['Black'] = ['r', 'n', 'b', 'q', 'k', 'b', 'n', 'r']
        for i in range(8):
            self.board[0][i] = pieces['White'][i]
            self.board[1][i] = 'P'
            self.board[6][i] = 'p'
            self.board[7][i] = pieces['Black'][i]
        return pieces

    def display_board(self):
        for row in self.board:
            print('|'.join(row))
            print('-' * 15)

    def move_piece(self, start, end):
        piece = self.board[start[0]][start[1]]
        if self.is_valid_move(start, end):
            self.board[end[0]][end[1]] = piece
            self.board[start[0]][start[1]] = ' '
            if self.is_in_check(self.current_turn):
                print(f'{self.current_turn} is in check!')
            if self.is_checkmate(self.current_turn):
                print(f'{self.current_turn} is in checkmate! Game over.')
                return
            self.current_turn = 'Black' if self.current_turn == 'White' else 'White'
        else:
            print('Invalid move')

    def is_valid_move(self, start, end):
        piece = self.board[start[0]][start[1]]
        if piece == 'P':
            return self.validate_pawn_move(start, end)
        elif piece == 'R':
            return self.validate_rook_move(start, end)
        elif piece == 'N':
            return self.validate_knight_move(start, end)
        elif piece == 'B':
            return self.validate_bishop_move(start, end)
        elif piece == 'Q':
            return self.validate_queen_move(start, end)
        elif piece == 'K':
            return self.validate_king_move(start, end)
        return False

    def is_in_check(self, color):
        return False

    def is_checkmate(self, color):
        # Logic to determine if the current player's king is in checkmate
        return not any(self.is_valid_move(king_position, move) for move in self.get_king_moves(color))

    def get_king_moves(self, color):
        # Placeholder for getting king's possible moves
        return []

    def validate_pawn_move(self, start, end):
        return True

    def validate_rook_move(self, start, end):
        return True

    def validate_knight_move(self, start, end):
        return True

    def validate_bishop_move(self, start, end):
        return True

    def validate_queen_move(self, start, end):
        return True

    def validate_king_move(self, start, end):
        return True

if __name__ == '__main__':
    game = Chess()
    game.display_board()
    while True:
        start = tuple(map(int, input('Enter start position (row col): ').split()))
        end = tuple(map(int, input('Enter end position (row col): ').split()))
        game.move_piece(start, end)
        game.display_board()