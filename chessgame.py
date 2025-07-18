### CHESS GAME
## (For a long duration project using NNs)

class Game:
    def __init__(self):
        self.board = [['r', 'n', 'b', 'q', 'k', 'b', 'n', 'r'], 
                      ['p', 'p', 'p', 'p', 'p', 'p', 'p', 'p'],
                      [' ', ' ', ' ', ' ', ' ', ' ', ' ', ' '],
                      [' ', ' ', ' ', ' ', ' ', ' ', ' ', ' '],
                      [' ', ' ', ' ', ' ', ' ', ' ', ' ', ' '],
                      [' ', ' ', ' ', ' ', ' ', ' ', ' ', ' '],
                      ['P', 'P', 'P', 'P', 'P', 'P', 'P', 'P'],
                      ['R', 'N', 'B', 'Q', 'K', 'B', 'N', 'R'],
                      ]
        self.eval=0
        self.turn=1
        self.white_in_check = False
        self.black_in_check = False
        self.checkmate=False
        self.AllGameMoves = []
        self.white_protection = [[0]*8 for _ in range(8)]
        self.black_protection = [[0]*8 for _ in range(8)]
        self.blackcastlingallowed = [True, True]
        self.whitecastlingallowed = [True, True]
        self.blackcastled = False
        self.whitecastled = False
    
    def convert_notation(self, ist, jst, iend, jend):
        return f"{chr(97+jst)}{8-ist}{chr(97+jend)}{8-iend}"

    def draw_board(self):
        for i in self.board:
            for j in i:
                print(j, end=" ")
            print()
    
    def check_boundaries(self, i, j):
        return (0 <= i <= 7 and 0 <= j <= 7)

    def check_pawn_moves(self, i, j):
        moves = []
        pp = []
        if self.board[i][j].lower() == 'p':
            color = self.board[i][j].islower()
            if not color:
                # Move Forward
                if self.check_boundaries(i-1, j) and self.board[i-1][j] == ' ':
                    moves.append(self.convert_notation(i, j, i-1, j))
                    if i == 6 and self.board[i-2][j] == ' ':
                        moves.append(self.convert_notation(i, j, i-2, j))
                
                # Kills
                if self.check_boundaries(i-1, j-1) and self.board[i-1][j-1].islower():
                    moves.append(self.convert_notation(i, j, i-1, j-1))
                if self.check_boundaries(i-1, j+1) and self.board[i-1][j+1].islower():
                    moves.append(self.convert_notation(i, j, i-1, j+1))
                
                # EnPassant
                if i == 3 and self.AllGameMoves: 
                    last_move = self.AllGameMoves[-1]
                    last_start_col = ord(last_move[0]) - 97
                    last_start_row = 8 - int(last_move[1])
                    last_end_col = ord(last_move[2]) - 97
                    last_end_row = 8 - int(last_move[3])
                    
                    if (self.board[last_end_row][last_end_col].lower() == 'p' and abs(last_start_row - last_end_row) == 2 and last_end_row == 3):  
                        if last_start_col == j - 1:
                            moves.append(self.convert_notation(i, j, i-1, j-1))
                        if last_start_col == j + 1:
                            moves.append(self.convert_notation(i, j, i-1, j+1)) 

            else:
                # Move Forward
                if self.check_boundaries(i+1, j) and self.board[i+1][j] == ' ':
                    moves.append(self.convert_notation(i, j, i+1, j))
                    if i == 1 and self.board[i+2][j] == ' ':
                        moves.append(self.convert_notation(i, j, i+2, j))
                
                # Kills
                if self.check_boundaries(i+1, j-1) and self.board[i+1][j-1].isupper():
                    moves.append(self.convert_notation(i, j, i+1, j-1))
                if self.check_boundaries(i+1, j+1) and self.board[i+1][j+1].isupper():
                    moves.append(self.convert_notation(i, j, i+1, j+1))
                
                # EnPassant
                if i == 4 and self.AllGameMoves: 
                    last_move = self.AllGameMoves[-1]
                    last_start_col = ord(last_move[0]) - 97
                    last_start_row = 8 - int(last_move[1])
                    last_end_col = ord(last_move[2]) - 97
                    last_end_row = 8 - int(last_move[3])
                    
                    if (self.board[last_end_row][last_end_col].lower() == 'p' and abs(last_start_row - last_end_row) == 2 and last_end_row == 4):
                        if last_start_col == j - 1:
                            moves.append(self.convert_notation(i, j, i+1, j-1))
                        if last_start_col == j + 1:
                            moves.append(self.convert_notation(i, j, i+1, j+1)) 

        else:
            return -1
        return moves, pp
    
    def check_knight_moves(self, i, j):
        moves = []
        pp=[]
        knight_moves = [(2, 1), (2, -1), (-2, 1), (-2, -1), (1, 2), (1, -2), (-1, 2), (-1, -2)]
        for move in knight_moves:
            di, dj = i + move[0], j + move[1]
            if self.check_boundaries(di, dj):
                if self.board[di][dj]!=' ':
                    if (self.board[di][dj].islower() != self.board[i][j].islower()):
                        moves.append(self.convert_notation(i, j, di, dj))
                    else: 
                        pp.append(self.convert_notation(i, j, di, dj))
                else:
                    moves.append(self.convert_notation(i, j, di, dj))
        return moves, pp

    def check_bishop_moves(self, i, j):
        moves = []
        pp = []
        directions = [(1, 1), (1, -1), (-1, 1), (-1, -1)]
        for direction in directions:
            x, y = direction
            di, dj = i, j
            while self.check_boundaries(di + x, dj + y):
                di += x
                dj += y
                if self.board[di][dj] != ' ':
                    if (self.board[di][dj].islower() != self.board[i][j].islower()):
                        move_str = self.convert_notation(i, j, di, dj)
                        if move_str != self.convert_notation(i, j, i, j):
                            moves.append(move_str)
                    else:
                        pp.append(self.convert_notation(i, j, di, dj))
                    break

                move_str = self.convert_notation(i, j, di, dj)
                if move_str != self.convert_notation(i, j, i, j):
                    moves.append(move_str)
        return moves, pp

    def check_rook_moves(self, i, j):
        moves = []
        pp = []
        directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]
        for direction in directions:
            x, y = direction
            di, dj = i, j
            while self.check_boundaries(di + x, dj + y):
                di += x
                dj += y
                if self.board[di][dj] != ' ':
                    if (self.board[di][dj].islower() != self.board[i][j].islower()):
                        move_str = self.convert_notation(i, j, di, dj)
                        if move_str != self.convert_notation(i, j, i, j):
                            moves.append(move_str)
                    else:
                        pp.append(self.convert_notation(i, j, di, dj))
                    break
                move_str = self.convert_notation(i, j, di, dj)
                if move_str != self.convert_notation(i, j, i, j):
                    moves.append(move_str)
        return moves, pp

    def check_queen_moves(self, i, j):
        moves = []
        pp = []
        bm, bpp = self.check_bishop_moves(i, j)
        rm, rpp = self.check_rook_moves(i, j)
        moves.extend(bm)
        moves.extend(rm)
        pp.extend(bpp)
        pp.extend(rpp)
        return moves, pp

    def check_king_moves(self, i, j):
        moves = []
        pp = []
        king_moves = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)]
        for move in king_moves:
            di, dj = i + move[0], j + move[1]
            if self.check_boundaries(di, dj):
                if self.board[di][dj] == ' ' or (self.board[di][dj].islower() != self.board[i][j].islower() and ((self.board[i][j].islower() and not self.white_protection[di][dj]) or (self.board[i][j].isupper() and not self.black_protection[di][dj]))):
                    moves.append(self.convert_notation(i, j, di, dj))
                else:
                    pp.append(self.convert_notation(i, j, di, dj))
        
        # Castling
        if self.board[i][j].islower() and not self.black_in_check:
            if self.blackcastlingallowed[1] and self.board[i][j+1] == ' ' and self.board[i][j+2] == ' ':
                if (self.white_protection[i][j+1]==0 and self.white_protection[i][j+2]==0):
                    moves.append(self.convert_notation(i, j, i, j+2))

            if self.blackcastlingallowed[0] and self.board[i][j-1] == ' ' and self.board[i][j-2] == ' ' and self.board[i][j-3] == ' ':
                if (self.white_protection[i][j-1]==0 and self.white_protection[i][j-2]==0):
                    moves.append(self.convert_notation(i, j, i, j-2))
        elif self.board[i][j].isupper() and not self.white_in_check:
            if self.whitecastlingallowed[1] and self.board[i][j+1] == ' ' and self.board[i][j+2] == ' ':
                if (self.black_protection[i][j+1]==0 and self.black_protection[i][j+2]==0):
                    moves.append(self.convert_notation(i, j, i, j+2))
            
            if self.whitecastlingallowed[0] and self.board[i][j-1] == ' ' and self.board[i][j-2] == ' ' and self.board[i][j-3] == ' ':
                if (self.black_protection[i][j-1]==0 and self.black_protection[i][j-2]==0):
                    moves.append(self.convert_notation(i, j, i, j-2))

        return moves, pp
    
    def possible_moves(self):
        white = ['P', 'K', 'Q', 'B', 'N', 'R']
        black = ['p', 'k', 'q', 'b', 'n', 'r']
        moves = []
        for i in range(8):
            for j in range(8):
                piece = self.board[i][j]
                if (self.turn == 1 and piece in white) or (self.turn == -1 and piece in black):
                    if piece.lower() == 'p':
                        moves.extend(self.check_pawn_moves(i, j)[0])
                    elif piece.lower() == 'n':
                        moves.extend(self.check_knight_moves(i, j)[0])
                    elif piece.lower() == 'b':
                        moves.extend(self.check_bishop_moves(i, j)[0])
                    elif piece.lower() == 'r':
                        moves.extend(self.check_rook_moves(i, j)[0])
                    elif piece.lower() == 'q':
                        moves.extend(self.check_queen_moves(i, j)[0])
                    elif piece.lower() == 'k':
                        moves.extend(self.check_king_moves(i, j)[0])
        return moves

    def is_in_check(self, update=True):
        king_position = None
        for i in range(8):
            for j in range(8):
                if (self.turn == 1 and self.board[i][j] == 'K') or (self.turn == -1 and self.board[i][j] == 'k'):
                    king_position = (i, j)
                    break
            if king_position:
                break

        if not king_position:
            return False
        opponent_pieces = ['p', 'n', 'b', 'r', 'q', 'k'] if self.turn == 1 else ['P', 'N', 'B', 'R', 'Q', 'K']
        for i in range(8):
            for j in range(8):
                piece = self.board[i][j]
                if piece in opponent_pieces:
                    if piece.lower() == 'p':
                        attack_positions = [(i + (1 if piece.islower() else -1), j - 1), (i + (1 if piece.islower() else -1), j + 1)]
                        for pos in attack_positions:
                            ai, aj = pos
                            if self.check_boundaries(ai, aj):
                                if (ai, aj) == king_position:
                                    if update:
                                        if self.board[i][j].islower():
                                            self.black_in_check=True
                                        else:
                                            self.white_in_check=True
                                    return True
                    elif piece.lower() == 'n':
                        moves = self.check_knight_moves(i, j)[0]
                        if self.convert_notation(i, j, king_position[0], king_position[1]) in moves:
                            if update:
                                if self.board[i][j].islower():
                                    self.black_in_check=True
                                else:
                                    self.white_in_check=True
                            return True
                    elif piece.lower() == 'b':
                        moves = self.check_bishop_moves(i, j)[0]
                        if self.convert_notation(i, j, king_position[0], king_position[1]) in moves:
                            if update:
                                if self.board[i][j].islower():
                                    self.black_in_check=True
                                else:
                                    self.white_in_check=True
                            return True
                    elif piece.lower() == 'r':
                        moves = self.check_rook_moves(i, j)[0]
                        if self.convert_notation(i, j, king_position[0], king_position[1]) in moves:
                            if update:
                                if self.board[i][j].islower():
                                    self.black_in_check=True
                                else:
                                    self.white_in_check=True
                            return True
                    elif piece.lower() == 'q':
                        moves = self.check_queen_moves(i, j)[0]
                        if self.convert_notation(i, j, king_position[0], king_position[1]) in moves:
                            if update:
                                if self.board[i][j].islower():
                                    self.black_in_check=True
                                else:
                                    self.white_in_check=True
                            return True
                    elif piece.lower() == 'k':
                        moves = self.check_king_moves(i, j)[0]
                        if self.convert_notation(i, j, king_position[0], king_position[1]) in moves:
                            if update:
                                if self.board[i][j].islower():
                                    self.black_in_check=True
                                else:
                                    self.white_in_check=True
                            return True

        return False

    def is_checkmate(self):
        if not self.is_in_check(update=False):
            return False
        for i in range(8):
            for j in range(8):
                piece = self.board[i][j]
                if (self.turn == 1 and piece.isupper()) or (self.turn == -1 and piece.islower()):
                    moves = self.possible_moves()
                    if (len(moves) == 0):
                        return 0.5
                    for move in moves:
                        start_col = ord(move[0]) - 97
                        start_row = 8 - int(move[1])
                        end_col = ord(move[2]) - 97
                        end_row = 8 - int(move[3])

                        original_piece = self.board[start_row][start_col]
                        target_piece = self.board[end_row][end_col]
                        self.board[end_row][end_col] = original_piece
                        self.board[start_row][start_col] = ' '

                        if not self.is_in_check(update=False):
                            self.board[start_row][start_col] = original_piece
                            self.board[end_row][end_col] = target_piece
                            return False

                        self.board[start_row][start_col] = original_piece
                        self.board[end_row][end_col] = target_piece
        self.checkmate=True
        return True

    def is_valid_move(self, start_row, start_col, end_row, end_col):
        piece = self.board[start_row][start_col]
        if piece == ' ':
            return False 

        if piece.lower() == 'p':
            moves = self.check_pawn_moves(start_row, start_col)[0]
        elif piece.lower() == 'n':
            moves = self.check_knight_moves(start_row, start_col)[0]
        elif piece.lower() == 'b':
            moves = self.check_bishop_moves(start_row, start_col)[0]
        elif piece.lower() == 'r':
            moves = self.check_rook_moves(start_row, start_col)[0]
        elif piece.lower() == 'q':
            moves = self.check_queen_moves(start_row, start_col)[0]
        elif piece.lower() == 'k':
            moves = self.check_king_moves(start_row, start_col)[0]
        else:
            return False

        move_str = self.convert_notation(start_row, start_col, end_row, end_col)
        return move_str in moves

    def update_protection_matrices(self):
        self.white_protection = [[0]*8 for _ in range(8)]
        self.black_protection = [[0]*8 for _ in range(8)]

        for i in range(8):
            for j in range(8):
                piece = self.board[i][j]
                if piece != ' ':
                    if piece.isupper():
                        if piece.lower() == 'p':
                            if self.check_boundaries(i-1, j-1):
                                self.white_protection[i-1][j-1] += 1
                            if self.check_boundaries(i-1, j+1):
                                self.white_protection[i-1][j+1] += 1
                        elif piece.lower() == 'n':
                            knight_moves = [(2, 1), (2, -1), (-2, 1), (-2, -1), (1, 2), (1, -2), (-1, 2), (-1, -2)]
                            for move in knight_moves:
                                di, dj = i + move[0], j + move[1]
                                if self.check_boundaries(di, dj):
                                    self.white_protection[di][dj] += 1
                        elif piece.lower() == 'b':
                            moves, pp = self.check_bishop_moves(i, j)
                            for move in moves+pp:
                                end_row = 8 - int(move[3])
                                end_col = ord(move[2]) - 97
                                self.white_protection[end_row][end_col] += 1
                        elif piece.lower() == 'r':
                            moves, pp = self.check_rook_moves(i, j)
                            for move in moves+pp:
                                end_row = 8 - int(move[3])
                                end_col = ord(move[2]) - 97
                                self.white_protection[end_row][end_col] += 1
                        elif piece.lower() == 'q':
                            moves, pp = self.check_queen_moves(i, j)
                            for move in moves+pp:
                                end_row = 8 - int(move[3])
                                end_col = ord(move[2]) - 97
                                self.white_protection[end_row][end_col] += 1
                        elif piece.lower() == 'k':
                            moves, pp = self.check_king_moves(i, j)
                            for move in moves+pp:
                                start_row = 8 - int(move[1])
                                start_col = ord(move[0]) - 97
                                end_row = 8 - int(move[3])
                                end_col = ord(move[2]) - 97
                                if abs(start_col-end_col)==1:
                                    self.white_protection[end_row][end_col] += 1
                    else:  # Black piece
                        if piece.lower() == 'p':
                            if self.check_boundaries(i+1, j-1):
                                self.black_protection[i+1][j-1] += 1
                            if self.check_boundaries(i+1, j+1):
                                self.black_protection[i+1][j+1] += 1
                        elif piece.lower() == 'n':
                            knight_moves = [(2, 1), (2, -1), (-2, 1), (-2, -1), (1, 2), (1, -2), (-1, 2), (-1, -2)]
                            for move in knight_moves:
                                di, dj = i + move[0], j + move[1]
                                if self.check_boundaries(di, dj):
                                    self.black_protection[di][dj] += 1
                        elif piece.lower() == 'b':
                            moves, pp = self.check_bishop_moves(i, j)
                            for move in moves+pp:
                                end_row = 8 - int(move[3])
                                end_col = ord(move[2]) - 97
                                self.black_protection[end_row][end_col] += 1
                        elif piece.lower() == 'r':
                            moves, pp = self.check_rook_moves(i, j)
                            for move in moves+pp:
                                end_row = 8 - int(move[3])
                                end_col = ord(move[2]) - 97
                                self.black_protection[end_row][end_col] += 1
                        elif piece.lower() == 'q':
                            moves, pp = self.check_queen_moves(i, j)
                            for move in moves+pp:
                                end_row = 8 - int(move[3])
                                end_col = ord(move[2]) - 97
                                self.black_protection[end_row][end_col] += 1
                        elif piece.lower() == 'k':
                            moves, pp = self.check_king_moves(i, j)
                            for move in moves+pp:
                                end_row = 8 - int(move[3])
                                end_col = ord(move[2]) - 97
                                self.black_protection[end_row][end_col] += 1

    def play_game(self):
        while True:
            self.draw_board()
            self.update_protection_matrices()
            in_check = self.is_in_check()
            if in_check:
                print(f"{'White' if self.turn == 1 else 'Black'} is in check!")
            cm = self.is_checkmate()
            if cm==0.5:
                print("Stalemate! It is a Draw!")
                break
            elif cm:
                self.checkmate = True
                print(f"{'White' if self.turn == 1 else 'Black'} is in checkmate! Game over.")
                break

            move = input(f"{'White' if self.turn == 1 else 'Black'}, enter your move (e.g., e2e4): ").strip()
            if len(move) != 4:
                print("Invalid move format. Please enter a move in the format 'e2e4'.")
                continue

            start_col = ord(move[0]) - 97
            start_row = 8 - int(move[1])
            end_col = ord(move[2]) - 97
            end_row = 8 - int(move[3])

            if not self.check_boundaries(start_row, start_col) or not self.check_boundaries(end_row, end_col):
                print("Move out of bounds. Try again.")
                continue

            if not self.is_valid_move(start_row, start_col, end_row, end_col):
                print("Invalid move for the selected piece. Try again.")
                continue

            piece = self.board[start_row][start_col]
            if (self.turn == 1 and piece.islower()) or (self.turn == -1 and piece.isupper()):
                print("It's not your turn!")
                continue
            
            # Castling validations
            if piece.lower() == 'r':
                if (not self.blackcastled) and piece.islower():
                    self.blackcastlingallowed[start_col==7] = False
                elif (not self.whitecastled) and piece.isupper():
                    self.whitecastlingallowed[start_col==7] = False
            
            elif piece.lower() == 'k':
                if piece.islower():
                    self.blackcastlingallowed = [False, False]
                else:
                    self.whitecastlingallowed = [False, False]
                
                if (end_col-start_col) == 2:
                    if (self.board[start_row][start_col].islower()):
                        self.blackcastled = True
                    else:
                        self.whitecastled = True
                    self.board[start_row][start_col+1] = self.board[end_row][end_col+1]
                    self.board[end_row][end_col+1] = ' '
                elif (start_col-end_col) == 2:
                    if (self.board[start_row][start_col].islower()):
                        self.blackcastled = True
                    else:
                        self.whitecastled = True
                    self.board[start_row][start_col-1] = self.board[start_row][end_col-2]
                    self.board[start_row][end_col-2] = ' '

            # EnPassant move
            elif piece.lower() == 'p':
                if (abs(start_row - end_row) == 1 and abs(start_col-end_col)==1 and self.board[end_row][end_col] == ' '):
                    if (self.turn == 1 and start_row == 3 and end_row == 2) or (self.turn == -1 and start_row == 4 and end_row == 5):
                        self.board[start_row][end_col] = ' '
                
            
            y = self.board[end_row][end_col]
            self.board[end_row][end_col] = piece
            self.board[start_row][start_col] = ' '

            # Promotions
            if (piece.lower() == 'p'):
                if (self.board[end_row][end_col].islower() and end_row==7):
                    while True:
                        h = input("What do you want the promotion to be: ")
                        if (h.lower() in ['q', 'r', 'b', 'n']):
                            self.board[end_row][end_col]=h.lower()
                            break
                        else:
                            print("Invalid Promotion")
                            continue
                elif (self.board[end_row][end_col].isupper() and end_row==0):
                    while True:
                        h = input("What do you want the promotion to be: ")
                        if (h.lower() in ['q', 'r', 'b', 'n']):
                            self.board[end_row][end_col]=h.upper()
                            break
                        else:
                            print("Invalid Promotion")
                            continue

            if in_check:
                if (self.is_in_check(update=False)):
                    self.board[start_row][start_col] = piece
                    self.board[end_row][end_col] = y
                    print("Illegal Move! You are in check")
                    continue

            self.AllGameMoves.append(move)
            self.turn *= -1

game = Game()
game.play_game()
