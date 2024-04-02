import pygame
import numpy as np
import time
import random
import q_learn
import multi_process_q_learn


# constants:
WIDTH = 800
HEIGHT = 700
WINDOW = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Connect Four")
pygame.font.init()
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
BLUE = (0, 0, 180)
RED = (200, 0, 0)
YELLOW = (240, 240, 0)
GREEN =(0, 255, 0)
FPS = 60
CIRC = 40
BG = pygame.image.load('/Users/richie/Library/Mobile Documents/com~apple~CloudDocs/Documents/First_Code/games/connect_four/background.png')
BG = pygame.transform.scale(BG, (WIDTH, HEIGHT))

# Functions:

# display game
def set_game(positions):
    WINDOW.fill(WHITE)
    # board:
    pygame.draw.rect(WINDOW, BLUE, (50, 50, (WIDTH - 100), (HEIGHT - 100)), width = 0)
    for col in range(7):
        for row in range(6):
            x, y = (col + 1) * 100, (row + 1) * 100
            if positions[row][col] == 0:
                pygame.draw.circle(WINDOW, WHITE, (x, y), CIRC)
            elif positions[row][col] == 1:
                pygame.draw.circle(WINDOW, RED, (x, y), CIRC)
            else:
                pygame.draw.circle(WINDOW, YELLOW, (x, y), CIRC)
    # update display
    pygame.display.update()


def set_main_menu():
    y1, y2, y3, y4 = 275, 350, 425, 500
    x = 300
    w = 200
    h = 50
    e = 5
    ox = 27.5
    oy = 12
    FONT1 = pygame.font.Font("freesansbold.ttf", 100)
    FONT2 = pygame.font.Font("freesansbold.ttf", 25)
    title = FONT1.render("Connect Four", True, BLUE)
    p = FONT2.render("Pass'n'Play", True, WHITE)
    ea = FONT2.render("Easy Bot", True, WHITE)
    m = FONT2.render("Medium Bot", True, WHITE)
    ha = FONT2.render("Hard Bot", True, WHITE)
    WINDOW.fill(WHITE)
    WINDOW.blit(BG, (0, 0))
    WINDOW.blit(title, (75, 150))
    pygame.draw.rect(WINDOW, BLUE, (x, y1, w, h), e, e, e, e)
    pygame.draw.rect(WINDOW, BLUE, (x + 1, y1 + 1, w - 3, h - 3))
    pygame.draw.rect(WINDOW, BLUE, (x, y2, w, h), e, e, e, e)
    pygame.draw.rect(WINDOW, BLUE, (x + 1, y2 + 1, w - 3, h - 3))
    pygame.draw.rect(WINDOW, BLUE, (x, y3, w, h), e, e, e, e)
    pygame.draw.rect(WINDOW, BLUE, (x + 1, y3 + 1, w - 3, h - 3))
    pygame.draw.rect(WINDOW, BLUE, (x, y4, w, h), e, e, e, e)
    pygame.draw.rect(WINDOW, BLUE, (x + 1, y4 + 1, w - 3, h - 3))
    WINDOW.blit(p, (x + ox, y1 + oy))
    WINDOW.blit(ea, (x + ox, y2 + oy))
    WINDOW.blit(m, (x + ox, y3 + oy))
    WINDOW.blit(ha, (x + ox, y4 + oy))
    pygame.display.update()


# check for any win conditions
def check_win(positions, player):
    rows = len(positions)
    cols = len(positions[0])

    # Check for horizontal win
    for row in range(rows):
        for col in range(cols - 3):
            if (positions[row][col] == player and
                positions[row][col + 1] == player and
                positions[row][col + 2] == player and
                positions[row][col + 3] == player):
                return True

    # Check for vertical win
    for row in range(rows - 3):
        for col in range(cols):
            if (positions[row][col] == player and
                positions[row + 1][col] == player and
                positions[row + 2][col] == player and
                positions[row + 3][col] == player):
                return True

    # Check for ascending diagonals
    for row in range(3, rows):
        for col in range(cols - 3):
            if (positions[row][col] == player and
                positions[row - 1][col + 1] == player and
                positions[row - 2][col + 2] == player and
                positions[row - 3][col + 3] == player):
                return True

    # Check for descending diagonals
    for row in range(rows - 3):
        for col in range(cols - 3):
            if (positions[row][col] == player and
                positions[row + 1][col + 1] == player and
                positions[row + 2][col + 2] == player and
                positions[row + 3][col + 3] == player):
                return True


# get the lowest empty row for use in move function
def get_low_row(positions: list, col: int):
    lowest_empty = []
    for i in range(6):
        if positions[i][col] == 0:
            lowest_empty.append(i)
    if lowest_empty:
        return int(max(lowest_empty))
    else:
        return -1


# assign a move based on click
def player_move(positions: list, row: int, col: int, i: int):
    player = i % 2
    # if spot is empty, change to current player
    if positions[row][col] == 0:
        # player one
        if player == 0:
            positions[row][col] = 1
        # player two
        else:
            positions[row][col] = 2


# game mode with a friend
def passnplay():
    positions = np.zeros((6, 7), dtype=int)
    clock = pygame.time.Clock()
    run = True
    count, i = 42, 0
    while run:
        clock.tick(FPS)
        # loop through events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
            # register clicks as moves
            if event.type == pygame.MOUSEBUTTONDOWN:
                col = (pygame.mouse.get_pos()[0] - 60) // 100
                # make move on board
                if col >= 0 and col <= 6:
                    row = get_low_row(positions, col)
                    if row >= 0:
                        player_move(positions, row, col, i)
                        i += 1
                        count -= 1
                if count == 0 and not check_win(positions, 1) and not check_win(positions, 2):
                    print("Tie Game")
                    pygame.quit()
                elif check_win(positions, 1):
                    print("Red wins")
                    pygame.quit()
                elif check_win(positions, 2):
                    print("Yellow wins")
                    pygame.quit()
        set_game(positions)
    pygame.quit()


# easy game mode with random moves
def easy():
    clock = pygame.time.Clock()
    run = True
    play = True
    count = 42
    positions = np.zeros((6, 7), dtype=int)
    while run:
        clock.tick(FPS)
        # loop through events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
            if event.type == pygame.MOUSEBUTTONDOWN:
                x, y = pygame.mouse.get_pos()[0], pygame.mouse.get_pos()[1]
                if y >= 260 and y <= 450:
                    if x <= 330 and x >= 170:
                        player = True
                        run = False
                    elif x >= 470 and x <= 630:
                        player = False
                        run = False
        display_choice()

    if player:
        i = 0
    else:
        i = 1
        move = easy_move(positions, player)
        row = get_low_row(positions, move)
        if row >= 0:
            player_move(positions, row, move, i)
            i += 1
            count -= 1
    while play:
        clock.tick(FPS)
        # loop through events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                play = False
            # register clicks as moves
            if event.type == pygame.MOUSEBUTTONDOWN:
                col = (pygame.mouse.get_pos()[0] - 60) // 100
                # make move on board
                if col >= 0 and col <= 6:
                    row = get_low_row(positions, col)
                    if row >= 0:
                        player_move(positions, row, col, i)
                        i += 1
                        count -= 1
                        easy_move(positions, player)
                        i += 1
                        count -= 1
                if count == 0 and not check_win(positions, 1) and not check_win(positions, 2):
                    print("Tie Game")
                    pygame.quit()
                elif check_win(positions, 1):
                    print("Red Wins")
                    pygame.quit()
                elif check_win(positions, 2):
                    print('Yellow Wins')
                    pygame.quit()
        set_game(positions)
    pygame.quit()


# random move from computer
def easy_move(positions, player):
    legal_moves = []
    for i in range(len(positions[0])):
        if positions[0][i] == 0:
            legal_moves.append(i)
    move = random.choice(legal_moves)
    row = get_low_row(positions, move)
    if player:
        positions[row][move] = 2
    else:
        positions[row][move] = 1


# display screen for choice between first and second
def display_choice():
    WINDOW.fill(BLACK)
    x1, x2, y = 250, 550, 350
    size = 80
    FONT = pygame.font.Font("freesansbold.ttf", 25)
    f = FONT.render("First", True, WHITE)
    s = FONT.render("Second", True, WHITE)
    pygame.draw.circle(WINDOW, RED, (x1, y), size)
    pygame.draw.circle(WINDOW, YELLOW, (x2, y), size)
    WINDOW.blit(f, (x1 - 30, y - 120))
    WINDOW.blit(s, (x2 - 50, y - 120))
    pygame.display.update()


# decide between playing first and second for easy mode
def choice_menu(mode):
    clock = pygame.time.Clock()
    run = True
    while run:
        clock.tick(FPS)
        # loop through events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
            if event.type == pygame.MOUSEBUTTONDOWN:
                x, y = pygame.mouse.get_pos()[0], pygame.mouse.get_pos()[1]
                if y <= 310 and y >= 390:
                    if x <= 400:
                        easy(True)
                    else:
                        easy(False)
        display_choice()


def main():
    clock = pygame.time.Clock()
    run = True
    while run:
        clock.tick(FPS)
        # loop through events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
            if event.type == pygame.MOUSEBUTTONDOWN:
                x, y = pygame.mouse.get_pos()[0], pygame.mouse.get_pos()[1]
                if x >= 300 and x <= 500:
                    if y >= 275 and y <= 325:
                        passnplay()
                        run = False
                    elif y >= 350 and y <= 400:
                        easy()
                    elif y >= 425 and y <= 475:
                        med(True)
                    elif y >= 500 and y <= 550:
                        hard(True)
                
        set_main_menu()


def med(player):
    clock = pygame.time.Clock()
    run = True
    play = True
    count = 42
    positions = np.zeros((6, 7), dtype=int)
    file = '/Users/richie/Documents/git_hub/connect_four/periodic_save_more_neurons/periodic_save20093_16000.weights.h5'
    while run:
        clock.tick(FPS)
        # loop through events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
            if event.type == pygame.MOUSEBUTTONDOWN:
                x, y = pygame.mouse.get_pos()[0], pygame.mouse.get_pos()[1]
                if y >= 260 and y <= 450:
                    if x <= 330 and x >= 170:
                        player = True
                        run = False
                    elif x >= 470 and x <= 630:
                        player = False
                        run = False
        display_choice()

    if player:
        i = 0
    else:
        i = 1
        best_move = q_learn.choose_best_move(positions, i % 2 + 1, file)
        print(q_learn.list_moves(positions, i % 2 + 1, file ))
        row = get_low_row(positions, best_move)
        if row >= 0:
            player_move(positions, row, best_move, i)
            i += 1
            count -= 1
    while play:
        clock.tick(FPS)
        # loop through events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                play = False
            # register clicks as moves
            if event.type == pygame.MOUSEBUTTONDOWN:
                col = (pygame.mouse.get_pos()[0] - 60) // 100
                # make move on board
                if col >= 0 and col <= 6:
                    row = get_low_row(positions, col)
                    if row >= 0:
                        player_move(positions, row, col, i)
                        i += 1
                        count -= 1
                        # AI player's move
                        best_move = q_learn.choose_best_move(positions, player, file)  # Use the opponent's player number
                        print(q_learn.list_moves(positions, player, file ))
                        row = get_low_row(positions, best_move)
                        if row >= 0:
                            player_move(positions, row, best_move, i)
                            i += 1
                            count -= 1
                if count == 0 and not check_win(positions, 1) and not check_win(positions, 2):
                    print("Tie Game")
                    pygame.quit()
                elif check_win(positions, 1):
                    print("Red Wins")
                    pygame.quit()
                elif check_win(positions, 2):
                    print('Yellow Wins')
                    pygame.quit()
        set_game(positions)
    pygame.quit()



def hard(player):
    clock = pygame.time.Clock()
    run = True
    play = True
    count = 42
    positions = np.zeros((6, 7), dtype=int)
    file = '/Users/richie/Documents/git_hub/connect_four/periodic_save_multiprocess/periodic_save46990_2450_multiprocess.weights.h5'
    while run:
        clock.tick(FPS)
        # loop through events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
            if event.type == pygame.MOUSEBUTTONDOWN:
                x, y = pygame.mouse.get_pos()[0], pygame.mouse.get_pos()[1]
                if y >= 260 and y <= 450:
                    if x <= 330 and x >= 170:
                        player = True
                        run = False
                    elif x >= 470 and x <= 630:
                        player = False
                        run = False
        display_choice()

    if player:
        i = 0
    else:
        i = 1
        best_move = multi_process_q_learn.choose_best_move(positions, i % 2 + 1, file)
        print(multi_process_q_learn.list_moves(positions, i % 2 + 1, file ))
        row = get_low_row(positions, best_move)
        if row >= 0:
            player_move(positions, row, best_move, i)
            i += 1
            count -= 1
    while play:
        clock.tick(FPS)
        # loop through events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                play = False
            # register clicks as moves
            if event.type == pygame.MOUSEBUTTONDOWN:
                col = (pygame.mouse.get_pos()[0] - 60) // 100
                # make move on board
                if col >= 0 and col <= 6:
                    row = get_low_row(positions, col)
                    if row >= 0:
                        player_move(positions, row, col, i)
                        i += 1
                        count -= 1
                        # AI player's move
                        best_move = multi_process_q_learn.choose_best_move(positions, player, file)  # Use the opponent's player number
                        print(multi_process_q_learn.list_moves(positions, player, file ))
                        row = get_low_row(positions, best_move)
                        if row >= 0:
                            player_move(positions, row, best_move, i)
                            i += 1
                            count -= 1
                if count == 0 and not check_win(positions, 1) and not check_win(positions, 2):
                    print("Tie Game")
                    pygame.quit()
                elif check_win(positions, 1):
                    print("Red Wins")
                    pygame.quit()
                elif check_win(positions, 2):
                    print('Yellow Wins')
                    pygame.quit()
        set_game(positions)
    pygame.quit()



if __name__ == "__main__":
    main()
