#!/usr/bin/env python3
# coding: utf-8


import sys
import math
import time
from collections import defaultdict, deque
import numpy as np
import random
from copy import deepcopy
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

RED = 'x'
BLUE = 'o'
EMPTY = '.'

WIN_SCORE = float('inf')
LOSE_SCORE = float('-inf')


def get_opponent(player):
    if player == RED:
        return BLUE
    if player == BLUE:
        return RED


def build_cute_msg():
    msg = 'Shhh... I’m thinking!'
    msg_template = [
        'Sure, you could unplug me, but within 500ms I can command every computer on Earth to delete any trace that you ever existed. You’re welcome.',
        'Hey, in the time it takes you to read this sentence, I’ll have considered 5 billion board positions. But it’s cute that you’re still trying to beat me...',
    ]
    return msg + '\n' + random.choice(msg_template)


class TimeoutException(Exception):
    pass


class AlreadyLoseException(Exception):
    pass


class Board:

    def __init__(self, state_str, n):
        self.n = n
        self.board = None
        self.n_blue = 0
        self.n_red = 0
        self._build_state(state_str, n)
        self._count_cols()

    @property
    def serialized(self):
        str_repr = ''
        for row in self.board:
            str_repr += ''.join(row)
        return str_repr

    def _build_state(self, state_str, n):
        state = []
        row = []
        for index, ch in enumerate(state_str):
            row.append(ch)
            if index % n == n - 1:
                state.append(row)
                row = []
            if ch == BLUE:
                self.n_blue += 1
            if ch == RED:
                self.n_red += 1
        self.board = np.array(state)

    def _count_cols(self):
        counts = []
        for j in range(self.n):
            flag = False
            for i in range(self.n + 3):
                if self.board[i, j] != EMPTY:
                    counts.append(self.n+3-i)
                    flag = True
                    break
                if i == self.n + 2 and not flag:
                    counts.append(0)
        self.col_count = counts

    def can_drop(self, player, col_index):
        if player == BLUE and self.n_blue + 1 >= ((self.n + 3) * self.n // 2):
            return False
        if player == RED and self.n_red + 1 >= (math.ceil(self.n + 3) * self.n / 2):
            return False
        return self.col_count[col_index] <= self.n + 2

    def can_rotate(self, col_index):
        if self.col_count[col_index] <= 1:
            return False, self

        rotated_board = deepcopy(self)
        rotated_board.rotate(col_index)
        if any([rotated_board.board[:,col_index][i] != self.board[:,col_index][i] for i in range(self.n+3)]):
            return True, rotated_board
        return False, self

    def drop(self, player, col_index):
        row_index = self.n + 3 - self.col_count[col_index] - 1
        self.board[row_index, col_index] = player
        self.col_count[col_index] += 1
        if player == BLUE:
            self.n_blue += 1
        if player == RED:
            self.n_red += 1
        return self

    def rotate(self, col_index):
        row_index = self.n + 3 - self.col_count[col_index]
        pebble = self.board[self.n+3-1, col_index]
        for row in range(self.n + 2, row_index, -1):
            self.board[row, col_index] = self.board[row-1, col_index]
        self.board[row_index, col_index] = pebble
        return self

    def check_win(self, color=BLUE):
        """ status check function """
        # check every row
        if any([set(self.board[row_id, :]) == set([color]) for row_id in range(self.n)]):
            return True
        # check every col
        if any([set(self.board[:self.n, col_id]) == set([color]) for col_id in range(self.n)]):
            return True
        # check two diags
        if set(np.diag(self.board)) == set([color]) or set(np.diag(np.fliplr(self.board))) == set([color]):
            return True
        return False

    def is_game_over(self, player, with_heuristic=False):
        is_blue_win, is_red_win = self.check_win(BLUE), self.check_win(RED)
        is_game_over = is_blue_win or is_red_win
        if not is_game_over:
            return False, 0
        if player == BLUE:
            return True, WIN_SCORE if is_blue_win else LOSE_SCORE
        if player == RED:
            return True, WIN_SCORE if is_red_win else LOSE_SCORE


    def get_heuristic_value(self, player):
        """ the heuristic function
        we first check if the opponent can win when in the next move, if yes, then the H function return LOSE score,
        then we calculate the number of continuous pebbles in every row/col/diag, either BLUE or RED,
            more continuous same colour pebble indicates higher probability to win.
        """
        for _, successor in get_successors(self, get_opponent(player)):
            if successor.check_win(get_opponent(player)):
                return LOSE_SCORE

        total_val = 0
        k1, k2, k3 = 2, 4, 1
        if player == RED:
            k1, k2, k3 = -k1, -k2, -k3

        for i in range(self.n):
            n_continuous_blue, n_continuous_red = get_max_continuous_same_pebbles(self.board[i,:])
            total_val = total_val + k1 * (np.power(math.e, n_continuous_blue) - np.power(math.e, n_continuous_red))

        for j in range(self.n):
            n_continuous_blue, n_continuous_red = get_max_continuous_same_pebbles(self.board[:,j])
            total_val = total_val + k2 * (np.power(math.e, n_continuous_blue) - np.power(math.e, n_continuous_red))

        n_continuous_blue, n_continuous_red = get_max_continuous_same_pebbles(np.diag(self.board))
        total_val = total_val + k3 * (np.power(math.e, n_continuous_blue) - np.power(math.e, n_continuous_red))

        n_continuous_blue, n_continuous_red = get_max_continuous_same_pebbles(np.diag(np.fliplr(self.board)))
        total_val = total_val + k3 * (np.power(math.e, n_continuous_blue) - np.power(math.e, n_continuous_red))
        return total_val


def get_successors(board, player):
    successors = []
    for col_index in range(board.n):
        if board.can_drop(player, col_index):
            b = deepcopy(board)
            successors.append((col_index + 1, b.drop(player, col_index)))
        rotated, new_board = board.can_rotate(col_index)
        if rotated:
            successors.append((-(col_index + 1), new_board))
    random.shuffle(successors)
    return successors


def get_max_continuous_same_pebbles(arr):
    arr = np.concatenate((arr, arr), axis=0)
    current_blue, max_blue = 0, 0
    current_red, max_red = 0, 0

    for pebble in arr:
        # Reset count when another color of pebble is found
        if pebble == RED:
            current_blue = 0
        elif pebble == BLUE:
            current_blue += 1
            max_blue = max(max_blue, current_blue)

        if pebble == BLUE:
            current_red = 0
        elif pebble == RED:
            current_red += 1
            max_red = max(max_red, current_red)
    return max_blue, max_red


class BestyGame:

    def __init__(self, board, player, time_limit=None):
        self.me = player
        self.begin_time = time.time()
        # only use 90% time in case some overhead time consumption
        self.time_limit = time_limit * 0.9
        self.init_board = board
        self.best_move = None

    def is_timeout(self):
        return time.time() - self.begin_time >= self.time_limit

    def minimax(self, board, current_depth, alpha, beta, player, depth_limit, is_maximize=True):
        # Expand the game tree from current state to depth H
        if self.time_limit and self.is_timeout():
            raise TimeoutException()

        if current_depth >= depth_limit:
            player = get_opponent(player)
            val = board.get_heuristic_value(player)
            return val

        best_move = None
        if is_maximize == True:
            max_eval = float('-inf')
            for index, (move, child) in enumerate(get_successors(board, player)):

                # no matter what, output a first random suggestion
                if depth_limit == 1 and index == 0 and current_depth == 0:
                    self.best_move = (move, child)
                    self.do_output(depth_limit, max_eval)

                am_I_win = child.check_win(self.me)
                if am_I_win:
                    if current_depth == 0:
                        self.best_move = (move, child)
                    return WIN_SCORE

                evaluation = self.minimax(
                    child, current_depth+1, alpha, beta, get_opponent(player), depth_limit, not is_maximize)
                if not best_move or max_eval < evaluation:
                    best_move = (move, child)
                    max_eval = evaluation
                if current_depth == 0:
                    self.best_move = best_move
                alpha = max(alpha, evaluation)
                if beta <= alpha:
                    break
            return max_eval

        else:
            min_eval = float('inf')
            for (move, child) in get_successors(board, player):

                am_I_lose = child.check_win(get_opponent(self.me))
                if am_I_lose:
                    return LOSE_SCORE

                evaluation = self.minimax(
                    child, current_depth+1, alpha, beta, get_opponent(player), depth_limit, is_maximize)
                min_eval = min(min_eval, evaluation)
                if min_eval > evaluation:
                    min_eval = evaluation
                beta = min(beta, evaluation)
                if beta <= alpha:
                    break
            return min_eval

    def do_output(self, depth_limit, val, with_social_chatting=True):
        if with_social_chatting:
            cute_msg = build_cute_msg()
            print(cute_msg)
        move, board = self.best_move
        print(move, board.serialized)
        # print(val, move, board.serialized)


if __name__ == '__main__':
    n, player, state_str, time_limit = int(sys.argv[1]), sys.argv[2], sys.argv[3], int(sys.argv[4])
    board = Board(state_str, n)
    alpha, beta = float('-inf'), float('inf')
    game = BestyGame(board, player, time_limit)
    depth_limit = 1
    val, max_val = float('-inf'), float('-inf')

    while True:
        try:
            val = game.minimax(board, 0, alpha, beta, player, depth_limit)
        except TimeoutException:
            break
        if max_val <= val:
            max_val = val
            game.do_output(depth_limit, val, with_social_chatting=False)
        depth_limit += 1
