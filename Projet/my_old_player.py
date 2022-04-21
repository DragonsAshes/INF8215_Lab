#!/usr/bin/env python3

"""
Team:
Cazou
Authors:
Virgile Retault - 2164296
Sebastien Foucher - 2162248
"""

"""
Quoridor agent.
Copyright (C) 2013, DragonsAshes & g33kex 

This program is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation; version 2 of the License.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program; if not, see <http://www.gnu.org/licenses/>.

"""

from quoridor import *

import math
import random
from operator import itemgetter

infinity = math.inf

import time
from functools import wraps
from functools import partial

times = {}
occurences = {}
def timeit(my_func):
    #This class is used to mesure the average time of function's execution
    @wraps(my_func)
    def timed(*args, **kw):
        name = my_func.__name__

        tstart = time.time()
        output = my_func(*args, **kw)
        tend = time.time()

        deltat = 1000*(tend-tstart)
        times[name] = times.get(name, 0) + deltat
        occurences[name] = occurences.get(name, 0) + 1

        print('"{}" took {:.3f} ms to execute ({:.2f} ms average)\n'.format(name, deltat, times[name]/occurences[name]))
        return output
    return timed

#Reimplementing some Board functions to make them faster
class MyBoard(Board):

    def __init__(self, *args, **kw):
        super().__init__(*args, **kw)
        self.interesting_walls = set()

    #@timeit
    def get_legal_pawn_moves(self, player):
        """Returns legal moves for the pawn of player."""
        (x, y) = self.pawns[player]
        positions = [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1),
        (x + 1, y + 1), (x - 1, y - 1), (x + 1, y - 1), (x - 1, y + 1),
        (x + 2, y), (x - 2, y), (x, y + 2), (x, y - 2)]
        moves = []
        for new_pos in positions:
            if self.is_pawn_move_ok(self.pawns[player], new_pos,
                self.pawns[(player + 1) % 2]):
                moves.append(('P', new_pos[0], new_pos[1]))
        return moves

    #@timeit
    def get_legal_wall_moves(self, player):
        """Returns legal wall placements (adding a wall
        somewhere) for player.
        """
        positions = []
        moves = []
        if self.nb_walls[player] <= 0:
            return moves
        for i in range(self.size - 1):
            for j in range(self.size - 1):
                positions.append((i, j))
        for pos in positions:
            if self.is_wall_possible_here(pos, True):
                moves.append(('WH', pos[0], pos[1]))
            if self.is_wall_possible_here(pos, False):
                moves.append(('WV', pos[0], pos[1]))
        return moves

    def is_wall_possible_here(self, pos, is_horiz):
        """
        Returns True if it is possible to put a wall in position pos
        with direction specified by is_horiz.
        """
        (x, y) = pos
        if x >= self.size - 1 or x < 0 or y >= self.size - 1 or y < 0:
            return False
        horiz_walls = set(self.horiz_walls)
        verti_walls = set(self.verti_walls)
        if (tuple(pos) in horiz_walls or tuple(pos) in verti_walls):
            return False
        wall_horiz_right = (x, y + 1) in horiz_walls
        wall_horiz_left = (x, y - 1) in horiz_walls
        wall_vert_up = (x - 1, y) in verti_walls
        wall_vert_down = (x + 1, y) in verti_walls

        if is_horiz:
            if wall_horiz_right or wall_horiz_left:
                return False
            # Doing paths_exist() only if the new wall is adjacent to existing walls
            adjacent_vert = {(x-1, y-1), (x-1, y), (x-1, y+1), (x, y-1), (x+1, y-1), (x+1, y), (x+1, y+1), (x, y+1)}
            adjacent_horiz = {(x, y-2), (x, y+2)}
            if adjacent_vert.intersection(verti_walls) or adjacent_horiz.intersection(horiz_walls):
                self.horiz_walls.append(tuple(pos))
                if not self.paths_exist():
                    a = self.horiz_walls.pop()
                    return False
                self.horiz_walls.pop()
        else:
            if wall_vert_up or wall_vert_down:
                return False
            # Doing paths_exist() only if the new wall is adjacent to existing walls
            adjacent_horiz = {(x-1, y-1), (x-1, y), (x-1, y+1), (x, y-1), (x+1, y-1), (x+1, y), (x+1, y+1), (x, y+1)}
            adjacent_vert = {(x, y-2), (x, y+2)}
            if adjacent_vert.intersection(verti_walls) or adjacent_horiz.intersection(horiz_walls):
                self.verti_walls.append(tuple(pos))
                if not self.paths_exist():
                    a = self.verti_walls.pop()
                    return False
                self.verti_walls.pop()

        return True

    def get_actions(self, player):
        """ Returns all the possible actions for player."""
        pawn_moves = self.get_legal_pawn_moves(player)
        wall_moves = self.get_legal_wall_moves(player)
        pawn_moves.extend(wall_moves)
        return pawn_moves

    def clone(self):
        """Return a clone of this object."""
        clone_board = MyBoard()
        clone_board.pawns[0] = self.pawns[0]
        clone_board.pawns[1] = self.pawns[1]
        clone_board.goals[0] = self.goals[0]
        clone_board.goals[1] = self.goals[1]
        clone_board.nb_walls[0] = self.nb_walls[0]
        clone_board.nb_walls[1] = self.nb_walls[1]
        clone_board.interesting_walls = self.interesting_walls
        for (x, y) in self.horiz_walls:
            clone_board.horiz_walls.append((x, y))
        for (x, y) in self.verti_walls:
            clone_board.verti_walls.append((x, y))
        return clone_board

    def play_action(self, action, player):
        """Play an action"""
        kind, x, y = action
        if kind == 'WH':
            self.add_wall((x, y), True, player)
            adjacent_horiz = {("WH", x-2, y), ("WH", x+2, y)}
            adjacent_vert = {("WV", x-1, y-1), ("WV", x-1, y), ("WV", x-1, y+1), ("WV", x, y-1), ("WV", x, y+1), ("WV", x+1, y-1), ("WV", x+1, y), ("WV", x+1, y+1)}
            self.interesting_walls |= adjacent_vert
            self.interesting_walls |= adjacent_horiz
        elif kind == 'WV':
            self.add_wall((x, y), False, player)
            adjacent_vert = {("WV", x-2, y), ("WV", x+2, y)}
            adjacent_horiz = {("WH", x-1, y-1), ("WH", x-1, y), ("WH", x-1, y+1), ("WH", x, y-1), ("WH", x, y+1), ("WH", x+1, y-1), ("WH", x+1, y), ("WH", x+1, y+1)}
            self.interesting_walls |= adjacent_vert
            self.interesting_walls |= adjacent_horiz
        elif kind == 'P':
            self.move_pawn((x, y), player)
        return self

class MyAgent(Agent):

    """My Quoridor agent."""

    def play(self, percepts, player, step, time_left):
        """
        This function is used to play a move according
        to the percepts, player and time left provided as input.
        It must return an action representing the move the player
        will perform.
        :param percepts: dictionary representing the current board
            in a form that can be fed to `dict_to_board()` in quoridor.py.
        :param player: the player to control in this step (0 or 1)
        :param step: the current step number, starting from 1
        :param time_left: a float giving the number of seconds left from the time
            credit. If the game is not time-limited, time_left is None.
        :return: an action
          eg: ('P', 5, 2) to move your pawn to cell (5,2)
          eg: ('WH', 5, 2) to put a horizontal wall on corridor (5,2)
          for more details, see `Board.get_actions()` in quoridor.py
        """
        print("percept:", percepts)
        print("player:", player)
        print("step:", step)
        print("time left:", time_left if time_left else '+inf')

        state = MyBoard(dict_to_board(percepts))

        # Setting the search's depth to 3
        max_depth = 2

        # Define the heuristic to get the cost of a move
        def heuristic(state):
            try:
                h = 1.1*state.min_steps_before_victory(1-player)-state.min_steps_before_victory(player) 
            except NoPath:
                h = +infinity
            return h

        # Define the heuristic to classify movement
        def move_heuristic(p, action):
          if action[0] == 'P':
                  return -1
          if action in state.interesting_walls :
            return abs(action[1] - percepts['pawns'][1-p][0]) + abs(action[2] - percepts['pawns'][1-p][1])
          return 2*(abs(action[1] - percepts['pawns'][1-p][0]) + abs(action[2] - percepts['pawns'][1-p][1]))

        # Remove pwn moves which are not the best
        # Used to avoid going back and forth
        def remove_useless_pawn_moves(L, p):
            best_pawn_move = state.get_shortest_path(p)[0]
            L2 = []
            for a in L:
                if a[0] != "P" or (a[1], a[2]) == best_pawn_move:
                    L2.append(a)
            return L2

        # If there is no path, play the best pwn move by default
        # Used to avoid the glitch where the path is blocked by one of the player
        if not state.paths_exist():
           return state.get_legal_pawn_moves(player)[0] 

        # If the player is closest to the goal than the over one, just rush
        if state.min_steps_before_victory(1-player) >= state.min_steps_before_victory(player) or state.nb_walls[player] == 0:
            move = ("P", *state.get_shortest_path(player)[0])
            print("Rush move", move)
            return move

        def max_value(state, alpha, beta, depth):
            # Return score if state is a goal state
            if state.is_finished():
                return (state.get_score(player), (0,0))
            # Return the heuristic of the state if max depth is reached
            if depth >= max_depth:
                return (heuristic(state), (0,0))

            values = []
            actions = state.get_actions(player)
            #Add some randomness to the moves
            random.shuffle(actions)
            #Sort moves using heuristic
            actions.sort(key=partial(move_heuristic, player))
            actions = remove_useless_pawn_moves(actions, player)
            # Only keep 20 first moves
            # actions = actions[:20]
            
            for action in actions:
                try:
                    values.append((min_value(state.clone().play_action(action, player), alpha, beta, depth+1)[0], action))
                except NoPath:
                    return (-infinity, action)
                if (alpha := min(alpha, values[-1][0]))>beta:
                    return (values[-1][0], action)

            result = max(values, key=itemgetter(0))
            return result
    
        def min_value(state, alpha, beta, depth):
            # Return score if state is a goal state
            if state.is_finished():
                return (state.get_score(player), (0,0))
            # Return the heuristic of the state if max depth is reached
            if depth >= max_depth:
                return (heuristic(state), (0,0))
            values = []
            actions = state.get_actions(1-player)
            #Add some randomness to the moves
            random.shuffle(actions)
            #Sort moves using heuristic
            actions.sort(key=partial(move_heuristic, 1-player))
            actions = remove_useless_pawn_moves(actions, 1-player)
            # Only keep 20 first moves
            # actions = actions[:20]
            for action in actions: 
                try:
                    values.append((max_value(state.clone().play_action(action, 1-player), alpha, beta, depth+1)[0], action))
                except NoPath:
                    return (+infinity, action)
                if alpha>(beta := max(beta, values[-1][0])):
                    return (values[-1][0], action)
    
            result = min(values, key=itemgetter(0))
            return result
    
        # Last check to see if move is valid befor playing. If not, play the best pwn move by default
        move = max_value(state, -infinity, +infinity, 0)[1]
        print("Trying to play:", move)
        state2 = Board(dict_to_board(percepts))
        if state2.is_action_valid(move, player):
            print("I can confirm this move is valid")
            return move
        move = ("P", *state.get_shortest_path(player)[0])
        print("move becoming : ", move)
        return move

if __name__ == "__main__":
    agent_main(MyAgent())
