#!/usr/bin/env python3
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

class MyBoard(Board):

    def __init__(self, *args, **kw):
        super().__init__(*args, **kw)
        self.interesting_walls = []

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
            adjacent_vert = {(x-2, y), (x+2, y)}
            adjacent_horiz = {(x-1, y-1), (x-1, y), (x-1, y+1), (x, y-1), (x, y+1), (x+1, y-1), (x+1, y), (x+1, y+1)}
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
            adjacent_vert = {(x-1, y-1), (x-1, y), (x-1, y+1), (x, y-1), (x+1, y-1), (x+1, y), (x+1, y+1), (x, y+1)}
            adjacent_horiz = {(x, y-2), (x, y+2)}
            self.interesting_walls += adjacent_vert
            self.interesting_walls += adjacent_horiz
        elif kind == 'WV':
            self.add_wall((x, y), False, player)
            adjacent_vert = {(x-2, y), (x+2, y)}
            adjacent_horiz = {(x-1, y-1), (x-1, y), (x-1, y+1), (x, y-1), (x, y+1), (x+1, y-1), (x+1, y), (x+1, y+1)}
            self.interesting_walls += adjacent_vert
            self.interesting_walls += adjacent_horiz
        elif kind == 'P':
            self.move_pawn((x, y), player)
        return self

class MyAgent(Agent):

    """My Quoridor agent."""

    def is_terminal(state):
        return state['pawns'][0][0] == 8 or state['pawns'][1][0] == 0

    def utility(state, player):
        assert is_terminal(state), "Utility can only be computed for a terminal state."
        return 1 if state['pawns'][player][0] == 8*(1-player) else -1


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
        '''
        if step==1:
            return ('WH', 5, 3)
        if step==2:
            return ('WH', 3, 3)
        if step == 3:
            return('WH', 5, 5)
        if step ==4:
            return('WH', 3, 5)
'''
        state = MyBoard(dict_to_board(percepts))

        max_depth = 3

        # if step > 30:
        #     max_depth = 2
        # if step > 40:
        #     max_depth = 3


        def heuristic_wall(state):
            return state.min_steps_before_victory(1-player)

        #def heuristic_move(state):
            #return -state.min_steps_before_victory(player)

        def move_heuristic(p, action):
          #if( action[0][0] == 'P' or ((abs(action[1] - percepts['pawns'][1][0]) + abs(action[2] - percepts['pawns'][1][1])) <= 2)):
          if action[0] == 'P':
              return -1
          if action in state.interesting_walls :
            return abs(action[1] - percepts['pawns'][1-p][0]) + abs(action[2] - percepts['pawns'][1-p][1])
          return 2 * (abs(action[1] - percepts['pawns'][1-p][0]) + abs(action[2] - percepts['pawns'][1-p][1]))
          

        
        if state.min_steps_before_victory(1-player) >= state.min_steps_before_victory(player) or state.nb_walls[player] == 0:
            #heuristic = heuristic_move

            move = ("P", *state.get_shortest_path(player)[0])
            print("Rush move", move)
            return move
        heuristic = heuristic_wall


        def max_value(state, alpha, beta, depth):
            if state.is_finished():
                return (state.get_score(player), (0,0))
            if depth >= max_depth:
                return (heuristic(state), (0,0))
            values = []
            actions = state.get_actions(player)
            #actions = state.get_legal_wall_moves(player)
            random.shuffle(actions)
            actions.sort(key=partial(move_heuristic, player))
            actions = actions[:20]
            for action in actions:
                #if( action[0][0] == 'P' or ((abs(action[1] - percepts['pawns'][1][0]) + abs(action[2] - percepts['pawns'][1][1])) <= 2)):
                values.append((min_value(state.clone().play_action(action, player), alpha, beta, depth+1)[0], action))
                if (alpha := min(alpha, values[-1][0]))>beta:
                    return (values[-1][0], action)
   
            result = max(values, key=itemgetter(0))
           # print("MAX:", result)
            return result
    
        def min_value(state, alpha, beta, depth):
            if state.is_finished():
                return (state.get_score(player), (0,0))
            if depth >= max_depth:
                return (heuristic(state), (0,0))
            values = []
            actions = state.get_actions(1-player)
            #actions = state.get_legal_wall_moves(1-player)
            random.shuffle(actions)
            actions.sort(key=partial(move_heuristic, 1-player))
            actions = actions[:20]
            for action in actions: 
                                
                #if( action[0][0] == 'P' or ((abs(action[1] - percepts['pawns'][0][0]) + abs(action[2] - percepts['pawns'][0][1])) <= 2)):
                values.append((max_value(state.clone().play_action(action, 1-player), alpha, beta, depth+1)[0], action))
                if alpha>(beta := max(beta, values[-1][0])):
                    return (values[-1][0], action)
    
            result = min(values, key=itemgetter(0))
           # print("MIN:", result)
            return result
    
        move = max_value(state, -infinity, +infinity, 0)[1]
        print(move)
        return move



if __name__ == "__main__":
    agent_main(MyAgent())
