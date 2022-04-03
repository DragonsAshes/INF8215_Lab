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

hack_step=0
class MyAgent(Agent):

    """My Quoridor agent."""
    """This agent exploits a bug in the quoridor code that if not fixed by the other players makes their agent crash if there're currently no path because our player is blocking the only path."""

    def play(self, percepts, player, step, time_left):
        global hack_step
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
        
        first_row = []
        for i in range(8):
            first_row.append([player*7, i])
        if hack_step < 3:
            print("Delaying opponent")
            oppo_y, oppo_x = state.pawns[1-player]
            oppo_goal_y = state.goals[1-player]
            wall_actions = state.get_legal_wall_moves(player)

            # find valid walls in front of opponent
            candidate_walls = []
            if oppo_goal_y < oppo_y:
                print("opponent moving north")
                for wall_action in wall_actions:
                    wall_dir, wall_y, wall_x = wall_action
                    if wall_dir == 'WH' and wall_y == oppo_y - 1 and wall_x in (oppo_x, oppo_x - 1):
                        candidate_walls.append(wall_action)
            else:
                print("opponent moving south")
                for wall_action in wall_actions:
                    wall_dir, wall_y, wall_x = wall_action
                    if wall_dir == 'WH' and wall_y == oppo_y and wall_x in (oppo_x, oppo_x - 1):
                        candidate_walls.append(wall_action)
            print(f"candidate walls: {candidate_walls}")

            if len(candidate_walls) > 0:
                choice = random.choice(candidate_walls)
                print(f"placing a wall: {choice}")
                hack_step += 1
                return choice


        wall_moves = []
        for action in state.get_actions(player):
            move, *coordinates = action
            if move == "WH" and coordinates in first_row:
                wall_moves.append(action)

        wall_moves.sort(key=lambda x: abs(x[1] - state.pawns[player][0])+abs(x[2] - state.pawns[player][1]))
        
        if wall_moves:
            print("Placing support walls")
            return wall_moves[0] 

        if hack_step==3:
            hack_step+=1
            if (player*7, 0) not in state.horiz_walls:
                y = 1
            if (player*7, 7) not in state.horiz_walls:
                y = 6
            x = 6 if player else 1
            action = ("WV", x, y)
            if action in state.get_actions(player):
                return action

        player_pos = state.pawns[player]
        if player_pos[1] != 0 and player_pos[1] != 8:
            print("Moving to the side")
            return ("P", *state.get_shortest_path(player)[0])

        if player == 0 and player_pos[0] == 0:
            return ("P", 1, player_pos[1])
        if player == 1 and player_pos[0] == 8:
            return ("P", 7, player_pos[1])

        if hack_step==4:
            wall_moves = []
            for action in state.get_actions(player):
                move, *coordinates = action
                if move == "WV" and coordinates in first_row:
                    wall_moves.append(action)

            wall_moves.sort(key=lambda x: abs(x[1] - state.pawns[player][0])+abs(x[2] - state.pawns[player][1]))
            
            if wall_moves:
                print("Placing last support wall")
                hack_step+=1
                return wall_moves[0] 

        if player == 0 and player_pos[0] == 1:
            return ("P", 0, player_pos[1])
        if player == 1 and player_pos[0] == 7:
            return ("P", 8, player_pos[1])

if __name__ == "__main__":
    agent_main(MyAgent())
