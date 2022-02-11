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
  #      if step==1:
  #          return ('WH', 5, 4)

        state = dict_to_board(percepts)

        max_depth = 2

        def heuristic(state):
            return state.get_score(player)

        def max_value(state, alpha, beta, depth):
            if state.is_finished():
                return (state.get_score(player), (0,0))
            if depth >= max_depth:
                return (heuristic(state), (0,0))
            values = []
            actions = state.get_actions(player)
            random.shuffle(actions)
            for action in actions:
                values.append((min_value(state.clone().play_action(action, player), alpha, beta, depth+1)[0], action))
                if (alpha := min(alpha, values[-1][0]))>beta:
                    return (values[-1][0], action)
   
            return max(values, key=itemgetter(0))
    
        def min_value(state, alpha, beta, depth):
            if state.is_finished():
                return (state.get_score(player), (0,0))
            if depth >= max_depth:
                return (heuristic(state), (0,0))
            values = []
            actions = state.get_actions(1-player)
            random.shuffle(actions)
            for action in actions: 
                values.append((max_value(state.clone().play_action(action, 1-player), alpha, beta, depth+1)[0], action))
                if alpha>(beta := max(beta, values[-1][0])):
                    return (values[-1][0], action)
    
            return min(values, key=itemgetter(0))
    
        return max_value(state, -infinity, +infinity, 0)[1]



if __name__ == "__main__":
    agent_main(MyAgent())
