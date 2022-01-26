from __future__ import annotations
# search.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util
from game import Directions
from dataclasses import dataclass

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

# This class allows to store the previous node while exploring for faster backtracking
class Node:

    def __init__(self, state):
        self.state = state
        self.previous = None
        self.direction = None

    def getState(self):
        return self.state 

    def setPrevious(self, previous):
        self.previous = previous

    def getPrevious(self):
        return self.previous

    # Direction to get from previous node to this one
    def setDirection(self, direction):
        self.direction = direction

    def getDirection(self):
        return self.direction

    def __eq__(self, other):
        return self.getState() == other.getState()

def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print("Start:", problem.getStartState())
    print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    print("Start's successors:", problem.getSuccessors(problem.getStartState()))
    """
    from game import Actions

    fringe = util.Stack()
    fringe.push(Node(problem.getStartState()))

    visited = []

    current_node = fringe.pop()

    while not problem.isGoalState(current_node.getState()):
        visited.append(current_node)

        for successor, direction, cost in problem.getSuccessors(current_node.getState()):
            next_node = Node(successor)
            if next_node not in visited:
                next_node.setPrevious(current_node)
                next_node.setDirection(direction)
                fringe.push(next_node)

        if fringe.isEmpty():
            return []
        current_node = fringe.pop()

    directions = []
    while current_node.getPrevious() is not None:
        directions.insert(0, current_node.getDirection())
        current_node = current_node.getPrevious()

    return directions

def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""


    '''
        INSÉREZ VOTRE SOLUTION À LA QUESTION 2 ICI

        avoir une liste d etat visites
        mettre en queue de fringe
        changer d'etat en prenant l element en tete de la fringe

    '''
    fringe = util.Queue()
    fringe.push(Node(problem.getStartState()))

    visited = []

    current_node = fringe.pop()

    while not problem.isGoalState(current_node.getState()):
        visited.append(current_node)

        for successor, direction, cost in problem.getSuccessors(current_node.getState()):
            next_node = Node(successor)
            if next_node not in visited:
                next_node.setPrevious(current_node)
                next_node.setDirection(direction)
                fringe.push(next_node)

        if fringe.isEmpty():
            return []
        
        # Don't visit an already visited node even if it is in the fringe
        while current_node in visited:
            current_node = fringe.pop()

    directions = []
    while current_node.getPrevious() is not None:
        directions.insert(0, current_node.getDirection())
        current_node = current_node.getPrevious()

    return directions

def uniformCostSearch(problem):
    """Search the node of least total cost first."""


    '''
        INSÉREZ VOTRE SOLUTION À LA QUESTION 3 ICI
    '''

    util.raiseNotDefined()

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

@dataclass(eq=False)
class ANode:
    """A structure used to store information about a state useful for the A* algorithm."""
    state: any #: The stat represented by the ANode
    previous: ANode = None #: The previous ANode we went through to go to this one
    direction: Direction = None #: The direction to go from the previous ANode to this one
    gcost: int = 0 #: The cost to go to that node from the start
    hcost: int = None #: The approximated cost to go from that node to the end

    def __eq__(self, other):
        return self.state == other.state


def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    fringe = util.PriorityQueue()
    visited = []
    directions = []
    
    fringe.push(ANode(problem.getStartState()), heuristic(problem.getStartState(), problem))

    while not problem.isGoalState((current_node := fringe.pop()).state):
        if current_node in visited:
            if fringe.isEmpty():
                break
            continue

        visited.append(current_node)

        for successor, direction, cost in problem.getSuccessors(current_node.state):
            next_node = ANode(successor)
            if next_node not in visited: # No need to put a visited node in the fringe
                next_node.previous = current_node
                next_node.direction = direction
                next_node.hcost = heuristic(next_node.state, problem)
                next_node.gcost = current_node.gcost + cost
                fringe.push(next_node, next_node.gcost + next_node.hcost)

        if fringe.isEmpty():
            break
    else:
        while current_node.previous is not None:
            directions.insert(0, current_node.direction)
            current_node = current_node.previous

    return directions

# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
