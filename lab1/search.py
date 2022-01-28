from __future__ import annotations # This allows to type hint recursive classes and to use typed lists
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

## Imports

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
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

@dataclass(eq=False)
class Node:
    """
    This data class allows to store the previous node and direction from that node while exploring for faster backtracking
    """
    state: any
    """The state represented by the Node. It must be hashable"""
    previous: ANode = None
    """
    The previous Node we went through to go to this one.
    """
    direction: Direction = None 
    """
    The direction to go from the previous Node to this one
    """

    def construct_path(self: Node) -> list[Direction]:
        """
        Returns the list of direction to go from the start node to this node.
        """

        if self.previous is not None:
            return self.previous.construct_path() + [self.direction]
        return []

    def __eq__(self, other):
        """
        Two nodes are equal if their states are equal
        """
        
        return self.state == other.state

    def __hash__(self):
        """
        Returns the hash of the Node which is the hash of its state
        """

        return hash(self.state)

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

    visited = set() # Set of visited nodes
    fringe = util.Stack() # LIFO fringe
    fringe.push(Node(problem.getStartState()))

    # Pop the fringe if it's not empty and stop if we arrive at the goal
    while not (fringe.isEmpty() or problem.isGoalState((current_node := fringe.pop()).state)):
        if current_node in visited: # Check if current node is visted because we search on a graph
            continue
        visited.add(current_node)
        for state, direction, cost in problem.getSuccessors(current_node.state):
            fringe.push(Node(state, current_node, direction)) # Add successors to the fringe
    
    # Reconstruct the path, if the last node is not the goal there is no solution
    return current_node.construct_path() if problem.isGoalState(current_node.state) else []

def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""

    visited = set() # Set of visited nodes
    fringe = util.Queue() # FIFO fringe
    fringe.push(Node(problem.getStartState()))

    # Pop the fringe if it's not empty and stop if we arrive at the goal
    while not (fringe.isEmpty() or problem.isGoalState((current_node := fringe.pop()).state)):
        if current_node in visited: # Check if current node is visted because we search on a graph
            continue
        visited.add(current_node)
        for state, direction, cost in problem.getSuccessors(current_node.state):
            fringe.push(Node(state, current_node, direction)) # Add successors to the fringe
    
    # Reconstruct the path, if the last node is not the goal there is no solution
    return current_node.construct_path() if problem.isGoalState(current_node.state) else []

@dataclass(eq=False)
class ANode(Node):
    """
    A structure used to store information about a state and is used for the A* algorithm and the UCS algorithm.
    """

    gcost: int = 0
    """
    The cost to go to that node from the start
    """

def uniformCostSearch(problem):
    """Search the node of least total cost first."""

    visited = set() # Set of visited nodes
    fringe = util.PriorityQueue() # FIFO fringe
    fringe.push(ANode(problem.getStartState()), 0)

    # Pop the fringe if it's not empty and stop if we arrive at the goal
    while not (fringe.isEmpty() or problem.isGoalState((current_node := fringe.pop()).state)):
        if current_node in visited: # Check if current node is visted because we search on a graph
            continue
        visited.add(current_node)
        for state, direction, cost in problem.getSuccessors(current_node.state):
            next_node = ANode(state, current_node, direction, current_node.gcost + cost)
            fringe.push(next_node, next_node.gcost) # Add successors to the fringe with priority gcost
    
    # Reconstruct the path, if the last node is not the goal there is no solution
    return current_node.construct_path() if problem.isGoalState(current_node.state) else []


def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""

    visited = set() # Set of visited nodes
    fringe = util.PriorityQueue() # FIFO fringe
    fringe.push(ANode(problem.getStartState()), heuristic(problem.getStartState(), problem))

    # Pop the fringe if it's not empty and stop if we arrive at the goal
    while not (fringe.isEmpty() or problem.isGoalState((current_node := fringe.pop()).state)):
        if current_node in visited: # Check if current node is visted because we search on a graph
            continue
        visited.add(current_node)
        for state, direction, cost in problem.getSuccessors(current_node.state):
            next_node = ANode(state, current_node, direction, current_node.gcost + cost)
            fringe.push(next_node, next_node.gcost + heuristic(state, problem)) # Add successors to the fringe with priority gcost+hcost
    
    # Reconstruct the path, if the last node is not the goal there is no solution
    return current_node.construct_path() if problem.isGoalState(current_node.state) else []

# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
