# Overview
# This program solves a maze of any size by reading it from a
# plain-text file and applying Breadth-First Search (BFS) to
# find the shortest path from a start cell ('S') to an end
# cell ('E').
#
# Why AI and not a typical decision-making algorithm?
# A conventional rule-based algorithm (e.g. "always turn left")
# can only handle simple, predictable mazes and fails when the
# layout changes. BFS belongs to the family of uninformed
# search algorithms used throughout AI because it models the
# problem as a state-space graph: each cell is a node and each
# valid move is an edge. This generalises perfectly to any maze
# shape or size without hard-coded rules.
#
# How BFS works
# BFS explores the maze level by level using a queue (FIFO).
# Starting from 'S', it enqueues all reachable neighbours,
# marks them as visited, then processes each one in turn,
# enqueuing their unvisited neighbours. Because every move has
# equal cost, the first time BFS reaches 'E' it is guaranteed
# to have taken the fewest steps possible — i.e. the shortest
# path. This is a key advantage over Depth-First Search (DFS),
# which finds a path but not necessarily the shortest one.
#
# Code structure
# The program is split into four focused functions:
#   read_maze()             — parses the text file into a 2-D list
#   is_safe()               — validates boundary, wall, and visited checks
#   breadth_first_search()  — implements the BFS algorithm
#   print_result()          — renders the solved maze to the screen
#
# Each path in the queue stores the full route taken so far, so
# when the goal is reached the solution can be returned directly
# without a separate back-tracking step. Visited cells are
# tracked in a parallel boolean grid to prevent revisiting and
# infinite loops.
#
# References:
#   BFS algorithm overview  : https://en.wikipedia.org/wiki/Breadth-first_search
#   Python collections.deque: https://docs.python.org/3/library/collections.html#collections.deque
#   2D matrix manipulation  : https://coderivers.org/blog/python-matrix-manipulation/
# ============================================================

from collections import deque
from typing import Optional


#Constants
WALL      = '#'
FREE      = ' '
START     = 'S'
END       = 'E'
PATH_MARK = '.'   # character used to draw the solution path


#1. Read the maze from a text file
def read_maze(filepath: str) -> list[list[str]]:
    """
    Read a maze from a plain-text file and return it as a 2-D
    list of single characters.

    Each row in the file becomes one row in the grid.
    Walls are '#', open cells are ' ', start is 'S', end is 'E'.

    Args:
        filepath: Path to the maze text file.

    Returns:
        2-D list (list of lists) representing the maze.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError:        If the maze contains no START or END cell.
    """
    maze = []
    with open(filepath, 'r') as f:
        for line in f:
            row = list(line.rstrip('\n'))   # keep spaces; strip newline only
            if row:                          # skip truly empty lines
                maze.append(row)

    # Validate that the maze has both a start and an end
    flat = [cell for row in maze for cell in row]
    if START not in flat:
        raise ValueError(f"Maze has no start cell '{START}'.")
    if END not in flat:
        raise ValueError(f"Maze has no end cell '{END}'.")

    return maze


#2. Check whether moving to (row, col) is safe
def is_safe(maze: list[list[str]], visited: list[list[bool]],
            row: int, col: int) -> bool:
    """
    Return True if (row, col) is a valid, unvisited, non-wall cell.

    Args:
        maze:    The 2-D maze grid.
        visited: Boolean grid tracking cells already explored.
        row:     Row index of the candidate cell.
        col:     Column index of the candidate cell.

    Returns:
        True if the cell can be entered, False otherwise.
    """
    num_rows = len(maze)
    num_cols = len(maze[0]) if maze else 0

    # Boundary check
    if row < 0 or row >= num_rows or col < 0 or col >= num_cols:
        return False

    # Wall check
    if maze[row][col] == WALL:
        return False

    # Already visited check
    if visited[row][col]:
        return False

    return True


#3. Breadth-First Search
def breadth_first_search(maze: list[list[str]]) -> Optional[list[tuple[int, int]]]:
    """
    Find the shortest path from START ('S') to END ('E') using BFS.

    BFS explores the maze level by level (one step at a time),
    guaranteeing that the first time END is reached, the path
    taken is the shortest possible one.

    Args:
        maze: The 2-D maze grid.

    Returns:
        A list of (row, col) tuples representing the shortest path
        from START to END (inclusive), or None if no path exists.
    """
    num_rows = len(maze)
    num_cols = len(maze[0])

    # Locate start and end positions
    start = end = None
    for r in range(num_rows):
        for c in range(num_cols):
            if maze[r][c] == START:
                start = (r, c)
            elif maze[r][c] == END:
                end = (r, c)

    # visited grid – all False initially
    visited = [[False] * num_cols for _ in range(num_rows)]

    # Each queue entry stores the path taken so far as a list of (r, c)
    # Using deque for O(1) popleft: https://docs.python.org/3/library/collections.html
    queue = deque()
    queue.append([start])           # start with a single-element path
    visited[start[0]][start[1]] = True

    # Four possible moves: up, down, left, right
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    while queue:
        path = queue.popleft()      # take the oldest (shallowest) path
        current = path[-1]          # last cell in that path

        # Goal check
        if current == end:
            return path

        row, col = current
        for dr, dc in directions:
            next_row, next_col = row + dr, col + dc
            if is_safe(maze, visited, next_row, next_col):
                visited[next_row][next_col] = True
                new_path = path + [(next_row, next_col)]
                queue.append(new_path)

    return None     # no path found


#4. Print the result to the screen
def print_result(maze: list[list[str]],
                 path: Optional[list[tuple[int, int]]]) -> None:
    """
    Display the maze with the solution path marked, or a failure
    message if no path exists.

    The path cells (excluding START and END) are replaced with '.'

    Args:
        maze: The original 2-D maze grid.
        path: Ordered list of (row, col) cells on the solution path,
              or None if no solution was found.
    """
    if path is None:
        print("No path found from START to END.")
        return

    # Copy the maze so the original is not modified
    display = [row[:] for row in maze]

    # Mark path cells (skip start and end to preserve 'S' / 'E' labels)
    for r, c in path[1:-1]:
        display[r][c] = PATH_MARK

    # Print the marked maze
    print("\n=== Maze Solution ===")
    for row in display:
        print(''.join(row))

    print(f"\nShortest path length: {len(path)} steps")
    print("Path coordinates (row, col):")
    print(" -> ".join(str(cell) for cell in path))


#Main entry point
def main():
    maze_file = "maze.txt"      # change to any maze file path

    print(f"Loading maze from '{maze_file}' ...")
    maze = read_maze(maze_file)

    print(f"Maze size: {len(maze)} rows x {len(maze[0])} columns")
    print("\n=== Original Maze ===")
    for row in maze:
        print(''.join(row))

    print("\nRunning Breadth-First Search ...")
    path = breadth_first_search(maze)

    print_result(maze, path)


if __name__ == "__main__":
    main()
