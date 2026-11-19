
from collections import deque
import time

graph = {
    'A': ['B', 'C'],
    'B': ['A', 'D', 'E'],
    'C': ['A', 'F'],
    'D': ['B'],
    'E': ['B', 'F'],
    'F': ['C', 'E']
}

def bfs(graph, start, goal):
    q = deque([[start]])
    visited = {start}
    count = 0
    t = time.time()

    while q:
        path = q.popleft()
        node = path[-1]
        count += 1

        if node == goal:
            return path, count, time.time() - t

        for n in graph[node]:
            if n not in visited:
                visited.add(n)
                q.append(path + [n])

    return None, count, time.time() - t


def dfs(graph, start, goal):
    stack = [[start]]
    visited = set()
    count = 0
    t = time.time()

    while stack:
        path = stack.pop()
        node = path[-1]

        if node not in visited:
            visited.add(node)
            count += 1

            if node == goal:
                return path, count, time.time() - t

            for n in reversed(graph[node]):
                if n not in visited:
                    stack.append(path + [n])

    return None, count, time.time() - t


start = 'A'
goal = 'F'

for name, algo in [('BFS', bfs), ('DFS', dfs)]:
    path, nodes, tm = algo(graph, start, goal)
    print("\n", name)
    print("Path:", path)
    print("Path Length:", len(path)-1 if path else 0)
    print("Nodes Explored:", nodes)
    print("Execution Time:", tm)
