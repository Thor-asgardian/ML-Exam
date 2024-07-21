
import heapq

def best_first_search(graph, start, goal, heuristic):
    priority_queue = [(heuristic[start], start)]
    visited = set()
    parent = {start: None}

    while priority_queue:
        _, current_node = heapq.heappop(priority_queue)

        if current_node == goal:
            break

        if current_node not in visited:
            visited.add(current_node)
            for neighbor in graph[current_node]:
                if neighbor not in visited:
                    heapq.heappush(priority_queue, (heuristic[neighbor], neighbor))
                    parent[neighbor] = current_node

    path = []
    node = goal
    while node:
        path.append(node)
        node = parent[node]
    path.reverse()

    total_heuristic = sum(heuristic[n] for n in path)
    return path, total_heuristic

# Graph definition
graph = {
    'A': ['B', 'C'],
    'B': ['D', 'E'],
    'C': ['F', 'G'],
    'D': [],
    'E': [],
    'F': [],
    'G': []
}

# Heuristic values
heuristic = {
    'A': 6,
    'B': 4,
    'C': 4,
    'D': 0,
    'E': 2,
    'F': 3,
    'G': 1
}

# Starting and goal nodes
start = 'A'
goal = 'D'

# Perform the search and print the result
path, total_heuristic = best_first_search(graph, start, goal, heuristic)
print("Best First Search Path:", path)
print("Total Heuristic Value:", total_heuristic)