import heapq

def astar(graph, start, goal, heuristic, cost):
    priority_queue = []
    heapq.heappush(priority_queue, (0 + heuristic[start], start))
    visited = set()
    g_cost = {start: 0}
    parent = {start: None}

    while priority_queue:
        current_cost, current_node = heapq.heappop(priority_queue)
        if current_node in visited:
            continue
        visited.add(current_node)
        if current_node == goal:
            break
        for neighbor in graph[current_node]:
            new_cost = g_cost[current_node] + cost[(current_node, neighbor)]
            if neighbor not in visited or new_cost < g_cost[neighbor]:
                g_cost[neighbor] = new_cost
                f_cost = new_cost + heuristic[neighbor]
                heapq.heappush(priority_queue, (f_cost, neighbor))
                parent[neighbor] = current_node

    path = []
    node = goal
    while node is not None:
        path.append(node)
        node = parent[node]
    
    path.reverse()
    return path

# Example Usage:
graph = {
    'A': ['B', 'C'],
    'B': ['D', 'E'],
    'C': ['F'],
    'D': [],
    'E': [],
    'F': []
}
heuristic = {
    'A': 10,
    'B': 9,
    'C': 8,
    'D': 0,
    'E': 7,
    'F': 6
}
cost = {
    ('A', 'B'): 1,
    ('A', 'C'): 4,
    ('B', 'D'): 2,
    ('B', 'E'): 5,
    ('C', 'F'): 3
}

start = 'A'
goal = 'D'

path = astar(graph, start, goal, heuristic, cost)
print("Optimal path =", path)


"""
If asked for optimal cost, it will be used
import heapq

def astar(graph, start, goal, heuristic, cost):
    priority_queue = []
    heapq.heappush(priority_queue, (0 + heuristic[start], start))
    visited = set()
    g_cost = {start: 0}
    parent = {start: None}

    while priority_queue:
        current_cost, current_node = heapq.heappop(priority_queue)
        if current_node in visited:
            continue
        visited.add(current_node)
        if current_node == goal:
            break
        for neighbor in graph[current_node]:
            new_cost = g_cost[current_node] + cost[(current_node, neighbor)]
            if neighbor not in visited or new_cost < g_cost[neighbor]:
                g_cost[neighbor] = new_cost
                f_cost = new_cost + heuristic[neighbor]
                heapq.heappush(priority_queue, (f_cost, neighbor))
                parent[neighbor] = current_node

    path = []
    node = goal
    while node is not None:
        path.append(node)
        node = parent[node]
    
    path.reverse()
    total_cost = g_cost[goal]
    return path, total_cost

# Example Usage:
graph = {
    'A': ['B', 'C'],
    'B': ['D', 'E'],
    'C': ['F'],
    'D': [],
    'E': [],
    'F': []
}
heuristic = {
    'A': 10,
    'B': 9,
    'C': 8,
    'D': 0,
    'E': 7,
    'F': 6
}
cost = {
    ('A', 'B'): 1,
    ('A', 'C'): 4,
    ('B', 'D'): 2,
    ('B', 'E'): 5,
    ('C', 'F'): 3
}

start = 'A'
goal = 'D'

path, total_cost = astar(graph, start, goal, heuristic, cost)
print("Optimal path =", path)
print("Total cost =", total_cost)
