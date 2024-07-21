def minmax(depth, nodeIndex, maximizingPlayer, values, alpha, beta, path):
    if depth == 3:
        return values[nodeIndex], path + [nodeIndex]
    
    if maximizingPlayer:
        best = float('-inf')
        best_path = []
        for i in range(2):
            val, new_path = minmax(depth + 1, nodeIndex * 2 + i, False, values, alpha, beta, path + [nodeIndex])
            if val > best:
                best = val
                best_path = new_path
            alpha = max(alpha, best)
            if beta <= alpha:
                break
        return best, best_path
    else:
        best = float('inf')
        best_path = []
        for i in range(2):
            val, new_path = minmax(depth + 1, nodeIndex * 2 + i, True, values, alpha, beta, path + [nodeIndex])
            if val < best:
                best = val
                best_path = new_path
            beta = min(beta, best)
            if beta <= alpha:
                break
        return best, best_path

values = [2, 5, 3, 10, 12, 23, 32]
optimal_value, optimal_path = minmax(0, 0, True, values, float('-inf'), float('inf'), [])

print("Optimal value =", optimal_value)
print("Optimal path =", optimal_path)