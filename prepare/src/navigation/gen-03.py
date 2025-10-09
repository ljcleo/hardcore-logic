import random
import pandas as pd
from collections import deque, defaultdict
from tqdm import tqdm


class NavigationPuzzleGenerator:
    LANDMARK_TYPES = ["store", "bank", "house", "cinema", "garden", "school", "stadium"]
    LANDMARK_TYPES_CN = {
        "store": "商店", "bank": "银行", "house": "住宅",
        "cinema": "电影院", "garden": "花园", "school": "学校", "stadium": "体育场"
    }
    DISTANCES = [100, 150, 200, 250, 300, 400, 500, 600]
    ALPHABETS = [chr(ord('A') + i) for i in range(26)] 
    def __init__(self, difficulty):
        if difficulty not in ['easy', 'hard']:
            raise ValueError("难度必须为 'easy' 或 'hard'")
        
        self.difficulty = difficulty
        if difficulty == 'easy':
            self.landmark_range = (7, 14)
            self.min_path_length = 3
        else:
            self.landmark_range = (12, 18)
            self.min_path_length = 4

    def generate_directed_graph(self, landmark_letters, root):
        """生成有向图结构，确保所有节点连通"""
        graph = defaultdict(list)
        remaining = set(landmark_letters) - {root}
        visited = {root}
        queue = deque([root])
 
        while remaining and queue:
            current = queue.popleft()
            num_connections = random.randint(1, 3)
            unvisited = list(remaining)
            random.shuffle(unvisited)
            connections = unvisited[:num_connections]
            
            for child in connections:
                if child not in visited:
                    visited.add(child)
                    remaining.discard(child)
                    queue.append(child)
                distance = random.choice(self.DISTANCES)
                graph[current].append((child, distance))
        
        all_nodes = list(landmark_letters)
        for node in all_nodes:
            connected = set(child for child, _ in graph[node])
            candidates = [n for n in all_nodes if n != node and n not in connected]
            if not candidates:
                continue
                
            extra_connections = min(random.randint(0, 3), len(candidates))
            for _ in range(extra_connections):
                child = random.choice(candidates)
                candidates.remove(child)
                distance = random.choice(self.DISTANCES)
                graph[node].append((child, distance))
        
        return graph

    def find_shortest_paths(self, graph, start, targets, landmarks):
        """Dijkstra算法：找start到所有target的最短路径（用于约束路径验证）"""
        distances = {node: float('inf') for node in landmarks}
        distances[start] = 0
        previous = {node: None for node in landmarks}
        priority_queue = [(0, start)]
        
        while priority_queue:
            current_dist, current_node = priority_queue.pop(0)
            
          
            if current_dist > distances[current_node]:
                continue
                
      
            for neighbor, dist in graph.get(current_node, []):
                if neighbor not in landmarks:
                    continue
                new_dist = current_dist + dist
                if new_dist < distances[neighbor]:
                    distances[neighbor] = new_dist
                    previous[neighbor] = current_node
                    priority_queue.append((new_dist, neighbor))
                    priority_queue.sort()  
        
        paths = {}
        for target in targets:
            if distances[target] == float('inf') or previous[target] is None:
                continue
            
            path = []
            current = target
            while current != start:
                path.append(current)
                current = previous[current]
                if current is None:
                    break
            else:
                path.reverse()
                paths[target] = (path, distances[target])
        
        return paths

    def is_constrained_shortest_path_unique(self, graph, start, target_type, pre_condition_type, landmark_types):
        """验证“必须经过前置节点”的约束路径是否唯一且有效"""
   
        targets = [node for node, typ in landmark_types.items() 
                  if typ == target_type and node != start]
        pre_condition_nodes = [node for node, typ in landmark_types.items() 
                              if typ == pre_condition_type]
        
        if not targets or not pre_condition_nodes:
            return False, None, None
        
        paths = self.find_shortest_paths(graph, start, targets, landmark_types.keys())
        if not paths:
            return False, None, None
   
        valid_paths = []
        for target, (path, dist) in paths.items():
            found_pre = any(landmark_types[node] == pre_condition_type for node in path[:-1])  # 排除目标节点本身
            if found_pre:
                valid_paths.append((target, path, dist))
        
        if not valid_paths:
            return False, None, None
        
        # 检查最短路径是否唯一
        min_dist = min(dist for _, _, dist in valid_paths)
        shortest_paths = [(t, p) for t, p, d in valid_paths if d == min_dist]
        if len(shortest_paths) == 1:
            return True, shortest_paths[0][1], min_dist
        return False, None, None

    def dijkstra(self, graph, start, landmarks, end=None):
        """基础Dijkstra算法（给find_direct_shortest_path调用）"""
        distances = {node: float('inf') for node in landmarks}
        distances[start] = 0
        previous = {node: None for node in landmarks}
        priority_queue = [(0, start)]
        
        while priority_queue:
            priority_queue.sort() 
            current_dist, current_node = priority_queue.pop(0)
            
            if current_dist > distances[current_node]:
                continue
            if end and current_node == end:
                break  
            
          
            for neighbor, dist in graph.get(current_node, []):
                if neighbor not in landmarks:
                    continue
                new_dist = current_dist + dist
                if new_dist < distances[neighbor]:
                    distances[neighbor] = new_dist
                    previous[neighbor] = current_node
                    priority_queue.append((new_dist, neighbor))
        
       
        def reconstruct_path(end_node):
            path = []
            current = end_node
            while current is not None:
                path.append(current)
                current = previous[current]
            return path[::-1]  
        
        if end:
            return reconstruct_path(end), distances[end]
        return previous, distances

    def find_direct_shortest_path(self, graph, start, target_type, landmark_types):
        """新增：无前置条件的直接最短路径（找最近的目标类型节点）"""

        target_nodes = [node for node, typ in landmark_types.items() 
                       if typ == target_type and node != start]
        if not target_nodes:
            return None, float('inf')
        
        min_direct_dist = float('inf')
        best_direct_path = None
        landmarks = landmark_types.keys()  
    
        for target_node in target_nodes:
            path, dist = self.dijkstra(graph, start, landmarks, target_node)
            if dist < min_direct_dist:
                min_direct_dist = dist
                best_direct_path = path
        

        direct_path_str = ''.join(best_direct_path[1:]) if (best_direct_path and len(best_direct_path) > 1) else None
        return direct_path_str, min_direct_dist

    def generate_puzzle(self):
        """生成单个谜题：确保约束路径解 ≠ 直接路径解"""
        max_attempts = 200  
        attempts = 0
        
        while attempts < max_attempts:

            num_landmarks = random.randint(*self.landmark_range)
            used_letters = random.sample(self.ALPHABETS, num_landmarks)
            landmark_types = {letter: random.choice(self.LANDMARK_TYPES) for letter in used_letters}
            root_letter = random.choice(used_letters)
            graph = self.generate_directed_graph(used_letters, root_letter)

            constrained_path = None
            constrained_solution = None  
            target_type = None
            pre_condition_type = None

            candidate_targets = random.sample(self.LANDMARK_TYPES, 3)
            for ct in candidate_targets:
                candidate_pres = [t for t in self.LANDMARK_TYPES if t != ct]
                if not candidate_pres:
                    continue
                cp = random.choice(candidate_pres)
                is_unique, path, dist = self.is_constrained_shortest_path_unique(
                    graph, root_letter, ct, cp, landmark_types
                )
                if is_unique and path and len(path) >= self.min_path_length:
                    constrained_path = path
                    constrained_solution = dist
                    target_type = ct
                    pre_condition_type = cp
                    break
            if not constrained_path:  
                attempts += 1
                continue
            _, direct_solution = self.find_direct_shortest_path(
                graph, root_letter, target_type, landmark_types
            )
            if direct_solution == float('inf') or direct_solution == constrained_solution:
                attempts += 1
                continue
            story_parts = [
                "There is a city with various landmarks.",
                f"The start point is {landmark_types[root_letter]} {root_letter}."
            ]
            edges = []
            for parent in graph:
                for child, dist in graph[parent]:
                    edges.append(
                        f"There is a road which is {dist} meters long from {landmark_types[parent]} {parent} to {landmark_types[child]} {child}."
                    )
            random.shuffle(edges)
            story_parts.extend(edges)
            story_parts.append("All roads are one-way.") 
            query = (f"From the start point, how to reach the nearest {target_type}? "
                     f"And you must visit {pre_condition_type} before visiting {target_type}.")
            return {
                "story": '\n'.join(story_parts),
                "query": query,
                "constrained_path_str": ''.join(constrained_path),  
                "constrained_solution": constrained_solution,      
                "direct_solution": direct_solution                
            }

        return self.generate_puzzle()

    def generate_puzzles(self, num=600):
        """生成指定数量的谜题，仅保存“两个答案不同”的谜题"""
        puzzles = []
        with tqdm(total=num, desc=f"生成{self.difficulty}难度谜题") as pbar:
            count = 1
            while len(puzzles) < num:
                puzzle = self.generate_puzzle()
                if (len(puzzle["constrained_path_str"]) >= self.min_path_length and
                    puzzle["constrained_solution"] != puzzle["direct_solution"]):
                    puzzle_id = f"gen-03-graph--{self.difficulty}-{count:03d}"
                    full_puzzle = f"{puzzle['story']}\n\n## Query:\n{puzzle['query']}"
                    
                    puzzles.append({
                        "id": puzzle_id,
                        "puzzle": full_puzzle,
                        "constrained_solution": puzzle["constrained_solution"],  
                        "direct_solution": puzzle["direct_solution"],          
                        "constrained_path_str": puzzle["constrained_path_str"]
                    })
                    count += 1
                    pbar.update(1)
        return puzzles

    @staticmethod
    def save_to_parquet(puzzles, filename):
        """保存谜题到Parquet文件（高效存储）"""
        df = pd.DataFrame(puzzles)
        df.to_parquet(filename, engine='pyarrow', index=False)


if __name__ == "__main__":
    easy_gen = NavigationPuzzleGenerator(difficulty='easy')
    easy_puzzles = easy_gen.generate_puzzles(5)
    easy_gen.save_to_parquet(easy_puzzles, r"navigation_puzzles.parquet")
    hard_gen = NavigationPuzzleGenerator(difficulty='hard')
    hard_puzzles = hard_gen.generate_puzzles(5)
    hard_gen.save_to_parquet(hard_puzzles, r"navigation_puzzles.parquet")
