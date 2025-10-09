import random
from collections import deque, defaultdict
import pandas as pd
from tqdm import tqdm
import pyarrow as pa
import pyarrow.parquet as pq


class NavigationPuzzleGenerator:
    def __init__(self, difficulty):
        self.difficulty = difficulty
        if difficulty == 'easy':
            self.landmark_types = ["store", "bank", "house", "cinema", "garden", "school", "stadium"]
            self.num_landmarks_range = (10, 14)
            self.total_puzzles = 50
        elif difficulty == 'hard':
            self.landmark_types = ["store", "bank", "house", "cinema", "garden", "school", "square", "stadium"]
            self.num_landmarks_range = (10, 18)
            self.total_puzzles = 50
        else:
            raise ValueError("难度必须是 'easy' 或 'hard'")
            
        self.landmark_types_cn = {
            "store": "商店",
            "bank": "银行",
            "house": "住宅",
            "cinema": "电影院",
            "garden": "花园",
            "school": "学校",
            "square": "广场",
            "stadium": "体育场"
        }
        self.distances = [100, 150, 200, 250, 300, 400, 500, 600]
        self.alphabets = [chr(ord('A') + i) for i in range(26)]  

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
                distance = random.choice(self.distances)
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
                distance = random.choice(self.distances)
                graph[node].append((child, distance))
        
        return graph

    def find_shortest_paths(self, graph, start, targets):
        """使用Dijkstra算法找到从start到所有target的最短路径和距离"""
        distances = {node: float('inf') for node in graph}
        distances[start] = 0
        previous = {node: None for node in graph}
        priority_queue = [(0, start)]
        
        while priority_queue:
            current_dist, current_node = priority_queue.pop(0)
            
            if current_dist > distances[current_node]:
                continue
                
            for neighbor, dist in graph[current_node]:
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
            else:  # 正常完成循环
                path.reverse()
                paths[target] = (path, distances[target])
        
        return paths

    def is_shortest_path_unique(self, graph, start, target_type, landmark_types):
        """验证到目标类型的最短路径是否唯一，并返回路径和距离"""
        targets = [node for node, typ in landmark_types.items() 
                  if typ == target_type and node != start]
        if not targets:
            return False, None, None
        paths = self.find_shortest_paths(graph, start, targets)
        if not paths:
            return False, None, None
        min_dist = min(dist for _, dist in paths.values()) 
        shortest_paths = []
        for target, (path, dist) in paths.items():
            if dist == min_dist:
                path_str = ''.join(path) + target
                shortest_paths.append((target, path, path_str, dist))
        unique_paths = set(path_str for _, _, path_str, _ in shortest_paths)
        if len(unique_paths) == 1:
            return True, shortest_paths[0][1], shortest_paths[0][3]
        else:
            return False, None, None

    def generate_puzzle(self):
        """生成单个谜题"""
        max_landmarks = 26
        num_landmarks = random.randint(*self.num_landmarks_range)
        num_landmarks = min(num_landmarks, max_landmarks)
        used_letters = random.sample(self.alphabets, num_landmarks)
        landmark_types = {letter: random.choice(self.landmark_types) for letter in used_letters}
        root_letter = random.choice(used_letters)
        graph = self.generate_directed_graph(used_letters, root_letter)
        target_type = None
        shortest_path = []
        shortest_distance = 0
        max_attempts = 100
        attempts = 0
        
        while attempts < max_attempts:
            candidate_type = random.choice(self.landmark_types)
            is_unique, path, distance = self.is_shortest_path_unique(graph, root_letter, candidate_type, landmark_types)
            
            if is_unique and path:
                target_type = candidate_type
                shortest_path = path
                shortest_distance = distance
                break
                
            attempts += 1
        
        if not shortest_path:
            return self.generate_puzzle()
        story_parts = []
        start_type = landmark_types[root_letter]
        story_parts.append(f"There is a city with various landmarks.")
        story_parts.append(f"The start point is {start_type} {root_letter}.")
        edges = []
        for parent in graph:
            for child, dist in graph[parent]:
                parent_type = landmark_types[parent]
                child_type = landmark_types[child]
                edges.append(f"There is a road which is {dist} meters long from {parent_type} {parent} to {child_type} {child}.")
        random.shuffle(edges)
        story_parts.extend(edges)
        story_parts.append("All roads are one-way.")
        if landmark_types[root_letter] == target_type:
            query = f"From the start point, how to reach the nearest {target_type} other than the start point in the shortest way?"
        else:
            query = f"From the start point, how to reach the nearest {target_type} in the shortest way?"
        
        story_parts.append("\n## Query:")
        story_parts.append(query)

        puzzle_text = "\n".join(story_parts)

        return {
            "puzzle": puzzle_text,
            "solution": shortest_distance,
            "path": "".join(shortest_path)
        }

    def generate_puzzles(self):
        """生成指定数量的谜题"""
        puzzles = []
        with tqdm(total=self.total_puzzles, desc=f"生成{self.difficulty}难度谜题") as pbar:
            count = 1
            while len(puzzles) < self.total_puzzles:
                puzzle = self.generate_puzzle()
                if len(puzzle["path"]) > 3:
                    puzzle_id = f"gen-01-graph--{self.difficulty}-{count}"
                    puzzles.append({
                        "id": puzzle_id,
                        "puzzle": puzzle["puzzle"],
                        "solution": puzzle["solution"]
                    })
                    count += 1
                    pbar.update(1)
        return puzzles

    def save_to_parquet(self, puzzles, filename):
        """将谜题保存为parquet文件"""
        df = pd.DataFrame(puzzles)
        table = pa.Table.from_pandas(df)
        pq.write_table(table, filename)


if __name__ == "__main__":
    easy_generator = NavigationPuzzleGenerator('easy')
    easy_puzzles = easy_generator.generate_puzzles()
    easy_generator.save_to_parquet(easy_puzzles, 'navigation_puzzles_easy.parquet')
    hard_generator = NavigationPuzzleGenerator('hard')
    hard_puzzles = hard_generator.generate_puzzles()
    hard_generator.save_to_parquet(hard_puzzles, 'navigation_puzzles_hard.parquet')
    