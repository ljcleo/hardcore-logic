import random
from collections import deque, defaultdict
from tqdm import tqdm
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

class NavigationPuzzleGenerator:
    # 常量定义
    LANDMARK_TYPES = ["store", "bank", "house", "cinema", "garden", "school", "stadium"]
    LANDMARK_TYPES_CN = {
        "store": "商店",
        "bank": "银行",
        "house": "住宅",
        "cinema": "电影院",
        "garden": "花园",
        "school": "学校",
        "stadium": "体育场"
    }
    DISTANCES = [100, 150, 200, 250, 300, 400, 500, 600]
    ALPHABETS = [chr(ord('A') + i) for i in range(26)]  # A-Z

    def __init__(self, difficulty):
        self.difficulty = difficulty
        if difficulty == "medium":
            self.landmark_range = (7, 14)
        elif difficulty == "hard":
            self.landmark_range = (10, 18)
        else:
            raise ValueError("难度必须是 'medium' 或 'hard'")

    def generate_directed_graph(self, landmark_letters, root):
        """生成有向图结构，确保所有节点连通"""
        graph = defaultdict(list)
        remaining = set(landmark_letters) - {root}
        visited = {root}
        queue = deque([root])
        
        # 步骤1: 构建连通图（类似树，但可能有多个父节点）
        while remaining and queue:
            current = queue.popleft()
            # 每个节点连接1-3个其他节点
            num_connections = random.randint(1, 3)
            # 优先连接未访问节点以确保连通性
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
        
        # 步骤2: 为已访问节点添加额外连接（最多3个）
        all_nodes = list(landmark_letters)
        for node in all_nodes:
            # 已经连接的节点
            connected = set(child for child, _ in graph[node])
            # 可连接的候选节点（排除自身和已连接的）
            candidates = [n for n in all_nodes if n != node and n not in connected]
            if not candidates:
                continue
                
            # 随机添加0-3个额外连接
            extra_connections = min(random.randint(0, 3), len(candidates))
            for _ in range(extra_connections):
                child = random.choice(candidates)
                candidates.remove(child)
                distance = random.choice(self.DISTANCES)
                graph[node].append((child, distance))
        
        return graph

    def generate_unsolvable_puzzle(self):
        """生成无解谜题：从起点无法到达目标地标"""
        # 限制地标数量不超过可用字母数量
        num_landmarks = random.randint(*self.landmark_range)
        
        # 分配唯一字母标识
        used_letters = random.sample(self.ALPHABETS, num_landmarks)
        
        # 为每个地标分配类型
        landmark_types = {letter: random.choice(self.LANDMARK_TYPES) for letter in used_letters}
        
        # 选择根节点作为起点
        root_letter = random.choice(used_letters)
        
        # 生成有向图结构
        graph = self.generate_directed_graph(used_letters, root_letter)
        
        # 选择目标类型（确保该类型只出现在无法到达的节点上）
        # 1. 选择1-3个节点作为隔离节点
        non_root_nodes = [node for node in used_letters if node != root_letter]
        if not non_root_nodes:
            return self.generate_unsolvable_puzzle()
            
        num_isolated = min(random.randint(1, 3), len(non_root_nodes))
        isolated_nodes = random.sample(non_root_nodes, num_isolated)
        
        # 2. 断开这些节点与其他节点的所有连接
        for node in list(graph.keys()):
            # 删除指向隔离节点的连接
            graph[node] = [(child, dist) for child, dist in graph[node] if child not in isolated_nodes]
        
        # 删除隔离节点本身的连接
        for node in isolated_nodes:
            if node in graph:
                del graph[node]
        
        # 3. 为隔离节点分配相同的特殊类型
        # 选择在非隔离节点中不存在的类型
        non_isolated_types = set(landmark_types[node] for node in used_letters if node not in isolated_nodes)
        available_types = set(self.LANDMARK_TYPES) - non_isolated_types
        if not available_types:
            # 如果没有可用类型，重新生成谜题
            return self.generate_unsolvable_puzzle()
        
        isolated_type = random.choice(list(available_types))
        
        # 4. 将隔离节点的类型设置为特殊类型
        for node in isolated_nodes:
            landmark_types[node] = isolated_type
        
        # 设置目标类型
        target_type = isolated_type
        
        # 生成英文故事文本
        story_parts = []
        start_type_en = landmark_types[root_letter]
        story_parts.append(f"There is a city with various landmarks.")
        story_parts.append(f"The start point is {start_type_en} {root_letter}.")
        
        # 列出所有地标及其类型
        landmarks_desc_en = [f"{landmark_types[letter]} {letter}" for letter in used_letters]
        story_parts.append(f"The landmarks include: {', '.join(landmarks_desc_en)}.")
        
        # 收集所有边的信息
        edges_en = []
        for parent in graph:
            for child, dist in graph[parent]:
                parent_type_en = landmark_types[parent]
                child_type_en = landmark_types[child]
                edges_en.append(f"There is a road which is {dist} meters long from {parent_type_en} {parent} to {child_type_en} {child}.")
        
        # 随机排序边的描述
        random.shuffle(edges_en)
        story_parts.extend(edges_en)
        
        # 添加所有道路都是单向的说明
        story_parts.append("All roads are one-way.")
        
        # 添加查询
        story_parts.append("\n## Query:")
        story_parts.append(f"From the start point, how to reach the nearest {target_type} in the shortest way?")
        
        puzzle = "\n".join(story_parts)
        
        return puzzle

    def generate_puzzles(self, num_puzzles=600):
        puzzles = []
        # 使用tqdm包装循环，显示进度条
        with tqdm(total=num_puzzles, desc=f"生成{self.difficulty}难度无解谜题") as pbar:
            for i in range(1, num_puzzles + 1):
                puzzle_id = f"unsolvable--{self.difficulty}-{i}"
                puzzle_content = self.generate_unsolvable_puzzle()
                puzzles.append({
                    "id": puzzle_id,
                    "puzzle": puzzle_content,
                    "solution": "null"
                })
                pbar.update(1)
        return puzzles

    @staticmethod
    def save_to_parquet(puzzles, filename):
        # 创建DataFrame
        df = pd.DataFrame(puzzles)
        # 转换为Parquet格式并保存
        table = pa.Table.from_pandas(df)
        pq.write_table(table, filename)


if __name__ == "__main__":
    # 生成中等难度谜题
    medium_generator = NavigationPuzzleGenerator("medium")
    medium_puzzles = medium_generator.generate_puzzles(5)
    medium_generator.save_to_parquet(medium_puzzles, r"navigation_puzzles_easy.parquet")
    
    # 生成高难度谜题
    hard_generator = NavigationPuzzleGenerator("hard")
    hard_puzzles = hard_generator.generate_puzzles(5)
    hard_generator.save_to_parquet(hard_puzzles, r"navigation_puzzles.parquet")
    