import random
import re
import heapq
import pandas as pd
from collections import deque, defaultdict
from tqdm import tqdm

class NavigationPuzzleGenerator:
    LANDMARK_TYPES = ["store", "bank", "house", "cinema", "garden", "school", "stadium"]
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
        targets = [node for node, typ in landmark_types.items() if typ == target_type and node != start]
        pre_condition_nodes = [node for node, typ in landmark_types.items() if typ == pre_condition_type]
        if not targets or not pre_condition_nodes:
            return False, None, None

        paths = self.find_shortest_paths(graph, start, targets, landmark_types.keys())
        if not paths:
            return False, None, None

        valid_paths = []
        for target, (path, dist) in paths.items():
            found_pre = any(landmark_types[node] == pre_condition_type for node in path[:-1])
            if found_pre:
                valid_paths.append((target, path, dist))
        if not valid_paths:
            return False, None, None

        min_dist = min(dist for _, _, dist in valid_paths)
        shortest_paths = [(t, p) for t, p, d in valid_paths if d == min_dist]
        if len(shortest_paths) == 1:
            return True, shortest_paths[0][1], min_dist
        return False, None, None

    def dijkstra(self, graph, start, landmarks, end=None):
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
        target_nodes = [node for node, typ in landmark_types.items() if typ == target_type and node != start]
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
            query = (f"From the start point, how to reach a {target_type} other than the start point, "
                     f"and then a {pre_condition_type} in the shortest way?")
            return {
                "story": '\n'.join(story_parts),
                "query": query,
                "constrained_path_str": ''.join(constrained_path),
                "constrained_solution": constrained_solution,
                "direct_solution": direct_solution,
                "puzzle": f"{'\n'.join(story_parts)}\n\n## Query:\n{query}"
            }
        return self.generate_puzzle()

    @staticmethod
    def node_type(name):
        return name.split()[0].lower() if name else None

    @staticmethod
    def parse_puzzle(text):
        start_m = re.search(r"The start point is ([\w ]+)\.", text)
        start = start_m.group(1).strip() if start_m else None

        road_re = re.compile(
            r"There is a road which is (\d+) meters long from ([\w ]+?) to ([\w ]+?)\."
        )
        edges = []
        nodes = set()
        for m in road_re.finditer(text):
            w = int(m.group(1))
            src = m.group(2).strip()
            dst = m.group(3).strip()
            edges.append((src, dst, w))
            nodes.add(src)
            nodes.add(dst)

        query_m = re.search(r"From the start point, how to reach a (\w+).*?and then a (\w+)", text, re.DOTALL)
        if query_m:
            first_type = query_m.group(1).lower()
            second_type = query_m.group(2).lower()
        else:
            first_type = second_type = None
        return start, edges, nodes, first_type, second_type

    @staticmethod
    def build_graphs(edges, nodes):
        g = defaultdict(list)
        rg = defaultdict(list)
        for src, dst, w in edges:
            g[src].append((dst, w))
            rg[dst].append((src, w))
        for n in nodes:
            g.setdefault(n, [])
            rg.setdefault(n, [])
        return g, rg

    @staticmethod
    def dijkstra_static(source, graph):
        dist = {}
        prev = {}
        pq = [(0, source)]
        dist[source] = 0
        while pq:
            d, u = heapq.heappop(pq)
            if d != dist.get(u, float('inf')):
                continue
            for v, w in graph[u]:
                nd = d + w
                if nd < dist.get(v, float('inf')):
                    dist[v] = nd
                    prev[v] = u
                    heapq.heappush(pq, (nd, v))
        return dist, prev

    @staticmethod
    def dijkstra_multi_source(sources, graph):
        dist = {}
        prev = {}
        pq = []
        for s in sources:
            dist[s] = 0
            prev[s] = None
            heapq.heappush(pq, (0, s))
        while pq:
            d, u = heapq.heappop(pq)
            if d != dist.get(u, float('inf')):
                continue
            for v, w in graph[u]:
                nd = d + w
                if nd < dist.get(v, float('inf')):
                    dist[v] = nd
                    prev[v] = u
                    heapq.heappush(pq, (nd, v))
        return dist, prev

    @staticmethod
    def reconstruct_from_prev(prev, start, end):
        if end == start:
            return [start]
        path = []
        cur = end
        while cur != start:
            path.append(cur)
            if cur not in prev:
                return []
            cur = prev[cur]
        path.append(start)
        path.reverse()
        return path

    @staticmethod
    def reconstruct_to_a_source(prev_rev, node, sources_set):
        path = [node]
        cur = node
        while cur not in sources_set:
            if cur not in prev_rev:
                return []
            cur = prev_rev[cur]
            path.append(cur)
        return path

    def solve_one(self, puzzle_text):
        start, edges, nodes, first_type, second_type = self.parse_puzzle(puzzle_text)
        if not start or not first_type or not second_type:
            return None
        graph, rev_graph = self.build_graphs(edges, nodes)
        first_targets = [n for n in nodes if self.node_type(n) == first_type and n != start]
        second_targets = [n for n in nodes if self.node_type(n) == second_type]
        if not first_targets or not second_targets:
            return None

        dist_start, prev_start = self.dijkstra_static(start, graph)
        dist_to_second, prev_rev = self.dijkstra_multi_source(second_targets, rev_graph)

        best_total = float('inf')
        best_path = None
        for u in first_targets:
            du = dist_start.get(u, float('inf'))
            d2 = dist_to_second.get(u, float('inf'))
            if du == float('inf') or d2 == float('inf'):
                continue
            total = du + d2
            if total < best_total:
                p1 = self.reconstruct_from_prev(prev_start, start, u)
                p2 = self.reconstruct_to_a_source(prev_rev, u, set(second_targets))
                if not p1 or not p2:
                    continue
                best_total = total
                best_path = p1 + p2[1:]
        if not best_path:
            return None
        return int(best_total)

    def generate_puzzles(self, num=50):
        puzzles = []
        with tqdm(total=num, desc=f"Generate {self.difficulty} difficulty puzzles") as pbar:
            count = 1
            while len(puzzles) < num:
                puzzle = self.generate_puzzle()
                if (len(puzzle["constrained_path_str"]) >= self.min_path_length and
                    puzzle["constrained_solution"] != puzzle["direct_solution"]):

                    solution = self.solve_one(puzzle["puzzle"])
                    if solution is None:
                        continue
                    if solution != puzzle["direct_solution"]:
                        puzzle_id = f"gen-03-graph--{self.difficulty}-{count:03d}"
                        puzzles.append({
                            "id": puzzle_id,
                            "puzzle": puzzle["puzzle"],
                            "solution": solution
                        })
                        count += 1
                        pbar.update(1)
        return puzzles

    @staticmethod
    def save_to_parquet(puzzles, filename):
        df = pd.DataFrame(puzzles)
        df.to_parquet(filename, engine="pyarrow", index=False)

if __name__ == "__main__":
    easy_gen = NavigationPuzzleGenerator("easy")
    easy_puzzles = easy_gen.generate_puzzles(50)
    easy_gen.save_to_parquet(easy_puzzles, r"navigation_puzzles.parquet")


