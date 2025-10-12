import random
import os
import pandas as pd

class SkyscraperUnsolvableGenerator:
    def __init__(self, output_path):
        self.output_path = output_path
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

    def generate_grid(self, size):
        grid = [[0 for _ in range(size)] for _ in range(size)]
        first_row = list(range(1, size + 1))
        random.shuffle(first_row)
        grid[0] = first_row
        for row in range(1, size):
            used = set()
            for col in range(size):
                possible = []
                for num in range(1, size + 1):
                    if num in used:
                        continue
                    if any(grid[r][col] == num for r in range(row)):
                        continue
                    possible.append(num)
                if not possible:
                    return self.generate_grid(size)
                chosen = random.choice(possible)
                grid[row][col] = chosen
                used.add(chosen)
        return grid

    def calculate_visibility(self, grid, size):
        vis = []
        top = []
        for c in range(size):
            max_h, cnt = 0, 0
            for r in range(size):
                if grid[r][c] > max_h:
                    max_h = grid[r][c]
                    cnt += 1
            top.append(cnt)
        vis.append(top)
        bottom = []
        for c in range(size):
            max_h, cnt = 0, 0
            for r in range(size - 1, -1, -1):
                if grid[r][c] > max_h:
                    max_h = grid[r][c]
                    cnt += 1
            bottom.append(cnt)
        vis.append(bottom)
        left = []
        for r in range(size):
            max_h, cnt = 0, 0
            for c in range(size):
                if grid[r][c] > max_h:
                    max_h = grid[r][c]
                    cnt += 1
            left.append(cnt)
        vis.append(left)
        right = []
        for r in range(size):
            max_h, cnt = 0, 0
            for c in range(size - 1, -1, -1):
                if grid[r][c] > max_h:
                    max_h = grid[r][c]
                    cnt += 1
            right.append(cnt)
        vis.append(right)
        return vis

    def make_unsolvable(self, vis, size):
        modified = [v.copy() for v in vis]
        method = random.randint(1, 4)
        if method == 1:
            col = random.randint(0, size - 1)
            modified[0][col] = size
            modified[1][col] = random.randint(2, size - 1) if size > 2 else 2
        elif method == 2 and size >= 2:
            if random.choice([True, False]):
                row = random.randint(0, size - 1)
                modified[2][row] = 1
                modified[3][row] = 1
            else:
                col = random.randint(0, size - 1)
                modified[0][col] = 1
                modified[1][col] = 1
        elif method == 3:
            d, p = random.randint(0, 3), random.randint(0, size - 1)
            modified[d][p] = size + random.randint(1, 3)

        else:
            if random.choice([True, False]):
                row1, row2 = random.sample(range(size), 2)
                modified[2][row1] = 1
                modified[2][row2] = 1
            else:
                col1, col2 = random.sample(range(size), 2)
                modified[1][col1] = 1
                modified[1][col2] = 1

        return modified

    def visibility_to_puzzle(self, vis, size):
        top, bottom, left, right = vis
        lines = []
        top_str = " ".join(f"{x:2}" for x in top)
        lines.append(f"-1|{top_str}|-1")

        for i in range(size):
            row_str = " ".join(" ." for _ in range(size))
            lines.append(f"{left[i]:2}|{row_str}|{right[i]:2}")

        bottom_str = " ".join(f"{x:2}" for x in bottom)
        lines.append(f"-1|{bottom_str}|-1")
        return "\n".join(lines)

    def generate_unsolvable_puzzles(self, sizes=[6, 7, 8], per_size=50):
        records = []
        count_global = 0
        for size in sizes:
            for i in range(1, per_size + 1):
                count_global += 1
                grid = self.generate_grid(size)
                vis = self.calculate_visibility(grid, size)
                unsolvable_vis = self.make_unsolvable(vis, size)
                puzzle_str = self.visibility_to_puzzle(unsolvable_vis, size)
                count_str = f"{i:02d}" 
                puzzle_id = f"unsolvable-1--{size}x{size}-{count_str}"
                records.append({
                    "id": puzzle_id,
                    "puzzle": puzzle_str,
                    "solution": "null",
                    "mode": "count"
                })

        df = pd.DataFrame(records)
        df.to_parquet(self.output_path, index=False)


if __name__ == "__main__":
    output_path = r"unsolvable_skyscraper.parquet"
    generator = SkyscraperUnsolvableGenerator(output_path)
    generator.generate_unsolvable_puzzles(sizes=[6], per_size=10)


