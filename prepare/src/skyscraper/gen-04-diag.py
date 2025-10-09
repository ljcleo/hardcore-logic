import random
import os
import pandas as pd

class SkyscraperPuzzleGenerator:
    def __init__(self):
        self.size = 0
        self.grid = []
        self.direction_clues = []  
        self.diagonal_clues = [] 

    def generate_grid(self, size):
        self.size = size
        grid = [[0 for _ in range(size)] for _ in range(size)]
        first_row = list(range(1, size + 1))
        random.shuffle(first_row)
        grid[0] = first_row
        for row in range(1, size):
            used_in_row = set()
            for col in range(size):
                possible = []
                for num in range(1, size + 1):
                    if num in used_in_row:
                        continue
                    in_col = any(grid[r][col] == num for r in range(row))
                    if not in_col:
                        possible.append(num)
                if not possible:
                    return self.generate_grid(size)
                chosen = random.choice(possible)
                grid[row][col] = chosen
                used_in_row.add(chosen)
        
        self.grid = grid
        return grid

    def calculate_direction_visibility(self):
        if not self.grid:
            raise ValueError("Please form a grid.")
        size = self.size
        visibility = []
        top = []
        for col in range(size):
            max_height, count = 0, 0
            for row in range(size):
                if self.grid[row][col] > max_height:
                    max_height = self.grid[row][col]
                    count += 1
            top.append(count)
        visibility.append(top)
        bottom = []
        for col in range(size):
            max_height, count = 0, 0
            for row in range(size - 1, -1, -1):
                if self.grid[row][col] > max_height:
                    max_height = self.grid[row][col]
                    count += 1
            bottom.append(count)
        visibility.append(bottom)
        left = []
        for row in range(size):
            max_height, count = 0, 0
            for col in range(size):
                if self.grid[row][col] > max_height:
                    max_height = self.grid[row][col]
                    count += 1
            left.append(count)
        visibility.append(left)
        right = []
        for row in range(size):
            max_height, count = 0, 0
            for col in range(size - 1, -1, -1):
                if self.grid[row][col] > max_height:
                    max_height = self.grid[row][col]
                    count += 1
            right.append(count)
        visibility.append(right)
        
        self.direction_clues = visibility
        return visibility

    def calculate_diagonal_visibility(self):
        if not self.grid:
            raise ValueError("Please form a grid.")
        size = self.size
        diagonal_clues = []
        max_h, count = 0, 0
        for i in range(size):
            if self.grid[i][i] > max_h:
                max_h = self.grid[i][i]
                count += 1
        diagonal_clues.append(count)
        max_h, count = 0, 0
        for i in range(size - 1, -1, -1):
            if self.grid[i][i] > max_h:
                max_h = self.grid[i][i]
                count += 1
        diagonal_clues.append(count)
        max_h, count = 0, 0
        for i in range(size):
            row, col = i, size - 1 - i
            if self.grid[row][col] > max_h:
                max_h = self.grid[row][col]
                count += 1
        diagonal_clues.append(count)
        max_h, count = 0, 0
        for i in range(size - 1, -1, -1):
            row, col = i, size - 1 - i
            if self.grid[row][col] > max_h:
                max_h = self.grid[row][col]
                count += 1
        diagonal_clues.append(count)
        self.diagonal_clues = diagonal_clues
        return diagonal_clues

    def format_puzzle_string(self):
        if not self.grid or not self.direction_clues or not self.diagonal_clues:
            raise ValueError("Please complete the generation of the grid and clues first.")
        size = self.size
        top_clues = self.direction_clues[0]  
        bottom_clues = self.direction_clues[1]  
        left_clues = self.direction_clues[2] 
        right_clues = self.direction_clues[3]  
        diag1, diag2, diag3, diag4 = self.diagonal_clues  
        top_str = "  ".join(map(str, top_clues))  
        first_line = f" {diag1}| {top_str}| {diag3}"
        middle_lines = []
        for i in range(size):
            dots = "  ".join(["." for _ in range(size)])  
            middle_line = f" {left_clues[i]}| {dots}| {right_clues[i]}"
            middle_lines.append(middle_line)
        bottom_str = "  ".join(map(str, bottom_clues)) 
        last_line = f" {diag4}| {bottom_str}| {diag2}"
        return "\n".join([first_line] + middle_lines + [last_line])

    def format_solution_string(self):
        if not self.grid:
            raise ValueError("Please form a grid.")
        return str(self.grid).replace(" ", "")

    def generate_single_puzzle(self, size):
        self.generate_grid(size)
        self.calculate_direction_visibility()
        self.calculate_diagonal_visibility()
        
        return {
            "puzzle": self.format_puzzle_string(),
            "solution": self.format_solution_string(),
            "mode": "count",  
            "size": size
        }

    def generate_and_save_puzzles(self, output_path, sizes, count_per_size):
        directory = os.path.dirname(output_path)
        if directory and not os.path.exists(directory):
            os.makedirs(directory)
        puzzles_data = []
        for size in sizes:
            for i in range(count_per_size):
                puzzle = self.generate_single_puzzle(size)
                original_id = f"{i+1:02d}"  
                puzzle_id = f"gen-04-diag--{size}x{size}-{original_id}"
                puzzles_data.append({
                    "id": puzzle_id,
                    "puzzle": puzzle["puzzle"],
                    "solution": puzzle["solution"],
                    "mode": puzzle["mode"]
                })
        df = pd.DataFrame(puzzles_data)
        df.to_parquet(output_path, engine='pyarrow', index=False)


if __name__ == "__main__":
    generator = SkyscraperPuzzleGenerator()
    output_file = r"skyscraper_sum_diag.parquet"
    sizes_to_generate = [6,7,8]  
    count_per_size = 50  
    generator.generate_and_save_puzzles(
        output_path=output_file,
        sizes=sizes_to_generate,
        count_per_size=count_per_size
    )