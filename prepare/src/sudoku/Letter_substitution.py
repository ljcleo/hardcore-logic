import csv
import copy
from pathlib import Path
#在gen-diag/discon/max_value中存在很少一部分原始约束下也是单解的谜题，这个代码把原始单解的给完全去掉了，并且加上了字母替换功能

class SudokuPostProcessor:
    def __init__(self, input_csvs, output_csv, enable_letter_encoding=True):
        """
        param input_csvs: 输入CSV文件路径列表
        param output_csv: 输出CSV路径
        param enable_letter_encoding: 是否启用A–I/Z替换
        """
        self.input_csvs = input_csvs
        self.output_csv = output_csv
        self.enable_letter_encoding = enable_letter_encoding

    @staticmethod
    def parse_board(board_str):
        nums = [int(x) for x in board_str.split()]
        if len(nums) != 81:
            raise ValueError(f"谜题数据长度错误，应为81个数，实际为{len(nums)}")
        return [nums[i * 9:(i + 1) * 9] for i in range(9)]

    @staticmethod
    def encode_board_AZ(board):
        mapping = {0: 'Z', 1: 'A', 2: 'B', 3: 'C', 4: 'D', 5: 'E', 6: 'F', 7: 'G', 8: 'H', 9: 'I'}
        return ' '.join(mapping[cell] for row in board for cell in row)

    @staticmethod
    def board_to_string(board):
        return ' '.join(str(c) for r in board for c in r)

    @staticmethod
    def is_valid(board, r, c, val):
        for i in range(9):
            if board[r][i] == val or board[i][c] == val:
                return False
        br, bc = (r // 3) * 3, (c // 3) * 3
        for i in range(br, br + 3):
            for j in range(bc, bc + 3):
                if board[i][j] == val:
                    return False
        return True

    def count_sudoku_solutions(self, board, limit=5):
        solutions = 0
        def backtrack():
            nonlocal solutions
            for r in range(9):
                for c in range(9):
                    if board[r][c] == 0:
                        for val in range(1, 10):
                            if self.is_valid(board, r, c, val):
                                board[r][c] = val
                                backtrack()
                                board[r][c] = 0
                                if solutions >= limit:
                                    return
                        return
            solutions += 1
        backtrack()
        return solutions

    def process(self):
        all_records = []
        for input_csv in self.input_csvs:
            if not Path(input_csv).exists():
                print(f"⚠️ 文件不存在: {input_csv}")
                continue
            with open(input_csv, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                fieldnames = reader.fieldnames or []
                has_target_pos = "目标最大子网格位置" in fieldnames
                for row in reader:
                    try:
                        board = self.parse_board(row["谜题数据"])
                    except Exception as e:
                        print(f"⚠️ 跳过格式错误行: {e}")
                        continue
                    sol_count = self.count_sudoku_solutions(copy.deepcopy(board), limit=2)
                    if sol_count > 1:
                        row["原始约束下解的个数"] = sol_count
                        if self.enable_letter_encoding:
                            row["转化后的谜题"] = self.encode_board_AZ(board)
                        else:
                            row["转化后的谜题"] = ""
                        all_records.append(row)
        if not all_records:
            print("❌ 没有检测到多解谜题。")
            return

        base_fields = ["谜题ID", "数独类型", "终盘数据", "谜题数据", "挖空数"]
        if "目标最大子网格位置" in all_records[0]:
            base_fields.append("目标最大子网格位置")
        base_fields += ["原始约束下解的个数", "转化后的谜题"]

        with open(self.output_csv, 'w', newline='', encoding='utf-8') as f_out:
            writer = csv.DictWriter(f_out, fieldnames=base_fields)
            writer.writeheader()
            for r in all_records:
                writer.writerow(r)

        print(f"✅ 处理完成，共保留 {len(all_records)} 个多解谜题。输出文件: {self.output_csv}")


if __name__ == "__main__":
    input_csvs = [r"C:\Users\ohhhh\Desktop\sudoku_extremal.csv"]
    output_csv = r"C:\Users\ohhhh\Desktop\sudoku_multisolution_filtered.csv"
    enable_letter_encoding = True

    processor = SudokuPostProcessor(input_csvs, output_csv, enable_letter_encoding)
    processor.process()