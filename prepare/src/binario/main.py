from subtasks.Binario_Gen01 import Binario_Gen01
from subtasks.Binario_Gen02 import Binario_Gen02
from subtasks.Binario_Uns import Binario_Uns
import pandas as pd

def generate_dataset(subtask, size, n):
    '''
    - subtask : gen-01/gen-02/uns
    - size : size of the puzzle grid (must be an even number)
    - num : number of puzzles to generate
    '''
    generator = {
        'gen-01': Binario_Gen01,
        'gen-02': Binario_Gen02,
        'uns': Binario_Uns,
    }

    datasets = []
    generator = generator[subtask](size)

    if subtask == 'gen-01':
        for i in range(n):
            id = f"{subtask}--{size}x{size}-{i}"
            puzzle, solution = generator.create_puzzle()
            datasets.append({'id': id, 'puzzle': puzzle, 'solution': solution})

    if subtask == 'gen-02':
        for i in range(n):
            id = f"{subtask}--{size}x{size}-{i}"
            data = generator.generate_formatted_puzzle()
            datasets.append({'id': id, 'puzzle': data['puzzle'], 'solution': data['solution']})

    if subtask == 'uns':
        for i in range(n):
            id = f"{subtask}--{size}x{size}-{i}"
            unsolvable_puzzle, original_solution = generator.create_unsolvable_puzzle()
            datasets.append({'id': id, 'puzzle': unsolvable_puzzle, 'solution': None})

    return pd.DataFrame(datasets)

def save_to_parquet(df, path):
    df.to_parquet(path)

if __name__ == '__main__':
    df = generate_dataset('gen-01', 6,5)
    print(df)