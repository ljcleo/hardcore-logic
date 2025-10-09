from subtasks.Hanoi_Gen01 import Hanoi_Gen01
from subtasks.Hanoi_Gen02 import Hanoi_Gen02
from subtasks.Hanoi_Gen03 import Hanoi_Gen03
from subtasks.Hanoi_Uns import Hanoi_Uns
import pandas as pd
import random

def generate_datasets(subtask, level, n):

    '''
    subtask: gen-01/gen-02/gen-03/uns
    level: small/medium/large/x-large
    n: number of puzzles
    '''

    generator = {
        'gen-01' : Hanoi_Gen01,
        'gen-02' : Hanoi_Gen02,
        'gen-03' : Hanoi_Gen03,
        'uns' : Hanoi_Uns,
    }

    gen = generator[subtask]

    configs = {
        'small' : [[3,3],[3,4],[4,4]],
        'medium' : [[3,5],[4,5],[5,5]],
        'large' : [[3,6],[4,6],[5,6],[6,6]],
        'x-large' : [[3,7],[4,7],[5,7],[6,7]]
    }

    datasets = []
    for i in range(n):
        lst = random.choice(configs[level])
        hanoi = gen(lst[0], lst[1])
        puzzle = hanoi.format_start_and_goal()
        datasets.append({
            'id' : f'{subtask}--{level}-{i}',
            'puzzle' : puzzle,
            'solution' : len(hanoi.solve_bfs()) if subtask != 'uns' else None,
            'order' : hanoi.new_target if subtask == 'gen-03' else list(range(1, lst[1]+1)),
            'right_only' : True if subtask == 'uns' else False
        })

    df = pd.DataFrame(datasets)
    return df

def save_to_parquet(df, path):
    df.to_parquet(path)

if __name__ == '__main__':
    df = generate_datasets('uns', 'small', 3)



