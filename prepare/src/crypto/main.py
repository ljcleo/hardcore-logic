from subtasks.KKA_Gen01 import KKA_Gen01
from subtasks.KPA_Gen01 import KPA_Gen01
from subtasks.KPA_Gen02 import KPA_Gen02
from subtasks.KKA_Gen03 import KKA_Gen03
from subtasks.KKA_Gen04 import KKA_Gen04
from subtasks.KPA_Uns import KPA_Uns
import pandas as pd

def generate_datasets(subtask, n, layers, parts):

    '''
    subtask: gen-01-kka/gen-01-kpa/gen-02-kpa/gen-03-kka/gen-04-kka/uns
    n: number of puzzles
    layers: Number of encryption layers,Only works on gen-03-kka
    parts: Number of encryption parts,Only works on gen-04-kka
    '''

    generator = {
        'gen-01-kka': KKA_Gen01,
        'gen-01-kpa': KPA_Gen01,
        'gen-02-kpa': KPA_Gen02,
        'gen-03-kka': KKA_Gen03,
        'gen-04-kka': KKA_Gen04,
        'uns': KPA_Uns,
    }

    gen = generator[subtask](layers, parts)
    datasets = []
    for i in range(n):
        puzzle ,solution = gen.generate_puzzle()
        if subtask == 'gen-03-kka':
            id = f"{subtask}--{layers}x-{i}"
        elif subtask == 'gen-04-kka':
            id = f"{subtask}--{parts}s-{i}"
        else:
            id = f"{subtask}--all-{i}"
        datasets.append({
            'id' : id,
            'puzzle' : puzzle,
            'solution' : solution,
            'ordered' : True if subtask == 'gen-03-kka' else False,
        })

    df = pd.DataFrame(datasets)
    return df

def save_to_parquet(df, path):
    df.to_parquet(path)

if __name__ == '__main__':
    df = generate_datasets('gen-03-kka',10,2,1)
