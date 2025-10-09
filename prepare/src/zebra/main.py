from subtasks.Zebra_Gen01 import Zebra_Gen01
from subtasks.Zebra_Gen02 import Zebra_Gen02
from subtasks.Zebra_Gen03 import Zebra_Gen03
from subtasks.Zebra_Uns import Zebra_Unsolvable
from NaturalLanguage_Transformation import base_naturalizer, llm_naturalizer
import pandas as pd
import json


def generate_dataset(subtask, num_per_config, model, judge_model, base_url, api_key):
    '''
    - subtask: gen-01/gen-02/gen-03/uns
    - num_per_config: number of puzzles in each config
    - model : Model for natural language conversion
    - judge_model : Judge model for natural language conversion
    - base_url : base url for Natural Language conversion
    - api_key : API key for Natural Language conversion
    '''
    generator = {'gen-01': Zebra_Gen01, 'gen-02': Zebra_Gen02,
                 'gen-03': Zebra_Gen03, 'uns': Zebra_Unsolvable}

    '''
    - N: number of houses
    - M: number of attributions
    - num : parameters set for gen-02, gen-03, and uns to control the number of new type constraints,
    - num_fakes: parameters set for uns to control the number of fake clues
    '''

    config = {
        'large': [[4, 5, 120, -1], [5, 3, 120, -1], [4, 6, 120, -1],[5, 4, 150, -1],[6, 3, 150, -1]],
        'x-large': [[5, 5, 150, -1],[6, 4, 180, -1],[5, 6, 180, -1],[6, 5, 200, -1],[6, 6, 200, -1]]
    }

    gen = generator[subtask]
    dataset = []
    for config_size, config_list in config.items():
        for cfg_idx, cfg in enumerate(config_list):
            N, M, num, num_fakes = cfg

            if subtask == 'uns':
                num_fakes = 1

            for i in range(num_per_config):
                zebra = gen(N,M,num,num_fakes)

                def format_clues(puzzle):
                    clue_map = {
                        "FOUNDAT": puzzle.original_foundat_clues,
                        "SAMEHOUSE": puzzle.original_samehouse_clues,
                        "NOTAT": puzzle.original_notat_clues,
                        "DIRECTADJ": puzzle.original_directadj_clues,
                        "LEFTRIGHTOF": puzzle.original_left_rightof_clues,
                        "BETWEEN": puzzle.original_between_clues,
                        "SIDEBYSIDE": puzzle.original_sidebyside_clues,
                        "OR": getattr(puzzle, "original_or_clues", []),
                        "IMPLICATE": getattr(puzzle, "original_implication_clues", []),
                        "EXACTLY_ONE": getattr(puzzle, "original_exactly_clues", [])
                    }

                    expr_to_type = {}
                    for clue_type, clues in clue_map.items():
                        for c in clues:
                            expr_to_type[str(c)] = clue_type

                    lines = []
                    for idx, clue in enumerate(puzzle.final_clues, 1):
                        clue_str = str(clue).replace("\n", "")
                        clue_type = expr_to_type.get(str(clue), "UNKNOWN")
                        lines.append(f"{idx}. {clue_type}: {clue_str}")
                    return "\n".join(lines)

                def format_solution(puzzle):
                    header = ["House"] + puzzle.attributes
                    rows = []
                    for i, house in enumerate(puzzle.original_solution, 1):
                        row = [str(i)] + [house[attr] for attr in puzzle.attributes]
                        rows.append(row)
                    return {"header": header, "rows": rows}

                dataset.append({
                    'size': f'{N}X{M}',
                    'clues': format_clues(zebra),
                    'solution': json.dumps(format_solution(zebra)),
                    'tag': 'all'
                })
        df = pd.DataFrame(dataset)
        rdf = base_naturalizer(df)
        ndf = llm_naturalizer(rdf, model, judge_model, base_url, api_key)
        return ndf

def save_to_parquet(df, path):
    df.to_parquet(path)

if __name__ == "__main__":
    df = generate_dataset('uns',1,'gpt-4.1-mini','gpt-4.1', 'base_url','api_key')
    print(df)

