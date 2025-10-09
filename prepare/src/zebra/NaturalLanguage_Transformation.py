import json
import re
from collections.abc import Iterable
from random import shuffle
from typing import Optional,Any
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
from typing import Any
from httpx import Client
from openai import OpenAI
from pydantic import BaseModel, Field, create_model
from tqdm import tqdm

'''
Convert the generated constraints into natural language form
'''

def base_naturalizer(df: pd.DataFrame) -> pd.DataFrame:
    def cv_sol(s: str) -> dict:
        d = eval(s)
        df = pd.DataFrame(d["rows"], columns=d["header"]).set_index("House")
        df.index = df.index.map(lambda x: f"House {x}")
        return df.to_dict(orient="index")

    sol: pd.Series = df["solution"].apply(cv_sol)

    pl: pd.Series = sol.apply(
        lambda d: "|".join([f"{k}_{v}" for h in d.values() for k, v in h.items()])
    )

    def match(src: str, parts: Iterable[str]) -> Optional[re.Match]:
        return re.match(r"\s*".join(("^", *parts, "$")), src)

    def parse_stmt(s):
        sd = s["id"]
        ps = f"({pl[sd]})"
        sp = (ps, "-", ps)
        ap = ("If", "\\(", *sp, ">", "0", ",", *sp, ",", "-", "\\(", *sp, "\\)", "\\)")
        hr = f"([1-{len(sol[sd])}])"
        ih = (ps, "==", hr)
        ii = ("If", "\\(", *ih, ",", "1", ",", "0", "\\)")
        od = ("first", "second", "third", "fourth", "fifth", "sixth")

        def leftrightof(st):
            m = match(st, (ps, "([<>])", ps))
            if m is None:
                return None

            lhs = m.group(1)
            op = m.group(2)
            rhs = m.group(3)

            return (
                f"The person '{lhs}' is somewhere to the {'left' if op == '<' else 'right'} "
                f"of the person '{rhs}'."
            )

        def notat(st):
            m = match(st, (ps, "!=", ps))
            if m is None:
                return None

            lhs = m.group(1)
            rhs = m.group(2)
            return f"The person '{lhs}' is not the person '{rhs}'."

        def directadj(st):
            m = match(st, (ps, "([+-])", "1", "==", ps))
            if m is None:
                return None

            lhs = m.group(1)
            op = m.group(2)
            rhs = m.group(3)

            return (
                f"The person '{lhs}' is directly {'left' if op == '+' else 'right'} "
                f"of the person '{rhs}'."
            )

        def between(st):
            m = match(st, (*ap, "==", "([23])"))

            if m is None:
                return None
            if m.group(1) != m.group(3) or m.group(1) != m.group(5) or m.group(3) != m.group(5):
                return None
            if m.group(2) != m.group(4) or m.group(2) != m.group(6) or m.group(4) != m.group(6):
                return None

            lhs = m.group(1)
            rhs = m.group(2)
            opv = int(m.group(7)) - 1

            return (
                f"There {'is one house' if opv == 1 else 'are two houses'} "
                f"between the person '{lhs}' and the person '{rhs}'."
            )

        def sidebyside(st):
            m = match(st, (*ap, "==", "1"))

            if m is None:
                return None
            if m.group(1) != m.group(3) or m.group(1) != m.group(5) or m.group(3) != m.group(5):
                return None
            if m.group(2) != m.group(4) or m.group(2) != m.group(6) or m.group(4) != m.group(6):
                return None

            lhs = m.group(1)
            rhs = m.group(2)
            return f"The person '{lhs}' and the person '{rhs}' are next to each other."

        def foundat(st):
            m = match(st, (ps, "(==|!=)", hr))
            if m is None:
                return None

            lhs = m.group(1)
            op = m.group(2)
            rhs = int(m.group(3)) - 1
            return f"The person '{lhs}' is{' ' if op == '==' else ' not '}in the {od[rhs]} house."

        def samehouse(st):
            m = match(st, (ps, "==", ps))
            if m is None:
                return None

            lhs = m.group(1)
            rhs = m.group(2)
            return f"The person '{lhs}' is the person '{rhs}'."

        def or_(st):
            m = match(st, ("Or", "\\(", *ih, ",", *ih, "\\)"))
            if m is None:
                return None

            lhs = m.group(1)
            lhh = int(m.group(2)) - 1
            rhs = m.group(3)
            rhh = int(m.group(4)) - 1

            return (
                f"Either the person '{lhs}' is in the {od[lhh]} house, "
                f"or the person '{rhs}' is in the {od[rhh]} house, or both."
            )

        def implicate(st):
            m = match(st, ("Implies", "\\(", *ih, ",", "(.+)", "\\)"))
            if m is None:
                return None

            lhs = m.group(1)
            lhh = int(m.group(2)) - 1
            rhs = m.group(3)
            rhv = None

            for f in (leftrightof, notat, directadj, between, sidebyside, foundat, samehouse):
                rhv = f(rhs)
                if rhv is not None:
                    break

            if rhv is None:
                return None

            rhv = f"{rhv[0].lower()}{rhv[1:]}"
            return f"If the person '{lhs}' is in the {od[lhh]} house, then {rhv}."

        def exactly_one(st):
            m = match(st, (*ii, "\\+", *ii, "\\+", *ii, "==", "1"))
            if m is None:
                return None

            lhs = m.group(1)
            lhh = int(m.group(2)) - 1
            mhs = m.group(3)
            mhh = int(m.group(4)) - 1
            rhs = m.group(5)
            rhh = int(m.group(6)) - 1

            return (
                f"Only one of these holds: the person '{lhs}' is in the {od[lhh]} house, "
                f"the person '{mhs}' is in the {od[mhh]} house, "
                f"or the person '{rhs}' is in the {od[rhh]} house."
            )

        fm = {
            "LEFTRIGHTOF": leftrightof,
            "NOTAT": notat,
            "DIRECTADJ": directadj,
            "BETWEEN": between,
            "SIDEBYSIDE": sidebyside,
            "FOUNDAT": foundat,
            "SAMEHOUSE": samehouse,
            "OR": or_,
            "IMPLICATE": implicate,
            "EXACTLY_ONE": exactly_one,
        }

        st = None

        if "type" in s:
            st = fm[s["type"]](s["stmt"])
            assert st is not None, (s["type"], s["stmt"], ps, hr)
        else:
            for f in fm.values():
                st = f(s["stmt"])
                if st is not None:
                    break

            assert st is not None, (s["stmt"], ps, hr)

        return st

    clues1 = df["clues"].str.split("\n").explode().str.partition(". ")[2]

    if (clues1.str.count(":") > 0).any():
        clues = (
            clues1.str.partition(": ")[[0, 2]]
            .rename(columns={0: "type", 2: "stmt"})
            .reset_index(names="id")
        )
    else:
        clues = clues1.rename(index="stmt").reset_index().rename(columns={"index": "id"})

    new_clues = (
        clues.assign(clue=clues.apply(parse_stmt, axis=1))
        .groupby("id")["clue"]
        .apply(lambda x: "\n".join((f"{i + 1}. {t}" for i, t in enumerate(x))))
    )

    new_sol = sol.map(lambda x: {h: {k: v.lower() for k, v in d.items()} for h, d in x.items()})
    keys = new_sol.map(lambda x: list(list(x.values())[0].keys()))

    def make_puzzle(idx):
        cur_sol = new_sol[idx]
        nh = len(cur_sol)
        ks = keys[idx]
        vs = {k: [] for k in ks}

        for d in cur_sol.values():
            for k, v in d.items():
                vs[k].append(f"`{v}`")

        for v in vs.values():
            shuffle(v)

        return "\n\n".join(
            (
                "\n".join(
                    (
                        f"There are {nh} houses, numbered 1 to {nh} from left to right, "
                        "as seen from across the street. Each house is occupied by a different person. "
                        "Each house has a unique attribute for each of the following characteristics:",
                        *(f"- Each person has a unique {k}: {', '.join((vs[k]))}" for k in ks),
                    )
                ),
                f"## Clues:\n{new_clues[idx]}",
            )
        )

    pz = df.index.to_series().map(make_puzzle)

    def dj(d):
        return json.dumps(d, ensure_ascii=False, separators=(",", ":"))

    ndf = df.assign(
        puzzle=pz,
        solution=new_sol.map(dj),
        house_ids=new_sol.map(lambda x: dj([int(s.partition(" ")[2]) for s in x.keys()])),
        keys=keys.map(dj),
    )[["size", "puzzle", "solution", "house_ids", "keys"]].sort_values("size", ignore_index=True)

    rdf = ndf.assign(id=ndf.index.map(lambda x: f"{int(x):04d}"))[["id", *ndf.columns]]

    return rdf

def llm_naturalizer(df:pd.DataFrame, model, judge_model, base_url, api_key):
    cd = {
        "Animal": "The person's favorite animal.",
        "Birthday": "The person's birthday.",
        "BookGenre": "The person's favorite book genre.",
        "CarModel": "The model of the person's car.",
        "Children": "What or how many children the person have.",
        "Cigar": "The person's favorite cigar brand.",
        "Color": "The person's favorite color.",
        "Drink": "The person's favorite drink.",
        "Education": "The person's education level.",
        "FavoriteSport": "The person's favorite sport.",
        "Flower": "The person's favorite flower.",
        "Food": "The person's breakfast/lunch/dinner. (choose one meal for all people)",
        "HairColor": "The person's hair color.",
        "Height": "The person's height.",
        "Hobby": "The person's hobby.",
        "HouseStyle": "The house style where the person lives.",
        "Mother": "The name of the person's mother.",
        "MusicGenre": "The person's favorite music genre.",
        "Name": "The person's name.",
        "Nationality": "The person's nationality.",
        "Occupation": "The person's occupation.",
        "Pet": "What pet the person have.",
        "PhoneModel": "The model of the person's phone.",
        "Smoothie": "The person's favorite smoothie flavor.",
        "Vacation": "The destination of the person's vacation.",
    }

    def cm(name: str, **kwargs: Any) -> type[BaseModel]:
        return create_model(
            name,
            __config__=None,
            __doc__=None,
            __base__=None,
            __module__=__name__,
            __validators__=None,
            __cls_kwargs__=None,
            **kwargs,
        )

    def make_prompt(s):
        cs = eval(s["keys"])
        nc = s["puzzle"].partition("\n\n")[2].count("\n")

        p = "\n\n--------\n\n".join(
            (
                "Here is a generated logic puzzle:",
                s["puzzle"],
                "Here are the explanations of the mentioned characteristics:",
                "\n".join((f"{c}: {cd[c]}" for c in cs)),
                "Please polish the puzzle description with natural language, "
                "more specifically the following parts:\n"
                "1. Characteristic description between the introduction and the clues, e.g. "
                "'Each person has a unique Computer' -> 'People use different computer brands'."
                "DO NOT INCLUDE THE ATTRIBUTES AFTER ':', "
                "AND DO NOT ADD ANY PUNCTUATIONS ('.', ':') AT THE END.\n"
                "2. Clues, e.g. "
                "'The person 'Name_Frieren' is directly right of the person 'Nationality_British'.' -> "
                "'Frieren lives directly right to the Brit.' "
                "DO NOT MODIFY THE RELATIONS OR MIX THE ATTRIBUTES.",
                "Your output should be in JSON format, e.g.: "
                '{"char_desc":{"Computer":"Each person uses a different computer brand","...":"..."},'
                '"clues":{"500":"Frieren lives directly right to the Brit.","...":"..."}}',
            )
        )

        m = cm(
            "PolishedContent",
            char_desc=cm("PolishedCharDesc", **{c: (str, Field(pattern="^[a-zA-Z ]+$")) for c in cs}),
            clues=cm("PolishedClues", **{str(i + 1): str for i in range(nc)}),
        )

        return p, m

    prompts = df.apply(make_prompt, axis=1).tolist()

    def make_client(base_url, api_key, proxy=None):
        http_client = None
        if proxy is not None:
            http_client = Client(proxy=proxy)

        return OpenAI(base_url=base_url, api_key=api_key, http_client=http_client)

    client = make_client(
        base_url, api_key=api_key
    )

    def query(prompt, format, model):
        try:
            completion = client.beta.chat.completions.parse(
                model=model, messages=[{"role": "user", "content": prompt}], response_format=format
            )

            message = completion.choices[0].message
            return message.refusal if message.refusal else message.parsed
        except Exception as e:
            return e

    def work(ps, model, pa):
        with tqdm(None, total=len(ps)) as pbar:
            def qw(p):
                result = query(p[0], p[1], model)
                pbar.update(1)
                return result

            with ThreadPoolExecutor(max_workers=pa) as pool:
                return list(pool.map(qw, ps))

    results = work(prompts, model, 96)

    def fix_work(idx, model):
        p, m = prompts[idx]
        results[idx] = query(p, m, model)

    for i, r in enumerate(results):
        if not isinstance(r, BaseModel):
            print(i)
            fix_work(i, model)

    df["polish"] = results

    def make_new_puzzle(s):
        intro, _, old_clue = s["puzzle"].partition("\n\n")
        head, *old_char = intro.split("\n")
        split_old_char = [char.partition(": ") for char in old_char]
        num_clue = old_clue.count("\n")
        clue_title = old_clue.partition("\n")[0]
        polish = s["polish"]
        polish_desc = polish.char_desc
        polish_clue = polish.clues

        new_char = [
            f"{getattr(polish_desc, desc.rsplit(' ', maxsplit=1)[-1])}{mid}{att}"
            for desc, mid, att in split_old_char
        ]

        new_intro = "\n".join((head, *new_char))
        new_clue_list = [f"{i + 1}. {getattr(polish_clue, f'{i + 1}')}" for i in range(num_clue)]

        new_clue = "\n".join(
            (clue_title, *(f"{c}{'' if c.endswith('.') else '.'}" for c in new_clue_list))
        )

        return f"{new_intro}\n\n{new_clue}"

    df["update"] = df.apply(make_new_puzzle, axis=1)

    def make_new_prompt(s):
        return "\n\n--------\n\n".join(
            (
                "Here is a generated logic puzzle:",
                s["puzzle"],
                "Someone has proposed a natural language version of it:",
                s["update"],
                "Examine the clues (indexed from 1) one by one, "
                "and determine whether the revised clues match the original ones. "
                "If not, fix any mismatched clues in the natural language version.",
                "Your output should be in JSON format, e.g.: "
                '{"reasoning":"...","same":true/false,"fix":[{"index":500,"newclue":"..."},...]}',
            )
        )

    new_prompts = df.apply(make_new_prompt, axis=1).tolist()

    judgement_model = create_model(
        "Judgment", reasoning=str, same=bool, fix=list[create_model("Clue", index=int, newclue=str)]
    )

    def new_work(ps, bm, model, pa):
        with tqdm(None, total=len(ps)) as pbar:
            def qw(p):
                result = query(p, bm, model)
                pbar.update(1)
                return result

            with ThreadPoolExecutor(max_workers=pa) as pool:
                return list(pool.map(qw, ps))

    new_results = new_work(new_prompts, judgement_model, judge_model, 96)

    def new_fix_work(idx, model):
        new_results[idx] = query(new_prompts[idx], judgement_model, model)

    for i, r in enumerate(new_results):
        if not isinstance(r, BaseModel):
            print(i)
            new_fix_work(i, judge_model)

    df["check"] = new_results

    def make_final_puzzle(s):
        check = s["check"]
        if check.same:
            return s["update"]
        fix = {clue_fix.index - 1: clue_fix.newclue for clue_fix in check.fix}
        intro, _, old_clue = s["puzzle"].partition("\n\n")
        head, *old_char = intro.split("\n")
        split_old_char = [char.partition(": ") for char in old_char]
        num_clue = old_clue.count("\n")
        clue_title = old_clue.partition("\n")[0]
        polish = s["polish"]
        polish_desc = polish.char_desc
        polish_clue = polish.clues

        new_char = [
            f"{getattr(polish_desc, desc.rsplit(' ', maxsplit=1)[-1])}{mid}{att}"
            for desc, mid, att in split_old_char
        ]

        new_intro = "\n".join((head, *new_char))

        new_clue_list = [
            f"{i + 1}. {fix.get(i, getattr(polish_clue, f'{i + 1}'))}" for i in range(num_clue)
        ]

        new_clue = "\n".join(
            (clue_title, *(f"{c}{'' if c.endswith('.') else '.'}" for c in new_clue_list))
        )

        return f"{new_intro}\n\n{new_clue}"

    df["final"] = df.apply(make_final_puzzle, axis=1)

    ndf = df[["id", "size", "final", "solution", "house_ids", "keys"]].rename(
        columns={"final": "puzzle"}
    )

    return ndf

