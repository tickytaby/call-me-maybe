import math
import re
import json
import sys


def fn_add_numbers(a: int | float, b: int | float) -> float:
    return a + b


def fn_greet(name: str) -> str:
    return f"Hello {name.capitalize()}!"


def fn_reverse_string(s: str) -> str:
    return s[::-1]


def fn_get_square_root(a: int | float) -> float:
    return math.sqrt(a)


def fn_substitute_string_with_regex(
    source_string: str, regex: str, replacement: str
) -> str:
    try:
        out = re.sub(regex, replacement, source_string)
    except Exception:
        print("Failed call to fn_sbustitute_string_with_regex")
        print(f"Source: {source_string}")
        print(f"Regex: {regex}")
        print(f"Replacement: {replacement}")
        return ""
    return out


def main() -> None:
    fn_store = {}
    fn_store["fn_add_numbers"] = fn_add_numbers
    fn_store["fn_greet"] = fn_greet
    fn_store["fn_reverse_string"] = fn_reverse_string
    fn_store["fn_get_square_root"] = fn_get_square_root
    fn_store["fn_substitute_string_with_regex"] = fn_substitute_string_with_regex

    path_pre = "./data/output/"

    try:
        with open(f"{path_pre}{sys.argv[1]}") as f:
            data = json.load(f)
        for call in data:
            try:
                out = fn_store[call["name"]](**call["parameters"])
                print(call)
                print("-->", out)
                print()
            except Exception:
                print(f"Failed call for {call}.")
                continue
    except Exception as e:
        print("ERROR")
        print(e)
        return


if __name__ == "__main__":
    main()
