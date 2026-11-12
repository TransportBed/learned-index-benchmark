from __future__ import annotations

from typing import Iterable, Sequence


def spacer(lines: int = 1) -> None:
    for _ in range(max(0, int(lines))):
        print("")


def banner(tag: str, text: str) -> None:
    body = f"--- {tag} - {text} ---"
    line = "-" * len(body)
    spacer(1)
    print(line)
    print(body)
    print(line)


def info(msg: str) -> None:
    print(f"  - {msg}")


def print_table(title: str, headers: Sequence[str], rows: Iterable[Sequence[str]]) -> None:
    table = [list(map(str, headers))]
    table.extend(list(map(str, row)) for row in rows)
    widths = [max(len(row[idx]) for row in table) for idx in range(len(headers))]

    def fmt(row: Sequence[str]) -> str:
        return "  " + " | ".join(str(cell).ljust(widths[idx]) for idx, cell in enumerate(row))

    spacer(1)
    print(title)
    print("  " + "-+-".join("-" * width for width in widths))
    print(fmt(table[0]))
    print("  " + "-+-".join("-" * width for width in widths))
    for row in table[1:]:
        print(fmt(row))
    print("  " + "-+-".join("-" * width for width in widths))
