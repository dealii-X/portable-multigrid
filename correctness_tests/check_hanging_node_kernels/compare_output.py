#!/usr/bin/env python3
"""Compare a deal.II test output with the expected one like deal.II's
testsuite does: the output is normalized like cmake/scripts/normalize.pl
does for these tests (JobId line emptied, -0.000 printed as 0.000), then
compared like numdiff -a 1e-6 -r 1e-8 with the same separators: numbers may
differ by an absolute 1e-6 or a relative 1e-8, everything else must match.

  compare_output.py output expected

Prints whether the files are byte-identical, else the first differing lines,
and exits with 0 if they agree.
"""

import re
import sys

ABS_TOL = 1e-6
REL_TOL = 1e-8
SEPARATORS = re.compile(r"[ \t\r\n=,:;<>\[\](){}^]+")


def normalize(text):
    """The rules of deal.II's cmake/scripts/normalize.pl that apply here."""
    lines = []
    for line in text.splitlines():
        line = re.sub(r"JobId.*", "", line)
        line = re.sub(r"-0\.(0+)(?!\d)", r"0.\1", line)
        lines.append(line)
    return "\n".join(lines) + "\n"


def tokens(line):
    return [t for t in SEPARATORS.split(line) if t]


def numbers_agree(a, b):
    try:
        x, y = float(a), float(b)
    except ValueError:
        return a == b
    diff = abs(x - y)
    return diff <= ABS_TOL or diff <= REL_TOL * max(abs(x), abs(y))


def main(output_file, expected_file):
    with open(output_file) as f:
        output = normalize(f.read())
    with open(expected_file) as f:
        expected = f.read()

    if output == expected:
        print("  byte-identical to the expected output")
        return 0

    output_lines = output.splitlines()
    expected_lines = expected.splitlines()
    if len(output_lines) != len(expected_lines):
        print(f"  {len(output_lines)} lines, expected {len(expected_lines)}")
        return 1

    n_differing = 0
    for number, (line, expected_line) in enumerate(
        zip(output_lines, expected_lines), start=1
    ):
        a, b = tokens(line), tokens(expected_line)
        if len(a) != len(b) or not all(map(numbers_agree, a, b)):
            n_differing += 1
            if n_differing <= 10:
                print(f"  line {number}:")
                print(f"    output:   {line}")
                print(f"    expected: {expected_line}")

    if n_differing:
        print(f"  {n_differing} lines differ beyond the numdiff tolerances")
        return 1

    print("  not byte-identical, but within the numdiff tolerances")
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    sys.exit(main(sys.argv[1], sys.argv[2]))
