"""
This helper script creates a documentation page from the materials database
that is stored in JSON format.

To run it, just do

    python ./make_materials_table.py

This will create the `pyroomacoustics.materials.database.rst` file in the doc.
"""

import json
from pathlib import Path

from tabulate import tabulate

datafile = Path("../pyroomacoustics/data/materials.json")
docfile = Path("./pyroomacoustics.materials.database.rst")


def format_freq(f):
    if f < 1000:
        return "{:d} Hz".format(f)
    else:
        return "{:d} kHz".format(f // 1000)


def print_section(title, data, headers, file):

    print(title, file=file)
    print("-" * len(title), file=file)
    print(file=file)

    for subtitle, materials in data.items():

        print(subtitle, file=file)
        print("^" * len(subtitle), file=file)
        print(file=file)

        table = []
        for keyword, p in materials.items():

            # fill the table
            row = [keyword, p["description"]]
            for c in p["coeffs"]:
                row.append("{:.2f}".format(c))

            # pad possibly missing coefficients
            while len(row) < len(headers):
                row.append("")

            table.append(row)

        print(tabulate(table, headers, tablefmt="rst"), file=file)
        print(file=file)


if __name__ == "__main__":

    with open(datafile, "r") as f:
        data = json.load(f)

    headers = ["keyword", "description"] + [
        format_freq(f) for f in data["center_freqs"]
    ]

    with open(docfile, "w") as f:

        print("Materials Database", file=f)
        print("==================", file=f)
        print(file=f)

        sections = {
            "absorption": "Absorption Coefficients",
            "scattering": "Scattering Coefficients",
        }

        for key, sectitle in sections.items():

            print_section(
                title=sectitle, data=data[key], headers=headers, file=f,
            )
