"""Convert an ngen per-catchment forcing CSV to the BMI's input units.

    python scripts/prepare_forcing.py <ngen_forcing.csv> <out.csv>

ngen converts units itself; this is only for the standalone drivers
(examples/run_forcing.rs and scripts/run_python_reference.py). Precipitation goes
from mm/s to mm/h and temperature from K to degC; other columns are kept, with
the unit suffix stripped from their names.
"""

import csv
import sys


def main(src: str, dst: str) -> None:
    with open(src, newline='') as f_in, open(dst, 'w', newline='') as f_out:
        reader = csv.reader(f_in)
        header = [h.split('[')[0] for h in next(reader)]
        writer = csv.writer(f_out)
        writer.writerow(header)
        p = header.index('precip_rate')
        t = header.index('TMP_2maboveground')
        for row in reader:
            row[p] = repr(float(row[p]) * 3600.0)
            row[t] = repr(float(row[t]) - 273.15)
            writer.writerow(row)


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
