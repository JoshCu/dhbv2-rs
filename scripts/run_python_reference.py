"""Run the Python dhbv2 BMI over a prepared forcing CSV (reference for comparisons).

    python scripts/run_python_reference.py <mts|daily> <bmi_config.yaml> <forcing.csv> <out.csv>

Requires the `dhbv2` package (and its torch / dmg / hydrodl2 dependencies). The
forcing CSV is the output of scripts/prepare_forcing.py.
"""

import csv
import sys
import time

import numpy as np

COLUMNS = {
    'atmosphere_water__liquid_equivalent_precipitation_rate': 'precip_rate',
    'land_surface_air__temperature': 'TMP_2maboveground',
    'atmosphere_air_water~vapor__relative_saturation': 'SPFH_2maboveground',
    'land_surface_radiation~incoming~longwave__energy_flux': 'DLWRF_surface',
    'land_surface_radiation~incoming~shortwave__energy_flux': 'DSWRF_surface',
    'land_surface_air__pressure': 'PRES_surface',
    'land_surface_wind__x_component_of_velocity': 'UGRD_10maboveground',
    'land_surface_wind__y_component_of_velocity': 'VGRD_10maboveground',
}
OUTPUT = 'land_surface_water__runoff_volume_flux'


def main(kind: str, config: str, forcing: str, out: str) -> None:
    if kind == 'mts':
        from dhbv2.mts_bmi import MtsDeltaModelBmi as Bmi
    elif kind == 'daily':
        from dhbv2.bmi import DeltaModelBmi as Bmi
    else:
        raise SystemExit(f"unknown model {kind}")

    with open(forcing, newline='') as f:
        rows = list(csv.DictReader(f))

    bmi = Bmi()
    t0 = time.time()
    bmi.initialize(config)
    t1 = time.time()
    dest = np.zeros(1)
    with open(out, 'w') as f:
        f.write('step,runoff_m_per_h\n')
        for step, row in enumerate(rows):
            for name, column in COLUMNS.items():
                bmi.set_value(name, np.float64(row[column]))
            bmi.update()
            bmi.get_value_at_indices(OUTPUT, dest, [0])
            f.write(f'{step},{dest[0]:e}\n')
    print(f'initialize: {t1 - t0:.3f}s, {len(rows)} updates: {time.time() - t1:.3f}s', file=sys.stderr)


if __name__ == '__main__':
    main(*sys.argv[1:5])
