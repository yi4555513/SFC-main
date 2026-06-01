from pathlib import Path

import numpy as np


class TLEGenerator:
    """Generate idealized Walker-style TLE records for the satellite snapshots."""

    def __init__(
        self,
        output_file,
        planes,
        satellites_per_plane,
        inclo,
        altitude,
        argpo_init=90,
        nodeo_phase=None,
        argpo_phase=True,
    ):
        self.output_file = Path(output_file)
        self.satellites_per_plane = int(satellites_per_plane)
        self.planes = int(planes)
        self.inclo = float(inclo)
        self.altitude = int(altitude)
        self.mean_motion = self._mean_motion(self.altitude)
        self.ecco = "0002000"
        self.bstar = "12222-5"
        self.argpo_init = argpo_init
        self.nodeo_phase = nodeo_phase
        self.argpo_phase = argpo_phase

    @staticmethod
    def _mean_motion(altitude):
        earth_radius = 6371.0
        gravitational_constant = 398600.4418
        semi_major_axis = earth_radius + altitude
        period = 2 * np.pi * np.sqrt((semi_major_axis**3) / gravitational_constant)
        return 1 / (period / (60 * 60 * 24))

    @staticmethod
    def tle_checksum(line):
        checksum = 0
        for char in line[:-1]:
            if char.isdigit():
                checksum += int(char)
            if char == "-":
                checksum += 1
        return checksum % 10

    def generate_tles(self):
        self.output_file.parent.mkdir(parents=True, exist_ok=True)
        with self.output_file.open("w", encoding="utf-8") as f:
            for plane_idx in range(self.planes):
                nodeo = (180 if self.inclo >= 75 else 360) / self.planes * plane_idx
                if self.nodeo_phase:
                    nodeo += 360 / self.planes / self.nodeo_phase[1] * self.nodeo_phase[0]

                for sat_idx in range(self.satellites_per_plane):
                    sat_id = plane_idx * 100 + sat_idx + 1
                    sat_name = f"Satellite_{self.altitude}_{plane_idx + 1}_{sat_idx + 1}"
                    f.write(sat_name + "\n")

                    line1 = (
                        "1 {0:05d}U 00000A   23121.00000000  .00000000  00000+0  {1} 0  999"
                    ).format(sat_id, self.bstar)
                    f.write(line1 + str(self.tle_checksum(line1 + " ")) + "\n")

                    mean_anomaly = 360.0 / self.satellites_per_plane * sat_idx
                    argpo = self.argpo_init
                    if plane_idx % 2 != 0 and self.argpo_phase:
                        argpo = self.argpo_init - 180 / self.satellites_per_plane

                    line2 = (
                        "2 {0:05d} {1:8.4f} {2:8.4f} {3} {4:8.4f} {5:8.4f} {6:11.8f}  999"
                    ).format(
                        sat_id,
                        self.inclo,
                        nodeo,
                        self.ecco,
                        argpo,
                        mean_anomaly,
                        self.mean_motion,
                    )
                    f.write(line2 + str(self.tle_checksum(line2 + " ")) + "\n")

        print(f"TLEs written to {self.output_file.resolve()}")
