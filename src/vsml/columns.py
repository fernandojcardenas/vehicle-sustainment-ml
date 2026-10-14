"""Column names and sensor metadata for the C-MAPSS turbofan data set.

The raw files have 26 space-separated columns with no header:
unit, cycle, 3 operational settings, 21 sensor measurements.
Names and units follow the data set's readme.txt and Saxena et al. (2008).
"""

from __future__ import annotations

UNIT = "unit"
CYCLE = "cycle"

OP_SETTINGS = ["op_setting_1", "op_setting_2", "op_setting_3"]

SENSORS = [f"sensor_{i}" for i in range(1, 22)]

# Raw column order, matching the files exactly.
RAW_COLUMNS = [UNIT, CYCLE, *OP_SETTINGS, *SENSORS]

# Short physical descriptions from the C-MAPSS documentation, for reports.
# (Index i here is sensor_i.)
SENSOR_DESCRIPTIONS = {
    "sensor_1": "Total temperature at fan inlet (R)",
    "sensor_2": "Total temperature at LPC outlet (R)",
    "sensor_3": "Total temperature at HPC outlet (R)",
    "sensor_4": "Total temperature at LPT outlet (R)",
    "sensor_5": "Pressure at fan inlet (psia)",
    "sensor_6": "Total pressure in bypass-duct (psia)",
    "sensor_7": "Total pressure at HPC outlet (psia)",
    "sensor_8": "Physical fan speed (rpm)",
    "sensor_9": "Physical core speed (rpm)",
    "sensor_10": "Engine pressure ratio (P50/P2)",
    "sensor_11": "Static pressure at HPC outlet (psia)",
    "sensor_12": "Ratio of fuel flow to Ps30 (pps/psi)",
    "sensor_13": "Corrected fan speed (rpm)",
    "sensor_14": "Corrected core speed (rpm)",
    "sensor_15": "Bypass ratio",
    "sensor_16": "Burner fuel-air ratio",
    "sensor_17": "Bleed enthalpy",
    "sensor_18": "Demanded fan speed (rpm)",
    "sensor_19": "Demanded corrected fan speed (rpm)",
    "sensor_20": "HPT coolant bleed (lbm/s)",
    "sensor_21": "LPT coolant bleed (lbm/s)",
}

# The four sub-data-sets and their operating-condition / fault-mode layout.
# Engine counts are the VERIFIED counts from the data files, not the readme:
# readme.txt lists FD004 as 248 train / 249 test, but the files actually hold
# 249 train / 248 test (and test always matches the RUL truth length). The
# n_conditions field drives per-regime normalisation.
SUBSETS = {
    "FD001": {"conditions": 1, "fault_modes": 1, "train_units": 100, "test_units": 100},
    "FD002": {"conditions": 6, "fault_modes": 1, "train_units": 260, "test_units": 259},
    "FD003": {"conditions": 1, "fault_modes": 2, "train_units": 100, "test_units": 100},
    "FD004": {"conditions": 6, "fault_modes": 2, "train_units": 249, "test_units": 248},
}

ALL_SUBSETS = list(SUBSETS)
