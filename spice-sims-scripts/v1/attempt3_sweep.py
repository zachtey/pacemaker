import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import re
from matplotlib.ticker import FuncFormatter

import analyzer_funcs_v1

# === Parameters ===
FILENAME = "multivibrator1.3.txt"
PEAK_HEIGHT = 1.8  # threshold for rising edge
V_ON_THRESHOLD = 1.8  # threshold for pulse width
analyzer_funcs_v1.analyze_sweep_file(FILENAME, param_name='C1', PEAK_HEIGHT=1.8, V_ON_THRESHOLD=1.8)

