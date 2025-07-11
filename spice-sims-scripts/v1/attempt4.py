import analyzer_funcs_v1

FILENAME = "multivibrator1.5.2.txt"
#analyzer_funcs_v1.plot_waveforms_from_step_file(FILENAME, voltage_threshold=1.8)
analyzer_funcs_v1.analyze_sweep_file("multivibrator1.5.3.txt",
                        param_name='R2',  # or 'C1', etc.
                        PEAK_HEIGHT=1.8,
                        RESET_THRESHOLD=0.2,
                        V_ON_THRESHOLD=1.8)
