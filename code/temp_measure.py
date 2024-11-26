from functions import *
freq_start = (2435-250)*1e6
freq_stop = (2435+250)*1e6
freq_points = 1001
volt = 45
time_steps = 120
dt = 30

measurement_time( freq_start,  freq_stop,  freq_points,  volt, time_steps, dt, power=-15, filename="Time_Measurement", plot=False)