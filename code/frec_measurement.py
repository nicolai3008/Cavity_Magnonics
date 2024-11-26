from functions import *
import time
freq_start = (2435-250)*1e6
freq_stop = (2435+250)*1e6
freq_points = 1001
volt_start = 35
volt_stop = 60
spacing = 0.1
volt_points = int((volt_stop - volt_start) / spacing) + 1

#measurement(freq_start, freq_stop, freq_points, volt_start, volt_stop, volt_points,avg=1, filename="Test_24-10-08", plot=False)
#viewer("Test_24-10-08", x_axis = "field")
#hall_probe_time()
powers = [-20,-15,-10,-5,0,5,10]
os.makedirs("data/Nicolai_Data/Power",exist_ok=True)
for power in powers:
    print(power)
    #time.sleep(600)
    measurement_no_lockin(freq_start, freq_stop, freq_points, volt_start, volt_stop, volt_points,avg=1, filename=f"Nicolai_Data/Power/Power_{power:5.2f}", plot=False, power=power)

for power in powers:
    viewer_no_lockin("Nicolai_Data/Power/Power_"+str(power), x_axis = "field")
#measurement(freq_start, freq_stop, freq_points, volt_start, volt_stop, volt_points,avg=1, filename="Test_24-10-08", plot=False)
#viewer_no_lockin("Nicolai_Data/24-10-15/4")
