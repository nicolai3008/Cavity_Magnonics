from functions import *

connections = connect()

osc = connections['osc']
osc_reset(osc)
#osc_pulse(osc)

time, data = osc_get_data(osc)

plt.plot(time, data)
plt.xlabel('Time (s)')
plt.ylabel('Voltage (V)')
plt.show()

close(connections)