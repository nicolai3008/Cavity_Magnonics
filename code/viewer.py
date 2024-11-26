from functions import *
from pandas import read_csv

file = "data/Nicolai_Data/Temperature/Time_Measurement.csv"

data = read_csv(file)

data = pd.read_csv(file,sep=';')
time = data.Time.to_numpy()
for i in range(len(time)):
    t = time[i].split("-")
    time[i] = int(t[0])*3600 + int(t[1])*60 + int(t[2])
    if i == 0:
        bottom = time[i]
    time[i] -= bottom
temp = data.Temp.to_numpy()
disp = data.Display.to_numpy()
avg = float(data.Averages[0])
volt = data.Voltage.to_numpy()
freq = []
re = []
im = []
for i in range(len(disp)):
    freq.append(np.array(data.Frequencies[i].split(","),dtype=float))
    re.append(np.array(data.Real[i].split(","),dtype=float))
    im.append(np.array(data.Img[i].split(","),dtype=float))
freq = np.array(freq)
re = np.array(re)
im = np.array(im)
amp = (re**2 + im**2)**0.5

folder = "data/Nicolai_Data/Temperature/"

fig, ax = plt.subplots(figsize=(10,5))
ax.plot(time,temp)
ax.set_xlabel('Time (s)')
ax.set_ylabel('Temperature (°C)')
ax.set_title('Temperature over time')
plt.savefig(folder + "Temperature_over_time.png")
plt.close()

x_ar = [time,temp]
labels = ['Time (s)','Temperature (°C)']

for i, x in enumerate(x_ar):

    fig, ax = plt.subplots(figsize=(15,15))
    x_extent = [x[0], x[-1]]
    y_extent = [freq[0][0]/1e9, freq[0][-1]/1e9]
    extent = x_extent + y_extent

    X, Y = np.meshgrid(x, freq[0]/1e9)

    plt.pcolormesh(X,Y,amp, cmap="plasma",extent=extent,aspect = 'auto',interpolation='nearest',origin='lower') #color is "plasma"
    #plt.plot([x[0], x[-1]], [cavity_freq, cavity_freq], 'w--', label=f"Cavity frequency: {cavity_freq/1e9:.3f} GHz")
    #plt.plot([-87.99,-85.5799],np.array([1.26215,1.31189]),'k--',label='Anticrossing')
    ylim = ax.get_ylim()
    #plt.plot(field, en1, 'k--', label='E1')
    #plt.plot(field, en2, 'k--', label='E2')
    ax.set_ylim(ylim)
    plt.xlabel(labels[i], fontsize=20)
    plt.ylabel("Frequency [GHz]", fontsize=20)
    ax.tick_params(axis='both', which='major', labelsize=15)
    cbar = plt.colorbar()
    cbar.set_label("Amplitude [dB]", fontsize=20)
    cbar.ax.tick_params(labelsize=15)
    plt.title("Amplitude over {}".format(labels[i].split(" ")[0]), fontsize=20)
    plt.savefig(folder + "Amplitude_over_{}.png".format(labels[i].split(" ")[0]))
    plt.close()