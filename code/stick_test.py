from functions import *

# # ----------------------------------------------
# # Main
# # ----------------------------------------------

file = "test_cesare"

#measure_no_field(4.2e9,4.6e9,20000,resolution_bandwidth=100,sets=100,avg=1,filename=file,plot=False)
file = "C:/Users/Administrator/Desktop/cavity_magnonics_TP4/data/"+file+'.csv'
data = pd.read_csv(file,sep=';')
time = np.array(data.Time)
avg = float(data.Averages[0])
freq = tuple(map(float, str(data["Frequencies[Hz]"][0]).split(",")))[::-1] 
freq = np.array(freq)/ 1e9
re = np.array(data["Real Amp"])
im = np.array(data["Img Amp"])
for i in range(len(re)):
    re[i] = tuple(map(float, re[i].split(",")))[::-1]
    im[i] = tuple(map(float, im[i].split(",")))[::-1]
    
re = np.array(re)
im = np.array(im)

# # ----------------------------------------------
# # Plot
# # ----------------------------------------------

os.makedirs("C:/Users/Administrator/Desktop/cavity_magnonics_TP4/data/stick_test",exist_ok=True)
for i in range(len(re)):
    plt.figure()
    plt.plot(freq,re[i])
    plt.plot(freq,im[i])
    plt.title("Real and Imaginary part of the spectrum")
    plt.xlabel("Frequency (GHz)")
    plt.ylabel("Amplitude")
    plt.legend(["Real","Imaginary"])
    plt.grid()
    plt.savefig("C:/Users/Administrator/Desktop/cavity_magnonics_TP4/data/stick_test/time_"+str(i)+".png")
    plt.close()
    amplitude = np.sqrt(np.array(re[i])**2+np.array(im[i])**2)
    plt.figure()
    plt.plot(freq,amplitude)
    plt.title("Amplitude of the spectrum")
    plt.xlabel("Frequency (GHz)")
    plt.ylabel("Amplitude")
    plt.grid()
    plt.savefig("C:/Users/Administrator/Desktop/cavity_magnonics_TP4/data/stick_test/amplitude_"+str(i)+".png")
    plt.close()
    
# # ----------------------------------------------
# # 2D plot of the spectrum through time
# # ----------------------------------------------

# Turn the data into a 2D array
re_array = np.zeros((len(re),len(re[0])))
im_array = np.zeros((len(im),len(im[0])))
for i in range(len(re)):
    for j in range(len(re[0])):
        re_array[i,j] = re[i][j]
        im_array[i,j] = im[i][j]
re = re_array
im = im_array

os.makedirs("C:/Users/Administrator/Desktop/cavity_magnonics_TP4/data/stick_test/2D",exist_ok=True)
plt.figure()
plt.imshow(re,aspect='auto',extent=[freq[0],freq[-1],len(time),0])
plt.colorbar()
plt.title("Real part of the spectrum through time")
plt.xlabel("Frequency (GHz)")
plt.ylabel("Measurement number")
plt.savefig("C:/Users/Administrator/Desktop/cavity_magnonics_TP4/data/stick_test/2D/real.png")
plt.close()

plt.figure()
plt.imshow(im,aspect='auto',extent=[freq[0],freq[-1],len(time),0])
plt.colorbar()
plt.title("Imaginary part of the spectrum through time")
plt.xlabel("Frequency (GHz)")
plt.ylabel("Measurement number")
plt.savefig("C:/Users/Administrator/Desktop/cavity_magnonics_TP4/data/stick_test/2D/imaginary.png")
plt.close()

plt.figure()
plt.imshow(np.sqrt(re**2+im**2),aspect='auto',extent=[freq[0],freq[-1],len(time),0])
plt.colorbar()
plt.title("Amplitude of the spectrum through time")
plt.xlabel("Frequency (GHz)")
plt.ylabel("Measurement number")
plt.savefig("C:/Users/Administrator/Desktop/cavity_magnonics_TP4/data/stick_test/2D/amplitude.png")

# # ----------------------------------------------
# # Min max scale the amplitude for each frequency
# # ----------------------------------------------
amplitude = np.sqrt(re**2+im**2)
for i in range(len(freq)):
    amplitude[:,i] = (amplitude[:,i]-np.min(amplitude[:,i]))/(np.max(amplitude[:,i])-np.min(amplitude[:,i]))


plt.figure()
plt.imshow(amplitude,aspect='auto',extent=[freq[0],freq[-1],len(time),0])
plt.colorbar()
plt.title("Amplitude of the spectrum through time")
plt.xlabel("Frequency (GHz)")
plt.ylabel("Measurement number")
plt.savefig("C:/Users/Administrator/Desktop/cavity_magnonics_TP4/data/stick_test/2D/amplitude_min_max.png")
plt.close()