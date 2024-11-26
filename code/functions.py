"""
Cavity Magnonics Project
Author: Nicolai Amin - 2024

Description: 
This file contains the functions to the Cavity Magnonics instrument.
Will be expanded upon as the project progresses.
Is linked to the GUI program: xxx

Groups of functions:
- VNA communication functions
- Lock-in Amplifier communication functions
- Power Supply communication functions
- Measurement functions
- Plotting functions

#TODO:
- Add theoretical functions for the cavity magnonics project
- 

"""
# ----------------------------------------------
# Import libraries for Cavity Magnonics
# ----------------------------------------------
import numpy as np
import os# to handle files in the folder
from RsInstrument import RsInstrument#<- both this or the above command works to import RsInstrument even if spyder complins about this one
from time import sleep
from time import sleep
import matplotlib.pyplot as plt
import re
from pathlib import Path
import sys
home = str(Path.home())
sys.path.insert(0, home+"//Desktop//python_libraries//VISA")
import pyvisa
import datetime
import pandas as pd
from tqdm import tqdm
import time
import datetime
from scipy.optimize import curve_fit
from numpy import random
# ----------------------------------------------
# Commands to control VNA
# Written by Cesare Soci - 2021
# ----------------------------------------------
def com_prep(znb):
    """Preparation of the communication (termination, etc...)"""

    print(f'VISA Manufacturer: {znb.visa_manufacturer}')  # Confirm VISA package to be chosen
    znb.visa_timeout = 5000  # Timeout for VISA Read Operations
    znb.opc_timeout = 5000  # Timeout for opc-synchronised operations
    znb.instrument_status_checking = True  # Error check after each command, can be True or False
    znb.clear_status()  # Clear status register

def com_check(znb):
    """Check communication with the device"""

    # Just knock on the door to see if instrument is present
    idnResponse = znb.query_str('*IDN?')
    sleep(1)
    print('Hello, I am ' + idnResponse)
        
def close(znb):
    """Close the VISA session"""

    znb.close()
    
def set_center(znb,central_freq):
    """set the central frequency [Hz]"""
    
    znb.write_str("FREQ:CENT "+str(central_freq))  
    
def set_span(znb,span_freq):
    """set the frequency span [Hz]"""
    
    znb.write_str("FREQ:SPAN "+str(span_freq))

def set_start_freq(znb,start_freq):
    """set the start frequency [Hz]"""
    
    znb.write_str("FREQ:START "+str(start_freq))    
    
def set_stop_freq(znb,stop_freq):  
    
    """set the stop frequency [Hz]"""
    
    znb.write_str("FREQ:STOP "+str(stop_freq)) 

def set_power(znb,Power):
    "set the source power in dbm (0dbm=1mW)" 
    znb.write_str('SOURce:POWer '+str(Power))

def average(znb,N):
    
    
    """set the number of trace average
    If N=0 turn off average
    """
    
    if N>0:
        znb.write_str("AVERAGE:COUNT "+str(int(N)))
        znb.write_str("AVERAGE ON")    
    else:
        znb.write_str("AVERAGE OFF")
      
def set_resolution_bandwidth(znb,RBW):
    """set the resolution bandwidth
    """
    znb.write_str("BAND "+str(RBW))

def set_sweep_points(znb,N):
    
    
    """set the the number of sweep points
    """
    znb.write_str("SWE:POINTS "+str(N))
    
def set_measure(znb,S='S12'):
    """choose the mesured S paramenter
    """
    if S == "S11":
        znb.write_str("CALCulate1:PARameter:MEASure 'Trc1','S11'")
    elif S == "S12":
        znb.write_str("CALCulate1:PARameter:MEASure 'Trc1','S12'")
    elif S == "S21":
        znb.write_str("CALCulate1:PARameter:MEASure 'Trc1','S21'")
    elif S == "S22":
        znb.write_str("CALCulate1:PARameter:MEASure 'Trc1','S22'")   
   
    
def autoscale(znb):
    "Automatically aggiust the Y scale "
    znb.write_str('DISPlay:WINDow1:TRACe1:Y:SCALe:AUTO ONCE')  # Enable auto scaling for trace 1
 
def preset(znb):
    "Preset the instrument"
    znb.write_str('SYSTem:PRESet')
    
# ----------------------------------------------
# Commands to control Lock-in Amplifier
# Written by Simon ... - 2024
# ----------------------------------------------
def clean_term(x):
    aux = repr(x)
    aux = re.sub('\\\\n\\\\.*', '', aux)[1:]
    return aux

def purge(inst):
    i = 0
    while True:
        try:
            inst.read_bytes(1)
            i += 1
        except:
            break
    sleep(1)
    return i

# ----------------------------------------------
# Measurement functions
# Written by Nicolai Amin - 2024
# ----------------------------------------------

# Connect to different instruments
def connect(verbose = False):
    # Read the VISA address of the instruments
    print("Searching for instruments...\n")
    rm = pyvisa.ResourceManager()
    instruments = rm.list_resources('?*')
    connections = {}
    possible_connections = ["znb", "lockin", "coms", "sg", "osc"]
    if verbose:
        print(instruments)
    # Connect to the instruments
    for inst in instruments:
        if "0x0AAD" in inst and "0x01E6" in inst:
            if 'znb' not in connections:
                znb = RsInstrument(inst)   # Create the connection with the instrument
                sleep (1)                               # Wait for some time since it it can take a few time to establish the connection
                if verbose:
                    print("VNA found\n")
                com_prep(znb)                           # Prepare the communication
                com_check(znb)                          # Check the communication
                connections["znb"] = znb
                
            else:
                print("More than one VNA found, please check the connections")
        elif "0x0AAD" in inst and "0x0054" in inst:
            if 'sg' not in connections:
                sg = RsInstrument(inst)   # Create the connection with the instrument
                sleep (1)                               # Wait for some time since it it can take a few time to establish the connection
                if verbose:
                    print("SG found\n")
                com_prep(sg)                           # Prepare the communication
                com_check(sg)                          # Check the communication
                connections["sg"] = sg
            else:
                print("More than one SG found, please check the connections")
        elif "0x0A2D" in inst:
            if 'lockin' not in connections:
                lockin = rm.open_resource(inst)
                sleep(1)
                if verbose:
                    print("Lock-in found\n")
                clean_term(lockin.query('MAG.'))
                connections["lockin"] = lockin
            else:
                print("More than one Lock-in found, please check the connections")
        elif "ASRL" in inst:
            if 'coms' not in connections:
                connections["coms"] = []
            com = rm.open_resource(inst)
            com.write_termination = '\r'
            com.read_termination = '\r'
            sleep (1)
            if verbose:
                print("ASLR found\n")
            connections["coms"].append(com)
        elif "0x0699::0x0364" in inst:
            osc = rm.open_resource(inst)
            sleep(1)
            if verbose:
                print("Oscilloscope found\n")
            connections["osc"] = osc
        else:
            print("Unknown instrument found")
            print(inst)
    for key in possible_connections:
        if key not in connections:
            print(f"{key} not found")

    sleep(2.5)
    return connections

# Close the connection to the instruments
def close(connections):
    for key in connections:
        if key == "coms":
            for com in connections[key]:
                com.close()
        else:
            connections[key].close()
    print("Connections closed")

# Set the current of the power supply
def set_current(com,current):
    com.write(f'CURR{current*10:0>3.0f}')
    com.read_bytes(3)
    sleep(1)
    return True

# Set the voltage of the power supply
def set_voltage(com,voltage):
    com.write(f'VOLT{voltage*10:0>3.0f}')
    com.read_bytes(3)
    sleep(1)
    return True

def get_display(com):
    disp = com.query("GETD")
    sleep(1)
    com.read_bytes(3)
    return disp

# Check connection to power supply
def check_connection(com):
    try:
        purge(com)
        set_current(com,10)
        set_voltage(com,1)
        return True
    except pyvisa.errors.VisaIOError:
        return False
    
def linear_fit(x, y):
    a, *args = curve_fit(lambda x, a, b: a*x+b, x, y)
    return a[0], a[1]

def hall_calibration():
    file = "C:/Users/Administrator/Desktop/cavity_magnonics_TP4/code/calibration.txt"
    with open(file, "r") as f:
        data = f.readlines()[1:]
        # Data in format V;B
        data = np.array([np.array(map(float, line.split(";"))) for line in data])
    return linear_fit(data[:,0], data[:,1])

def anticrossing(E0,E1,omega):
    # Hamiltonian
    H = np.array([[E0, omega],[omega, E1]])
    # Eigenvalues
    eigvals = np.linalg.eigvalsh(H)
    return eigvals[1], eigvals[0]

#----------------------------------------------
# Oscilloscope functions
#----------------------------------------------
def osc_reset(osc):
    osc.timeout = 100000 # ms
    osc.encoding = 'latin_1'
    osc.read_termination = '\n'
    osc.write_termination = None
    osc.write('*cls') 
    sleep(2)
    idn = osc.query('*idn?')
    
    r = osc.write('*rst')
    t1 = time.perf_counter()
    r = osc.query('*opc?')
    t2 = time.perf_counter()
    #print(f'oscilloscope reset time: {t2-t1:.3f} s')
    return idn

def osc_autoset(osc):
    r = osc.write('autoset EXECUTE')
    t1 = time.perf_counter()
    r = osc.query('*opc?')
    t2 = time.perf_counter()
    # print(f'oscilloscope autoset time: {t2-t1:.3f} s')
    return 


def osc_io(osc, ch=1):
    osc.write('header 0')
    osc.write('data:encdg RIBINARY')
    osc.write(f'data:source CH{str(ch)}') # channel
    osc.write('data:start 1') # first sample
    record = int(osc.query('wfmpre:nr_pt?'))
    #print(f'oscilloscope record length: {record}')
    osc.write('data:stop {}'.format(record)) # last sample
    osc.write('wfmpre:byt_n 1') # 1 byte per sample
    return record

def osc_pulse(osc, tscale=500e-6, vscale=50):

    osc.write("HORizontal:SCAle {}".format(tscale))
    osc.write("HORizontal:POSITION {}".format(tscale*4))
    osc.write("CH1:SCAle {}".format(vscale))
    osc.write("CH1:POSition -3")
    osc.write('acquire:state 0') # stop
    
    p = osc.query('TRIGger:MAIn:PULse?')
    print(p)
    osc.write("TRIGger:MAIn:TYPe PULse")
    osc.write('TRIGger:MAIn:PULse:SOUrce EXT')
    osc.write('TRIGger:MAIn:PULse:WIDth:WHEN NOTEqual')
    
    t5 = time.perf_counter()
    osc.query('*opc?') # sync
    t6 = time.perf_counter()
    print(f'oscilloscope pulse set: {t6-t5:.3f} s')

def osc_acq(osc):
    osc.write('acquire:state 0') # stop
    osc.write('acquire:stopafter SEQUENCE') # single
    osc.write('acquire:state 1') # run

    t5 = time.perf_counter()
    r = osc.query('*opc?') # sync
    t6 = time.perf_counter()
    print(f'oscilloscope acquire time: {t6-t5:.3f} s')

def osc_data_transfer(osc):
    t7 = time.perf_counter()
    bin_wave = osc.query_binary_values('curve?', datatype='b', container=np.array)
    t8 = time.perf_counter()
    print(f'oscilloscope data transfer time: {t8-t7:.3f} s')
    return bin_wave

def osc_scale(osc):
    tscale = float(osc.query('wfmpre:xincr?'))
    tstart = float(osc.query('wfmpre:xzero?'))
    vscale = float(osc.query('wfmpre:ymult?')) # volts / level
    voff = float(osc.query('wfmpre:yzero?')) # reference voltage
    vpos = float(osc.query('wfmpre:yoff?')) # reference position (level)
    return tscale, tstart, vscale, voff, vpos

def osc_error(osc):
    # error checking
    r = int(osc.query('*esr?'))
    if r != 0:
        print('event status register: 0b{:08b}'.format(r))
    r = osc.query('allev?').strip()
    print(r)

def osc_get_data(osc,ch=1):
    record = osc_io(osc, ch)
    # acq config
    #osc_acq(osc)
    osc_pulse(osc)

    # data query
    data = osc_data_transfer(osc)

    # retrieve scaling factors
    tscale, tstart, vscale, voff, vpos = osc_scale(osc)
    print(f'tscale: {tscale}, tstart: {tstart}, vscale: {vscale}, voff: {voff}, vpos: {vpos}')

    osc_error(osc)

    # create scaled vectors
    # horizontal (time)
    total_time = tscale * record
    tstop = tstart + total_time
    scaled_time = np.linspace(tstart, tstop, num=record, endpoint=False)
    # vertical (voltage)
    unscaled_wave = np.array(data, dtype='double') # data type conversion
    scaled_wave = (unscaled_wave - vpos) * vscale + voff

    return scaled_time, scaled_wave 


# ----------------------------------------------
# Measurement functions
# ----------------------------------------------

def measure_no_field(
    freq_start: float,
    freq_stop: float,
    freq_points: int,
    resolution_bandwidth: int = 1000,
    sets: int = 100,
    avg=1, filename=None, plot=False):
    connections = connect()
    
    # Set the filename and create the file
    if filename == None:
        # Filename set to date and time YYYY-MM-DD_HH-MM-SS
        filename = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        filename += ".csv"
    file="C:/Users/Administrator/Desktop/cavity_magnonics_TP4/data/"+filename+".csv"
    if os.path.isfile(file):
        print("File already exists")
        x = input("Replace file? [y/n]")
        if x == "y":
            print("File replaced")
        else:
            print("Please choose another filename")
            return False
    # Set the VNA parameters
    znb = connections["znb"]
    set_start_freq(znb,freq_start)
    set_stop_freq(znb,freq_stop)
    set_sweep_points(znb,freq_points)
    set_measure(znb, "S12")
    set_resolution_bandwidth(znb, resolution_bandwidth)
    znb.write_str('INITiate1:CONTinuous 0')
    average(znb,avg)
    
    # Start the measurement
    with open(file, "w") as f:
        f.write('Time;Averages;Frequencies[Hz];Real Amp;Img Amp\n')
        for i in range(sets):            
            # Measure the VNA
            sleep(1)
            print(i)
            znb.write_str("INITiate:IMMediate:ALL")
            znb.query_opc(timeout=5*60*1000)
            
            freq_str=znb.query_str('CALC:DATA:STIM?')#get the string of the stimulus frequencies
            trace_str=znb.query_str('CALC1:DATA? SDAT')# the string is written as Re(S_point1),Im(S_point1),Re(S_point2)....
            time = znb.query_str('SYSTem:TIME?').replace(",",":")

            real = ','.join(trace_str.split(",")[0::2])
            img = ','.join(trace_str.split(",")[1::2])
            
            f.write(str(time)+";"+str(avg)+";"+str(freq_str)+";"+str(real)+";"+str(img)+"\n")

        
    close(connections)
    print("Measurement finished")
    print(f"Data saved in {file}")
    if plot:
        print("Plotting data...")
        viewer(filename)
    return True

def lockin_field_calibration(file = 'C:/Users/Administrator/Desktop/cavity_magnonics_TP4/code/calibration.txt', power=3):
    data = np.loadtxt(file, delimiter=";",)
    B = data[:,0]/10 # Convert to mT
    V = data[:,1]    # Voltage
    poly = np.polyfit( V,B, power)
    return np.poly1d(poly)


# Take a measurement
def measurement(
    freq_start: float, 
    freq_stop: float, 
    freq_points: int, 
    volt_start: float, 
    volt_stop: float, 
    volt_points: int,
    field_start: float= None, 
    field_stop: float = None, 
    field_points: int = None, 
    field_calibration = None,
    lockin_func = lockin_field_calibration,
    avg=1, filename=None, plot=False):
    
    connections = connect()

    # Set the filename and create the file
    if filename == None:
        # Filename set to date and time YYYY-MM-DD_HH-MM-SS
        filename = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        filename += ".csv"
    file="C:/Users/Administrator/Desktop/cavity_magnonics_TP4/data/"+filename+".csv"
    
    if os.path.isfile(file):
        print("File already exists")
        x = input("Replace file? [y/n]")
        if x == "y":
            print("File replaced")
        else:
            print("Please choose another filename")
            return False
    
    # Set the VNA parameters
    znb = connections["znb"]
    set_start_freq(znb,freq_start)
    set_stop_freq(znb,freq_stop)
    set_sweep_points(znb,freq_points)
    znb.write_str('INITiate1:CONTinuous 0')
    average(znb,avg)
    
    # Set the power supply parameters
    lockin = connections["lockin"]
    if field_calibration != None and field_start != None and field_stop != None and field_points != None:
        volt_start, volt_stop, volt_points = field_calibration(field_start, field_stop, field_points)
    volt_sweep = np.linspace(volt_start, volt_stop, volt_points)
    
    # Test the connection to the power supply
    coms = connections["coms"]
    connected = [check_connection(com) for com in coms]
    coms = [com for com, conn in zip(coms, connected) if conn]
    poly = lockin_field_calibration()
    print("Measuring...")
    # Start the measurement
    with open(file, "w") as f:
        f.write('Time;Display;Averages;Field[mT];Frequencies[Hz];Real Amp;Img Amp\n')
        for volt in tqdm(volt_sweep):
            # Set the voltage of the power supply
            
            for com in coms:
                set_voltage(com, volt)
            sleep(1)
            disps = get_display(coms[0])
            
            # Measure field from lock-in
            volt_ose = float(clean_term(lockin.query('MAG.')))
            field = poly(volt_ose)
            
            # Measure the VNA
            znb.write_str("INITiate:IMMediate:ALL")
            znb.query_opc(timeout=5*60*1000)
            
            freq_str=znb.query_str('CALC:DATA:STIM?')#get the string of the stimulus frequencies
            trace_str=znb.query_str('CALC1:DATA? SDAT')# the string is written as Re(S_point1),Im(S_point1),Re(S_point2)....
            time = znb.query_str('SYSTem:TIME?').replace(",",":")

            real = ','.join(trace_str.split(",")[0::2])
            img = ','.join(trace_str.split(",")[1::2])
            
            f.write(str(time)+";"+str(disps)+";"+str(avg)+";"+str(field)+";"+str(freq_str)+";"+str(real)+";"+str(img)+"\n")
            
        for com in coms:
            set_voltage(com, 1)
            set_current(com, 0.5)
        
    close(connections)
    print("Measurement finished")
    print(f"Data saved in {file}")
    if plot:
        print("Plotting data...")
        viewer(filename)
    return True

# Read the data from the file
def viewer(filename, x_axis = "field"): 
    file="C:/Users/Administrator/Desktop/cavity_magnonics_TP4/data/"+filename+".csv"
    isFile = os.path.isfile(file)
    if isFile:
        data = pd.read_csv(file,sep=';')
        time = data.Time.to_numpy()
        disp = data.Display.to_numpy()
        avg = float(data.Averages[0])
        field = data.Field.to_numpy()
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
        if x_axis == "time":
            x = time    
        else:
            x = field
        
        # Decide the sweep
        if x[0] > x[-1]:
            sweep = -1
        else:
            sweep = 1
        
        # Data analysis section
        # ----------------------------------------------
        # Find cavity resonance frequency
        # ----------------------------------------------
        min_left = np.argmin(amp[0])
        min_right = np.argmin(amp[-1])

        # ----------------------------------------------
        # Find the gradient of the anticrossing
        # ----------------------------------------------

        # Plot the data
        fig, ax = plt.subplots(figsize=(15,15))
        
        x_extent = [x[0], x[-1]]
        y_extent = [freq[0][0]/1e9, freq[0][-1]/1e9]
        extent = x_extent[::sweep] + y_extent
        
        plt.imshow(amp.T, cmap="plasma",extent=extent,aspect = 'auto',interpolation='nearest') #color is "plasma"
        #plt.plot([x[0], x[-1]], [cavity_freq, cavity_freq], 'w--', label=f"Cavity frequency: {cavity_freq/1e9:.3f} GHz")
        #plt.plot([-87.99,-85.5799],np.array([1.26215,1.31189]),'k--',label='Anticrossing')
        ylim = ax.get_ylim()
        #plt.plot(field, en1, 'k--', label='E1')
        #plt.plot(field, en2, 'k--', label='E2')
        ax.set_ylim(ylim)
        plt.xlabel("Field [mT]", fontsize=20)
        plt.ylabel("Frequency [GHz]", fontsize=20)
        ax.tick_params(axis='both', which='major', labelsize=15)
        plt.legend(fontsize=15)
        cbar = plt.colorbar()
        cbar.set_label("Amplitude [dB]", fontsize=20)
        cbar.ax.tick_params(labelsize=15)
        
        plt.show()
        #print(f"Cavity frequency: {cavity_freq/1e9:.3f} GHz")
        
        print("Data plotted")
        return 1
    else:
        print(f"File: {file} does not exist :/")
        return 0
    
def hall_probe_time():
    connections = connect()
    
    # Set the filename and create the file
    filename = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")    
    file="C:/Users/Administrator/Desktop/cavity_magnonics_TP4/data/"+filename+".csv"
    
    # No VNA, only measure voltage from lock-in with time
    # Switch field once in a while to see if it vaies with field as well
    
    poly = lockin_field_calibration()
    print("Measuring...")
    
    lockin = connections["lockin"]
    
    with open(file, "w") as f:
        f.write('Time;Field[mT];Voltage[V]\n')
        i = 0
        while True:
            # Measure field from lock-in
            volt_ose = float(clean_term(lockin.query('MAG.')))
            field = poly(volt_ose)
            
            # Measure the coms
            time = i*6
            
            f.write(str(time)+";"+";"+str(field)+";"+str(volt_ose)+"\n")
            
            sleep(6)
            i+=1
            if i > 7200:
                break   
    
    close()
    
    print("Measurement finished")


def measurement_no_lockin(
    freq_start: float, 
    freq_stop: float, 
    freq_points: int, 
    volt_start: float, 
    volt_stop: float, 
    volt_points: int,
    power: float = -5,
    band: float = 1e3,
    avg=1, filename=None, plot=False):
    
    connections = connect()

    # Set the filename and create the file
    if filename == None:
        # Filename set to date and time YYYY-MM-DD_HH-MM-SS
        filename = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        filename += ".csv"
    file="C:/Users/Administrator/Desktop/cavity_magnonics_TP4/data/"+filename+".csv"
    
    if os.path.isfile(file):
        print("File already exists")
        x = input("Replace file? [y/n]")
        if x == "y":
            print("File replaced")
        else:
            print("Please choose another filename")
            return False
    
    # Set the VNA parameters
    znb = connections["znb"]
    set_start_freq(znb,freq_start)
    set_stop_freq(znb,freq_stop)
    set_sweep_points(znb,freq_points)
    znb.write_str('INITiate1:CONTinuous 0')
    average(znb,avg)
    set_power(znb,power)
    set_resolution_bandwidth(znb,band)
    
    # Set the power supply parameters
    volt_sweep = np.linspace(volt_start, volt_stop, volt_points)

    # Test the connection to the power supply
    coms = connections["coms"]
    connected = [check_connection(com) for com in coms]
    coms = [com for com, conn in zip(coms, connected) if conn]
    
    print("Measuring...")
    # Start the measurement
    with open(file, "w") as f:
        f.write('Time;Display;Averages;Voltage;Frequencies;Real;Img\n')
        for volt in tqdm(volt_sweep):
            # Set the voltage of the power supply
            
            for com in coms:
                set_voltage(com, volt)
            disps = get_display(coms[0])
            
            # Measure the VNA
            znb.write_str("INITiate:IMMediate:ALL")
            znb.query_opc(timeout=5*60*1000)
            
            freq_str=znb.query_str('CALC:DATA:STIM?')#get the string of the stimulus frequencies
            trace_str=znb.query_str('CALC1:DATA? SDAT')# the string is written as Re(S_point1),Im(S_point1),Re(S_point2)....
            time = znb.query_str('SYSTem:TIME?').replace(",",":")

            real = ','.join(trace_str.split(",")[0::2])
            img = ','.join(trace_str.split(",")[1::2])
            
            f.write(str(time)+";"+str(disps)+";"+str(avg)+";"+str(volt)+";"+str(freq_str)+";"+str(real)+";"+str(img)+"\n")
            
        for com in coms:
            set_voltage(com, 1)
            set_current(com, 0.1)
        
    close()
    print("Measurement finished")
    print(f"Data saved in {file}")
    if plot:
        print("Plotting data...")
        viewer_no_lockin(filename)
    return True

def viewer_no_lockin(filename, x_axis = "field",save=True,display=True): 
    file="C:/Users/Administrator/Desktop/cavity_magnonics_TP4/data/"+filename+".csv"
    isFile = os.path.isfile(file)
    if isFile:
        data = pd.read_csv(file,sep=';')
        time = data.Time.to_numpy()
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
        if x_axis == "time":
            x = time    
        else:
            x = volt
        
        # Decide the sweep
        if x[0] > x[-1]:
            sweep = -1
        else:
            sweep = 1
        
        # Data analysis section
        # ----------------------------------------------
        # Find cavity resonance frequency
        # ----------------------------------------------
        min_left = np.argmin(amp[0])
        min_right = np.argmin(amp[-1])

        # ----------------------------------------------
        # Find the gradient of the anticrossing
        # ----------------------------------------------

        # Plot the data
        fig, ax = plt.subplots(figsize=(15,15))
        
        x_extent = [x[0], x[-1]]
        y_extent = [freq[0][0]/1e9, freq[0][-1]/1e9]
        extent = x_extent[::sweep] + y_extent
        
        plt.imshow(amp.T, cmap="plasma",extent=extent,aspect = 'auto',interpolation='nearest',origin='lower') #color is "plasma"
        #plt.plot([x[0], x[-1]], [cavity_freq, cavity_freq], 'w--', label=f"Cavity frequency: {cavity_freq/1e9:.3f} GHz")
        #plt.plot([-87.99,-85.5799],np.array([1.26215,1.31189]),'k--',label='Anticrossing')
        ylim = ax.get_ylim()
        #plt.plot(field, en1, 'k--', label='E1')
        #plt.plot(field, en2, 'k--', label='E2')
        ax.set_ylim(ylim)
        plt.xlabel("Voltage [V]", fontsize=20)
        plt.ylabel("Frequency [GHz]", fontsize=20)
        ax.tick_params(axis='both', which='major', labelsize=15)
        plt.legend(fontsize=15)
        cbar = plt.colorbar()
        cbar.set_label("Amplitude [dB]", fontsize=20)
        cbar.ax.tick_params(labelsize=15)
        if save==True:
            plt.savefig("C:/Users/Administrator/Desktop/cavity_magnonics_TP4/data/"+filename+".jpg") 
        if display==True:   
            plt.show()
        else:
            plt.close()
        #print(f"Cavity frequency: {cavity_freq/1e9:.3f} GHz")
        
        print("Data plotted")
        return 1
        print("Data plotted")
    else:
        print(f"File: {file} does not exist :/")
        return 0

def measurement_time(
    freq_start: float, 
    freq_stop: float, 
    freq_points: int, 
    volt: int,
    time_steps: int,
    dt: int,
    power: float = -5,
    band: float = 1e3,
    avg=1, filename=None, plot=False):
    
    connections = connect()

    # Set the filename and create the file
    if filename == None:
        # Filename set to date and time YYYY-MM-DD_HH-MM-SS
        filename = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        filename += ".csv"
    file="C:/Users/Administrator/Desktop/cavity_magnonics_TP4/data/"+filename+".csv"
    
    if os.path.isfile(file):
        print("File already exists")
        x = input("Replace file? [y/n]")
        if x == "y":
            print("File replaced")
        else:
            print("Please choose another filename")
            return False
    
    # Set the VNA parameters
    znb = connections["znb"]
    set_start_freq(znb,freq_start)
    set_stop_freq(znb,freq_stop)
    set_sweep_points(znb,freq_points)
    znb.write_str('INITiate1:CONTinuous 0')
    average(znb,avg)
    set_power(znb,power)
    set_resolution_bandwidth(znb,band)
    
    # Set the power supply parameters
    # Test the connection to the power supply
    coms = connections["coms"]
    connected = [check_connection(com) for com in coms]
    coms = [com for com, conn in zip(coms, connected) if conn]
    
    for com in coms:
        set_voltage(com, volt)
    disps = get_display(coms[0])
    
    print("Measuring...")
    # Start the measurement
    with open(file, "w") as f:
        f.write('Time;Temp;Display;Averages;Voltage;Frequencies;Real;Img\n')
        for volt in range(time_steps):            
            # Measure the VNA
            znb.write_str("INITiate:IMMediate:ALL")
            znb.query_opc(timeout=5*60*1000)
            
            freq_str=znb.query_str('CALC:DATA:STIM?')#get the string of the stimulus frequencies
            trace_str=znb.query_str('CALC1:DATA? SDAT')# the string is written as Re(S_point1),Im(S_point1),Re(S_point2)....
            #time = znb.query_str('SYSTem:TIME?').replace(",",":")

            real = ','.join(trace_str.split(",")[0::2])
            img = ','.join(trace_str.split(",")[1::2])
            
            ttime = datetime.datetime.now().strftime("%H-%M-%S")
            
            print(ttime)
            temp = input("Temperature at time {}: ".format(ttime))
            
            f.write(str(ttime)+";"+str(temp)+";"+str(disps)+";"+str(avg)+";"+str(volt)+";"+str(freq_str)+";"+str(real)+";"+str(img)+"\n")
            time.sleep(dt) 
        for com in coms:
            set_voltage(com, 1)
            set_current(com, 0.1)
        
    close()
    print("Measurement finished")
    print(f"Data saved in {file}")
    return True


