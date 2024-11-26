#import pyvisa as visa
#rm = visa.ResourceManager()
import re

def clean_term(x):
    aux = repr(x)
    aux = re.sub('\\\\n\\\\.*', '', aux)[1:]
    return aux

def query_mag(id):
    '''
    Returns you the magitude of the voltage signal read by the Lock In
    '''
    return float(clean_term(id.query('MAG.')))

def query_xy(id):
    '''
    Returns the X and Y values measured by the Lock In
    '''
    aux = id.query('XY.')
    aux = aux.split(',')
    x = float(clean_term(aux[0]))
    y = float(clean_term(aux[1]))
    return x,y

def query_x(id):
    '''
    Returns you the signal read by the Lock In at the X channel
    '''
    aux = id.query('X.')
    
    return float(clean_term(aux))

def query_y(id):
    '''
    Returns you the signal read by the Lock In at the Y channel
    '''
    aux = id.query('Y.')
    return float(clean_term(aux))

def query_f(id):
    '''
    Returns you the frequency read by the Lock In
    '''
    aux = id.query('FRQ.')
    return float(clean_term(aux))

def query_phase(id):
    '''
    Returns you the Phase read by the Lock In
    '''
    aux = id.query('PHA.')
    return float(clean_term(aux))

def set_vmode(id,n):
    '''
    Set the mode of the voltage that is input by the lock in to:
    "GROUND": both A and B are grounded 0
    "A": measure A against ground 1
    "-B": measure B against ground 2 
    "A-B": measure A against B 3
    0<= n <= 3
    '''

    aux = ['GROUND', 'A', '-B', 'A-B']
    g = aux.index(n.upper())
    id.write(f'VMODE{g}')
    id.query('VER')
    return

def set_harmonic(id,n):
    '''
    Sets the harmonic in which the lock in measurements will be measured

    n = 1 : 1st harmonic
    n = 2: 2nd harmonic
    etc
    '''
    id.write(f'REFN{n}')
    id.query('VER')
    return

def auto_sensitivity(id):
    '''
    Sets the instrument to automatic sensitivity adjustment mode
    '''
    id.write('AS')
    id.query('VER')
    return

def set_v(id,v):
    '''
    Set the amplitude of the peak voltage
    max vpeak is ~ 7V.
    '''
    # v is in volts, but the command reads in mV
    id.write(f'OA.{v}')
    id.query('VER')
    return

def set_f(id,f):
    '''
    sets the frequency to f Hz
    '''
    id.write(f'OF.{f}')
    id.query('VER')
    return

def set_sen(id,n):
    '''
    Sets the sensitivity according to the table in page G-16 on the manual

    1 <= n < = 27
    '''
    id.write(f'SEN{n}')
    id.query('VER')
    return

def set_time_constant(id,n):
    '''
    Sets the time constant according to the table on page G-19 in the manual
    0 <= n <= 29
    '''
    id.write("TC "+ str(n))
    id.query('VER')
    
def set_phase(id,n):
    '''
    Sets the reference phase in degrees
    '''
    id.write("REFP."+str(n))
    id.query('VER')

def set_gain(id,n):
    '''
    Sets the gain, you can send either "AUTO" or the desired decade, i.e., 10 dB would be 1.
    '''
    if str(n).upper() == "AUTO":
        id.write("AUTOMATIC 1")
        id.query('VER')
    else:
        id.write("AUTOMATIC 0")
        id.write("ACGAIN"+str(n))
        id.query('VER')

def set_reference(id,ref):
    '''
    Sets the reference mode to either "INT", "EXT LOGIC"  or "EXT"
    '''
    aux = ["INT", "EXT LOGIC", "EXT"]
    id.write("IE "+str(aux.index(str(ref).upper())))
    id.query('VER')
    return

def DCCOUPLE(id, a):
    '''
    Sets the coupling to AC or DC

    Or reads the coupling
    '''
    if a == "?":
        return id.query('CP?')
    b = ["AC", "DC"]
    c = a.upper().strip()
    c = str(b.index(c))
    id.write('DCCOUPLE'+c)
    id.query('VER')

def status(id):
    count = 0
    t0 = time.time()
    while count < 10:
        N = id.query('ST')
        aux = bin(int(N)).replace('0b','')
        sleep(1)
        if aux == str(0):
            count += 1
        else:
            if time.time() - t0 > 10*60:
                return False
            auto_sensitivity(sr1)
            sleep(5)
            count = 0
    return False

def set_float(id,n):
    '''
    Sets either "FLOAT" or "GROUND"
    '''
    aux = ["GROUND", "FLOAT"]
    i = aux.index(n.upper())
    id.write("FLOAT "+str(i))
    id.query('VER')
    
def set_fet(id,n):
    '''
    Sets either "BIPOLAR" or "FET"
    '''
    aux = ["BIPOLAR", "FET"]
    i = aux.index(n.upper())
    id.write("FET "+str(i))
    id.query('VER')
