import sys
import time
import os
import ctypes
import struct
import pandas as pd


TOTAL_BAR_LENGTH = 65.
last_time = time.time()
begin_time = last_time

def generate_kappa_schedule_MNIST():

    # kappa_schedule = 2400*[1] # warm-up phase
    # kappa_value = 1.0
    # step = 0.5/57600
    
    # for i in range(57600):
    #     kappa_value = kappa_value - step
    #     kappa_schedule.append(kappa_value)

    # for i in range(60000):
    #     kappa_schedule.append(0.5)
    kappa_schedule = 2400*[1] # warm-up phase
    for i in range(57600+60000):
        kappa_schedule.append(0.5)
    return kappa_schedule

def generate_epsilon_schedule_MNIST(epsilon_train):
    
    epsilon_schedule = 2400*[0]
    step = epsilon_train/12000
    
    for i in range(12000):
        epsilon_schedule.append(i*step) #ramp-up phase
    
    for i in range(45600+60000):
        epsilon_schedule.append(epsilon_train)
        
    return epsilon_schedule


def get_terminal_size():
    try:
        # Get the handle to the standard output
        h = ctypes.windll.kernel32.GetStdHandle(-12)
        # Create a buffer to store the console screen buffer information
        csbi = ctypes.create_string_buffer(22)
        # Retrieve the console screen buffer information
        ctypes.windll.kernel32.GetConsoleScreenBufferInfo(h, csbi)
        # Unpack the buffer into console screen buffer info fields
        (x, y, _, _, _, _, _, _, _, _, _) = struct.unpack("hhhhHhhhhhh", csbi.raw)
        return y, x
    except Exception as e:
        print(f"Error getting terminal size: {e}")
        return 25, 80  # default size

# Example usage
term_height, term_width = get_terminal_size()
#print(f"Terminal width: {term_width}, height: {term_height}")

term_width = int(term_width)


def format_time(seconds):
    days = int(seconds / 3600/24)
    seconds = seconds - days*3600*24
    hours = int(seconds / 3600)
    seconds = seconds - hours*3600
    minutes = int(seconds / 60)
    seconds = seconds - minutes*60
    secondsf = int(seconds)
    seconds = seconds - secondsf
    millis = int(seconds*1000)

    f = ''
    i = 1
    if days > 0:
        f += str(days) + 'D'
        i += 1
    if hours > 0 and i <= 2:
        f += str(hours) + 'h'
        i += 1
    if minutes > 0 and i <= 2:
        f += str(minutes) + 'm'
        i += 1
    if secondsf > 0 and i <= 2:
        f += str(secondsf) + 's'
        i += 1
    if millis > 0 and i <= 2:
        f += str(millis) + 'ms'
        i += 1
    if f == '':
        f = '0ms'
    return f


def progress_bar(current, total, msg=None):
    global last_time, begin_time
    if current == 0:
        begin_time = time.time()  # Reset for new bar.

    cur_len = int(TOTAL_BAR_LENGTH*current/total)
    rest_len = int(TOTAL_BAR_LENGTH - cur_len) - 1

    sys.stdout.write(' [')
    for i in range(cur_len):
        sys.stdout.write('=')
    sys.stdout.write('>')
    for i in range(rest_len):
        sys.stdout.write('.')
    sys.stdout.write(']')

    cur_time = time.time()
    step_time = cur_time - last_time
    last_time = cur_time
    tot_time = cur_time - begin_time

    L = []
    L.append('  Step: %s' % format_time(step_time))
    L.append(' | Tot: %s' % format_time(tot_time))
    if msg:
        L.append(' | ' + msg)

    msg = ''.join(L)
    sys.stdout.write(msg)
    for i in range(term_width-int(TOTAL_BAR_LENGTH)-len(msg)-3):
        sys.stdout.write(' ')

    # Go back to the center of the bar.
    for i in range(term_width-int(TOTAL_BAR_LENGTH/2)+2):
        sys.stdout.write('\b')
    sys.stdout.write(' %d/%d ' % (current+1, total))

    if current < total-1:
        sys.stdout.write('\r')
    else:
        sys.stdout.write('\n')
    sys.stdout.flush()


class DictExcelSaver:
    def __init__(self):
        pass

    @staticmethod
    def save(d, fp, sheet_name='Sheet1'):
        df = pd.DataFrame(data=d.values(), index=d.keys())
        df = df.T

        df.to_excel(fp, sheet_name=sheet_name, index=False)