import os
import sys
import time

cmd = 'bash scripts/train.sh'


def gpu_info():
    gpu_status = os.popen('nvidia-smi | grep %').read().split('|')
    gpu_memory = int(gpu_status[2].split('/')[0].split('M')[0].strip())
    gpu_power = int(gpu_status[1].split('   ')[-1].split('/')[0].split('W')[0].strip())
    return gpu_power, gpu_memory


def narrow_setup(interval=2):
    gpu_power, gpu_memory = gpu_info()
    i = 0
    ok_count = 0
    while gpu_memory > 1000 or gpu_power > 40 or ok_count < 5:  # set waiting condition
        gpu_power, gpu_memory = gpu_info()
        if gpu_memory <= 1000 and gpu_power <= 40:
            ok_count += 1
        else:
            ok_count = 0
        i = i % 5
        symbol = 'monitoring: ' + '>' * i + ' ' * (10 - i - 1) + '|'
        gpu_power_str = 'gpu power:%d W |' % gpu_power
        gpu_memory_str = 'gpu memory:%d MiB |' % gpu_memory
        ok_count_str = 'ok count:%d | ' % ok_count
        sys.stdout.write('\r' + gpu_memory_str + ' ' + gpu_power_str + ' ' + \
            ok_count_str + ' ' + symbol)
        sys.stdout.flush()
        time.sleep(interval)
        i += 1
    print('\n' + cmd)
    os.system(cmd)


if __name__ == '__main__':
    narrow_setup()