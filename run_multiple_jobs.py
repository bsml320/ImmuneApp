import os
import subprocess
from multiprocessing import Pool

def Job(jd):
    command = jd['command']
    working_directory = jd['working_directory']
    id = jd['id']
    returncode = None
    stdout = ''
    stderr = ''

    if working_directory is not None:
        os.chdir(working_directory)

    command = command.split(' ') if isinstance(command, str) else command
    p = subprocess.Popen(command, stderr=subprocess.PIPE, stdout=subprocess.PIPE)
    stdout, stderr = p.communicate()
    returncode = p.returncode
    return returncode

def run_multiple_processes(jds, n_processes):
    pool = Pool(n_processes)
    returns = pool.map(Job, jds)
    pool.close()
    return returns