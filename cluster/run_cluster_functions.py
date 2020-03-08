# -*- coding: utf-8 -*-
#!/usr/bin/env python3

import subprocess
import time

vec_sub = list(range(12))

arch = []
# arch.append('net_sgd_000')
arch.append('net_sgd_001')
# arch.append('net_sgd_002')
# arch.append('net_sgd_003')
# arch.append('net_sgd_004')
# arch.append('net_sgd_005')
# arch.append('net_sgd_006')


# function = 'run_main_multiple_subject'
# hours = int(len(arch)*1.5)

function = 'run_main_across_subjects'
hours = int(12)

n_cpu = 4
memory = 6  # this is per CPU
do_submit = False
ix_config = 1

if hours > 12:
    hours = 12

sep = '/'
process0 = subprocess.Popen('pwd', stdout=subprocess.PIPE)
cwd = process0.communicate()[0].decode('UTF-8').rstrip()
if not cwd.split(sep)[-1] == 'cluster':
    raise Exception('We have to run this from the cluster directory in the repo.')
if not isinstance(vec_sub, list): vec_sub = [vec_sub]
for ix_sub in vec_sub:
    job_name = 's{:02d}'.format(ix_sub)

    g = {
        'g_account_g': 'dsi',
        'g_job_name_g': job_name,
        'g_c_g': '{}'.format(n_cpu),
        'g_time_g': '{hours:02}:{minutes:02}:00'.format(hours=hours, minutes=0),
        'g_mem_per_cpu_g': '{}gb'.format(memory),
        'g_arch_g': ' '.join(arch),
        'g_ix_sub_g': '{}'.format(ix_sub),
        'g_function_g': function,
        'g_ix_config_g': '{}'.format(ix_config),
    }

    fabs_input = cwd + sep + 'cluster_template.sh'
    f_output = 'b_{}.sh'.format(job_name)
    fabs_output = cwd + sep + f_output

    with open(fabs_input, 'r') as handle:
        filedata = handle.read()

    # Replace the target string

    for key in g:
        filedata = filedata.replace(key, g[key])

    with open(f_output, 'w') as handle:
        handle.write(filedata)

    if do_submit:
        time.sleep(2.5)
        bashCommand = 'sbatch {}'.format(f_output)
        process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
        output, error = process.communicate()
        time.sleep(12.5)

process = subprocess.Popen('pwd', stdout=subprocess.PIPE)
output = process.communicate()[0].decode('UTF-8')