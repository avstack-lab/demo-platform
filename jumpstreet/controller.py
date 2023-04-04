"""
Author: Nate Zelter
Date: April 2023

"""


import os
import subprocess



### data_broker
"""
    .PHONY: data_broker
    data_broker: $(INSTALL_STAMP)
            $(POETRY) run python jumpstreet/broker.py \
                lb_with_xsub_extra_xpub --frontend 5550 \
                --backend 5551 --backend_other 5552 --verbose
"""
subprocess.run(['pwd'])
cmd = ['poetry', 'run']
cmd.extend(('python3', 'jumpstreet/broker.py'))
cmd.extend(('lb_with_xsub_extra_xpub', '--frontend', '5550'))
cmd.extend(('--backend', '5551', '--backend_other', '5552', '--verbose'))
result = subprocess.Popen(cmd)
result.wait()

