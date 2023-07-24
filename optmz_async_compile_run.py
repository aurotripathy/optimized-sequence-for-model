#!/usr/bin/env python3
"""
run it as:
./optmz_async_compile_run.py --bs 1 --model_path nxo_mobile_wb2.dfg
"""

import logging
import os
from furiosa import runtime
from furiosa.runtime import session
import numpy as np
import time

LOGLEVEL = os.environ.get('FURIOSA_LOG_LEVEL', 'INFO').upper()
logging.basicConfig(level=LOGLEVEL)

import argparse

parser = argparse.ArgumentParser(
                    prog='async_compile_run.py',
                    description='compile and run',
                    epilog='look for inferences/sec')

parser.add_argument('--bs', required=True, type=int)
parser.add_argument('--dfg_path', required=True, type=str)
args = parser.parse_args()

nb_batches = 10000

def run_async():
    runtime.__full_version__
    compiler_config = { "permute_input": [[0, 3, 1, 2]] }
    submitter, queue = session.create_async(str(args.dfg_path),
                                            worker_num=16,
                                            compiler_config=compiler_config,
                                            input_queue_size=nb_batches * args.bs, # requests you can submit without blocking
                                            output_queue_size=nb_batches * args.bs)
    print(f'input: {submitter.inputs()[0]}')
    print(f'output: {submitter.outputs()[0]}')    
    
    input_tensor = submitter.inputs()[0]

    # Create input tensors
    inputs = []
    for i in range(0, nb_batches):
        inputs.append(np.random.randint(0, 256, input_tensor.shape, dtype=np.uint8))


    tic = time.perf_counter()
    # Submit the inference requests asynchronously
    for idx, input in enumerate(inputs):
        submitter.submit(input, context=idx)
    
    # Receive the results asynchronously
    #context_list = []
    #output_list = []
    for i in range(0, nb_batches):
        context, outputs = queue.recv(400) # provide timeout param. If None, queue.recv() will be blocking.
        #context_list.append(context)
        #output_list.append(outputs.numpy())  # https://github.com/furiosa-ai/furiosa-sdk-private/issues/439)
    
    toc = time.perf_counter()
    
    # housekeeping
    if queue:
        queue.close()
    if submitter:
        submitter.close()
    
    print(f"Completed {nb_batches} batches of inference with batch size {args.bs} in {toc - tic:0.4f} seconds")
    print(f"Time per image: {(toc - tic) / nb_batches:0.4f} seconds")
    print(f'Model inference through-put: {args.bs * (1 / ((toc - tic) / nb_batches)):0.3f} inferences/sec')

if __name__ == "__main__":
    run_async()
