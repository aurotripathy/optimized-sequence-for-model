#### Steps to run the optimized version 

You'll need the ONNX file, `nxo_mobile_wb2.onnx`. Please make a request.

Steps are:

`./optmz_quantize.py --bs 1 --model_path nxo_mobile_wb2.onnx `

`./optmz_async_compile_run.py --bs 1 --dfg_path generated_nxo_mobile_wb2.dfg`


