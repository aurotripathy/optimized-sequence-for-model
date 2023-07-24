#### Steps to run the optimized version 

You'll need the ONNX file, `nxo_mobile_wb2.onnx`. Please make a request.

Steps are:

`./optmz_quantize.py --bs 1 --model_path nxo_mobile_wb2.onnx `

`./optmz_async_compile_run.py --bs 1 --dfg_path generated_nxo_mobile_wb2.dfg`


#### Environment
```
(tf) furiosa@demo-server-1:~/nexoptics/07-07-dfg$ pip list | grep furiosa
furiosa-cli                   0.9.1
furiosa-common                0.9.1
furiosa-litmus                0.9.1
furiosa-optimizer             0.9.1
furiosa-quantizer             0.9.1
furiosa-quantizer-impl        0.9.2
furiosa-registry              0.9.1
furiosa-runtime               0.9.1
furiosa-sdk                   0.9.0
furiosa-tools                 0.9.1
```
```
(tf) furiosa@demo-server-1:~/nexoptics/07-07-dfg$ dpkg -l | grep furiosa | awk '{print $2 "\t\t" $3}'
furiosa-compiler		0.9.0-3
furiosa-driver-pdma		1.9.0-3
furiosa-driver-warboy		1.9.0-3
furiosa-firmware-image		1.7.0
furiosa-firmware-tools		1.5.0-3
furiosa-libcompiler		0.9.1-3
furiosa-libhal-warboy		0.11.0-3
furiosa-libnux		0.9.1-3
furiosa-toolkit		0.10.2-3
```
