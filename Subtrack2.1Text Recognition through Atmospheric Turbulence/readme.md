The following instructions describe how to run the code, which is used to process all distorted sequences：

The distorted_sequences folder contains distorted sequences, while the method folder includes the code for each step of the method as well as intermediate results.

Run the following Matlab and Python files in order. In each of the code files, you need to change the paths in the beginning of the file to point to your local drive. 

======================================================================================================

1-Select the sharpest frames：Run "method\step1-select_correct geometric distortion\files_frame_selection.m", and the results are in: "method\step1-select_correct geometric distortion\Intermediate results\sharpest_sequences"

2-Compute optical flow and image registration: Run "method\step1_correct geometric distortion\files_average_opticalflow.m", and the results are in: "method\step1-select_correct geometric distortion\Intermediate results\sharpest_registerted_sequences"

3-Remove blur using  wavelet-based fusion: Run "method\step2-waveletfusion\waveletfusion.m", and the results are in: "method\step2-waveletfusion\Intermediate results\waveletfusion_result"

4-Remove remianing artifacts: Run "method\step3-remove artifact\main_test_fbcnn_color_real.py", and the results are in: "\method\step3-remove artifact\test_results\Real_fbcnn_color"

Note that the final results will be in: "\method\step3-remove artifact\test_results\Real_fbcnn_color" 

======================================================================================================

Our code is designed to handle all sequences, so it may take some time to run. We apologize for any inconvenience this may cause.
