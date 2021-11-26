# micromobility-iot-ml
Using Sensor Data and Machine Learning to Solve Problems in the Micromobility Transportation Space

## Generate Heatmaps for Unsafe Parking Detection
### Setup
1. Download radar raw data files (.bin and .txt) for all scenarios from the [Google Drive here](https://drive.google.com/drive/folders/1YWizSIF95OdjgWbZH3o0JDlbs2yZyrIL).
2. Put the downloaded scenario directories under the _OriginalData_ dir.
3. Create a new dir named _PostProcessData_ under the _awr2243_dca1000evm_ dir.
4. Run the _process_bins_batch.m_ MATLAB script. The heatmaps will be generated and saved under _PostProcessData_.
