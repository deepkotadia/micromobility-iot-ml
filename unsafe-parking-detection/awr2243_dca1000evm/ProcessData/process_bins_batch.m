clear; clc;
close all;

file_ref_map = '../OriginalData/file_ref_map.csv';
output_base_path = '../PostProcessData';

T = readtable(file_ref_map);
table_size = size(T);
rows = table_size(1);

% Traverse through the ref map, process bins, generate heatmaps, save
for row = 1:rows
    id = T(row, :).Id{1};
    bin_filename = T(row, :).bin_Filename{1};
    log_filename = T(row, :).log_Filename{1};
    params_filename = '../OriginalData/parameters.m';
    scenario = T(row, :).Scenario{1};
    bin_input_path = ['../OriginalData' '/' scenario '/' bin_filename];
    log_input_path = ['../OriginalData' '/' scenario '/' log_filename];

    % Create dir for scenario
    heatmaps_scenario_out_dir = [output_base_path '/' scenario];
    if ~exist(heatmaps_scenario_out_dir, 'dir')
        mkdir(heatmaps_scenario_out_dir)
    end

    % Create dir for heatmaps
    heatmaps_out_dir = [heatmaps_scenario_out_dir '/' id];
    mkdir(heatmaps_out_dir)

    % Create heatmaps and save to dir
    raw_data_process(bin_input_path, log_input_path, params_filename, heatmaps_out_dir)

end
