function raw_data_process(binFilePath, logFilePath, paramsFilePath, heatmapsOutDir)

% clear; clc;
% close all;

if ~exist(heatmapsOutDir, 'dir')
    mkdir(heatmapsOutDir)
end

targetName = ['adc_data'];

logFile = logFilePath;

disp(logFile);

dataFile = binFilePath;
paraFile = paramsFilePath;
postFile = [targetName '_post.mat'];
sizeFile = [targetName '_postsize.mat'];
trackFile = [targetName '_track.mat'];

%% Load parameters
if ~isfile(paraFile)
    error('Parameter file does not exist!');
else
    run(paraFile);
end

%% Compute variables
% Range resolution
Lc = c / Fc;
adcSampleTime = nSamples / Fs;
bw = adcSampleTime * slope;
F1 = Fc + slope * adcStartTime;
F2 = F1 + bw;
dRes = c / (2*bw);
dMax = dRes * nSamples;
% Velocity resolution
Tc = idleTime + rampEndTime;
Tf = nChirps * Tc;
vMax = Lc / (4*Tc);
vRes = Lc / (2*Tf);
% Angle resolution
dRx = Lc / 2;
phaseFFT = 256;

% resolution
dRange = 0:dRes:dMax-dRes;
vRange = -vMax:vRes:vMax-vRes;
vNzRange = vRange(vRange ~= 0);
phaseRes = 2*pi / phaseFFT;
phaseRange = -pi:phaseRes:pi-phaseRes;
thetaRange = asin(phaseRange/(2*pi) * Lc/dRx);

%% Raw data process
if ~isfile(postFile)
    %%
    fprintf('Post process start.\n'); tic;
    if isfile(dataFile)
        adcData = readDCA1000(dataFile);
    else
        dataFilePre = [targetName];
        if ~isfile([dataFilePre '_0.bin'])
            dataFilePre = [targetName '_Raw'];
        end
        
        adcData = [];
        fileIndex = 0;
        dataFileTemp = [dataFilePre '_' num2str(fileIndex) '.bin'];
        while isfile(dataFileTemp)
            adcDataTemp = readDCA1000(dataFileTemp);
            adcData = [adcData adcDataTemp];
            fileIndex = fileIndex + 1;
            dataFileTemp = [targetName '_' num2str(fileIndex) '.bin'];
        end
        clear('fileIndex', 'dataFileTemp', 'adcDataTemp');
    end
    adcData = reshape(adcData.', nSamples, nTx, nChirps, [], nLanes);
    if nTx == 3
        adcData(:, 3, :, :, :) = [];
        nTx = 2;
    end
    nFrames = size(adcData, 4);

    % Most probable velocity at each location
    vMxv = zeros(nSamples, phaseFFT, nFrames);
    vIdx = zeros(nSamples, phaseFFT, nFrames);
    % Most probable non-zero velocity at each location
    vNzMxv = zeros(nSamples, phaseFFT, nFrames);
    vNzIdx = zeros(nSamples, phaseFFT, nFrames);
    % Static objects at each location
    vZMxv = zeros(nSamples, phaseFFT, nFrames);
    for iFrame = 0:nFrames-1
        rxData = permute(adcData(:, :, :, iFrame+1, :), [1 3 5 2 4]);
        rxData = reshape(rxData, nSamples, nChirps, nTx * nLanes);
        angleData = fft(rxData, nSamples, 1);
        angleData = fftshift(fft(angleData, nChirps, 2), 2);
        angleData = fftshift(fft(angleData, phaseFFT, 3), 3);
        [vMxv(:,:,iFrame+1), vIdx(:,:,iFrame+1)] = max(angleData, [], 2);
        [vNzMxv(:,:,iFrame+1), vNzIdx(:,:,iFrame+1)] = max(angleData(:,vRange~=0,:), [], 2);
        vZMxv(:,:,iFrame+1) = squeeze(angleData(:,vRange==0,:));
    end
    fprintf('Post process end. '); toc;

    clear('adcData', 'rxData', 'angleData');

    vMxv = single(vMxv);
    vIdx = single(vIdx);
    vNzMxv = single(vNzMxv);
    vNzIdx = single(vNzIdx);
    vZMxv = single(vZMxv);
    save(postFile, 'vMxv', 'vIdx', 'vNzMxv', 'vNzIdx', 'vZMxv');

    if ~isfile(sizeFile)
        save(sizeFile, 'nFrames', 'phaseFFT');
    end
else
    fprintf('Load post process file.\n');
    load(postFile);
    load(sizeFile);
    phaseFFT = size(vMxv, 2);
    phaseRes = 2*pi / phaseFFT;
    phaseRange = -pi:phaseRes:pi-phaseRes;
    thetaRange = asin(phaseRange/(2*pi) * Lc/dRx);
end

%% Radar heatmap process
% rotate
thetaRange2 = pi/2 - thetaRange;
% to dB
vMxvdb = mag2db(abs(vMxv));
vZMxvdb = mag2db(abs(vZMxv));
vNzMxvdb = mag2db(abs(vNzMxv));

if 1
    %% pcolor heatmap
    % scale to range of interest
    dShow = 30;
    % pol2cart
    angleDataX = dRange.' * cos(thetaRange2);
    angleDataY = dRange.' * sin(thetaRange2);

    figure('Position', [100 100 2000 1000]);
    % all velocity
    h1_hax = subplot(2,2,1);
    h1 = pcolor(angleDataX, angleDataY, vMxvdb(:,:,1));
%             imshow(h1, 'InitialMagnification', 'fit')
%             print -djpeg matlabfigure;
    set(h1, 'EdgeColor', 'none');
    axis equal;
    xlim([-dShow dShow]);
    ylim([0 dShow]);
    caxis([min(vMxvdb, [], 'all'), max(vMxvdb, [], 'all')]);
    t1 = title('');
    
    % static objects
    h2_hax = subplot(2,2,2);
    h2 = pcolor(angleDataX, angleDataY, vZMxvdb(:,:,1));
    set(h2, 'EdgeColor', 'none');
    axis equal;
    xlim([-dShow dShow]);
    ylim([0 dShow]);
    caxis([min(vZMxvdb, [], 'all'), max(vZMxvdb, [], 'all')]);
    title('Static objects');

    % dynamic objects
    h3_hax = subplot(2,2,3); 
    hold on;
    h3 = pcolor(angleDataX, angleDataY, vNzMxvdb(:,:,1));
    set(h3, 'EdgeColor', 'none');
    axis equal;
    xlim([-dShow dShow]);
    ylim([0 dShow]);
    caxis([min(vNzMxvdb, [], 'all'), max(vNzMxvdb, [], 'all')]);
    title('Dynamic objects');

    if exist('detections', 'var')
        h3d = scatter(detections{1}(1,:), detections{1}(2,:), 'wx', ...
            'LineWidth', 2);
    end
%     if exist('positions', 'var')
%         h3p = scatter(positions{1}(1,:), positions{1}(2,:), 'w+', ...
%             'LineWidth', 2);
%     end

    for iFrame = 0:nFrames-1
        t1.String = [num2str(iFrame) ' frame, ' num2str(Tp*iFrame) ' sec'];
        h1.CData = vMxvdb(:,:,iFrame+1);
        h2.CData = vZMxvdb(:,:,iFrame+1);
        h3.CData = vNzMxvdb(:,:,iFrame+1);
        if exist('h3d', 'var')
            h3d.XData = detections{iFrame+1}(1,:);
            h3d.YData = detections{iFrame+1}(2,:);
        end
        if exist('h3p', 'var')
            h3p.XData = positions{iFrame+1}(1,:);
            h3p.YData = positions{iFrame+1}(2,:);
        end
        pause(Tp);
        
        % Save plots as PNG
        h1_fig = figure('Visible','off');
        h1_hax_new = copyobj(h1_hax, h1_fig);
        set(h1_hax_new, 'Position', get(0, 'DefaultAxesPosition'));
        axis off
        title '' Visible off
        exportgraphics(h1_fig, [heatmapsOutDir '/' 'h1_' num2str(iFrame) '.png'])

        h2_fig = figure('Visible','off');
        h2_hax_new = copyobj(h2_hax, h2_fig);
        set(h2_hax_new, 'Position', get(0, 'DefaultAxesPosition'));
        axis off
        title '' Visible off
        exportgraphics(h2_fig, [heatmapsOutDir '/' 'h2_' num2str(iFrame) '.png'])

        h3_fig = figure('Visible','off');
        h3_hax_new = copyobj(h3_hax, h3_fig);
        set(h3_hax_new, 'Position', get(0, 'DefaultAxesPosition'));
        axis off
        title '' Visible off
        exportgraphics(h3_fig, [heatmapsOutDir '/' 'h3_' num2str(iFrame) '.png'])
    end

%             writerObj = VideoWriter([outfolderName '/' targetName '.mp4'], 'MPEG-4');
%             writerObj.FrameRate = 1/Tp;
%             open(writerObj);
%             for iFrame = 0:nFrames-1
%                 t1.String = [num2str(Tp*iFrame) ' sec'];
%                 h1.CData = vMxvdb(:,:,iFrame+1);
%                 h2.CData = vZMxvdb(:,:,iFrame+1);
%                 h3.CData = vNzMxvdb(:,:,iFrame+1);
%                 writeVideo(writerObj, getframe(gcf));
%             end
%             close(writerObj);

    % Cleanup
    delete(postFile);
    delete(sizeFile);
end