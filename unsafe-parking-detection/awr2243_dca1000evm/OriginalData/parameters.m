% MIMO and data
nTx = 3;
nLanes = 4;
isReal = 0;
nADCBits = 16;
% EM property
c = 3e8;
Fc = 77e9;
slope = 60.012e12;
idleTime = 50e-6;
adcStartTime = 6e-6;
rampEndTime = 50e-6;
Fs = 10000e3;
nSamples = 256;
% Frame property
nChirps = 2;
Tp = 100e-3;