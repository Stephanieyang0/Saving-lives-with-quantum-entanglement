function [interpolatedWavenumber, interpolatedCellData, interpolatedWindowData] = interpolateData(wavenumbers, cellData, windowData, windowNumber)
% Sub-Function: Interpolates spectra over same ranges for both cells and
% windows
%% Find lower bound for wavenumber overlap between cells and windows
minWVNM = max(min(windowData{windowNumber}.wavenumber),min(wavenumbers));
%% Find upper bound for wavenumber overlap between cells and windows
maxWVNM = min(max(windowData{windowNumber}.wavenumber),max(wavenumbers));
%% Interpolate window data
interpolatedWavenumber = linspace(minWVNM,maxWVNM,10001);
originalWindowWavenumber = windowData{windowNumber}.wavenumber;
interpolatedWindowData.ABS = interp1(originalWindowWavenumber, windowData{windowNumber}.ABS,interpolatedWavenumber);
%% Interpolate cell data
originalCellWavenumber = wavenumbers;
interpolatedCellData.ABS = interp1(originalCellWavenumber, cellData,interpolatedWavenumber);
end