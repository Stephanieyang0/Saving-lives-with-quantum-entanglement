function digistainIndex = getDigistainIndex(wavenumbers, cellDataSet, windowData, sampleNumber)
% Sub-Function: Uses "interpolateData" to find same length spectra then calculates 
% transmittance of sample and window and multiplies them for each window.
% Uses results to find DI.
    cellData = cellDataSet{sampleNumber};
    n = 4;
    interpolatedWindowData = cell(1,n);
    interpolatedCellData = cell(1,n);
    interpolatedWavenumbers = cell(1,n);
    output = cell(1,n);
    for i = 1:n 
        %% Interpolate spectra 
        [interpolatedWavenumbers{i}, interpolatedCellData{i}, interpolatedWindowData{i}] = interpolateData(wavenumbers{sampleNumber}, cellData, windowData, i);
        %% Calculate cell transmittance
        interpolatedCellData{i}.TRN = 10.^(-interpolatedCellData{i}.ABS);
        %% Calculate window transmittance
        interpolatedWindowData{i}.TRN = 10.^(-interpolatedWindowData{i}.ABS);
        %% Multiply spectra
        output{i} = interpolatedCellData{i}.TRN .* interpolatedWindowData{i}.TRN;
    end
    %% Integrate under curve
    amidePeakIntegratedIntensity = sum(output{1});
    phosphatePeakIntegratedIntensity = sum(output{2});
    phosphateBaseIntegratedIntensity = sum(output{3});
    amideBaseIntegratedIntensity = sum(output{4});
    %% Get peak heights DI = (A - B) / (C - D)
    A = log10(amidePeakIntegratedIntensity);
    B = log10(amideBaseIntegratedIntensity);
    C = log10(phosphatePeakIntegratedIntensity);
    D = log10(phosphateBaseIntegratedIntensity);
    %% Digistain Index (Amide/Phosphate)
    digistainIndex = (A - B) ./ (C - D);
end