function windowData = getUntrimmedWindowData
% Sub-Function: Reads window data from 4 csv files. Writes each csv file in
% a cell of a cell array. 1 is amide peak, 2 phosphate peak, 3 phosphate 
% base and 4  is amide base.
    n = 4;  
    windowData = cell(1,n);
    for i = 1:n
        filename = strcat('Window', num2str(i), '.csv');
        windowData{i} = tdfread(filename, ',');
    end
end