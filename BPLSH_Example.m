clc;
clear all;
close all;
format long g;

%% Reading Data
Data = dlmread('SampleData.txt');

%% Simple Example

M = 100; 
L = 40; 
W = 1;
Selected_Data_Index = BPLSH(Data, M, L, W);
Selected_Data = Data(Selected_Data_Index, :);

figure,
scatter(Data(:,1), Data(:,2), [], Data(:,3), 'filled')
title('Original Datatset')

figure,
scatter(Selected_Data(:,1), Selected_Data(:,2), [], Selected_Data(:,3), 'filled')
title('Selected Dataset')


%%
disp('Calculating Tables 1 and 2 in the journal paper')

TimeC = [];
Preserved_Size = [];
W = 1;
L_Vector = [10 20 30];
M_Vector = [20 60 100];
% For measuring Time and preservation rate
for iteration = 1:10
    iteration
    Counter1 = 0;
    for L = L_Vector
        Counter1  = Counter1 + 1;
        Counter2 = 0;
        for M = M_Vector
            Counter2  = Counter2 + 1;
            tic
            Selected_Index = BPLSH(Data,M,L,W);
            TimeC(Counter1, Counter2, iteration) = toc;
            Preserved_Size(Counter1, Counter2, iteration) = 100 * numel(Selected_Index)/size(Data,1);
        end
    end
end
TimeC_Average = mean(TimeC, 3)
Preserved_Size_Average = mean(Preserved_Size, 3)



