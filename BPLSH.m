% function Selected_Data_Index = BPLSH(Data,M,L,W =1)
%
% This is a function for selecting boundary points using Locality sensitive hashing (BPLSH).
%
% The algorithm is developed by Aslani and Seipel 2021 in: 
% "Mohammad Aslani, Stefan Seipel, Efficient and Decision Boundary Aware Instance Selection for Support Vector Machines, Information Sciences, 2021, ISSN 0020-0255"
% https://doi.org/10.1016/j.ins.2021.07.015.
% 
% The developed instance selection method (BPLSH) is applicable to large
% datasets and effectively balances reduction rate and classification
% accuracy. It preserves border patterns and a few interior data points to 
% accelerate the training phase of machine learning.
%
% In BPLSH, the nearest patterns belonging to opposite classes of
% a given instance are regarded as border samples and are preserved, 
% whereas instances that are far from opposite classes are considered as 
% interior ones and are removed. The closeness of
% instances to a given sample is measured by the similarity index (SI), 
% which is defined based on partitioning the feature space using locality sensitive hashing.

% It relies on the idea that a border instance has heterogeneous neighbors
% and is the nearest neighbor of a pattern from another class. Generally,
% BPLSH consists of two major phases: 1- identifying the buckets of each
% sample, and 2- finding border samples and eliminating dispensable instances. 

% In the first phase, each instance is assigned to a set of buckets by
% using a group of hash function families (Bucket_Index_Decimal_All). 

% The second phase is composed of six minor steps summarized as follows:
% I) Extract the neighbors of sample xi .
% II) Calculate the similarity index of the neighbors.
% III) If xi and its neighbors are homogeneous (they belong to the same class), save xi, remove
% the neighbors with SimIndex >=2 from the training dataset, and go to step VI.
% IV) If xi and its neighbors are heterogeneous and the nearest neighbors of xi from opposite
% classes are "quite close" to it, save xi and the nearest neighbors from each opposite class and
% go to step VI.
% V) If xi and its neighbors are heterogeneous and the nearest neighbors of xi from opposite
% classes are not quite close to it, save the nearest neighbors from each opposite class, remove
% the quite close neighbors to xi with class yi from the training dataset, and go to step
% VI.
% VI) Repeat steps I to V until all instances in the training dataset are either investigated
% or deleted.


% Inputs:
% Data is a matrix that its rows show the instances and columns show the
% features. For instance, if there are 1000 instances with 7 features, the
% dimension of the matrix is 1000 by 8 (one output).
% M is the number of hash functions (Hyperplanes) for partitioning the space
% L is the number of Layers of hash functions
% W is the bucket size, if Data is normalized between 0 and 1, W = 1 is
% fairly good.

%**************************************************************************
% Note: Data Matrix should include both input features (X) and output class (Y)
%**************************************************************************

% Example:
% M=30; L=10; W=1;
% Selected_Data_Index = BPLSH(Data,M,L,W);
% Selected_Data = Data(Selected_Data_Index, :);


function Selected_Data_Index = BPLSH(Data,M,L,W)

%----------Data Reduction----------%
%Normalizing the data between 0 and 1
maximum = max(Data(:,1:end-1));
minimum = min(Data(:,1:end-1));
maxmin = maximum-minimum;
maxmin(maxmin==0) = 1;
Data(:,1:end-1) = (Data(:,1:end-1) - minimum)./maxmin;


Dimension = size(Data(:,1:end-1),2); % Number of features
M; % Number of hash functions in each table
L; % Number of hash tables
W; % Bucket size


%s = rng; %Reset Random Number Generator
a = normrnd(0,1, [M*L , Dimension]); % Generate a in floor((ax+b)/W)
b = W.*rand(M*L,1); % Generate b in floor((ax+b)/W)

% Calculating the buckets of samples
% disp('Bucket Assigning');
Bucket_Index_Decimal_All = int32(zeros(L,size(Data(:,1:end-1),1)));
for i = 1:L
    j = (1+(i-1)*M):i*M;
    Bucket_Index = int16( floor( (a(j,:)*(Data(:,1:end-1))' + b(j,1))/W ) );
    BI = (Bucket_Index);
    %--For splitting BI matrix into PartsNo to make the search faster and GPU%
    Bucket_Index_uniqued = ([]);
    partsNo1 = 1;  % In the case of any error regarding memory, change partsNo1
    vectLength1 = size(BI,2);
    splitsize1 = 1/partsNo1*vectLength1;
    for ijj = 1:partsNo1
        idxs1 = [floor(round((ijj-1)*splitsize1)):floor(round((ijj)*splitsize1))-1]+1;
        Bucket_Index_uniqued = [Bucket_Index_uniqued, unique(  (  (BI(:,idxs1))  )' , 'rows'  )'];             % gpuArray can be used
    end
    Bucket_Index_uniqued  = (unique(Bucket_Index_uniqued', 'rows'))'                                     ;
    %--For splitting BI matrix into PartsNo to make the search faster and GPU%
    
    
    %--For splitting BI matrix into PartsNo to make the search faster and GPU%
    partsNo = 1; % In the case of any error regarding memory, change partsNo
    ss = 0;
    vectLength = size(BI,2);
    splitsize = 1/partsNo*vectLength;
    for ij = 1:partsNo
        idxs = [floor(round((ij-1)*splitsize)):floor(round((ij)*splitsize))-1]+1;
        BI_Part = (BI(:,idxs))                                                                                                 ; % gpuArray can be used here
        [~, Bucket_Index_Decimal]= ismember((BI_Part'), (Bucket_Index_uniqued'), 'rows');
        Bucket_Index_Decimal = int32(Bucket_Index_Decimal')                                                                  ;
        Bucket_Index_Decimal_All(i,ss+1:ss+size(Bucket_Index_Decimal,2)) = gather(Bucket_Index_Decimal)                     ;
        ss = ss + size(Bucket_Index_Decimal, 2);
    end
    %---For splitting BI matrix into PartsNo to make the search faster and GPU%
    
    
    % Only for memory
    BI = [];
    BI_Part = [];
    Bucket_Index_uniqued = [];
    Bucket_Index_Decimal = [];
end

if (max(max(Bucket_Index_Decimal_All)) < 32767)
    Bucket_Index_Decimal_All = int16(Bucket_Index_Decimal_All)                                          ;
end


% Instance Selection
%disp('Data Slaughtering Step...');
iii = int32(1);
I = int32(1:size(Bucket_Index_Decimal_All,2))                                                            ;
EP = int32([]);
Bucket_Index_Decimal_All = (Bucket_Index_Decimal_All);
Classes = (Data(:,end));
TRS = size(Bucket_Index_Decimal_All,2)+1; Temporal_Removed_Samples = TRS;
Point_Extent = [];
Samples_OppositeClass_NearBoundary = [];
SI = 1;

while iii<numel(I)
    
    Current_Sample_Bucket_Index_Decimal = Bucket_Index_Decimal_All(:, iii)                                ;
    Bucket_Index_Decimal_All(:, iii)  = -1;
    Number_of_Common_Buckets = sum((Bucket_Index_Decimal_All - Current_Sample_Bucket_Index_Decimal)==0,1);
    Index_Neighbors = Number_of_Common_Buckets>0;
    Frequency_Neighbors = (Number_of_Common_Buckets(Index_Neighbors))'                               ;
    uniqued_Neighbors = (I(Index_Neighbors))'                                                           ;
    Bucket_Index_Decimal_All(:, iii)  = Current_Sample_Bucket_Index_Decimal;
    
    
    Class_Neighbors = Classes(uniqued_Neighbors)' ;
    Class_Current = Classes(I(iii))      ;
    if (sum(diff(sort([Class_Neighbors Class_Current]))~=0)+1>1)
        %disp('Fully Mixed')
        %break;
        
        Classes_Neighbors_Unique = unique(Class_Neighbors)                                                                                  ;
        
        OppositeClasses = (Classes_Neighbors_Unique (Classes_Neighbors_Unique~=Class_Current))'                                               ;
        SI_OppositeClasses = (Class_Neighbors == OppositeClasses).*repmat(Frequency_Neighbors',numel(OppositeClasses),1)                     ;
        Maximum_SI_OppositeClasses  = max(SI_OppositeClasses,[], 2)                                                                    ;
        Samples_OppositeClass_NearBoundary = [Samples_OppositeClass_NearBoundary; uniqued_Neighbors( any(SI_OppositeClasses == Maximum_SI_OppositeClasses, 1) )]         ;
        
        if max(Maximum_SI_OppositeClasses) >= 1.0*L
            Samples_OppositeClass_NearBoundary = [Samples_OppositeClass_NearBoundary; I(iii)];
        else
            Very_Close_Samples_SI =  (int32(uniqued_Neighbors(Frequency_Neighbors >= 1.0*L)))';
            Temporal_Removed_Samples = [Temporal_Removed_Samples Very_Close_Samples_SI];
            %--------------Just for making the algorithm fast--------------%
            % it causes not to call "ismember" each time. calling ismemebr each
            % time needs 0.05 or 0.04 sec. But calculating min is faster.
            if ( min(Temporal_Removed_Samples) <= I(iii+1) || numel(EP) > 2000)
                [aa, ~] = ismember(I,[Temporal_Removed_Samples EP]);
                I(aa) = []                                                                                           ;
                Bucket_Index_Decimal_All(:,aa) = []                                                                 ;
                Temporal_Removed_Samples = TRS;
                iii = iii - numel(EP)                                                                             ;
                EP = [];
                %numel(I)
            end
            %--------------Just for making the algorithm fast--------------%
        end
    else
        %disp('Unmixed')
        Very_Close_Samples_SI =  (int32(uniqued_Neighbors(Frequency_Neighbors >= SI + 1)))';
        Temporal_Removed_Samples = [Temporal_Removed_Samples Very_Close_Samples_SI];
        EP = [EP I(iii)];
        Point_Extent = [Point_Extent I(iii)];
        %--------------Just for making the algorithm fast--------------%
        % it causes not to call "ismember" each time. calling ismemebr each
        % time needs 0.05 or 0.04 sec. But calculating min is faster.
        if ( min(Temporal_Removed_Samples) <= I(iii+1) || numel(EP) > 2000)
            [aa, ~] = ismember(I,[Temporal_Removed_Samples EP]);
            I(aa) = []                                                                                           ;
            Bucket_Index_Decimal_All(:,aa) = []                                                                 ;
            Temporal_Removed_Samples = TRS;
            iii = iii - numel(EP)                                                                             ;
            EP = [];
            %numel(I)
        end
        %--------------Just for making the algorithm fast--------------%
    end
    iii= iii+1;
end

%----------Data Reduction----------%
Selected_Data_Index = unique([Point_Extent Samples_OppositeClass_NearBoundary']');

