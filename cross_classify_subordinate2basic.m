%Andrew Bauer
%092715

clear all
close all

%% set up

spm('defaults','fmri');

addpath /usr/cluster/software/ccbi/neurosemantics/CCBI4.0/
addpath /usr/cluster/software/ccbi/neurosemantics/CCBI4.0/Utils/
addpath /usr/cluster/software/ccbi/neurosemantics/CCBI4.0/fmri_core_new/
addpath /usr/cluster/software/ccbi/neurosemantics/CCBI4.0/fmri_core_new/Utils

mFile='detrend.MPSC_wraf336_ALL_w69.mat';
dlmReadIn = '/scratch/subordinate_basic/CmatFiles/SPM_paradigm_bas2SubMean_crossCls_newCmat.txt';

subjPool = {'04305S', '04307S', '04315S', '04337S','04342S', '04364S', '04365S', '04381S', '04390S', '04394S'};

%% set voxel number, mask, and classifier parameters

voxNoPool = [20 30 40 50 60 70];

write_racc_flag = 1;

mask_pool = textread('./AAL_ROIs_WSA_presMeans_wFolds','%s','delimiter','\n');

nonWL_mask = zeros(size(mask_pool)); %indicate if nonWL mask 0-no 1-yes
nonWL_mask(51:62) = 1;
maskLoc0 = '/usr/cluster/projects3/thirty_mammals/analysis/ROIs/adpted_sourceAAL/indivAALROIs';
maskLoc1_nonWL_prefix = '/usr/cluster/projects3/subordinate_basic/analysis/ROIs';

useImgMask = 1;

%classifierString='nbayesPooled';
%classifierString='svmlight';
classifierString='logisticRegression';

if strcmp(classifierString,'svmlight')
    addpath /usr/cluster/home/wang/methods/SVM/SVM/bin /usr/cluster/home/wang/methods/SVM/SVM/matlab
    c=1;classifierParameters{c} = {'linear',[],'crossValidation'}; 
end

%% set the single typical and atypical ind for each category

typIndEachCat = [2 1 1 1 2];
%oak, robin, bass, taxi, sneakers

atypIndEachCat = [3 2 3 2 4];
%palm, woodpecker, minnow, limousine, sandals

subTypePool = {'typical', 'atypical'};

%% lastly, specify feature/train/test parameters

trainCat_i = 3; %train and test can't be same, folding here assumes diff datasets
testCat_i = 2;
featureFromCat_i = trainCat_i; %right now, train and feat cat have to be same

noTestItems2Avg = 2;

%% go

for subType_i = 1:numel(subTypePool)
    subType = char(subTypePool(subType_i));
    
    if strcmp(subType, 'typical')
        subIndEachCat = typIndEachCat;
    elseif strcmp(subType, 'atypical')
        subIndEachCat = atypIndEachCat;
    end
    
    racc_subjXmaskXVox_subType = nan(numel(subjPool), size(mask_pool, 1), numel(voxNoPool));

    for mask_i = 1:size(mask_pool,1)
        mask_ID = char(mask_pool(mask_i, 1));
        
        if nonWL_mask(mask_i)
            maskLocation = strcat(maskLoc1_nonWL_prefix, '/nonWLMask_from_', mask_ID);
        else
            maskLocation = maskLoc0;
        end
        
        for voxNo_i = 1:numel(voxNoPool)
            topNStableVox = voxNoPool(voxNo_i);

            for sbj_i = 1:numel(subjPool)
                sbj = char(subjPool(sbj_i));
                
                if nonWL_mask(mask_i)
                    mask_ID_sbj = strcat('nonWLMask_from_', mask_ID, '_', sbj);
                else
                    mask_ID_sbj = mask_ID;
                end
                
                %load subj data        
                inpf = sprintf('/usr/cluster/projects3/subordinate_basic/normalMpsc/%s/%s',sbj,mFile);
                S = load(inpf);
                subj_voxXYZ_beforeTopN = S.meta.colToCoord;
                
                [C_default_doNotUse,subjData_beforeTopN]=transformIDMtoMPSC(S);
                C = dlmread(dlmReadIn);
                Cdlmread = C;
                
                %% modify C matrix and Mpsc

                %cat 2 = basic-level items
                %cat 3 = subordinate trees, 4 = birds, 5 = fish, 6 = cars, 7 = shoes       
                
                newC = C(C(:,1) == 2,:);                
                newSubjData_beforeTopN = subjData_beforeTopN(C(:,1) == 2,:);
                
                for sCat = 3:7
                    typInd_sCat = subIndEachCat(sCat - 2);
                    sCat_C = [repmat(3,6,1) repmat(sCat-2,6,1) [1:6]'];
                    sCat_subjDat_temp = arrayfun(@(p) subjData_beforeTopN(C(:,1) == sCat & C(:,2) == typInd_sCat & C(:,3) == p,:), [1:6]', 'UniformOutput', 0);
                    sCat_subjDat = cell2mat(sCat_subjDat_temp);
                    
                    newC = [newC; sCat_C];
                    newSubjData_beforeTopN = [newSubjData_beforeTopN; sCat_subjDat];
                end
                
                C = newC;
                subjData_beforeTopN = newSubjData_beforeTopN;
                
                if ~exist('noFolds', 'var')
                    trainPresMat = combnk(unique(C(:, 3)), numel(unique(C(:, 3))) - noTestItems2Avg);
                    noFolds = size(trainPresMat, 1);
                end
                
                rankAcc_folds = [];
                
                for f = 1:noFolds
                    
                    trainPresInd = transpose(ismember(transpose(C(:, 3)), trainPresMat(f, :)));
                    testPresInd = ~trainPresInd;
                    
                    %% get features (voxels)
                    
                    featFromCatInd = find(C(:,1) == featureFromCat_i & trainPresInd == 1);
                    [subjData, subj_voxXYZ] = getStableVoxData_forCrossCls(C, featFromCatInd, subjData_beforeTopN, subj_voxXYZ_beforeTopN, mask_ID_sbj, maskLocation, useImgMask, topNStableVox);

                    %% set up training
                    
                    word_labels_train = C(C(:,1) == trainCat_i & trainPresInd == 1,2);                
                    trainDat = subjData(C(:,1) == trainCat_i & trainPresInd == 1,:);
                    
                    trainDat = trainDat - ...
                        repmat(mean(trainDat),size(trainDat,1),1);
                    trainDat = trainDat./ ...
                        repmat(std(trainDat),size(trainDat,1),1);
                    
                    %% set up testing, and classify
                    
                    word_labels_test = C(C(:,1) == testCat_i & testPresInd == 1,2);
                    testDat = subjData(C(:,1) == testCat_i & testPresInd == 1,:);

                    testDat = testDat - ...
                              repmat(mean(testDat),size(testDat,1),1);
                    testDat = testDat./ ...
                              repmat(std(testDat),size(testDat,1),1);
                    
                    testDat_temp = arrayfun(@(w) mean(testDat(find(word_labels_test == w),:),1), unique(word_labels_test), 'UniformOutput', 0);
                    testDat = cell2mat(testDat_temp);
                    
                    [classifier]=trainClassifier(trainDat, word_labels_train, classifierString);
                    [predictions]=applyClassifier(testDat, classifier);
                    [results,predictedLabels,trace]=summarizePredictions(predictions, ...
                                                                      classifier,'averageRank', unique(word_labels_test));
                    
                    rankAcc_folds = [rankAcc_folds; 1 - results{1}];
                end
                
                racc_subjXmaskXVox_subType(sbj_i, mask_i, voxNo_i) = mean(rankAcc_folds);
            end
        end
    end
    
    if strcmp(subType, 'typical')
        racc_subjXmaskXVox.typ = racc_subjXmaskXVox_subType;
    elseif strcmp(subType, 'atypical')
        racc_subjXmaskXVox.atyp = racc_subjXmaskXVox_subType;
    end 
end

%% save

disp(['display results: racc_subjXmaskXVox']);

save(strcat('SAVE_', mfilename), 'racc_subjXmaskXVox', 'subjPool', 'mask_pool', 'voxNoPool');

write_racc_flag = 0;
if write_racc_flag
    clear all
    load SAVE_WSA_allCode_sub_wFolds_crossCls_AAL.mat
    write_racc_out_sub_wFolds_crossCls_AAL(racc_subjXmaskXVox, mask_pool, voxNoPool, numel(subjPool));

    raccThresh = 0.57;
    load SUB_WFOLDS_AAL_storeRACCTypMeans_nTtests.mat
    [typ, atyp, cmp] = find_threshRACC_wTtests_sub_wFolds_crossCls_AAL(raccThresh, racc_typMeans_subjXmaskXVox, storeTtest_tNp_btTyp_sameMask, mask_pool, voxNoPool);
end

disp(strcat(mfilename,': done'))