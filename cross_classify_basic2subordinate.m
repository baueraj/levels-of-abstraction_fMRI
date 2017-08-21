%Andrew Bauer
%081715

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

%% specify feature/train/test parameters

trainCat_i = 2; %MUST BE 2
testCat_i = 3; %MUST BE 3
featureFromCat_i = 2; %MUST BE 2

noTestItems2Avg = 2;

%% go

racc_eachWord_subjXmaskXVox = nan(numel(subjPool), 20, size(mask_pool, 1), numel(voxNoPool));

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
            
            [C_default_doNotUse, origSubjData_beforeTopN]=transformIDMtoMPSC(S);
            origC = dlmread(dlmReadIn);
            
            if ~exist('noFolds', 'var')
                trainPresMat = combnk(unique(origC(:, 3)), numel(unique(origC(:, 3))) - noTestItems2Avg);
                noFolds = size(trainPresMat, 1);
            end
            
            rankACcWords_folds = [];
            
            for f = 1:noFolds
                
                trainPresInd = transpose(ismember(transpose(origC(:, 3)), trainPresMat(f, :)));
                testPresInd = ~trainPresInd;
                
                %% modify C matrix and Mpsc

                %cat 2 = basic-level items
                %cat 3 = subordinate trees, 4 = birds, 5 = fish, 6 = cars, 7 = shoes       
                
                newC = origC(origC(:,1) == 2 & trainPresInd == 1,:);
                newSubjData_beforeTopN = origSubjData_beforeTopN(origC(:,1) == 2 & trainPresInd == 1,:);
                
                noW = 4;
                for sCat = 3:7
                    sCat_C = [repmat(3,noW,1) repmat(sCat-2,noW,1) [1:noW]'];
                    sCat_subjDat_temp = arrayfun(@(w) mean(origSubjData_beforeTopN(origC(:,1) == sCat & origC(:,2) == w & testPresInd == 1,:),1), [1:noW]', 'UniformOutput', 0);
                    sCat_subjDat = cell2mat(sCat_subjDat_temp);
                    
                    newC = [newC; sCat_C];
                    newSubjData_beforeTopN = [newSubjData_beforeTopN; sCat_subjDat];
                end
                
                C = newC;
                subjData_beforeTopN = newSubjData_beforeTopN;
                
                %% get features (voxels)
                
                featFromCatInd = find(C(:,1) == featureFromCat_i);
                [subjData, subj_voxXYZ] = getStableVoxData_forCrossCls(C, featFromCatInd, subjData_beforeTopN, subj_voxXYZ_beforeTopN, mask_ID_sbj, maskLocation, useImgMask, topNStableVox);

                %% set up training
                
                word_labels_train = C(C(:,1) == trainCat_i,2);                
                trainDat = subjData(C(:,1) == trainCat_i,:);
                
                trainDat = trainDat - ...
                    repmat(mean(trainDat),size(trainDat,1),1);
                trainDat = trainDat./ ...
                    repmat(std(trainDat),size(trainDat,1),1);
                
                %% set up testing, and classify
                
                word_labels_test = C(C(:,1) == testCat_i,2);
                testDat = subjData(C(:,1) == testCat_i,:);

                testDat = testDat - ... 
                          repmat(mean(testDat),size(testDat,1),1);
                testDat = testDat./ ...
                          repmat(std(testDat),size(testDat,1),1);
                
                [classifier]=trainClassifier(trainDat, word_labels_train, classifierString);
                [predictions]=applyClassifier(testDat, classifier);
                [results,predictedLabels,trace]=summarizePredictions(predictions, ...
                                                                  classifier,'averageRank', word_labels_test);
                
                %% save accuracies 
                
                addMultiples5_forIDs = transpose(5:5:(20*5)) - 5;
                
                rankPredictionList=trace{1} + repmat(addMultiples5_forIDs, 1, 5);
                trueLabels=word_labels_test + addMultiples5_forIDs;
                MeasureEx=results{2}; %this is rank distance from correct prediction

                truelabelsConfusion =trueLabels;
                predictedLabelsConfusion =rankPredictionList;
                uniqwordLabels_conf = unique(trueLabels); 
                rankACcWords_temp =uniqwordLabels_conf ; % dummy assignment
                for i_con =1:length(uniqwordLabels_conf)
                    MeasureEx_i_con = MeasureEx(find(truelabelsConfusion==uniqwordLabels_conf(i_con)));
                    rankACcWords_temp(i_con) =1 - (sum(MeasureEx_i_con)/length(MeasureEx_i_con)-1)/(length(uniqwordLabels_conf)/4-1);
                end
                %rankAccWords = [uniqwordLabels_conf , rankACcWords_temp];
                
                %[~, sortInd] = sort(truelabelsConfusion, 'ascend');
                %rankACcWords = rankACcWords_temp(sortInd);
                rankACcWords = rankACcWords_temp;
                rankACcWords_folds = [rankACcWords_folds; transpose(rankACcWords)];
            end
            
            racc_eachWord_subjXmaskXVox(sbj_i, :, mask_i, voxNo_i) = mean(rankACcWords_folds,1);
        end
    end
end

%% save

disp(['display results: racc_eachWord_subjXmaskXVox']);

save(strcat('SAVE_', mfilename), 'racc_eachWord_subjXmaskXVox', 'subjPool', 'mask_pool', 'voxNoPool');

write_racc_flag = 0;
if write_racc_flag
    clear all
    load SAVE_WSA_allCode_presMeans_wFolds_crossCls_AAL.mat
    write_racc_out_presMeans_wFolds_crossCls_AAL(racc_eachWord_subjXmaskXVox, mask_pool, voxNoPool, numel(subjPool));

    raccThresh = 0.57;
    load PRESMEANS_WFOLDS_AAL_storeRACCTypMeans_nTtests.mat
    [typ, atyp, cmp] = find_threshRACC_wTtests_presMeans_wFolds_crossCls_AAL(raccThresh, racc_typMeans_subjXmaskXVox, storeTtest_tNp_btTyp_sameMask, mask_pool, voxNoPool);
end

disp(strcat(mfilename,': done'))