%MODELING
data = readtable('data_end.csv');

%Categorize Features
data.job = categorical(data.job);
data.marital = categorical(data.marital);
data.month = categorical(data.month);
data.day_of_week = categorical(data.day_of_week);
data.poutcome = categorical(data.poutcome);
data.education = categorical(data.education);
data.loan = categorical(data.loan);
data.housing = categorical(data.housing);
data.contact = categorical(data.contact);
data.campaign = categorical(data.campaign);
data.pdays = categorical(data.pdays);
data.previous = categorical(data.previous);

%Normalization of social and economic context attributes using normalize()
data.emp_var_rate = normalize(data.emp_var_rate);
data.cons_price_idx = normalize(data.cons_price_idx);
data.cons_conf_idx = normalize(data.cons_conf_idx);
data.euribor3m = normalize(data.euribor3m);
data.nr_employed = normalize(data.nr_employed);

%Random Train-Test-Split with the function randperm
%--> randomly change the index order and save it in a vector --> use this
%vector to sample our train and test sets
[rows,columns] = size(data) ;
P = 0.70 ;
idxMinVal = randperm(rows)  ;
train = data(idxMinVal(1:round(P*rows)),:) ; 
test = data(idxMinVal(round(P*rows)+1:end),:);

%Train Test Split 
%prdictor and outcom variable Training
predictorsTrain = train(:, 1:19);   %Predictor columns
outcomeVarTrain = train(:, 20);     %Outcome variable

%prdictor and outcom variable Test
testPredictors =test(:, 1:19);      %Predictor columns
testOutcomeVar = test(:, 20);       %Outcome variable

%%%%% Fit Model for the first time (NAIVE BAYES, 'out of the box model')
Model = fitcnb(predictorsTrain, outcomeVarTrain, 'ClassNames', {'1','0'});
%out-of-sample misclassification rate 
Loss = loss(Model, testPredictors, testOutcomeVar);

% CONFUSION MATRIX
%first prediction test set
result_1 = predict(Model, testPredictors);
  
result_1 = cell2mat(result_1);
result_1 = double(result_1=='1')';
            
outcomeVariable= table2array(testOutcomeVar);
outcomeVariable2 = double(outcomeVariable);

%Confustion Matrix
confusionMat = confusionmat(outcomeVariable2,result_1);

%TrueNegative | TruePositive | FalseNegative | FalsePositive
TN1=confusionMat(1,1);
TP1=confusionMat(2,2);
FN1=confusionMat(2,1);
FP1=confusionMat(1,2);
            
%Matthews
Matthews1 = (TP1 .* TN1 - FP1 .* FN1) ./ ...
sqrt( (TP1 + FP1) .* (TP1 + FN1) .* (TN1 + FP1) .* (TN1 + FN1) );

%Recall and precision
Recall1 = TP1 / (TP1+FN1);
Precision1 = TP1 / (TP1+FP1);
         
%F-Score
F_score1=2*Recall1*Precision1/(Precision1+Recall1); 
           
%Accuracy
Accuracy1 = (TP1+TN1)/(TP1+TN1+FN1+FP1);


%%%%    HYPERPARAMETER TUNING

%distribution for numerical predictors -> Kernel 'normal' distribution assumed
distribution = {'kernel', 'mvmn','mvmn','mvmn','mvmn','mvmn','mvmn','mvmn','mvmn','kernel','mvmn','mvmn','mvmn','mvmn', 'kernel','kernel','kernel','kernel','kernel'};

kern_mdl = fitcnb(predictorsTrain, outcomeVarTrain,'Distribution', distribution, 'ClassNames', {'1','0'});
kern_Loss = loss(kern_mdl, testPredictors, testOutcomeVar);

%%%%%GRID SEARCH 
%-> Kernel parameter selection 
%First numerical features age and duration will be distributed in the same way; 
%the last 5 features are social and economic context attributes and will also be
%distributed in the same way

kernel_distributions = {'normal', 'box', 'triangle','epanechnikov'};

%MATRIX WITH ALL POSSIBLE COMBINATIONS for 2 different distributions
dist_matrix = nchoosek(repmat(kernel_distributions, 1, 4), 2);
dist_matrix = cell2table(dist_matrix);
dist_matrix = unique(dist_matrix);
dist_matrix = table2array(dist_matrix);
size_matr = size(dist_matrix);

%manipulation of the parameter width
width = [0.1 0.2 0.5 0.7];

c = cvpartition(train.y, 'KFold',10);
modelKFolds={};
errorKFolds={};
tableGrid=[];
result_dist=[];
counter=0;

%grid search, 1st loop results with k different cross validated | 2nd loop
%uses various numbers of width | 3dr loop uses different kernel distributions 
%10-Fold Cross-Validation, k training and k validation sets
%used to compute the cross validation
for i = 1:c.NumTestSets
        trainIdx = c.training(i);
        validationIdx = c.test(i);
        
        %specifiying different width during grid search
        for y = width 
            % going through all possible combinations of distributions
            % during grid search to see which performes best
            for x = 1:size_matr(1)   
            counter = counter+1;
            tic;
            value = dist_matrix(x,:);
            distr_kern = {value{1}, 'mvmn', 'mvmn','mvmn', 'mvmn','mvmn', 'mvmn','mvmn', 'mvmn', value{1}, 'mvmn', 'mvmn', 'mvmn','mvmn', value{2}, value{2}, value{2}, value{2}, value{2}};
            time1(counter)= toc;
            
            %Fit Model
            tic;
            NBmodelCV{counter} = fitcnb(predictorsTrain(trainIdx,:), outcomeVarTrain(trainIdx,1), 'Distribution', distribution,'Kernel',distr_kern,'Width',y,'ClassNames', {'1','0'});
            time2(counter)= toc;
            % calculate classification error by resubstitution
            errorKFolds{counter} = resubLoss(NBmodelCV{counter}, 'LossFun', 'classiferror');
            
            tic;
            %prediction validation set
            result_NB = predict(NBmodelCV{counter}, predictorsTrain(validationIdx,:));
            result_NB_Train = predict(NBmodelCV{counter}, predictorsTrain(trainIdx,:));
            
            result_NB=cell2mat(result_NB);
            result_NB=double(result_NB=='1')';
            
            result_NB_Train=cell2mat(result_NB_Train);
            result_NB_Train=double(result_NB_Train=='1')';
            
            outcomeVarTrain1= table2array(outcomeVarTrain(validationIdx,1));
            outcomeVarTrain2 = double(outcomeVarTrain1);
            
            outcomeVarTrain3= table2array(outcomeVarTrain(trainIdx,1));
            outcomeVarTrain4 = double(outcomeVarTrain3);
            
            %Confusion Matrix
            confMat = confusionmat(outcomeVarTrain2,result_NB);
            confMatTrain = confusionmat(outcomeVarTrain4,result_NB_Train);
            
            %TrueNegative | TruePositive | FalseNegative | FalsePositive 
            TN=confMat(1,1);
            TP=confMat(2,2);
            FN=confMat(2,1);
            FP=confMat(1,2);
            
            TN2=confMatTrain(1,1);
            TP2=confMatTrain(2,2);
            FN2=confMatTrain(2,1);
            FP2=confMatTrain(1,2);
            
            %Matthews (good evaluation method for unbalanced datasets)
            Matthews = (TP .* TN - FP .* FN) ./ ...
            sqrt( (TP + FP) .* (TP + FN) .* (TN + FP) .* (TN + FN) );
            
            %Recall and precision
            Recall = TP / (TP+FN);
            Precision = TP / (TP+FP);
            
            RecallTrain = TP2 / (TP2+FN2);
            PrecisionTrain = TP2 / (TP2+FP2);
         
            %calculating F-Score (good evaluation method for unbalanced datasets)
            F_score=2*Recall*Precision/(Precision+Recall);
            F_score_Train=2*RecallTrain*PrecisionTrain/(PrecisionTrain+RecallTrain);
           
            %Accuracy
            accuracy = (TP+TN)/(TP+TN+FN+FP);
            accuracy_Train = (TP2+TN2)/(TP2+TN2+FN2+FP2);
            
            time3(counter)= toc;
            %Table with partition(1), distribution(2), width(3), error(4),
            %F_score(5), Matthews(6) and Accuracy(7)
            %tableGrid(counter,:)=[i,x,y,errorKFolds{counter},F_score,Matthews, accuracy]
            tableGrid(counter,:)=[i,x,y,errorKFolds{counter},F_score,F_score_Train,Matthews, accuracy,accuracy_Train]
            
            end
    end
end

%min error in tableGrid:
[min_vals, min_idx] = min(tableGrid(:,4));
min_error = tableGrid(min_idx,4);
%max FScore in tableGrid:
[max_f, max_idxF] = max(tableGrid(:,5));
max_FScore = tableGrid(max_idxF,5);
%max FScore Train in tableGrid:
[max_ft, max_idxFT] = max(tableGrid(:,6));
max_FScoreT = tableGrid(max_idxF,6);
%max Matthews in tableGrid:
[max_f, max_idxM] = max(tableGrid(:,7));
max_Matthews = tableGrid(max_idxM,7);
%max Accuracy in tableGrid:
[max_a, max_idxA] = max(tableGrid(:,8));
max_Accuracy = tableGrid(max_idxA,8);


% the most accurate model 
[~,NBModelidx1]=max(tableGrid(:,8));  
NBAccuracyMdl = NBmodelCV{NBModelidx1};

% model with the lowes error
[~,NBModelidx2]=min(tableGrid(:,4));  
NBLowErrorMdl = NBmodelCV{NBModelidx2};

% the model with the best Fscore 
[~,NBmodelidx3]=max(tableGrid(:,5));
NBF_ScoreMdl=NBmodelCV{NBmodelidx3};

%Results 'Which is the best model' in terms of accuracy:
disp('The NB model with the best accuracy has : ');
str1 = ['Distribution = Dist_matrix column :',num2str(tableGrid(NBModelidx1,2))];
disp(str1);
str2 = ['width : ',num2str(tableGrid(NBModelidx1,3))];
disp(str2);
str3 = ['Cross validated training n : ',num2str(tableGrid(NBModelidx1,1))];
disp(str3);

%Results 'Which is the best model' in terms of f1-score:
disp('The NB model with the best F_Score has : ');
str1 = ['Distribution = Dist_matrix column :',num2str(tableGrid(NBmodelidx3,2))];
disp(str1);
str2 = ['width : ',num2str(tableGrid(NBmodelidx3,3))];
disp(str2);
str3 = ['Cross validated training n : ',num2str(tableGrid(NBmodelidx3,1))];
disp(str3);


%%%%%%% GRAPHS
%group by width and distribution
groupsW=findgroups(tableGrid(:,3));
groupsD=findgroups(tableGrid(:,2));

%plot max f1-score - width --> ValidationSet
MaxFS_Width = splitapply(@max,tableGrid(:,5),groupsW);
StdFS_Width = splitapply(@std,tableGrid(:,5),groupsW);
errorbar(MaxFS_Width,StdFS_Width)
hold on
%plot max f1-score - width --> TrainingSet
MaxFS_Width2 = splitapply(@max,tableGrid(:,6),groupsW);
StdFS_Width2 = splitapply(@std,tableGrid(:,6),groupsW);
errorbar(MaxFS_Width2,StdFS_Width2)
hold off

%plot max f1-score - distribution --> ValidationSet
MaxFS_Dist = splitapply(@max,tableGrid(:,5),groupsD);
stem(MaxFS_Dist)
hold on
%plot max f1-score - distribution --> TrainingSet
MaxFS_Dist2 = splitapply(@max,tableGrid(:,6),groupsD);
stem(MaxFS_Dist2)
hold off

%%%%%%%%Select best model -> train & final test
Best_Kern = ['normal' 'triangle'];
BestModel = fitcnb(predictorsTrain, outcomeVarTrain,'Distribution', distribution,'Kernel',Best_Kern,'Width',0.2, 'ClassNames', {'1','0'});

Best_Mdl_result= predict(BestModel, testPredictors);

Best_Mdl_result = cell2mat(Best_Mdl_result);
Best_Mdl_result = double(Best_Mdl_result=='1')';
            
outcomeVariableBest= table2array(testOutcomeVar);
outcomeVariableBest2 = double(outcomeVariableBest);

%Confustion Matrix
Best_confusionMat = confusionmat(outcomeVariableBest2,Best_Mdl_result);

%TrueNegative | TruePositive | FalseNegative | FalsePositive
TN_Best=Best_confusionMat(1,1);
TP_Best=Best_confusionMat(2,2);
FN_Best=Best_confusionMat(2,1);
FP_Best=Best_confusionMat(1,2);

%Recall and precision
Best_Recall = TP_Best / (TP_Best+FN_Best);
Best_Precision = TP_Best / (TP_Best+FP_Best);
         
%F-Score
Best_F_score=2*Best_Recall*Best_Precision/(Best_Precision+Best_Recall); 
           
%Accuracy
Best_Accuracy = (TP_Best+TN_Best)/(TP_Best+TN_Best+FN_Best+FP_Best);




%%%% DIFFERENT TESTS

%selection for Kernel parameters - Bayesian Optimization:
% Kernel1 = optimizableVariable('Kernel1', {'normal', 'box', 'triangle', 'epanechnikov'},'Type', 'categorical');
% Kernel2 = optimizableVariable('Kernel2', {'normal', 'box', 'triangle', 'epanechnikov'},'Type', 'categorical');
% Kernel3 = optimizableVariable('Kernel3', {'normal', 'box', 'triangle', 'epanechnikov'},'Type', 'categorical');
% distribution2 ={'kernel1', 'mvmn','mvmn','mvmn','mvmn','mvmn','mvmn','mvmn','mvmn','kernel2','mvmn','mvmn','mvmn','mvmn', 'kernel3','kernel3','kernel3','kernel3','kernel3'};
% 
% k_holdOut_mdl3 = fitcnb(predictorsH, outcomeVarH,'Distribution', distribution2, 'ClassNames', {'1','0'});
% kern3_Loss = @(x) loss(k_holdOut_mdl3, testPred, testOutcome);
% 
% results_kern3_Loss = bayesopt(kern3_Loss, [Kernel1, Kernel2, Kernel3], 'Verbose',1);


%BINNING
%binning of the numerical features (age/duration) -> did not show any
%segnificant effect on the accuracy

% Binning age
% dataAgeAsCat=data;
% catnames = {'0_10','11_20' '21_30', '31_40', '41_50', '51_60','61_70','71_80','81_90','91_100'};
% binned = discretize(data.age,[0 11 21 31 41 51 61 71 81 91 101], 'categorical', catnames);  
% dataAgeAsCat.age = binned;      

% catnames2 = {'0_150','151_300' '301_500', '501_650', '651_800', '801_950','951_1300','1301_2000'};
% binned = discretize(data.duration,[0 151 301 501 651 801 951 1301 2001], 'categorical', catnames2);  
% dataAgeAsCat.duration = binned;      

% train1 = dataAgeAsCat(idx(1:round(P*rows)),:) ; 
% test1 = dataAgeAsCat(idx(round(P*rows)+1:end),:) ;
% testPred1 =test1(:, 1:19);
% testOutcome1 = test1(:, 20);
% predictorsH1 = train1(:, 1:19);
% outcomeVarH1 = train1(:, 20);
% 
% BinnedMdl1 = fitcnb(predictorsH1, outcomeVarH1, 'ClassNames', {'1','0'});
% outOfSampleRate1 = loss(BinnedMdl1, testPred1, testOutcome1);
% 
% %confusion matrix 2
% figure(3)
% label1 = predict(BinnedMdl1,testPred1);
% table(testOutcome1,label1);
% label1=cell2mat(label1);
% label1=double(label1=='1')';
% testOutcome1 = table2array(testOutcome1)';
% plotconfusion(testOutcome1,label1)


% Binning duration
% dataDurationAsCat=data;
% catnames2 = {'0_150','151_300' '301_500', '501_650', '651_800', '801_950','951_1300','1301_2000'};
% binned = discretize(data.duration,[0 151 301 501 651 801 951 1301 2001], 'categorical', catnames2);  
% dataDurationAsCat.duration = binned;      

% train2 = dataDurationAsCat(idx(1:round(P*rows)),:) ; 
% test2 = dataDurationAsCat(idx(round(P*rows)+1:end),:) ;
% testPred2 =test2(:, 1:19);
% testOutcome2 = test2(:, 20);
% predictorsH2 = train2(:, 1:19);
% outcomeVarH2 = train2(:, 20);
% 
% BinnedMdl2 = fitcnb(predictorsH2, outcomeVarH2, 'ClassNames', {'1','0'});
% outOfSampleRate2 = loss(BinnedMdl2, testPred2, testOutcome2);
% 
% %confusion matrix 3
% figure(3)
% label2 = predict(BinnedMdl2,testPred2);
% table(testOutcome2,label2);
% label2=cell2mat(label2);
% label2=double(label2=='1')';
% testOutcome2 = table2array(testOutcome2)';
% plotconfusion(testOutcome2,label2)
