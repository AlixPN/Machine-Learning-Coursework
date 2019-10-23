% the sampled dataset, containing 75% of 'no' and 25% of 'yes', is loaded as
% table
data=readtable('data_end.csv');

%categorical features are converted using to_categorical function
data.job = categorical(data.job);
data.marital = categorical(data.marital);
data.month = categorical(data.month);
data.day_of_week = categorical(data.day_of_week);
data.poutcome = categorical(data.poutcome);    

%Normalisation of numerical values using normalize(), the code has been executed using 
%either normalised and not normalised numerical features without concreate
%changes. To run it with normalised values decomment line 16-20 

% data.emp_var_rate = normalize(data.emp_var_rate);
% data.cons_price_idx = normalize(data.cons_price_idx);
% data.cons_conf_idx = normalize(data.cons_conf_idx);
% data.euribor3m = normalize(data.euribor3m);
% data.nr_employed = normalize(data.nr_employed);

%Dataset splitted in 70% training set and 30% test set. Randperm is a function
%that randomly changes the index order and save it in a vector. The vector
%is after used to sample our train and test sets

[rows,columns] = size(data);
P = 0.70 ;
idx = randperm(rows);
train = data(idx(1:round(P*rows)),:) ; 
test = data(idx(round(P*rows)+1:end),:) ;

%training and test features are splitted from the classification feature
train_x=train(:,1:19);
train_y=train(:,20);

test_x=test(:,1:19);
test_y=test(:,20);


%First application of the model using 100 trees and calculating the
%out-of-bag-error.

Md1= TreeBagger(100,train_x,train_y,'OOBPrediction','On');
results=predict(Md1,test_x);

%The figure plots the out of bag error of our model, while the trees are
%ensambled
figure(1);
oobPredict(Md1); 
oobErrorBaggedEnsemble = oobError(Md1);
plot(oobErrorBaggedEnsemble)
xlabel 'Number of grown trees';
ylabel 'Out-of-bag classification error';

%Error of each tree, the avg gives an extimation of the accuracy error
err=error(Md1,test_x,test_y);
avgerr=mean(err);


%Confusion matrix of the model, variables transformation required to
%compute confusionmat(), our input are predictions and targets
test_ya=table2array(test_y);    
test_yd=double(test_ya);

results_a=cell2mat(results);
results_d=double(results_a=='1');

C = confusionmat(test_yd,results_d);

test_ydt=test_yd.'; 
results_d=results_d.';

figure(2)
plotconfusion(test_ydt,results_d);

%cv partition (k=10 by default), it creates k training and k validation sets
%used afterwards to compute the cross validation
c = cvpartition(train.y,'KFold',10);

%various transformations (dummy variables,tables to array...) to compute
%the cross validated error

train_xcv=train_x;
train_xcv.marital=dummyvar(train_x.marital);
train_xcv.month=dummyvar(train_x.month);
train_xcv.poutcome=dummyvar(train_x.poutcome);
train_xcv.day_of_week=dummyvar(train_x.day_of_week);
train_xcv.job=[];

train_xcv= table2array(train_xcv);
train_ycv=table2array(train_y);
test_ycv=table2array(test_y);

%function used with all different partitions to have the final CV error 
classfun=@(train_x,train_y,test_x)predict(TreeBagger(100,train_xcv,train_ycv,'OOBPrediction','On'),test_x);
cverror=crossval('mcr',train_xcv,train_ycv,'predfun',classfun,'partition',c);


% GRID SEARCH ALGORITHM, inner loop using various number of predictor, the
% one in middle allows to change number of trees, while the outer one is
% used to have our results with k different cross validated 

errorgrid={};      % used in the innerloop to save all trees err
cvmodels={};       %used afterwards to save all the trained models
t = zeros(1,1000);

counter=0; % Subsequently used to save all the iterations as table observations( j*i*z ) 
 
 for z= 1:c.NumTestSets
     trainIdx = c.training(z);
     validIdx = c.test(z);
     for i= [1,5,20:20:200]  
           for j= 1:10    
                        tic;
                        counter=counter+1;
                        cvmodels{counter}= TreeBagger(i,train_x(trainIdx ,:),train_y(trainIdx,1),'OOBPrediction','On','NumPredictorsToSample',j);
                        t(counter)=toc; %save the time initialised to zero at tic, gives us the training time 
                   
                        err = error(cvmodels{counter},train_x(validIdx ,:),train_y(validIdx,1));   %error of each tree of the validation set
                        errorgrid{counter}=mean(err); %averaged error
                        
                        %varius transformations required to commpute the
                        %confusion matrix. We did twice to compute training and validation results.
                        
                        
                        results_j=predict(cvmodels{counter},train_x(validIdx ,:));
                        results_jtrain=predict(cvmodels{counter},train_x(trainIdx ,:));
                        
                            
                        results_a=cell2mat(results_j);
                        results_atrain=cell2mat(results_jtrain);
                        
                        
                        results_d=double(results_a=='1');
                        results_dtrain=double(results_atrain=='1');
                        
                        
                        train_ya=table2array(train_y(validIdx,1));
                        train_yatrain=table2array(train_y(trainIdx,1));
                        
                        
                        train_yd=double(train_ya);
                        train_ydtrain=double(train_yatrain);
                        
                        %Confusion matrices and its values using validation
                        %set to test results                   
                        confusion = confusionmat(train_yd,results_d);
                        
                        %TrueNegative | TruePositive | FalseNegative | FalsePositive
                        TN=confusion(1,1);
                        TP=confusion(2,2);
                        FN=confusion(2,1);
                        FP=confusion(1,2);
                        
                        %Confusion matrices and its values using training
                        %set to test results
                        confusion2 = confusionmat(train_ydtrain,results_dtrain);
                        TN2=confusion2(1,1);
                        TP2=confusion2(2,2);
                        FN2=confusion2(2,1);
                        FP2=confusion2(1,2);
                        
                       
                        %Accuracy
                        Accuracy=(TN+TP)/(TN+TP+FN+FP);
                        AccuracyTrain=(TN2+TP2)/(TN2+TP2+FN2+FP2);
                       
                        
                        %Caluculation of Fscore
                        precision=TP/(TP+FP);
                        recall= TP/(TP+FN);
                        Fscore=2*precision*recall/(precision+recall);

                        precision2=TP2/(TP2+FP2);
                        recall2= TP2/(TP2+FN2);
                        FscoreTrain=2*precision2*recall2/(precision2+recall2);
                        
                        %Calculation of Matthews coefficient,  used in machine learning as a measure of 
                        %the quality of binary (two-class) classifications.
                        %Results proportional to F1-score. Thus we decide to use F1_Score as evaluation parameter
                        
                        Matthews = (TP .* TN - FP .* FN) ./ ...
                        sqrt( (TP + FP) .* (TP + FN) .* (TN + FP) .* (TN + FN) );
                    
                        MatthewsTrain = (TP2 .* TN2 - FP2 .* FN2) ./ ...
                        sqrt( (TP2 + FP2) .* (TP2 + FN2) .* (TN2 + FP2) .* (TN2 + FN2) );
                        
                        table(counter,:)=[z,i,j,errorgrid{counter},AccuracyTrain,Accuracy,FscoreTrain,Fscore,MatthewsTrain,Matthews,t(counter)]
                        
                        
           end  
     end
 end

% the most accurate model is selected
[~,modelidx]=max((table(:,6)));  
Accmodel= cvmodels{modelidx};

% the model with the best Fscore is selected
[~,modelidx2]=max((table(:,8)));
Fscoremodel=cvmodels{modelidx2};

% Best accuracy model and its hyperparameters
disp('The model with the best accuracy has : ');

Atrees=table(modelidx,2);
str1 = ['Number of trees :',num2str(Atrees)];
disp(str1);

AFeatures=table(modelidx,3);
str2 = ['Number of features per node : ',num2str(AFeatures)];
disp(str2);

cvAcc=table(modelidx,1);
str3 = ['Cross validated training n : ',num2str(cvAcc)];
disp(str3);


% Best F1-Score model and its hyperparameters
disp('The model with the best Fscore has : '); 
Ftrees=table(modelidx2,2);
str1 = ['Number of trees :',num2str(Ftrees)];
disp(str1);

FFeatures=table(modelidx2,3);
str2 = ['Number of features per node : ',num2str(FFeatures)];
disp(str2);

cvFscore=table(modelidx2,1);
str3 = ['Cross validated training n : ',num2str(cvFscore)];
disp(str3);



% 3D graph creation
% the table is filtered because we want to plot the 3D graphs of the most
% accurate and best F1-Score models. to do that we selected results of the
% cv partion that gave us those models.

tableAIdx=(table(:,1)==cvAcc);
tableA=table(tableAIdx,:);

tableFIdx=(table(:,1)==cvFscore);
tableF=table(tableFIdx,:);

%data grid to plot a 3D figure
[xq,yq]=meshgrid([1,5,20:20:200],1:10);


%2D matrices of both accuracy and F1-Score
foo=0;
for i=1:12
    for j=1:10
        foo=foo+1;
        matrixA(j,i)=tableA(foo,6);   %accuracy
        matrixB(j,i)=tableA(foo,8);   %Fscore
    end
end

%F1-Score 3D figure, it is possible to plot the accuracy changing matrixB
%with matrix A
figure(3)
surf(xq,yq,matrixB,'FaceAlpha',0.5)
colormap cool

%INVESTIGATION OF PREDICTORS IMPORTANCE

%Model created using Ftrees and FFeatures (hyperparameters that best fit the
%model), to predict features importance
FImportancemodel =TreeBagger(Ftrees,train_x,train_y,'OOBPredictorImportance','on','NumPredictorsToSample',FFeatures);
Importance = FImportancemodel.OOBPermutedPredictorDeltaError;

figure(4)
bar(Importance)
xlabel('Features')
ylabel('Out-of-Bag Feature Importance')


%samples of training dataset wihouth dominant variable and noisy variable (train_x2 and train_x3)
trainIdx = c.training(cvFscore);
validIdx = c.test(cvFscore);

train_x2= train_x;
train_x2.duration=[];

train_x3=train_x;
train_x3=train_x;

train_x3.marital=[];
train_x3.job=[];
train_x3.housing=[];
train_x3.loan=[];
train_x3.campaign=[];
train_x3.previous=[];

% Iteration of 3 models with different datasets to check the differences
% using different predictors. Same of the inner loop of our grid search
% algorithm

for j=1:10 
    example1 =TreeBagger(Ftrees,train_x(trainIdx ,:),train_y(trainIdx,1),'OOBPredictorImportance','on','NumPredictorsToSample',j);
    example2 =TreeBagger(Ftrees,train_x2(trainIdx ,:),train_y(trainIdx,1),'OOBPredictorImportance','on','NumPredictorsToSample',j);
    example3 =TreeBagger(Ftrees,train_x3(trainIdx ,:),train_y(trainIdx,1),'OOBPredictorImportance','on','NumPredictorsToSample',j);
    
    %prediction train_x
    results_j=predict(example1,train_x(validIdx ,:));                  
    results_a=cell2mat(results_j);
    results_d=double(results_a=='1');
    
    %prediction train_x2    
    results_j2=predict(example2,train_x2(validIdx ,:));
    results_a2=cell2mat(results_j2);
    results_d2=double(results_a2=='1');
    
    %prediction train_x3
    results_j3=predict(example3,train_x3(validIdx ,:));                  
    results_a3=cell2mat(results_j3);
    results_d3=double(results_a3=='1');                 
    
    
    train_ya=table2array(train_y(validIdx,1));
    train_yd=double(train_ya);

    %confusion matrix
    confusion1 = confusionmat(train_yd,results_d);
    TN1=confusion1(1,1);
    TP1=confusion1(2,2);
    FN1=confusion1(2,1);
    FP1=confusion1(1,2);
    
    confusion2 = confusionmat(train_yd,results_d2);
    TN2=confusion2(1,1);
    TP2=confusion2(2,2);
    FN2=confusion2(2,1);
    FP2=confusion2(1,2);
    
    confusion3 = confusionmat(train_yd,results_d3);
    TN3=confusion3(1,1);
    TP3=confusion3(2,2);
    FN3=confusion3(2,1);
    FP3=confusion3(1,2);


    %caluculation of Fscore
    precision1=TP1/(TP1+FP1);
    recall1= TP1/(TP1+FN1);
    Fscorej(j)=2*precision1*recall1/(precision1+recall1)
    
    precision2=TP2/(TP2+FP2);
    recall2=TP2/(TP2+FN2);
    Fscorej2(j)=2*precision2*recall2/(precision2+recall2);
    
    precision3=TP3/(TP3+FP3);
    recall3= TP3/(TP3+FN3);
    Fscorej3(j)=2*precision3*recall3/(precision3+recall3);

end

%Line plot of results 
figure (5)
n=[1:10];
plot(n,Fscorej,'-ob',n,Fscorej2,'-or',n,Fscorej3,'-og','MarkerIndices',n,'LineWidth',1.3);
xlabel('predictors sampled at each node')
ylabel('F1-Score') 
legend('All Features','NO Dominant Feature','NO Noisy Features')


%prediction of the testset using our best model (F1-Score)
Best_Mdl_result= predict(Fscoremodel,test_x);

 
Best_Mdl_result = cell2mat(Best_Mdl_result);
Best_Mdl_result = double(Best_Mdl_result=='1')';

            
outcomeVariableBest= table2array(test_y);
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

         
%F1-Score
Best_F_score=2*Best_Recall*Best_Precision/(Best_Precision+Best_Recall); 

           
%Accuracy
Best_Accuracy = (TP_Best+TN_Best)/(TP_Best+TN_Best+FN_Best+FP_Best);


%Evaluation of Cross validation impact on result. We filter a subtable
%that contains the vth partition and we calculate the average F-score of it
for v=1:10 
    tableVIdx=(table(:,1)==v);
    tableV=table(tableVIdx,:);
    mean(tableV(:,8))
    CVEstimation(v)=max(tableV(:,8))
end

disp(CVEstimation);
