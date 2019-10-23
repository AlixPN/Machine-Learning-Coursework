%Dataset Balancing
data=readtable('clean.csv');


%Random train test split of the data (70/30)
P = 0.70 ;
idx = randperm(size(data.y,1));
[rows,col] = size(data);
train = data(idx(1:round(P*rows)),:) ; 
test = data(idx(round(P*rows)+1:end),:) ;

%prdictor and outcom variable Test
testx=test(:,1:19);
testy=test(:,20);

%prdictor and outcom variable Training
train_yes=train((train.y==1),:);
train_no=train((train.y==0),:);

%number of yes/no
[rows_no, col_no]=size(train_no);
[rows_yes, col_yes]=size(train_yes);



MatthewsRF=zeros(11,1);
MatthewsNB=zeros(11,1);

w=[1 1.5 2 2.5 3 3.5 4 5.8 7.9];
counter=0;
for i=w      % 50 60 66 71 75 77 80 85 89    percent of 'no'
    counter=counter+1;
    
    Index=randperm(rows_no,round(rows_yes*i));
    train_noi=train_no(Index,:);

    %merging no and yes observations
    data_i = [train_noi; train_yes];
    [rows_i,col_i] = size(data_i);

    %split
    Index2 = randperm(rows_i);
    train_i = data_i(Index2,:); 

    train_xi=train_i(:,1:19);
    train_yi=train_i(:,20);

    for j=1:15    
        % RF model and solution with balanced training data
        RFmd= TreeBagger(100,train_xi,train_yi,'OOBPrediction','On');
        results_i=predict(RFmd,testx);

        test_ya=table2array(testy);    
        test_yd=double(test_ya);

        results_a=cell2mat(results_i);
        results_d=double(results_a=='1');
        
        %confusion matrix with TrueNegative | TruePositive | FalseNegative | FalsePositive
        C = confusionmat(test_yd,results_d);
        TN=C(1,1);
        TP=C(2,2);
        FN=C(2,1);
        FP=C(1,2);
        
        %Fscore
        precisionRF=TP/(TP+FP);
        recallRF= TP/(TP+FN);
        FscoreRF(j)=2*precisionRF*recallRF/(precisionRF+recallRF);
        %Accuracy
        AccurRF(j)=(TN+TP)/(TN+TP+FN+FP);
        %Matthews
        MatthewsRF(j) = (TP .* TN - FP .* FN) ./ ...
        sqrt( (TP + FP) .* (TP + FN) .* (TN + FP) .* (TN + FN) );                   %matthes correlation 

        test_yt=test_yd.';
        results_t=results_d.';
        
        %lot confusion matrix
        figure(1)
        plotconfusion(test_yt,results_t);

        % NB model and solution with balanced training data
        NBmd = fitcnb(train_xi,train_yi,'ClassNames', {'1','0'});
        results_inb=predict(NBmd,testx);

        results_anb=cell2mat(results_inb);
        results_dnb=double(results_anb=='1');
        
        %confusion matrix with TrueNegative | TruePositive | FalseNegative | FalsePositive
        C2 = confusionmat(test_yd,results_dnb);
        TN2=C2(1,1);
        TP2=C2(2,2);
        FN2=C2(2,1);
        FP2=C2(1,2);
        
        %Fscore
        precisionNB=TP2/(TP2+FP2);
        recallNB= TP2/(TP2+FN2);
        FscoreNB(j)=2*precisionNB*recallNB/(precisionNB+recallNB);
        
        %Accuracy
        AccurNB(j)=(TN2+TP2)/(TN2+TP2+FN2+FP2);
        
        %matthews correlation 
        MatthewsNB(j) = (TP2 .* TN2 - FP2 .* FN2) ./ ...
        sqrt( (TP2 + FP2) .* (TP2 + FN2) .* (TN2 + FP2) .* (TN2 + FN2) );                  
        
        
        test_yt=test_yd.';
        results_tnb=results_dnb.';
        
        %plot confusion matrix
        figure(2)
        plotconfusion(test_yt,results_tnb);
    end
    
    %calculating the average accuracy, fscore and matthews 
    avgAcc_RF(counter)=mean(AccurRF);
    avgF_RF(counter)=mean(FscoreRF);
    avgM_RF(counter)=mean(MatthewsRF);
    
    avgAcc_NB(counter)=mean(AccurNB);
    avgF_NB(counter)=mean(FscoreNB);
    avgM_NB(counter)=mean(MatthewsNB);
    
    
end

%plottingthe results (accuracy, fscore, matthews) of both methods for
%differently balanced datasets
x=[50 60 66 71 75 77 80 85 89];
figure(4)
hold on 
plot(x,avgAcc_RF,'b:',x,avgF_RF,'b--',x,avgM_RF,'b','LineWidth',1.5);
plot(x,avgAcc_NB,'g:',x,avgF_NB,'g--',x,avgM_NB,'g','LineWidth',1.5);
title('Performance of differently sampled Datasets');
xlabel('Percentage of majority class (No) within the dataset');
legend('RF Accuracy','RF Fscore','RF Matthews coeff','NB Accuracy','NB Fscore','NB Matthews coeff');
set(gca,'Xticklabel',{ '50%', '60%', '66%', '71%', '75%', '77%', '80%', '85%', '88%', '89%',});
hold off

%after comparing different balanced datasets (from totally balanced 50/50 to 
%unbalanced 89/21 the best one found for both methods is 75%/25% 
data_yes=data((data.y==1),:);
data_no=data((data.y==0),:);

[rows_no, col_no]=size(data_no);
[rows_yes, col_yes]=size(data_yes);

Index=randperm(rows_no,round(rows_yes*3));
sampled_no=data_no(Index,:);

%merge yes and no observations 
data_end = [sampled_no; data_yes];

%save balanced dataset as 'data_end'
writetable(data_end,'data_end.csv');



    