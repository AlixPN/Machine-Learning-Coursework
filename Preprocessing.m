%Load the data
data = readtable('bank-additional-full.csv');

%Shape: 
[noRows, noCols] = size(data);  

%CLEANING OF THE DATASET
%Delete 'unknowns' in columns Job, Marital, Education, Housing and Loan
toDelete1 = strcmp(data.job,'unknown');
data(toDelete1,:) = [];

toDelete2 = strcmp(data.marital,'unknown');
data(toDelete2,:) = [];

toDelete3 = strcmp(data.education,'unknown');
data(toDelete3,:) = [];

toDelete4 = strcmp(data.housing,'unknown');
data(toDelete4,:) = [];

toDelete5 = strcmp(data.loan,'unknown');
data(toDelete5,:) = [];

%Delete default column
data(:,5) = [];

%Change categorical features into 0 and 1 (instead of 'no' and 'yes')
%Reformat outcome value (to 1 | 0)
success = 'yes';
data.y = double(ismember(data.y,success));

%Reformat housing value (to 1 | 0)
house = 'yes';
data.housing = double(ismember(data.housing,house));

%Reformat loan value (to 1 | 0)
loan1 = 'yes';
data.loan = double(ismember(data.loan,loan1));

%Reformat contact value (to 1 | 0) (telephone = 1 | cellular = 0)
howToContact = 'telephone';
data.contact = double(ismember(data.contact,howToContact));

%delete duplicate observations 
data=unique(data);

%Categorize Features
data.job = categorical(data.job);
data.marital = categorical(data.marital);
data.education = categorical(data.education);
data.month = categorical(data.month);
data.day_of_week = categorical(data.day_of_week);
data.poutcome = categorical(data.poutcome);

%Outliers check | deleting outliers
figure(1)
scatter(data.age,data.duration)
OutToDel =(data.duration>=2500);
data(OutToDel,:)=[];

%Education: categorical feature to numerical (no of years education)
%categorical to number 
data.education = grp2idx(data.education);

%1=basic4 | 2=basic6 | 3=basic9 | 4=high.school | 5=illitterate | 6=professional.course |
%7=university.degree
data.education(data.education == 4,:)= 12;
data.education(data.education == 6,:)= 17;
data.education(data.education == 1,:)= 4;
data.education(data.education == 2,:)= 6;
data.education(data.education == 3,:)= 9;
data.education(data.education == 5,:)= 0;
data.education(data.education == 7,:)= 16;


% ANALYSIS AND BASIC STATISTICS
% first summary of the data
SumData = summary(data);

%Outcome variable: Number of 'yes' and 'no'
Number_Yes = sum(data.y);
Number_No = noRows - Number_Yes;


%Group data (y)
G = findgroups(data.y);

%calculate descriptive statistics
MeanAge = splitapply(@mean,data.age,G);
StdAge = splitapply(@std,data.age,G);
SkewAge = splitapply(@skewness,data.age,G);
MaxAge = splitapply(@min,data.age,G);
MinAge = splitapply(@min,data.age,G);

MeanDuration = splitapply(@mean,data.duration,G);
StdDuration = splitapply(@std,data.duration,G);
SkewDuration = splitapply(@skewness,data.duration,G);
MaxDuration = splitapply(@max,data.duration,G);
MinDuration = splitapply(@min,data.duration,G);

MeanEducation = splitapply(@mean,data.education,G);
StdEducation = splitapply(@std,data.education,G);
SkewEducation = splitapply(@skewness,data.education,G);
MaxEducation = splitapply(@max,data.education,G);
MinEducation = splitapply(@min,data.education,G);

%Graphs
%age distribution
figure(2)
hold on
h1age = histogram(data.age(data.y==1,:), 20, 'Normalization','probability');
h2age = histogram(data.age(data.y==0,:), 20, 'Normalization','probability');
title('Age distribution')
legend('Yes','No')
hold off

%duration distribution
figure(3)
hold on
h1duration = histogram(data.duration(data.y==1,:),30,'Normalization','probability');
h2duration = histogram(data.duration(data.y==0,:),30,'Normalization','probability');
title('Call duration distribution')
legend('Yes','No')
hold off

%data split
data_yes = data(data.y==1,:);
data_no = data(data.y==0,:);

%education level
figure(4)
hold on
y=[sum(data_yes.education==0), sum(data_no.education==0);sum(data_yes.education==4), sum(data_no.education==4); sum(data_yes.education==6), sum(data_no.education==6); sum(data_yes.education==9), sum(data_no.education==9); sum(data_yes.education==12), sum(data_no.education==12); sum(data_yes.education==16), sum(data_no.education==16); sum(data_yes.education==17), sum(data_no.education==17)];
y(:,1)=y(:,1)/size(data_yes,1);  %normalisation
y(:,2)=y(:,2)/size(data_no,1) ;  %normalisation
b=bar(y,'Facecolor','flat');
title('Education Level');
legend('Yes','No');
set(gca,'Xticklabel',{'Illiterate','Basic_4y','basic_6y','Basic_9y','High School','University','Specialisation'});
xtickangle(35);
hold off


writetable(data,'clean.csv')