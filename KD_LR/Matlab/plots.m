%% WRSampleMF-Based Analysis | Weights
fileID = fopen('F:\Python_workspace\KD_LR\datasets\yahooR3\user.txt');
train = textscan(fileID,'%f%f%f','delimiter',',');
train = cell2mat(train);
fclose(fileID);

fileID = fopen('F:\Python_workspace\KD_LR\datasets\yahooR3\random.txt');
test = textscan(fileID,'%f%f%f','delimiter',',');
test = cell2mat(test);
fclose(fileID);

fileID = fopen('wrsamplemf_samples.txt');
samples = textscan(fileID,'%f%f','delimiter',' ');
samples = cell2mat(samples);
fclose(fileID);

fileID = fopen('wrsamplemf_weights.txt');
weights = textscan(fileID,'%f');
weights = cell2mat(weights);
fclose(fileID);
weights(weights<0) = 0;
weights(weights>1) = 1; 

[userSet, p] = numunique(train(:,1));
userAct = arrayfun(@(x) length(p{x}),1:length(userSet));
[~,userSortIdx] = sort(userAct);
[itemSet, p] = numunique(train(:,2));
itemPop = arrayfun(@(x) length(p{x}),1:length(itemSet));
[~,itemSortIdx] = sort(itemPop);

newSamples = zeros(size(samples,1),2);
[samplesUserSet, p] = numunique(samples(:,1));
for u = 1:length(samplesUserSet)
    newSamples(p{u},1) = find(userSet(userSortIdx)==samplesUserSet(u));
end
[samplesItemSet, p] = numunique(samples(:,2));
for i = 1:length(samplesItemSet)
    newSamples(p{i},2) = find(itemSet(itemSortIdx)==samplesItemSet(i));
end

unifIdx = ismember(samples,test(:,1:2),'rows');
newSamples(unifIdx,:) = [];
weights(unifIdx) = [];

figure(1)
gx = 0:50:15400;
gy = 0:5:1000;
Smoothness = 0.005;
g = RegularizeData3D(newSamples(:,1),newSamples(:,2),weights,gx,gy,'interp', 'bicubic','smoothness',Smoothness);
sl = surf(gx,gy,g,'LineStyle',':','facealpha', 0.75);
xlabel('User ID');
ylabel('Item ID');
zlabel('Weight');
set(gca,'box','off');
set(gca,'FontName','Arial Rounded MT Bold','FontSize',15,'linewidth',3);
axis tight

%% RefineLabelMF-Based Analysis | Prediction
fileID = fopen('F:\Python_workspace\KD_LR\datasets\yahooR3\user.txt');
train = textscan(fileID,'%f%f%f','delimiter',',');
train = cell2mat(train);
fclose(fileID);

[userSet, p] = numunique(train(:,1));
userAct = arrayfun(@(x) length(p{x}),1:length(userSet));
[~,userSortIdx] = sort(userAct);
[itemSet, p] = numunique(train(:,2));
itemPop = arrayfun(@(x) length(p{x}),1:length(itemSet));
[~,itemSortIdx] = sort(itemPop);

fileID = fopen('valid.txt');
valid = textscan(fileID,'%f%f%f','delimiter',' ');
valid = cell2mat(valid);
fclose(fileID);

fileID = fopen('refinelabelmf_prediction.txt');
prediction = textscan(fileID,'%f');
prediction = cell2mat(prediction);
fclose(fileID);

fileID = fopen('biasedmf_prediction.txt');
oriPrediction = textscan(fileID,'%f');
oriPrediction = cell2mat(oriPrediction);
fclose(fileID);

posIdx = find(valid(:,3)==1);

[~,oriSortedIdx] = sort(oriPrediction);
oriRank = arrayfun(@(x) find(oriSortedIdx==posIdx(x)),1:length(posIdx));

[~,sortedIdx] = sort(prediction);
rank = arrayfun(@(x) find(sortedIdx==posIdx(x)),1:length(posIdx));

figure(2)
sortValidItem = arrayfun(@(x) find(itemSet(itemSortIdx)==valid(posIdx(x),2)), 1:length(posIdx));
hold on
scatter(sortValidItem,rank-oriRank,60,'s','filled');
line([0 1000], [0 0],'LineWidth',3,'Color','r')
xlabel('Item ID');
ylabel('Rank Difference');
set(gca,'box','off');
set(gca,'FontName','Arial Rounded MT Bold','FontSize',15,'linewidth',3);