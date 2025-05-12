%%
imgSize = [64,64,1];

%% YaleFaces
numFacesTrainYale = 100;
dataTrainYale = zeros([imgSize, numFacesTrainYale]);
labelsTrainYale = strings(numFacesTrainYale,1);

% 
for i = 1:numFacesTrainYale
    img = imresize(double(yalefaces(:,:,i))/255, imgSize(1:2));
    dataTrainYale(:,:,1,i) = img;
    labelsTrainYale(i) = "face";
end

labelsTrainYale = categorical(labelsTrainYale);

%% Zbiór Caltech101
load("101pliki.mat");

%% 
dataTrain = cat(4, dataTrainYale, dataTrainCaltech);
labelsTrain = [labelsTrainYale; labelsTrainCaltech];

%% CNN
layers = [
    imageInputLayer(imgSize)
    convolution2dLayer(3,8,'Padding','same')
    batchNormalizationLayer
    reluLayer
    maxPooling2dLayer(2,'Stride',2)
    
    convolution2dLayer(3,16,'Padding','same')
    batchNormalizationLayer
    reluLayer
    maxPooling2dLayer(2,'Stride',2)
    
    fullyConnectedLayer(2)
    softmaxLayer
    ];

options = trainingOptions('adam',...
    'MaxEpochs',10,...
    'MiniBatchSize',16,...
    'Verbose',false,...
    'Plots','training-progress', ...
    'Metrics','accuracy');
