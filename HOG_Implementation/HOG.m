tic;
filename = 'sat-4-full.mat';
S = load(filename);
train_x=S.train_x;
train_y=S.train_y;
test_x=S.test_x;
test_y=S.test_y;

% 2-barren land
% 3-trees
% 4-grassland

exImage=train_x(:,:,1:3,4);
processedImage = imbinarize(rgb2gray(exImage));
[featureVector,hogVisualization] = extractHOGFeatures(processedImage);
% figure;
[hog_2x2, vis2x2] = extractHOGFeatures(exImage,'CellSize',[2 2]);
[hog_4x4, vis4x4] = extractHOGFeatures(exImage,'CellSize',[4 4]);
[hog_5x5, vis5x5] = extractHOGFeatures(exImage,'CellSize',[5 5]);
[hog_8x8, vis8x8] = extractHOGFeatures(exImage,'CellSize',[8 8]);
% 
% subplot(1,2,1); imshow(exImage);

% % Visualize the HOG features
% subplot(1,2,2);  
% plot(vis2x2); 
% title({'CellSize = [2 2]'; ['Length = ' num2str(length(hog_2x2))]});

% subplot(2,3,5);
% plot(vis4x4); 
% title({'CellSize = [4 4]'; ['Length = ' num2str(length(hog_4x4))]});

% subplot(2,3,5);
% plot(vis5x5); 
% title({'CellSize = [5 5]'; ['Length = ' num2str(length(hog_4x4))]});
% 
% subplot(2,3,6);
% plot(vis8x8); 
% title({'CellSize = [8 8]'; ['Length = ' num2str(length(hog_8x8))]});

cellSize = [4 4];
hogFeatureSize = length(hog_4x4);


numImages=size(train_x,4);
% for i = 1:numImages
%     OriImg = train_x(:,:,1:3,i);
%     img = rgb2gray(OriImg);
%     
%     % Apply pre-processing steps
%     img = imbinarize(img);
%     
%     [trainingFeatures(i, :),visualization] = extractHOGFeatures(img, 'CellSize', cellSize);  
%     fprintf("%d\n",i);
% end
% 
% trainingLabels=train_y;
filename = 'sat-4-trainingLabels.mat';
trainingLabels = load(filename);
% fitcecoc uses SVM learners and a 'One-vs-One' encoding scheme.
classifier = fitcecoc(trainingFeatures, trainingLabels.training_labels);
save('classifier.mat','classifier')
% Extract HOG features from the test set. The procedure is similar to what
% was shown earlier and is encapsulated as a helper function for brevity.
% [testFeatures, testLabels] = helperExtractHOGFeaturesFromImageSet(testSet, hogFeatureSize, cellSize);
[testFeatures, testLabels] = helperExtractHOGFeaturesFromImageSet(testSet, hogFeatureSize, cellSize);
% numImages = numel(testSet.Files);
% testFeatures = zeros(numImages, hogFeatureSize, 'single');
% for i=1:numImages
%     img = readimage(testSet, i);
%     img = rgb2gray(img);
%     
%     % Apply pre-processing steps
%     img = imbinarize(img);
%     testFeatures(i, :) = extractHOGFeatures(img,'CellSize',cellSize);
%     
% end
% 
% % Get labels for each image.
% testLabels = testSet.Labels;


% Make class predictions using the test features.
load('classifier.mat')
predictedLabels = predict(classifier, testFeatures);

% Tabulate the results using a confusion matrix.
confMat = confusionmat(testLabels, predictedLabels);

helperDisplayConfusionMatrixNew(confMat)

toc;
