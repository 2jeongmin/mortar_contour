%% model training
imageSize = [1080 1440 3];
numClasses = 2;
network = 'resnet18';
lgraph = deeplabv3plusLayers(imageSize,numClasses,network, ...
             'DownsamplingFactor',16);
pximds = pixelLabelImageDatastore(gTruth); %gTruth from image labeler
opts = trainingOptions('sgdm',...
    'MiniBatchSize',4,...
    'MaxEpochs',10, ...
    'Plots','training-progress'); %training_options
net = trainNetwork(pximds,lgraph,opts); %training start
%% test_result visualization (mortar embossing detection)
I = imread(['test\test_01.jpg']); %change test_**.jpg if necessary
[C, ~, all_scores] = semanticseg(I,net);
figure(1023); imagesc(all_scores(:,:,1))
%% make ROI mask
SE = strel('rectangle' , [40 3])
Se = strel('disk', 10, 8)

R = labeloverlay(I, imdilate(all_scores(:,:,1) > 0.66, SE));
figure(1022); imshow(R)
%% rough contour lines
r = imdilate(all_scores(:,:,1) > 0.66, SE);
re = imerode(r, Se);
o = all_scores(:,:,1) > 0.66;

IV = re-o;
IV(IV < 0 ) = 0;
E = labeloverlay(I,IV);
figure(1021); imshow(E)
%% calculate each centroid
centers_list = [];

for i = 1:1430
    
    body_portion = IV;
    body_portion(:, 1:i) = 0;
    body_portion(:, i+2:end) = 0;
    figure(1020); imshow(body_portion);
    title([ 'X axis =' num2str(i-1) ] )
    
    centers = regionprops(bwlabel(body_portion), 'centroid');
    centers_list = [centers_list; centers];
    
end
%% contour lines
A = zeros(1080,1440, 'uint8');
for iter = 1:size(centers_list, 1) 
    
    x_cord_cent = int64(centers_list(iter).Centroid(2));
    y_cord_cent = int64(centers_list(iter).Centroid(1));
    
    A(x_cord_cent, y_cord_cent) = 255; 
    
end

bw = bwareaopen(A, 15);
%bw2 = bwperim(bw,8); 
%bw3 = imdilate(bw2, strel('disk',1)); % contour lines 1px to 3px for visualization

D = labeloverlay(I, bw);
figure(1019); imagesc(D);
%% loss calculation
AnsbyHand_2px = imread(['wrinklebyHand\40_1_bcl.png']); % hand-drawing by 2px pen, change ***_*_bcl.png if necessary
AnsbyHand_1px = imbinarize(rgb2gray(AnsbyHand_2px));
AnsbyHand = bwskel(AnsbyHand_1px); % 2px to 1px hand-drawing

denominator = sum(AnsbyHand, 'all')
numerator = sum(bw, 'all')
loss = 100*(1 - numerator/denominator)