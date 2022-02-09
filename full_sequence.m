
imageSize = [1080 1440 3];
numClasses = 2;
network = 'resnet18';
lgraph = deeplabv3plusLayers(imageSize,numClasses,network, ...
             'DownsamplingFactor',16);
%%
%gTruth from image labeler

pximds = pixelLabelImageDatastore(gTruth); 

%%
%training_options

opts = trainingOptions('sgdm',...
    'MiniBatchSize',4,...
    'MaxEpochs',10, ...
    'Plots','training-progress');
%%

net = trainNetwork(pximds,lgraph,opts);

%%

%test_result visualization

I = imread(['test\test_07.jpg']);

[C, ~, all_scores] = semanticseg(I,net);

%figure(1023); imagesc(all_scores(:,:,1))

%%
%thresholding
% 
% se = strel('rectangle', [3 30]);
% B = labeloverlay(I, imerode(all_scores(:,:,1) > 0.66, se )); %embossing B
% 
% K = imerode(all_scores(:,:,1) > 0.66, se);
% [x_cord, y_cord]= find(K);
% convex = convhull(x_cord, y_cord);
% convex_x_cord = x_cord(convex);
% convex_y_cord = y_cord(convex);
% 
% polymask = poly2mask(convex_y_cord, convex_x_cord, size(K, 1), size(K, 2));
% 
% 
% J = 1-K;
% J(polymask == 0) = 0;
% J = imerode(J, se);
% B = labeloverlay(I, J); %Wrinkles B
% figure(1022); imshow(B)         
%%
%let the mortar embossing get thicker, and make them to stick together
%,then they'll be ROI mask
SE = strel('rectangle' , [40 3])
Se = strel('disk', 10, 8)

R = labeloverlay(I, imdilate(all_scores(:,:,1) > 0.66, SE));
figure(1021); imshow(R)

r = imdilate(all_scores(:,:,1) > 0.66, SE);
re = imerode(r, Se);
o = all_scores(:,:,1) > 0.66;

IV = re-o;
IV(IV < 0 ) = 0;
E = labeloverlay(I,IV);
figure(1020); imshow(E)

%%
centers_list = [];

for i = 1:1430
    
%     sE = strel('disk',2);
%     IV = imopen(IV,sE);
    body_portion = IV;
    body_portion(:, 1:i) = 0;
    body_portion(:, i+2:end) = 0;
%     figure(1019); imshow(body_portion);
%     title([ 'X axis =' num2str(i-1) ] )
    
    centers = regionprops(bwlabel(body_portion), 'centroid');
    centers_list = [centers_list; centers];
    
end

%%
A = zeros(1080,1440, 'uint8');
for iter = 1:size(centers_list, 1) 
    
    x_cord_cent = int64(centers_list(iter).Centroid(2));
    y_cord_cent = int64(centers_list(iter).Centroid(1));
    
    A(x_cord_cent, y_cord_cent) = 255; 
    
end

bw = bwareaopen(A, 15);

D = labeloverlay(I, bw);
figure(1018); imagesc(D);



% D = imfuse(A,I, 'ColorChannels' , [1 2 0]);
% figure(1021); imshow(D);