testImagesDir = fullfile('C:\Users\oh978\Desktop\classification\definition');
testLabelsDir = fullfile('C:\Users\oh978\Desktop\classification\PixelLabelData_25');

imds = imageDatastore(testImagesDir);

classNames = ["Girder", "Pier" ,"Background"];
labelIDs = [1 2 3];

pxdsTruth = pixelLabelDatastore(testLabelsDir,classNames,labelIDs);

net = load(fullfile('network','200803_net_epochs20_batchsize_2_net'))
net = net.net;

pxdsResults = semanticseg(imds,net,"WriteLocation",tempdir);

metrics = evaluateSemanticSegmentation(pxdsResults,pxdsTruth);

metrics.ClassMetrics

metrics.NormalizedConfusionMatrix