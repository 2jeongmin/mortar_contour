%%
AnsbyHand_2px = imread(['wrinklebyHand\130_1_bcl.png']);
%imshow(AnsbyHand_2px)
AnsbyHand_1px = imbinarize(rgb2gray(AnsbyHand_2px));
%imshow(AnsbyHand_1px)
AnsbyHand = bwskel(AnsbyHand_1px);
%imshow(AnsbyHand)
%imshow(bw)

%% accuracy index
% upper = sum(AnsbyHand == 1 & bw == 1,'all');
% %imshow(AnsbyHand == 1 & bw == 1)
% lower = sum(AnsbyHand, 'all');
% 
% acc = (upper/lower)*100 

%% loss index
denominator = sum(AnsbyHand, 'all');
numerator = sum(bw, 'all');

loss = 100 - (numerator/denominator)*100

%%
vis = strel('disk', 1)
bw_vis = imdilate(bw, vis);
figure(1); imshow(bw_vis)
imwrite(bw_vis, 'hand_drawing_comparison\det\130_1_det.png');
AbH_vis = imdilate(AnsbyHand, vis);
figure(2); imshow(AbH_vis)
imwrite(AbH_vis, 'hand_drawing_comparison\AbH\130_1_AbH.png');