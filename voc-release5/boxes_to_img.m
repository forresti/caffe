
%call showboxes on all boxes.

cls = 'inriaperson';
conf = voc_config();
cachedir = conf.paths.model_dir;
[pos, neg, impos] = pascal_data(cls, conf.pascal.year);


[model pos] = root_model(cls, pos); %create model; assign templateSize to all 'pos'
%spos = precompute_gt_bbox_features(pos, {pos}, model);
spos = precompute_gt_bbox_features(pos(1:10), {pos(1:10)}, model); %small toy example


for imgId = 1:length(spos{1})

    figure(1)
    img = imread(pos(imgId).im);
    showboxes(img, pos(imgId).boxes); %pos(i).boxes is really just 1 box
    title('ground truth box')


    figure(2)
    rounded_box = spos{1}(imgId).roundedBox_in_px;
    rounded_box_vec = [rounded_box.x1, rounded_box.y1, rounded_box.x2, rounded_box.y2];
    showboxes(img, rounded_box_vec);
    title('box rounded to nearest descriptor')

    
    pause(1)
    %TODO: save...
end


