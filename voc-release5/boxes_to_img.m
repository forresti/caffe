
%call showboxes on all boxes.

cls = 'inriaperson';
conf = voc_config();
cachedir = conf.paths.model_dir;
[pos, neg, impos] = pascal_data(cls, conf.pascal.year);

for imgId = 1:length(pos)

    img = imread(pos(imgId).im);
    showboxes(img, [pos(i).boxes]); %pos(i).boxes is really just 1 box

    %TODO: save...
    pause(1);
end


