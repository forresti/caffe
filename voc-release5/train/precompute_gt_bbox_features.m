
% pull out each ground truth bbox from the pyra, for one image.
% use aspect ratios selected by split()

% @param spos{:} -- is split by aspect ratio. contains templateShape information.
% @param pos -- is sorted in order of image ID; boxes with same image ID are grouped together.

% @return spos, with features embedded as spos{component}(boxIdx).feat
function spos = precompute_gt_bbox_features(pos, spos, model)

    imageNames = unique({pos.im});
    
    for imgIdx = 1:length(imageNames)
        im = imread(imageNames{i});

        pyra = featpyramid(im, model); 

        for component = 1:length(spos)
            for pos_example_id = 1:length(spos{component})
                pos_example = spos{component}(pos_example_id);

                if( strcmp(pos_example.image, imageNames{i})   == 1 ) %if pos_example uses the current image

                    templateSize = pos_example.templateSize;
                    bbox = pos_example; %includes {x1 y1 x2 y2}, plus other debris.
                    [featureSlice, scale] = get_featureSlice(pyra, bbox, templateSize);
                    spos{component}(pos_example_id).feat = featureSlice;
                end
            end
        end

    end

    %TODO: list of gt bboxes per image.
    %TODO: index between spos and pos?

end




