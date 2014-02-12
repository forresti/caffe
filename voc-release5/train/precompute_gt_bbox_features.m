
% pull out each ground truth bbox from the pyra, for one image.
% use aspect ratios selected by split()

% @param spos{:} -- is split by aspect ratio. contains templateShape information.
% @param pos -- is sorted in order of image ID; boxes with same image ID are grouped together.

% @return spos, with features embedded as spos{component}(boxIdx).feat
%function spos = precompute_gt_bbox_features(pos, spos, model)
function spos = precompute_gt_bbox_features(pos, spos, model)

    imageNames = unique({pos.im});
    %imageNames = imageNames(200:end); % TEST

    % you need to add this to your LD_LIBRARY_PATH (for dependencies of caffe.mex):
    %  export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/path/to/caffe/src/stitch_pyramid

  %DenseNet / Caffe init
    %model_def_file = '../python/caffe/imagenet/imagenet_rcnn_batch_1_input_2000x2000_output_conv5.prototxt'
    %model_file = '../examples/alexnet_train_iter_470000'; % NOTE: you'll have to get the pre-trained ILSVRC network
    %caffe('init', model_def_file, model_file); % init caffe network (spews logging info)
    %caffe('set_mode_gpu');
    %caffe('set_phase_test'); 
    DenseNet_setup();

    for imgIdx = 1:length(imageNames)
        im = imread(imageNames{imgIdx});
imageNames{imgIdx}

      %HOG pyra
        %pyra = featpyramid(im, model);
        %pyra.sbin = model.sbin;

      %DenseNet pyra
        pyra = convnet_featpyramid(imageNames{imgIdx});

        for component = 1:length(spos)
            for pos_example_id = 1:length(spos{component})
                pos_example = spos{component}(pos_example_id);

                if( strcmp(pos_example.im, imageNames{imgIdx})   == 1 ) %if pos_example uses the current image

                    templateSize = pos_example.templateSize;
                    bbox = pos_example; %includes {x1 y1 x2 y2}, plus other debris.
                    [featureSlice, scale, roundedBox_in_px] = get_featureSlice(pyra, bbox, templateSize);
                    spos{component}(pos_example_id).feat = featureSlice;
                    spos{component}(pos_example_id).roundedBox_in_px = roundedBox_in_px;

                    % optional printouts
                    sizeStr = sprintf('size( spos{component = %d}(example id = %d) ) = %s', component, pos_example_id, mat2str(size(featureSlice)));
                    display([sizeStr ''])


                    % optional visualization (if HOG):
                    %figure()
                    %w = foldHOG( spos{component}(pos_example_id).feat );
                    %visualizeHOG(w);

                end
            end
        end

    end

    %TODO: list of gt bboxes per image.
    %TODO: index between spos and pos?
    %TODO: add 'for flip = 0:1 ...' and do with and without flipped image.
end

