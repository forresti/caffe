
%from here: http://www.cs.berkeley.edu/~rbg/d_and_t.tgz

function model = pascal_train(cls, note)
% Train a model.
%   model = pascal_train(cls, note)
%
% Trains a Dalal & Triggs model.
%
% Arguments
%   cls           Object class to train and evaluate
%                 (The final model has 2*n components)
%   note          Save a note in the model.note field that describes this model

% At every "checkpoint" in the training process we reset the 
% RNG's seed to a fixed value so that experimental results are 
% reproducible.
seed_rand();

% Default to no note
if nargin < 2
  note = '';
end

conf = voc_config();
cachedir = conf.paths.model_dir;

% Load the training data
[pos, neg, impos] = pascal_data(cls, conf.pascal.year);

max_num_examples = conf.training.cache_example_limit;
num_fp           = conf.training.wlssvm_M;
fg_overlap       = conf.training.fg_overlap;

% Select a small, random subset of negative images
% All data mining iterations use this subset, except in a final
% round of data mining where the model is exposed to all negative
% images
num_neg   = length(neg);
neg_large = neg; % use all of the negative images

save_file = [cachedir cls '_final'];
try
  load(save_file);
catch
  [model pos] = root_model(cls, pos, note); %create model; assign templateSize to all 'pos'

  % pos{}.feat = precompute features all positives, in the aspect ratio of appropriate all components
  spos = precompute_gt_bbox_features(pos, {pos}, model);
  %spos = precompute_gt_bbox_features(pos(1:10), {pos(1:10)}, model); %small toy example
  pos = spos{1}; %has extra indirection for the multi-component use case.  

  % Get warped positives and random negatives
  model = train(model, pos, neg_large, true, true, 1, 1, ...
                max_num_examples, fg_overlap, 0, false, 'init');
  % Finish training by data mining on all of the negative images
  model = train(model, pos, neg_large, false, false, 1, 10, ...
                max_num_examples, fg_overlap, num_fp, true, 'hard_neg');
  save(save_file);
end
