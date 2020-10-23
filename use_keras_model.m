model_base_path = '/users/lww/data/tom/checkpoint/3D_hanh-original-no5_split-method_1lpgo_cond_pass-zeroth_conv_7-5-3_basech_8_act_relu_pool_max_betw_flatten_fc_0_zscore_True_batch_32_output_1_DO_075_BN_X_backup/LPGO/None/01_p';

subject_data_base_path = '/users/kdy/tmp';


%%% Load models to cell array
loocv_model_list = dir(model_base_path);
valid_index_list = [];
temp_i = 1;
for i = 1:length(loocv_model_list)
    if endsWith(loocv_model_list(i).name, 'th_fold') == 1
        valid_index_list(temp_i) = i;
        temp_i = temp_i + 1;
    end;
end;
loocv_model_list = loocv_model_list(valid_index_list);

all_models = {};
for i = 1:length(loocv_model_list)
    loocv_model_path = fullfile(model_base_path, loocv_model_list(i).name);
    temp_model_list = dir(loocv_model_path);
    model_filename = temp_model_list(length(temp_model_list)).name;
    model_full_path = fullfile(loocv_model_path, model_filename);
    
%     disp(model_full_path)    
% 
%     layers = [
%         image3dInputLayer([53, 63, 46, 1])
% 
%         convolution3dLayer(7,8,'Padding','same')
%         reluLayer()
%         maxPooling3dLayer(2,'Stride',2)
% 
%         convolution3dLayer(5,16,'Padding','same')
%         reluLayer()
%         maxPooling3dLayer(2,'Stride',2)
% 
%         convolution3dLayer(3,32,'Padding','same')
%         reluLayer()
%         maxPooling3dLayer(2,'Stride',2)
%         
%         nnet.keras.layer.FlattenCStyleLayer('flatten_layer')
%         dropoutLayer(0.75)
% 
%         fullyConnectedLayer(1)
%         nnet.keras.layer.SigmoidLayer('output_layer')
%     ];

    net = importKerasLayers(model_full_path, 'ImportWeights', true);
    placeholderLayers = findPlaceholderLayers(net);
    for layer_i = 1:length(placeholderLayers)
        target_layer = placeholderLayers(layer_i,1) ;
        target_name = target_layer.Name;
        if endsWith(target_name, 'input_layer') == 1
            net = replaceLayer(net,target_name,image3dInputLayer([53, 63, 46, 1], 'name',target_name, 'Normalization', 'none'));
        elseif endsWith(target_name, 'conv') == 1
            target_weights = permute(target_layer.Weights.kernel, [5 4 3 2 1]);
            target_bias = reshape(permute(target_layer.Weights.bias, [2 1]), 1, 1, 1, []);
            target_shape = size(target_weights);
            net = replaceLayer(net,target_name,convolution3dLayer(target_shape(1),target_shape(5), 'Padding','same', 'name',target_name, 'Weights',target_weights, 'Bias',target_bias));
        elseif endsWith(target_name, 'pool') == 1
            net = replaceLayer(net,target_name,maxPooling3dLayer(2,'Stride',2, 'name',target_name));
        end
    end
    target_name = 'flatten_layer';
    net = replaceLayer(net,target_name,FlattenCStyle3dLayer(target_name));    

    convnet = assembleNetwork(net);
    
    all_models{i} = convnet;

%     figure
%     plot(net);
%     title('DAG Network Architecture')    

%     t=[];
%     for i = 1:1000
%         t0 = clock();
%         
%         input_sample = rand(53, 63, 46, 1, 1);
%         
%         y_preds = predict(convnet, input_sample);
% 
%         t=[t; etime(clock, t0)];
%     end
end;


%%% Reconstruct oiriginal performance
base_subejct_list = [
    'sub02'
    'sub05'
    'sub06'
    'sub07'
    'sub09'
    'sub10'
    'sub11'
    'sub12'
    'sub17'
    'sub18'
    'sub20'
    'sub21'
    'sub22'
    'sub24'
    'sub26'
    'sub30'
    'sub32'
    'sub33'
];
mV = spm_read_vols(spm_vol(fullfile(subject_data_base_path, 'mask.img')));

all_preds = [];
all_reals = [];
for i = 1:length(base_subejct_list)
    subject_data_path = fullfile(subject_data_base_path, base_subejct_list(i,:), 'stat/beta_mtx_tom3_dnn.mat');
    load(subject_data_path);
    tV = zeros([size(mV) 1 20]);
    tmp_tV = zeros(size(mV));
    idx_tmp = [11:20 31:40];
    for j = idx_tmp
        tscored = betas(j,:);
        zscored = zeros(size(tscored));
        idx_nan = isnan(tscored);
        zscored(~idx_nan) = zscore(tscored(~idx_nan));
        tmp_tV(find(mV)) = zscored;
        tV(:,:,:,1,find(idx_tmp==j)) = tmp_tV;
    end

    y_preds = predict(all_models{i}, tV);
    all_preds(i,:) = squeeze(y_preds);
    all_reals(i,1:10) = 0;
    all_reals(i,11:20) = 1;
end

all_preds(all_preds>=0.5) = 1;
all_preds(all_preds<0.5) = 0;

all_results = (all_reals == all_preds);
accuracy_per_subject = mean(all_results, 2);

% mean(accuracy_per_subject) %%% 0.7639...?
figure();
bar(accuracy_per_subject);
xticks([1:length(base_subejct_list)]);
xticklabels(base_subejct_list);
xtickangle(90);
xlabel('Accuracy');
ylim([0 1]);
title(sprintf('Accuracy per subject, Average accuracy: %0.4f', mean(accuracy_per_subject)));


%%% Evaluate other 5 subject
base_subejct_list = [
    'sub03' 
    'sub25' 
    'sub27' 
    'sub28' 
    'sub29' 
];
mV = spm_read_vols(spm_vol(fullfile(subject_data_base_path, 'mask.img')));

all_preds = [];
all_reals = [];
for i = 1:length(base_subejct_list)
    subject_data_path = fullfile(subject_data_base_path, base_subejct_list(i,:), 'stat/beta_mtx_tom3_dnn.mat');
    load(subject_data_path);
    tV = zeros([size(mV) 1 20]);
    tmp_tV = zeros(size(mV));
    idx_tmp = [11:20 31:40];
    for j = idx_tmp
        tscored = betas(j,:);
        zscored = zeros(size(tscored));
        idx_nan = isnan(tscored);
        zscored(~idx_nan) = zscore(tscored(~idx_nan));
        tmp_tV(find(mV)) = zscored;
        tV(:,:,:,1,find(idx_tmp==j)) = tmp_tV;
    end
    for model_index = 1:length(all_models)
        y_preds = predict(all_models{model_index}, tV);
        all_preds(i,model_index,:) = squeeze(y_preds);
        all_reals(i,model_index,1:10) = 0;
        all_reals(i,model_index,11:20) = 1;
    end
end;

% Each LOSOCV model's accuracy
temp_all_preds = all_preds;
temp_all_preds(temp_all_preds>=0.5) = 1;
temp_all_preds(temp_all_preds<0.5) = 0;

all_results = (all_reals == temp_all_preds);
accuracy_per_subject = mean(all_results, 3);

accuracy_per_model = mean(accuracy_per_subject, 1)

target_ticks = {};
for tick_i = 1:length(accuracy_per_model) 
    target_ticks{tick_i} = sprintf('Model %02d', tick_i);
end

% figure();
% bar(accuracy_per_model);
% xticks([1:length(accuracy_per_model) ]);
% xticklabels(target_ticks);
% xtickangle(90);
% xlabel('Accuracy');
% ylim([0 1]);
% title('Accuracy per model');

%%% Soft voting
temp_all_preds = all_preds;
temp_all_soft_preds = squeeze(mean(temp_all_preds, 2));
temp_all_soft_preds(temp_all_soft_preds>=0.5) = 1;
temp_all_soft_preds(temp_all_soft_preds<0.5) = 0;

soft_voting_results = (squeeze(all_reals(:,1,:)) == temp_all_soft_preds);
soft_voting_accuracy_per_subject = mean(soft_voting_results, 2);

%%% Hard voting
temp_all_preds = all_preds;
temp_all_preds(temp_all_preds>=0.5) = 1;
temp_all_preds(temp_all_preds<0.5) = 0;

hard_voting_results = squeeze(sum(temp_all_preds, 2));
hard_voting_results(hard_voting_results>=10) = 1;
hard_voting_results(hard_voting_results<10) = 0;

hard_voting_results = (squeeze(all_reals(:,1,:)) == hard_voting_results);
hard_voting_accuracy_per_subject = mean(hard_voting_results, 2);

accuracy_per_condition = accuracy_per_model;
accuracy_per_condition(length(accuracy_per_model)+1) = mean(soft_voting_accuracy_per_subject);
accuracy_per_condition(length(accuracy_per_model)+2) = mean(hard_voting_accuracy_per_subject);

all_target_ticks = target_ticks
tick_i = tick_i+1;
all_target_ticks{tick_i} = 'Soft voting';
tick_i = tick_i+1;
all_target_ticks{tick_i} = 'Hard voting';

figure();
bar(accuracy_per_condition);
xticks([1:length(accuracy_per_condition) ]);
xticklabels(all_target_ticks);
xtickangle(90);
xlabel('Accuracy');
ylim([0 1]);
title('Accuracy per ensemble type');


% Each subject performance
for i = 1:length(base_subejct_list)
    target_losocv_result = squeeze(accuracy_per_subject(i,:));
    target_soft_voting_result = soft_voting_accuracy_per_subject(i);
    target_hard_voting_result = hard_voting_accuracy_per_subject(i);
    
    accuracy_per_condition = target_losocv_result;
    accuracy_per_condition(length(accuracy_per_model)+1) = mean(target_soft_voting_result);
    accuracy_per_condition(length(accuracy_per_model)+2) = mean(target_hard_voting_result);
    
    figure();
    bar(accuracy_per_condition);
    xticks([1:length(accuracy_per_condition) ]);
    xticklabels(all_target_ticks);
    xtickangle(90);
    xlabel('Accuracy');
    ylim([0 1]);
    title(sprintf('Accuracy for each subject (%s)', base_subejct_list(i,:)));
    saveas(gcf,sprintf('matlab_vis/Accuracy_for_each_subject_%s.png', base_subejct_list(i,:)));
end;


% Each subject and sample performance for voting
for i = 1:length(base_subejct_list)
    soft_voting_temp = squeeze(mean(all_preds, 2));
    target_soft_voting_result = soft_voting_temp(i,:);
    hard_voting_temp = squeeze(sum(temp_all_preds, 2));
    target_hard_voting_result = hard_voting_temp(i,:);
    
    figure();
    bar(target_soft_voting_result);
%     xticks([1:length(accuracy_per_condition) ]);
%     xticklabels(all_target_ticks);
%     xtickangle(90);
    xlabel('Probability');
    ylim([0 1]);
    title(sprintf('Soft voting for each subject (%s)', base_subejct_list(i,:)));
    saveas(gcf,sprintf('matlab_vis/Soft_voting_for_each_subject_%s.png', base_subejct_list(i,:)));
    
    figure();
    bar(target_hard_voting_result);
%     xticks([1:length(accuracy_per_condition) ]);
%     xticklabels(all_target_ticks);
%     xtickangle(90);
    xlabel('Count');
    ylim([0 20]);
    title(sprintf('Hard voting for each subject (%s)', base_subejct_list(i,:)));
    saveas(gcf,sprintf('matlab_vis/Hard_voting_for_each_subject_%s.png', base_subejct_list(i,:)));    
end;
