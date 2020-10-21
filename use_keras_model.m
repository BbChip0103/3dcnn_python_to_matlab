model_base_path = '/users/lww/data/tom/checkpoint/3D_hanh-original-no5_split-method_1lpgo_cond_pass-zeroth_conv_7-5-3_basech_8_act_relu_pool_max_betw_flatten_fc_0_zscore_True_batch_32_output_1_DO_075_BN_X_backup/LPGO/None/01_p'
loocv_model_list = dir(model_base_path)

for i = 1:length(loocv_model_list)
    if endsWith(loocv_model_list(i).name, 'th_fold') == 0
        continue
    end
    loocv_model_path = fullfile(model_base_path, loocv_model_list(i).name)
    temp_model_list = dir(loocv_model_path)
    model_filename = temp_model_list(length(temp_model_list)).name
    model_full_path = fullfile(loocv_model_path, model_filename)
    
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

    input_sample = rand(53, 63, 46, 1, 3)

    net = importKerasLayers(model_full_path, 'ImportWeights', true);
    placeholderLayers = findPlaceholderLayers(net);
    for layer_i = 1:length(placeholderLayers)
        target_layer = placeholderLayers(layer_i,1) 
        target_name = target_layer.Name
        if endsWith(target_name, 'input_layer') == 1
            net = replaceLayer(net,target_name,image3dInputLayer([53, 63, 46, 1], 'name',target_name, 'Normalization', 'none'));
        elseif endsWith(target_name, 'conv') == 1
            target_weights = permute(target_layer.Weights.kernel, [5 4 3 2 1])
            target_bias = reshape(permute(target_layer.Weights.bias, [2 1]), 1, 1, 1, [])
            target_shape = size(target_weights)
            net = replaceLayer(net,target_name,convolution3dLayer(target_shape(1),target_shape(5), 'Padding','same', 'name',target_name, 'Weights',target_weights, 'Bias',target_bias));
        elseif endsWith(target_name, 'pool') == 1
            net = replaceLayer(net,target_name,maxPooling3dLayer(2,'Stride',2, 'name',target_name));
        end
    end
    target_name = 'flatten_layer'
    net = replaceLayer(net,target_name,FlattenCStyle3dLayer(target_name));    

    convnet = assembleNetwork(net)

%     figure
%     plot(net);
%     title('DAG Network Architecture')    
    
    y_preds = predict(convnet, input_sample)
    y_preds
end
