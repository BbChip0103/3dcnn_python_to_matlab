classdef FlattenCStyle3dLayer < nnet.layer.Layer
% FlattenCStyle3dLayer
    
    %   Copyright 2017-2018 The MathWorks, Inc.
    methods
        function this = FlattenCStyle3dLayer(name)
            this.Name = name;
            this.Description = getString(message('nnet_cnn_kerasimporter:keras_importer:FlattenDescription'));
            this.Type = getString(message('nnet_cnn_kerasimporter:keras_importer:FlattenType'));
        end
        
        function Z = predict( this, X )
            % X is size [H W D C N].
            % Z is size [1 1 HWDC N].
            [sz1, sz2, sz3, sz4, sz5] = size(X);
            Z = reshape(permute(X,[4 3 2 1 5]), [1 1 1 sz1*sz2*sz3*sz4 sz5]);
        end
        
        function dLdX = backward( this, X, Z, dLdZ, memory )
            % dLdZ is size [1 1 HWDC N].
            % dLdX and X are size [H W D C N].
            [sz1, sz2, sz3, sz4, sz5] = size(X);
            dLdX = permute(reshape(dLdZ, [sz4 sz3 sz2 sz1 sz5]), [4 3 2 1 5]);
        end
    end
end

