from keras.engine.topology import Layer
from keras import initializers
from keras import backend as K
from keras import constraints
from keras import regularizers

# TimeDistributed vs attentionLayer
# 1. how attention works with both timedistributed and bidirectional?
# 2. how to make the sentence pay attention on an sentence level not word level

# Hierarchical Model with Attention
class AttLayer(Layer):
    def __init__(self, init='glorot_uniform', kernel_regularizer=None, 
                 bias_regularizer=None, kernel_constraint=None, 
                 bias_constraint=None,  **kwargs):
        self.supports_masking = True
        self.init = initializers.get(init)
        self.kernel_initializer = initializers.get(init)
        
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(kernel_regularizer)
        
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)
        
        super(AttLayer, self).__init__(** kwargs)

    def build(self, input_shape):
        assert len(input_shape)==3
        self.W = self.add_weight((input_shape[-1], 1),
                                 initializer=self.kernel_initializer,
                                 name='{}_W'.format(self.name),
                                 regularizer=self.kernel_regularizer,
                                 constraint=self.kernel_constraint)
        self.b = self.add_weight((input_shape[1],),
                                 initializer='zero',
                                 name='{}_b'.format(self.name),
                                 regularizer=self.bias_regularizer,
                                 constraint=self.bias_constraint)
        self.u = self.add_weight((input_shape[1],),
                                 initializer=self.kernel_initializer,
                                 name='{}_u'.format(self.name),
                                 regularizer=self.kernel_regularizer,
                                 constraint=self.kernel_constraint)
        
        self.built = True
        
    def compute_mask(self, input, input_mask=None):
        return None

    def call(self, x, mask=None):
        uit = K.dot(x, self.W) # (x, 40, 1)
        uit = K.squeeze(uit, -1) # (x, 40)
        uit = uit + self.b # (x, 40) + (40,)
        uit = K.tanh(uit) # (x, 40)

        ait = uit * self.u # (x, 40) * (40, 1) => (x, 1)
        ait = K.exp(ait) # (X, 1)

        if mask is not None:
            mask = K.cast(mask, K.floatx()) #(x, 40)
            ait = mask*ait #(x, 40) * (x, 40, )

        ait /= K.cast(K.sum(ait, axis=1, keepdims=True) + K.epsilon(), K.floatx())
        ait = K.expand_dims(ait)
        weighted_input = x * ait
        output = K.sum(weighted_input, axis=1)
        return output
    
    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[-1])
