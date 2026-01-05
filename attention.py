from tensorflow.keras.layers import Layer, Conv2D, Multiply

class AttentionLayer(Layer):
    def build(self, input_shape):
        filters = input_shape[-1]
        self.context_conv1 = Conv2D(filters, (1,1), padding='same', activation='relu')
        self.context_conv2 = Conv2D(filters, (1,1), padding='same', activation='sigmoid')
        super().build(input_shape)

    def call(self, inputs):
        context = self.context_conv1(inputs)
        context = self.context_conv2(context)
        return Multiply()([inputs, context])
