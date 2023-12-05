from keras.layers import Embedding, Reshape, Activation, Input, dot
from keras.models import Sequential, Model

def create_model(graph_nodes_num, dimension):
    left_input = Input(shape=(1,))
    right_input = Input(shape=(1,))
    left_model = Sequential()
    left_model.add(Embedding(input_dim=graph_nodes_num + 1, output_dim=dimension, input_length=1, mask_zero=False))
    left_model.add(Reshape((dimension,)))

    right_model = Sequential()
    right_model.add(Embedding(input_dim=graph_nodes_num + 1, output_dim=dimension, input_length=1, mask_zero=False))
    right_model.add(Reshape((dimension,)))

    left_embed = left_model(left_input)
    right_embed = left_model(right_input)

    left_right_dot = dot(inputs=[left_embed, right_embed], axes=1, name="left_right_dot")
    model = Model(inputs=[left_input, right_input], outputs=[left_right_dot])
    embed_generator = Model(inputs=[left_input, right_input], outputs=[left_embed, right_embed])

    return model, embed_generator
