from keras.layers import Embedding, Reshape, Merge, Activation, Input, merge
from keras.models import Sequential, Model


def create_model(numNodes, factors):

    left_input = Input(shape=(1,))
    right_input = Input(shape=(1,))

    left_model = Sequential()
    left_model.add(Embedding(input_dim=numNodes + 1, output_dim=factors, input_length=1, mask_zero=False))
    left_model.add(Reshape((factors,)))

    right_model = Sequential()
    right_model.add(Embedding(input_dim=numNodes + 1, output_dim=factors, input_length=1, mask_zero=False))
    right_model.add(Reshape((factors,)))

    left_embed = left_model(left_input)
    right_embed = left_model(right_input)

    left_right_dot = merge([left_embed, right_embed], mode="dot", dot_axes=1, name="left_right_dot")

    model = Model(input=[left_input, right_input], output=[left_right_dot])
    embed_generator = Model(input=[left_input, right_input], output=[left_embed, right_embed])

    return model, embed_generator