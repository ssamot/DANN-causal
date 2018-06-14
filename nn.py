from keras.layers import Input, Dense, Activation, BatchNormalization, Dropout
from keras.layers.merge import concatenate
from keras.models import Model
from keras.optimizers import Adam,SGD
from utils import NormalizedSGD


def build_models(shape,n_neurons):
    #print(shape)
    feature_input = Input(shape = shape)
    treatment_input = Input(shape = (1,))
    internal_feature_input = Input(shape = (2,))

    feature_x = Dense(n_neurons, activation='linear', use_bias=False)(feature_input)
    feature_x = BatchNormalization()(feature_x)
    feature_x = Activation("elu") (feature_x)
    feature_x = Dropout(0.5)(feature_x)


    feature_x = Dense(2, activation='linear', use_bias=False)(feature_x)
    feature_x = BatchNormalization()(feature_x)
    feature_x = Activation("tanh") (feature_x)
    #feature_x = Dropout(0.5)(feature_x)

    effect_regressor = concatenate([treatment_input, feature_x])

    effect_regressor = Dense(n_neurons, activation="linear",  use_bias=False)(effect_regressor)
    effect_regressor = BatchNormalization()(effect_regressor)
    effect_regressor = Activation("elu")(effect_regressor)
    # effect_regressor = Dropout(0.5) (effect_regressor)
    # effect_regressor = Dense(n_neurons, activation="linear", use_bias=False)(effect_regressor)
    # effect_regressor = BatchNormalization()(effect_regressor)
    # effect_regressor = Activation("elu")(effect_regressor)
    # effect_regressor = Dropout(0.5) (effect_regressor)


    effect_regressor = Dense(1, activation="linear", name = "mo", use_bias=True)(effect_regressor)
    #effect_regressor = BatchNormalization(name = "mo")(effect_regressor)


    dom_layers = [
        Dense(n_neurons, activation='linear', name="do4", use_bias=False),
        BatchNormalization(name="do5", center=False, scale=False),
        Activation("elu", name="do6"),
        Dropout(0.5, name="do7"),

        Dense(2, activation='linear', name="do8", use_bias=False),
        BatchNormalization(name="do9", center=False, scale=False),
        Activation("softmax", name="do"),

    ]

    prev_layer = feature_x
    for layer in dom_layers:
        prev_layer = layer(prev_layer)



    model = Model(inputs=[feature_input, treatment_input] , outputs=[effect_regressor, prev_layer])
    model.compile(optimizer=NormalizedSGD(lr = 0.001),
                  loss={'mo': 'mse', 'do': 'categorical_crossentropy'},
                  loss_weights={'mo': 1.0, 'do': 7.0}, metrics=['accuracy'], )

    regressor_model = Model(inputs=[feature_input, treatment_input], outputs=[effect_regressor])
    regressor_model.compile(optimizer=NormalizedSGD(lr = 0.001),
                                        loss={'mo': 'mse'}, metrics=['accuracy'], )

    prev_layer = internal_feature_input
    for layer in dom_layers:
        prev_layer = layer(prev_layer)

    domain_classification_model = Model(inputs=[internal_feature_input], outputs=[prev_layer])
    domain_classification_model.compile(optimizer=NormalizedSGD(lr = 0.001),
                                        loss={'do': 'categorical_crossentropy'}, metrics=['accuracy'], )

    embeddings_model = Model(inputs=[feature_input], outputs=[feature_x])
    embeddings_model.compile(optimizer="SGD", loss='categorical_crossentropy', metrics=['accuracy'])

    return model, regressor_model, domain_classification_model, embeddings_model

