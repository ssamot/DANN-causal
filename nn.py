from keras.layers import Input, Dense, Activation, BatchNormalization, Dropout
from keras.layers.merge import concatenate
from keras.models import Model


def build_models(shape,n_neurons):
    #print(shape)
    feature_input = Input(shape = shape)
    treatment_input = Input(shape = (1,))

    feature_x = Dense(n_neurons, activation='linear')(feature_input)
    feature_x = BatchNormalization()(feature_x)
    feature_x = Activation("elu") (feature_x)
    feature_x = Dropout(0.5)(feature_x)
    feature_x = BatchNormalization()(feature_x)
    feature_x = Activation("elu")(feature_x)
    feature_x = Dropout(0.5)(feature_x)
    feature_x = Dense(2, activation='linear')(feature_x)
    feature_x = BatchNormalization()(feature_x)
    feature_x = Activation("tanh") (feature_x)
    #feature_x = Dropout(0.5)(feature_x)

    effect_regressor = concatenate([treatment_input, feature_x])

    effect_regressor = Dense(n_neurons, activation="linear")(effect_regressor)
    effect_regressor = BatchNormalization()(effect_regressor)
    effect_regressor = Activation("elu")(effect_regressor)
    effect_regressor = Dropout(0.5) (effect_regressor)
    effect_regressor = Dense(n_neurons, activation="linear")(effect_regressor)
    effect_regressor = BatchNormalization()(effect_regressor)
    effect_regressor = Activation("elu")(effect_regressor)
    effect_regressor = Dropout(0.5) (effect_regressor)

    effect_regressor = Dense(1, activation="linear", name = "mo3")(effect_regressor)
    effect_regressor = BatchNormalization(name = "mo")(effect_regressor)

    domain_classifier = Dense(n_neurons, activation='linear', name="do4")(feature_x)
    domain_classifier = BatchNormalization(name="do5")(domain_classifier)
    domain_classifier = Activation("elu", name="do6")(domain_classifier)
    domain_classifier = Dropout(0.5, name="do7")(domain_classifier)

    domain_classifier = Dense(2, activation='linear', name="do8")(domain_classifier)
    domain_classifier = BatchNormalization(name="do9")(domain_classifier)
    domain_classifier = Activation("softmax", name = "do")(domain_classifier)

    model = Model(inputs=[feature_input, treatment_input] , outputs=[effect_regressor, domain_classifier])
    model.compile(optimizer="Adam",
                  loss={'mo': 'mse', 'do': 'categorical_crossentropy'},
                  loss_weights={'mo': 1, 'do': 1}, metrics=['accuracy'], )

    regressor_model = Model(inputs=[feature_input, treatment_input], outputs=[effect_regressor])
    regressor_model.compile(optimizer="Adam",
                                        loss={'mo': 'mse'}, metrics=['accuracy'], )

    domain_classification_model = Model(inputs=[feature_input], outputs=[domain_classifier])
    domain_classification_model.compile(optimizer="Adam",
                                        loss={'do': 'categorical_crossentropy'}, metrics=['accuracy'], )

    embeddings_model = Model(inputs=[feature_input], outputs=[feature_x])
    embeddings_model.compile(optimizer="SGD", loss='categorical_crossentropy', metrics=['accuracy'])

    return model, regressor_model, domain_classification_model, embeddings_model

