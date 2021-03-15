#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 15 17:42:28 2021

Script to build and train an LSTM network for generating melodies/riffs, using
the dataset generated from a collection of MIDI files. Requires preprocess.py
to run and generate training sequences. Upon completion of training, the model
is saved in .h5 format.

Expanded upon the original work of Valerio Velardo/musikalkemist for 
non-commercial use.

@author: musikalkemist/nk
"""
#%% Dependencies: %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
import tensorflow.keras as keras
from preprocess import generate_training_sequences, SEQUENCE_LENGTH

#%% Paths/Variables/Parameters: %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
NUM_UNITS = [256]
LOSS = "sparse_categorical_crossentropy"
LEARNING_RATE = 0.001
EPOCHS = 90
BATCH_SIZE = 64
SAVE_MODEL_PATH = "RiffGen_model.h5"

#%% Define Model: %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def build_model(output_units, num_units, loss, learning_rate):
    """Builds and compiles model
    :param output_units (int): Num output units
    :param num_units (list of int): Num of units in hidden layers
    :param loss (str): Type of loss function to use
    :param learning_rate (float): Learning rate to apply
    :return model (tf model): Where the magic happens :D
    """

    # create the model architecture
    input = keras.layers.Input(shape=(None, output_units))
    x = keras.layers.LSTM(num_units[0])(input)
    x = keras.layers.Dropout(0.2)(x)

    output = keras.layers.Dense(output_units, activation="softmax")(x)

    model = keras.Model(input, output)

    # compile model
    model.compile(loss=loss,
                  optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
                  metrics=["accuracy"])

    model.summary()

    return model

#%% Training Method: %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if __name__ == "__main__":
    '''
    Instead of a train function I have chosen this method, which allows for
    greater flexibility depending on the number of different pitches and chords
    that are included in the dataset (OUTPUT_UNITS parameter).
    '''
    # load the training sequences
    INPUTS, TARGETS = generate_training_sequences(SEQUENCE_LENGTH)
    
    # Calculate output units
    OUTPUT_UNITS = INPUTS.shape[-1]
    
    # Build model
    RiffGen_model = build_model(OUTPUT_UNITS, NUM_UNITS, LOSS, LEARNING_RATE)
    
    # Fit model on data
    RiffGen_model.fit(INPUTS, TARGETS, epochs=EPOCHS, batch_size=BATCH_SIZE)
    
    # Save trained model
    RiffGen_model.save(SAVE_MODEL_PATH)