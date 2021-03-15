#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 15 17:14:20 2021

Preprocessing functions for the melody/riff generator model. When run, this 
script will read in the MIDI files stored in a directory and generate a single
file dataset containing the encoded information of the notes, as well as a 
corresponding json file for the mappings. It will accept polyphony, as long as
the information is stored within a single channel of the MIDI file.

Expanded upon the original work of Valerio Velardo/musikalkemist for 
non-commercial use.

@author: musikalkemist/nk
"""
#%% Dependencies: %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
import os
import json
import music21 as m21
import numpy as np
import tensorflow.keras as keras

#%% Paths/Variables/Parameters: %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
MIDI_DATASET_PATH = "./MIDI/"
SAVE_DIR = "./dataset"
SINGLE_FILE_DATASET = "file_dataset"
MAPPING_PATH = "mapping.json"
SEQUENCE_LENGTH = 64

#%% Preprocessing Functions: %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def load_songs_in_midi(dataset_path):
    """Loads all midi pieces in dataset using music21.
    :param dataset_path (str): Path to dataset
    :return songs (list of m21 streams): List containing all pieces
    
    NK - the original function was designed to load from .krn files, 
    I have modified this to load regular MIDIs (.mid).
    """
    songs = []

    # go through all the files in dataset and load them with music21
    for path, subdirs, files in os.walk(dataset_path):
        for file in files:

            # consider only MIDI files
            if file[-3:] == "mid":
                song = m21.converter.parse(os.path.join(path, file))
                songs.append(song)
    return songs

def encode_song(song, time_step=0.25):
    """Converts a score into a time-series-like music representation. 
    Each item in the encoded list represents 'min_duration' quarter lengths. 
    The symbols used at each step are: integers for MIDI notes, including the 
    root of a chord, 'r' for representing a rest, and '_' for representing 
    durations carried over into a new time step. Here's a sample encoding:
        ["r", "_", "60", "_", "_", "_", "72" "_"]
    :param song (m21 stream): Piece to encode
    :param time_step (float): Duration of each time step in quarter length
    :return:
        
    NK - The original version was strictly monophonic. I have modified this to
    consider chords as well. They are encoded as strings, with the pitches of
    the notes comprising them separated by "c" characters.
    """

    encoded_song = []

    for event in song.flat.notesAndRests:

        # handle notes
        if isinstance(event, m21.note.Note):
            symbol = event.pitch.midi # eg. 47
        # handle chords 
        elif isinstance(event, m21.chord.Chord):
            symbol = "c"
            for pitch in event.pitches:
                symbol += str(pitch.midi)+"c" # eg. c40c44c47c
        # handle rests
        elif isinstance(event, m21.note.Rest):
            symbol = "r"

        # convert the note/rest into time series notation
        steps = int(event.duration.quarterLength / time_step)
        for step in range(steps):

            # if it's the first time we see a note/rest, let's encode it. 
            # Otherwise, it means we're carrying the same
            # symbol in a new time step
            if step == 0:
                encoded_song.append(symbol)
            else:
                encoded_song.append("_")

    # cast encoded song to str
    encoded_song = " ".join(map(str, encoded_song))

    return encoded_song

def preprocess(dataset_path):

    # load folk songs
    print("Loading songs...")
    songs = load_songs_in_midi(dataset_path)
    print(f"Loaded {len(songs)} songs.")

    for i, song in enumerate(songs):

        # encode songs with music time series representation
        encoded_song = encode_song(song)

        # save songs to text file
        save_path = os.path.join(SAVE_DIR, str(i))
        with open(save_path, "w") as fp:
            fp.write(encoded_song)

        if i % 10 == 0:
            print(f"Song {i} out of {len(songs)} processed")
            
#%% Dataset Functions: %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

def load_song(file_path):
    '''
    NK - changed the name of this function from "load" so that the generic name
    does not interfere with other built-in utilities
    '''
    with open(file_path, "r") as fp:
        song = fp.read()
    return song

def create_single_file_dataset(dataset_path, file_dataset_path, sequence_length):
    """Generates a file collating all the encoded songs and adding new piece delimiters.
    :param dataset_path (str): Path to folder containing the encoded songs
    :param file_dataset_path (str): Path to file for saving songs in single file
    :param sequence_length (int): # of time steps to be considered for training
    :return songs (str): String containing all songs in dataset + delimiters
    """

    new_song_delimiter = "/ " * sequence_length
    songs = ""

    # load encoded songs and add delimiters
    for path, _, files in os.walk(dataset_path):
        for file in files:
            file_path = os.path.join(path, file)
            song = load_song(file_path)
            songs = songs + song + " " + new_song_delimiter

    # remove empty space from last character of string
    songs = songs[:-1]

    # save string that contains all the dataset
    with open(file_dataset_path, "w") as fp:
        fp.write(songs)

    return songs

def generate_training_sequences(sequence_length):
    """Create input and output data samples for training. Each sample is a sequence.
    :param sequence_length (int): Length of each sequence. With a quantisation at 16th notes, 64 notes equates to 4 bars
    :return inputs (ndarray): Training inputs
    :return targets (ndarray): Training targets
    """

    # load songs and map them to int
    songs = load_song(SINGLE_FILE_DATASET)
    int_songs = convert_songs_to_int(songs)

    inputs = []
    targets = []

    # generate the training sequences
    num_sequences = len(int_songs) - sequence_length
    for i in range(num_sequences):
        inputs.append(int_songs[i:i+sequence_length])
        targets.append(int_songs[i+sequence_length])

    # one-hot encode the sequences
    vocabulary_size = len(set(int_songs))
    # inputs size: (# of sequences, sequence length, vocabulary size)
    inputs = keras.utils.to_categorical(inputs, num_classes=vocabulary_size)
    targets = np.array(targets)

    print(f"There are {len(inputs)} sequences.")

    return inputs, targets

#%% Mapping Functions: %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def create_mapping(songs, mapping_path):
    """Creates a json file that maps the symbols in the song dataset onto ints
    :param songs (str): String with all songs
    :param mapping_path (str): Path where to save mapping
    :return:
    """
    mappings = {}

    # identify the vocabulary
    songs = songs.split()
    vocabulary = list(set(songs))

    # create mappings
    for i, symbol in enumerate(vocabulary):
        mappings[symbol] = i

    # save voabulary to a json file
    with open(mapping_path, "w") as fp:
        json.dump(mappings, fp, indent=4)
        
def convert_songs_to_int(songs):
    int_songs = []

    # load mappings
    with open(MAPPING_PATH, "r") as fp:
        mappings = json.load(fp)

    # transform songs string to list
    songs = songs.split()

    # map songs to int
    for symbol in songs:
        int_songs.append(mappings[symbol])

    return int_songs

#%% MAIN: %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def main():
    '''
    Calls functions in order to process data and generate a dataset ready to be
    used for training.
    '''
    preprocess(MIDI_DATASET_PATH)
    songs = create_single_file_dataset(SAVE_DIR, SINGLE_FILE_DATASET, SEQUENCE_LENGTH)
    create_mapping(songs, MAPPING_PATH)
    '''
    Depending on the extent of the dataset, the size of "inputs" might be quite
    large, so instead of storing these arrays, it might be preferrable to 
    generate them during the training process.
    '''
    #inputs, targets = generate_training_sequences(SEQUENCE_LENGTH)
        
if __name__ == "__main__":
    main()