# Guitar Riff Generator

A machine learning model used to generate guitar-centric riffs, trained using a database of MIDI files.
The code included in this repository is based on the Melody Generation tutorial project by Valerio Velardo 
[https://github.com/musikalkemist/generating-melodies-with-rnn-lstm/], the video playlist for which can be found available on
Youtube [https://www.youtube.com/playlist?list=PL-wATfeyAMNr0KMutwtbeDCmpwvtul-Xz].

My modifications to this model, as of the time writing this, regard the form of input for the generator network, as well as the addition
of polyphony in the melodies by the inclusion of chords. In particular:
* The model is now directly trained using MIDI (.mid) files, instead of .krn scores. This makes it much easier to assemble a training database,
since .mid files are in abundance and freely available all around the internet. They can also be easily modified and brought to a desirable
form through a multitude of programs.
* The original model could only accept a single monophonic melody. My modifications make it so that it can be trained on multiple voices,
through the consideration of chord structures made possible by the Music21 library, so long as everything is contained within a single MIDI channel.
This polyphony is also carried over to the model's generative capabilities.

## Approach:

Recurrent Neural Networks have long been used for generation of melodies. Although a complex network is required to produce a coherrent output of
significant length, comparable to that of a regular song, simple networks have proved capable of producing a variety of valid, yet disconnected, phrases.
My take on the matter is that such a feature could be utilized to generate some guitar riffs, provided that the network is trained on an appropriate dataset of
guitar tracks. Since the expectation for each riff is that it only needs to last for a few bars, even a simple network should be able to handle this task.

The model uses a database of MIDI files from which to extract encoded features and train itself. MIDIs are perhaps the most widely available
and accessible form of music on the internet, and thanks to their structure and small size are ideal for implementation in AI music composition.
The processing performed on each MIDI prior to its handling by the model, is to single out the main guitar track and transpose it to the tonality of E.
Using a single tonality will greatly help the coherrence of our results, and given the guitar-centric nature of this task, E is the most common key encountered.
Although we are only using a single track in each file, this doesn't mean that we are using strictly monophonic information. The model is designed to encode and
handle polyphony and chords.

Using the Music21 library, information from these MIDI files can be encoded in string or integer format. In particular, each note is converted to an 
integer corresponding to its pitch, each chord is encoded in a string containing the same corresponding pitch information for each note in it, and rests
are simply represented by a single character "r". 
The duration of each event is included by the addition of "\_" characters, each denoting the extention of the previous instance by duration of a 16th.

The model used is a very simple network consisting of a single Long-Short Term Memory layer followed by a Fully Connected layer with Softmax activation.
Dropout of order 0.2 is included to avoid severe overfitting. 
The trained model can be called to generate a riff of its own, of specified length, by providing an initial seed corresponding to a valid encoding, 
which might be as simple as a single note. 

## Results:

The example output file, **riff.mid**, was produced using a single note (E2) seed. 
This file serves as a demo to the capabilities this model: the generated melody contains a decipherable rhythmic metre, notes of varying duration and pitch, rests and coherrent chords.

Still, we are a long way to go from a full length song that maintains consistency, but given the simplicity of the current model and the size of the training dataset, it is rather satisfactory.
My planned improvements for this model begin from the expansion of the dataset, by adding more songs, and also by making sure the content of the songs is appropriate for training;
eg. no huge pauses when the guitar track is absent in the song, the trimming of solos in favor of including just the riffs, and the merging of guitar tracks to contain harmonized melodies within a single track.

As far as the network itself, I plan on trying out more complex structures, adding attention and encoder-decoder architectures, but these only when the dataset is large enough to actually make use of their capabilities.

*So, that's it for now. Rather basic but promising. I hope that by expanding my database and designing some more advanced models I will soon have an update with more impressive results.*

## Contents:

* *MIDI* : Directory containing the MIDI files constituting the main database. In our case the selected files are guitar tracks taken from 
songs by the bands Megadeth, Metallica and Slayer.
These have been processed beforehand using the Qtractor DAW, so that they all consist of a single track, and have been transposed to the same tonality (E).
* *dataset* : Directory containing the dataset in the form of individual encoded text files. Each file corresponds to a song in the intiial database.
* **file_dataset** : Single file dataset containing the encoded information from all the entries, to be used for training the model.
* **mapping.json** : Dictionary file containing the mapping for the encoded information, to be used by the model.
* **preprocess.py** : Script containing the utility functions used to process the MIDI dataset and produce the single file dataset and mapping dictionary.
* **train.py** : Script used to build and train the LSTM network for melody generation, using the files produced above.
* **melodygenerator.py** : Script for generating a melody using the trained network, by entering an initial seed corresponding to an encoded musical sequence.
