import speech_recognition as sr
import re
import pyaudio
import os
import wave
import pickle
from sys import byteorder
from array import array
from struct import pack
from scipy.io import wavfile
import soundfile as sf  # Import the soundfile library
from language_tool_python import LanguageTool
from utils import extract_feature

THRESHOLD = 500
# CHUNK_SIZE = 1024
CHUNK_SIZE = 2048

FORMAT = pyaudio.paInt16
RATE = 16000

SILENCE = 18

# Define a noise profile
noise_profile = None


def is_silent(snd_data):
    "Returns 'True' if below the 'silent' threshold"
    return max(snd_data) < THRESHOLD


def normalize(snd_data):
    "Average the volume out"
    MAXIMUM = 16384
    # MAXIMUM = 32768

    times = float(MAXIMUM) / max(abs(i) for i in snd_data)

    r = array('h')
    for i in snd_data:
        r.append(int(i * times))
    return r


def trim(snd_data):
    "Trim the blank spots at the start and end"

    def _trim(snd_data):
        snd_started = False
        r = array('h')

        for i in snd_data:
            if not snd_started and abs(i) > THRESHOLD:
                snd_started = True
                r.append(i)

            elif snd_started:
                r.append(i)
        return r

    # Trim to the left
    snd_data = _trim(snd_data)

    # Trim to the right
    snd_data.reverse()
    snd_data = _trim(snd_data)
    snd_data.reverse()
    return snd_data


def add_silence(snd_data, seconds):
    "Add silence to the start and end of 'snd_data' of length 'seconds' (float)"
    r = array('h', [0 for i in range(int(seconds * RATE))])
    r.extend(snd_data)
    r.extend([0 for i in range(int(seconds * RATE))])
    return r


def record():
    """
    Record a word or words from the microphone and 
    return the data as an array of signed shorts.

    Normalizes the audio, trims silence from the 
    start and end, and pads with 0.5 seconds of 
    blank sound to make sure VLC can play
    it without getting chopped off.
    """
    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT, channels=1, rate=RATE,
                    input=True, output=True,
                    frames_per_buffer=CHUNK_SIZE)

    num_silent = 0
    snd_started = False

    r = array('h')

    while 1:
        # little endian, signed short
        snd_data = array('h', stream.read(CHUNK_SIZE))
        if byteorder == 'big':
            snd_data.byteswap()
        r.extend(snd_data)

        silent = is_silent(snd_data)

        if silent and snd_started:
            num_silent += 1
        elif not silent and not snd_started:
            snd_started = True

        if snd_started and num_silent > SILENCE:
            break

    sample_width = p.get_sample_size(FORMAT)
    stream.stop_stream()
    stream.close()
    p.terminate()

    r = normalize(r)
    r = trim(r)
    r = add_silence(r, 0.5)
    return sample_width, r
# Function to convert the first letter to uppercase
def capitalize_first_letter(match):
    return match.group(1) + match.group(2).upper()
def check_grammar(text):
    tool = LanguageTool('en-US')  # Load English language rules
    matches = tool.check(text)
    return matches



if __name__ == "__main__":
    # load the saved model (after training)
    model = pickle.load(open("result/mlp_classifier2.pkl", "rb"))

    # load audio file in wav format size <400kb
    # filename = "03-01-04-01-02-01-02.wav"
    #3rd index is the voice mood here 04
    # save_path = "data/Actor_02/"
    # filename="baby.wav"
    filename="1.wav"
    save_path="TestAudio/"
    file_path = os.path.join(save_path, filename)
    rate, data = wavfile.read(file_path)




    # extract features and reshape it
    features = extract_feature(file_path, mfcc=True, chroma=True, mel=True).reshape(1, -1)
    # predict
    result = model.predict(features)[0]
    # show the result children mood!
    print("result mood:", result)





    # Now start speech to text and check its grammer
    # ****************************************************************
    # Save the NumPy array as a temporary WAV file
    temp_wav_file = "temp_audio.wav"
    sf.write(temp_wav_file, data, rate)

    # Recognize speech from the temporary WAV file using recognize_google
    r = sr.Recognizer()

    with sr.AudioFile(temp_wav_file) as source:
        try:
            audio_data = r.record(source)  # Record the entire audio file
            text = r.recognize_google(audio_data)
            print("---------------")
            print("Recognized text:", text)
            recognized_text = text
            # recognized_text="your recognized text goes here"

            # Regular expression pattern to match the first letter of the string
            pattern = r"(^|[.!?]\s+)([a-z])"

            # Convert the first letter to uppercase using regular expression substitution
            recognized_text = re.sub(pattern, capitalize_first_letter, recognized_text)
            print("preprocess recognized text ",recognized_text)
            # Check the length of the recognized text
            recognized_text_length = len(recognized_text.split())  # Split text into words and count the words

            print("Number of words in recognized text:", recognized_text_length)

            # Check if the number of words is less than 4 and say "wrong"
            if recognized_text_length < 4:
                print("Wrong sentence-  word count is less than 4");
            else:
                print("Correct word count")
                # Check grammar
                grammar_matches = check_grammar(recognized_text)

                # Print grammar mistakes (if any)
                if grammar_matches:
                    print("Grammar mistakes found:")
                    for match in grammar_matches:
                        print(match)
                else:
                    print("No grammar mistakes found.")

            print("---------------")
        except sr.UnknownValueError:
            print("Google Web Speech API could not understand the audio")
        except sr.RequestError as e:
            print("Could not request results from Google Web Speech API; {0}".format(e))


    # ****************************************************************