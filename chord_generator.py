import mido
from mido import Message, MidiFile, MidiTrack
import subprocess
import wave
import os
import sys



# Define 20 chord progressions. Each progression is a list of chords,
# and each chord is represented as a list of note strings.
# These progressions use common harmonies (mostly in C major).
progressions = [
    # 1. I - IV - V - I (C major)
    [
        ["C4", "E4", "G4"],
        ["F4", "A4", "C5"],
        ["G4", "B4", "D5"],
        ["C4", "E4", "G4"]
    ],
    # 2. I - vi - IV - V
    [
        ["C4", "E4", "G4"],
        ["A3", "C4", "E4"],
        ["F4", "A4", "C5"],
        ["G4", "B4", "D5"]
    ],
    # 3. I - V - vi - IV
    [
        ["C4", "E4", "G4"],
        ["G4", "B4", "D5"],
        ["A3", "C4", "E4"],
        ["F4", "A4", "C5"]
    ],
    # 4. ii - V - I (Dm - G - C)
    [
        ["D4", "F4", "A4"],
        ["G4", "B4", "D5"],
        ["C4", "E4", "G4"]
    ],
    # 5. I - IV - vi - V
    [
        ["C4", "E4", "G4"],
        ["F4", "A4", "C5"],
        ["A3", "C4", "E4"],
        ["G4", "B4", "D5"]
    ],
    # 6. I - vi - ii - V
    [
        ["C4", "E4", "G4"],
        ["A3", "C4", "E4"],
        ["D4", "F4", "A4"],
        ["G4", "B4", "D5"]
    ],
    # 7. I - V - I - V (alternating)
    [
        ["C4", "E4", "G4"],
        ["G4", "B4", "D5"],
        ["C4", "E4", "G4"],
        ["G4", "B4", "D5"]
    ],
    # 8. I - V - IV - I
    [
        ["C4", "E4", "G4"],
        ["G4", "B4", "D5"],
        ["F4", "A4", "C5"],
        ["C4", "E4", "G4"]
    ],
    # 9. I - vi - iii - IV
    [
        ["C4", "E4", "G4"],
        ["A3", "C4", "E4"],
        ["E4", "G4", "B4"],
        ["F4", "A4", "C5"]
    ],
    # 10. I - iii - IV - V
    [
        ["C4", "E4", "G4"],
        ["E4", "G4", "B4"],
        ["F4", "A4", "C5"],
        ["G4", "B4", "D5"]
    ],
    # 11. I - IV - I - V
    [
        ["C4", "E4", "G4"],
        ["F4", "A4", "C5"],
        ["C4", "E4", "G4"],
        ["G4", "B4", "D5"]
    ],
    # 12. I - vi - ii - V - I
    [
        ["C4", "E4", "G4"],
        ["A3", "C4", "E4"],
        ["D4", "F4", "A4"],
        ["G4", "B4", "D5"],
        ["C4", "E4", "G4"]
    ],
    # 13. I - IV - V - vi
    [
        ["C4", "E4", "G4"],
        ["F4", "A4", "C5"],
        ["G4", "B4", "D5"],
        ["A3", "C4", "E4"]
    ],
    # 14. vi - IV - I - V
    [
        ["A3", "C4", "E4"],
        ["F4", "A4", "C5"],
        ["C4", "E4", "G4"],
        ["G4", "B4", "D5"]
    ],
    # 15. ii - vi - V - I
    [
        ["D4", "F4", "A4"],
        ["A3", "C4", "E4"],
        ["G4", "B4", "D5"],
        ["C4", "E4", "G4"]
    ],
    # 16. I - V - ii - IV
    [
        ["C4", "E4", "G4"],
        ["G4", "B4", "D5"],
        ["D4", "F4", "A4"],
        ["F4", "A4", "C5"]
    ],
    # 17. I - ii - V - I
    [
        ["C4", "E4", "G4"],
        ["D4", "F4", "A4"],
        ["G4", "B4", "D5"],
        ["C4", "E4", "G4"]
    ],
    # 18. I - vi - IV - iii
    [
        ["C4", "E4", "G4"],
        ["A3", "C4", "E4"],
        ["F4", "A4", "C5"],
        ["E4", "G4", "B4"]
    ],
    # 19. iii - vi - IV - V
    [
        ["E4", "G4", "B4"],
        ["A3", "C4", "E4"],
        ["F4", "A4", "C5"],
        ["G4", "B4", "D5"]
    ],
    # 20. I - V - vi - IV - V
    [
        ["C4", "E4", "G4"],
        ["G4", "B4", "D5"],
        ["A3", "C4", "E4"],
        ["F4", "A4", "C5"],
        ["G4", "B4", "D5"]
    ]
]




class ChordGenerator():
    SYNTH_PATH = r"./fluidsynth-2.4.3-win10-x64/bin/fluidsynth.exe"
    WORKING_MIDI = MidiFile()
    soundfont = ""
    tempo = 500000        # microseconds per beat (120 BPM)
    midiFileName = "myMidi.mid"
    wavFileName = "myWaveFile.wav"



    def __init__(self, soundfont : str):
        self.soundfont = soundfont
        if not os.path.exists(self.soundfont):
            print("Path to sf2 does not exist. Try again.")

    
    def SetSf2(self, sf2):
        self.soundfont = sf2

    
    def note_to_midi(self, note):
        """
        Convert a note string (e.g., 'C4', 'D#4', 'Bb3') to a MIDI note number.
        MIDI note numbers: C4 is 60.
        """
        note = note.strip()
        if len(note) < 2:
            raise ValueError("Invalid note: " + note)
        
        letter = note[0].upper()
        accidental = ""
        octave_str = ""
        
        if len(note) >= 3 and note[1] in ['#', 'b']:
            accidental = note[1]
            octave_str = note[2:]
        else:
            octave_str = note[1:]
        
        try:
            octave = int(octave_str)
        except ValueError:
            raise ValueError("Invalid octave in note: " + note)
        
        # Define base note numbers (C = 0, C# = 1, etc.)
        note_base = {
            'C': 0,
            'C#': 1,
            'D': 2,
            'D#': 3,
            'E': 4,
            'F': 5,
            'F#': 6,
            'G': 7,
            'G#': 8,
            'A': 9,
            'A#': 10,
            'B': 11
        }
        
        # Handle flats by converting to the equivalent sharp
        if accidental == 'b':
            natural_value = note_base.get(letter)
            if natural_value is None:
                raise ValueError("Invalid note letter: " + note)
            base_value = (natural_value - 1) % 12
        else:
            key = letter + accidental  # accidental will be empty if none
            base_value = note_base.get(key)
            if base_value is None:
                raise ValueError("Invalid note: " + note)
        
        # MIDI note number calculation:
        # C4 is 60, and MIDI numbers increase by one per semitone.
        midi_number = 12 * (octave + 1) + base_value
        return midi_number


    def GenerateWavFromMidi(self, sf2 : str, midiFileName : str, wavFileName : str, sampleRate = 44100):
        # Build and run the FluidSynth command.
        command = [
            self.SYNTH_PATH,
            "-ni",
            sf2,
            midiFileName,
            "-F",
            wavFileName,
            "-r",
            str(sampleRate)
        ]

        print(f"Converting {midiFileName} to {wavFileName} using FluidSynth...")
        self.WORKING_MIDI.save(midiFileName)
        result = subprocess.run(command)
        
        if result.returncode == 0:
            print(f"WAV file saved as: {wavFileName}")
        else:
            print("An error occurred while rendering the WAV file. Please check your FluidSynth installation and SoundFont path.")

        return result

    def SelectProgression(self, idx):
        self.create_midi(progressions[idx], 2, self.midiFileName)
    
    def create_midi(self, chords, chord_duration, output_file="progression.mid", tempo=500000):
        """
        Creates a MIDI file from a chord progression.
        
        chords: list of chords, where each chord is a list of note strings.
        chord_duration: duration of each chord in seconds.
        tempo: microseconds per beat (default 500000, corresponding to 120 BPM).
        """
        # mid = MidiFile()
        track = MidiTrack()
        
        # mid.tracks.append(track)
        self.WORKING_MIDI.tracks.append(track)
        
        # Set tempo
        track.append(mido.MetaMessage('set_tempo', tempo=tempo, time=0))
        
        ticks_per_beat = self.WORKING_MIDI.ticks_per_beat
        # Calculate chord duration in ticks:
        # Seconds per beat = tempo / 1e6, so beats per chord = chord_duration / (tempo/1e6)
        beats_per_chord = chord_duration / (tempo / 1e6)
        ticks_per_chord = int(beats_per_chord * ticks_per_beat)
        
        for chord in chords:
            # Turn notes on (all at the same time)
            for note in chord:
                midi_note = self.note_to_midi(note)
                track.append(Message('note_on', note=midi_note, velocity=64, time=0))
            # Wait for the duration of the chord
            track.append(Message('note_off', note=0, velocity=0, time=ticks_per_chord))
            # Turn notes off immediately after the wait
            for note in chord:
                midi_note = self.note_to_midi(note)
                track.append(Message('note_off', note=midi_note, velocity=64, time=0))



def trim_wav(input_wav, output_wav, target_duration, sample_rate=44100):
    """
    Trims the WAV file to exactly target_duration seconds.
    """
    target_frames = int(target_duration * sample_rate)
    
    with wave.open(input_wav, 'rb') as in_wav:
        params = in_wav.getparams()
        # Read all frames, calculate how many frames to keep
        all_frames = in_wav.readframes(params.nframes)
    
    frame_size = params.sampwidth * params.nchannels
    trimmed_frames = all_frames[:target_frames * frame_size]
    
    with wave.open(output_wav, 'wb') as out_wav:
        out_wav.setparams(params)
        out_wav.writeframes(trimmed_frames)
    print(f"Trimmed {input_wav} to {target_duration} seconds for perfect looping.")



def main():

    ans = "y"
    midi_filename = "my_midi.mid"
    wav_filename = "my_wav.wav"
    soundfont = ""

    while(True):

        # We are done adding tracks. Generate midi
        if (ans.lower().strip() == "n" or ans.lower().strip() == "no"):
            break
        
        soundfont = input("Input the path to the sf2 you want to use ").strip()

        if not os.path.exists(soundfont):
            print("Path to sf2 does not exist. Try again.")
            ans = input("Add another track? 'y'/'n': ").strip()
            continue

        # Settings
        chord_duration = 2.0  # seconds per chord
        tempo = 500000        # microseconds per beat (120 BPM)
        
        # Path to FluidSynth and the soundfont.
        sample_rate = 44100

        selectedProgressions = input("Type in your desired progressions. 0-20 is supported as of now ").split()

        # Concatenate the list of the chords from the selected numbers
        chords = [progressions[int(idx)] for idx in selectedProgressions]

        for chord in chords:
            create_midi(chord, chord_duration, midi_filename, tempo)
        


        ans = input("Add another track? 'y'/'n': ")
   
    command = [
        SYNTH_PATH,
        "-ni",
        soundfont,
        midi_filename,
        "-F",
        wav_filename,
        "-r",
        str(sample_rate)
    ]

    # result = subprocess.run(command)
    result = GenerateWavFromMidi(soundfont, midi_filename, wav_filename)
    # if result.returncode == 0:
    #     # Calculate expected total duration for this progression
    #     target_duration = chord_duration * len(chords)
    #     # Trim the WAV file to remove any extra tail so it loops perfectly.
    #     temp_wav = f"trimmed_{wav_filename}"
    #     trim_wav(wav_filename, temp_wav, target_duration, sample_rate)
    #     # Replace the original WAV with the trimmed version.
    #     os.replace(temp_wav, wav_filename)
    # else:
    #     print("An error occurred while rendering the WAV file. Please check your FluidSynth installation and SoundFont path.")
            
    
    print("\nDone generating and trimming all chord progressions.")

if __name__ == "__main__":
    main()
