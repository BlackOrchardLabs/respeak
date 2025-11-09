from pydub import AudioSegment

sound = AudioSegment.from_mp3("samples/eric_voice.mp3")
sound.export("samples/eric_voice.wav", format="wav")
print("Converted to WAV!")
