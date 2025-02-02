from transformers import pipeline


print("Hello World!")
pipe = pipeline("audio-classification", model="nickprock/music_genres_classification-finetuned-gtzan")

result = pipe("sample.wav")
print("Old town Road:", result[:3])

result = pipe("sample2.wav")
print("TS:", result[:3])


result = pipe("sample3.wav")
print("Blues:", result[:3])




