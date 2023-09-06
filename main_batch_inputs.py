from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
import torch
import soundfile as sf

model = SpeechT5ForTextToSpeech.from_pretrained(f"microsoft/speecht5_tts")
processor = SpeechT5Processor.from_pretrained(f"microsoft/speecht5_tts")
vocoder = SpeechT5HifiGan.from_pretrained(f"microsoft/speecht5_hifigan")
speaker_embedding = torch.load(f"speaker_embeddings/spk_embed_default.pt")

def text2speech(texts, output_audio_path):
    # The original one that should succeed to convert text to audio
    # inputs = processor(text=text, return_tensors="pt")
    # The one that use padding and will finally convert text to wrong audio because of the attention mask is not well handled in modeling_speecht5.py
    inputs = processor(text=texts, padding='max_length', max_length=128, return_tensors="pt")

    with torch.no_grad():
        # pass attention mask or not should always be fine, and compatible to the previous example usage
        # Following parameter patterns are all tested to be correctly executed
        # spectrograms = model.generate_speech(input_ids=inputs["input_ids"], speaker_embeddings=speaker_embedding, attention_mask=inputs['attention_mask'],)
        # spectrograms = model.generate_speech(input_ids=inputs["input_ids"], speaker_embeddings=speaker_embedding)
        # spectrograms = model.generate_speech(inputs["input_ids"], speaker_embedding)
        spectrograms = model.generate_speech(inputs["input_ids"], speaker_embedding, inputs['attention_mask'])
        for idx, spectrogram in enumerate(spectrograms):
            speech = vocoder(spectrogram)
            sf.write(f"{output_audio_path}_{idx}.wav", speech.cpu().numpy(), samplerate=16000)

        # speech = model.generate_speech(inputs["input_ids"], speaker_embedding, inputs['attention_mask'], vocoder=vocoder)
        # for idx, speech in enumerate(speech):
        #     sf.write(f"{output_audio_path}_{idx}.wav", speech.cpu().numpy(), samplerate=16000)

if __name__ == "__main__":
    texts = ["I have a dream, do you?", "He likes drinking coffee.", "Hello!"]
    text2speech(texts, "output")