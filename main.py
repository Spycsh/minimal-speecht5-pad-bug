from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
import torch
import soundfile as sf


model = SpeechT5ForTextToSpeech.from_pretrained(f"microsoft/speecht5_tts")
processor = SpeechT5Processor.from_pretrained(f"microsoft/speecht5_tts")
vocoder = SpeechT5HifiGan.from_pretrained(f"microsoft/speecht5_hifigan")
speaker_embedding = torch.load(f"speaker_embeddings/spk_embed_default.pt")

def text2speech(text, output_audio_path):
    # The original one that should succeed to convert text to audio
    inputs = processor(text=text, return_tensors="pt")
    # The one that use padding and will finally convert text to wrong audio because of the attention mask is not well handled in modeling_speecht5.py
    # inputs = processor(text=text, padding='max_length', max_length=128, return_tensors="pt")

    with torch.no_grad():
        speech = model.generate_speech(inputs["input_ids"], speaker_embedding, vocoder=vocoder)
    sf.write(output_audio_path, speech.cpu().numpy(), samplerate=16000)

if __name__ == "__main__":
    text = "I have a dream, do you."
    text2speech(text, "output.wav")