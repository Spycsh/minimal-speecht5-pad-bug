from transformers import (
    SpeechT5Processor,
    SpeechT5ForTextToSpeech,
    SpeechT5HifiGan,
    set_seed,
)
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
    inputs = processor(
        text=texts, padding="max_length", max_length=128, return_tensors="pt"
    )

    with torch.no_grad():
        ########### use without vocoder
        # spectrograms, spectrogram_lengths = model.generate_speech(inputs["input_ids"], speaker_embedding, inputs['attention_mask'])
        # cur_idx = 0
        # for idx, spectrogram_length in enumerate(spectrogram_lengths):
        #     spectrogram = spectrograms[cur_idx: cur_idx+spectrogram_length]
        #     speech = vocoder(spectrogram)
        #     sf.write(f"{output_audio_path}_{idx}.wav", speech.cpu().numpy(), samplerate=16000)
        #     cur_idx += spectrogram_length
        ###########
        waveforms, waveform_lengths = model.generate_speech(
            input_ids=inputs["input_ids"],
            # speaker_embeddings=torch.ones([1, 512]),
            speaker_embeddings=speaker_embedding,
            attention_mask=inputs["attention_mask"],
            vocoder=vocoder,
        )

        cur_idx = 0
        for idx, spectrogram_length in enumerate(waveform_lengths):
            speech = waveforms[cur_idx : cur_idx + spectrogram_length]
            sf.write(
                f"{output_audio_path}_{idx}.wav", speech.cpu().numpy(), samplerate=16000
            )
            cur_idx += spectrogram_length


if __name__ == "__main__":
    # texts = ["I have a dream, do you?", "He likes drinking coffee.", "Hello!"]
    texts = [
        "mister quilter is the apostle of the middle classes and we are glad to welcome his gospel",
        "nor is mister quilter's manner less interesting than his matter",
        "he tells us that at this festive season of the year with christmas and rosebeaf looming before us",
    ]
    set_seed(555)
    text2speech(texts, "output")
