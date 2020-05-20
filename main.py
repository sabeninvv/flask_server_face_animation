from encoder_image import Encoder
from img_generator import Generator
from rnn_audio import RNN
from utils import *
from flask import Flask, request, send_file
import face_alignment
import torch
from time import time
import io

app = Flask(__name__)

gpu = 0

device = torch.device("cpu")  # device = torch.device("cuda:" + str(gpu))
model_dict = torch.load('models/crema.dat', map_location=lambda storage, loc: storage)
fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, device="cpu:" + str(gpu), flip_input=False)

audio_feat_len = 0.2
audio_feat_samples = 3200
aud_enc_dim = 256
rnn_gen_dim = 256
id_enc_dim = 128
aux_latent = 10
sequential_noise = True
img_size = (128, 96)
audio_rate = 16000
video_rate = 30
# stablePntsIDs = [33, 36, 39, 42, 45]
stablePntsIDs = [30, 33, 36, 39, 42, 45, 48, 54]
mean_face = model_dict['mean_face']
conversion_dict = {'s16': np.int16, 's32': np.int32}

encoder = RNN(audio_feat_len, aud_enc_dim, rnn_gen_dim,
              audio_rate, init_kernel=0.005, init_stride=0.001)
encoder.to(device)
encoder.load_state_dict(model_dict['encoder'])

encoder_id = Encoder(id_enc_dim, img_size)
encoder_id.to(device)
encoder_id.load_state_dict(model_dict['encoder_id'])

skip_channels = list(encoder_id.channels)
skip_channels.reverse()

generator = Generator(img_size, rnn_gen_dim, condition_size=id_enc_dim,
                      num_gen_channels=encoder_id.channels[-1],
                      skip_channels=skip_channels, aux_size=aux_latent,
                      sequential_noise=sequential_noise)
generator.to(device)
generator.load_state_dict(model_dict['generator'])

encoder.eval()
encoder_id.eval()
generator.eval()


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        filenames = save_input_files(request.files['img'], request.files['audio'])

        start = time()
        speech = audio_preproc(f'temp/{filenames["audio_name"]}', conversion_dict, audio_rate)
        print(f'Предобработка img: {round(time() - start, 3)} сек')
        start = time()
        frame = image_preproc(f'temp/{filenames["img_name"]}', fa, stablePntsIDs, mean_face, img_size, device)
        print(f'Предобработка audio: {round(time() - start, 3)} сек')

        start = time()
        cutting_stride = int(audio_rate / float(video_rate))
        audio_seq_padding = audio_feat_samples - cutting_stride
        audio_feat_seq = cut_sequence(speech, cutting_stride, audio_seq_padding, device)
        frame = frame.unsqueeze(0)
        audio_feat_seq = audio_feat_seq.unsqueeze(0)
        audio_feat_seq_length = audio_feat_seq.size()[1]
        print(f'Нарезка последовательностей: {round(time() - start, 3)} сек')

        start = time()
        z = encoder(audio_feat_seq, [audio_feat_seq_length])  # Encoding for the motion
        print(f'Нейронка encoder: {round(time() - start, 3)} сек')

        start = time()
        noise = torch.FloatTensor(1, audio_feat_seq_length, aux_latent).normal_(0, 0.33).to(device)
        print(f'Нейронка noise: {round(time() - start, 3)} сек')

        start = time()
        z_id, skips = encoder_id(frame, retain_intermediate=True)
        print(f'Нейронка encoder_id: {round(time() - start, 3)} сек')

        skip_connections = []
        for skip_variable in skips:
            skip_connections.append(broadcast_elements(skip_variable, z.size()[1]))
        skip_connections.reverse()
        z_id = broadcast_elements(z_id, z.size()[1])

        start = time()
        gen_video = generator(z, c=z_id, aux=noise, skip=skip_connections)
        print(f'Нейронка generator: {round(time() - start, 3)} сек')

        start = time()
        returned_audio = ((2 ** 15) * speech.detach().cpu().numpy()).astype(np.int16)
        gen_video = 125 * gen_video.squeeze().detach().cpu().numpy() + 125
        print(f'denorm audio+video: {round(time() - start, 3)} сек')

        start = time()
        save_video(video=gen_video, audio=returned_audio, path=f'temp/{filenames["video_name"]}', video_rate=video_rate,
                   audio_rate=audio_rate)
        print(f'save video: {round(time() - start, 3)} сек')

        with open(f'temp/{filenames["video_name"]}', 'rb') as bites:
            video_in_ram = io.BytesIO(bites.read())

        del_input_files(filenames)

        return send_file(
            video_in_ram,
            attachment_filename='video.mp4',
            mimetype='video/mp4')


if __name__ == '__main__':
    app.run(host='0.0.0.0')
