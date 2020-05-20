import torch
from skimage import transform as tf
from PIL import Image
from scipy import signal
import scipy.io.wavfile as wav
from torchvision import transforms
from pydub import AudioSegment
from pydub.utils import mediainfo
import os
import numpy as np
import shutil
import tempfile
import ffmpeg
import contextlib
import skvideo.io as sio
from math import ceil
from time import time


def save_input_files(file_img, file_audio):
    img_bytes = file_img.read()
    audio_bytes = file_audio.read()
    temp_prefix = f'{time()}{"".join(np.random.choice("h f k s l y".split(" "), 5))}'
    filenames = {'img_name': f'img_{temp_prefix}.jpeg',
                 'audio_name': f'audio_{temp_prefix}.wav',
                 'video_name': f'video_{temp_prefix}.mp4'}
    with open(f'temp/{filenames["img_name"]}', 'wb') as f:
        f.write(img_bytes)
    with open(f'temp/{filenames["audio_name"]}', 'wb') as f:
        f.write(audio_bytes)
    return filenames


def del_input_files(filenames):
    for key in filenames.keys():
        os.remove(f'temp/{filenames[key]}')


@contextlib.contextmanager
def cd(newdir, cleanup=lambda: True):
    prevdir = os.getcwd()
    os.chdir(os.path.expanduser(newdir))
    try:
        yield
    finally:
        os.chdir(prevdir)
        cleanup()


@contextlib.contextmanager
def tempdir():
    dirpath = tempfile.mkdtemp()

    def cleanup():
        shutil.rmtree(dirpath)

    with cd(dirpath, cleanup):
        yield dirpath


def save_video( video, audio, path, video_rate=30, audio_rate=16000, overwrite=True, experimental_ffmpeg=False, scale=None):
    if not os.path.isabs(path):
        path = os.getcwd() + "/" + path;

    with tempdir() as dirpath:
        # Save the video file
        writer = sio.FFmpegWriter(dirpath + "/tmp.avi",
                                    inputdict={'-r': str(video_rate) + "/1", },
                                    outputdict={'-r': str(video_rate) + "/1", }
                                    )
        for i in range(video.shape[0]):
            frame = np.rollaxis(video[i, :, :, :], 0, 3)
            if scale is not None:
                frame = tf.rescale(frame, scale, anti_aliasing=True, multichannel=True, mode='reflect')
            writer.writeFrame(frame)
        writer.close()

        # Save the audio file
        wav.write(dirpath + "/tmp.wav", audio_rate, audio)

        in1 = ffmpeg.input(dirpath + "/tmp.avi")
        in2 = ffmpeg.input(dirpath + "/tmp.wav")
        if experimental_ffmpeg:
            out = ffmpeg.output(in1['v'], in2['a'], path, strict='-2', loglevel="panic")
        else:
            out = ffmpeg.output(in1['v'], in2['a'], path, loglevel="panic")

        if overwrite:
            out = out.overwrite_output()
        out.run()


def preprocess_img(img, fa, stablePntsIDs, mean_face, img_size):
    src = fa.get_landmarks(img)[0][stablePntsIDs, :]
    dst = mean_face[stablePntsIDs, :]
    tform = tf.estimate_transform('similarity', src, dst)  # find the transformation matrix
    warped = tf.warp(img, inverse_map=tform.inverse, output_shape=img_size)  # wrap the frame image
    warped = warped * 255  # note output from wrap is double image (value range [0,1])
    warped = warped.astype('uint8')
    return warped


def cut_sequence(seq, cutting_stride, pad_samples, device, audio_feat_samples=3200):
    pad_left = torch.zeros(pad_samples // 2, 1)
    pad_right = torch.zeros(pad_samples - pad_samples // 2, 1)
    seq = torch.cat((pad_left, seq), 0)
    seq = torch.cat((seq, pad_right), 0)
    stacked = seq.narrow(0, 0, audio_feat_samples).unsqueeze(0)
    iterations = (seq.size()[0] - audio_feat_samples) // cutting_stride + 1
    for i in range(1, iterations):
        stacked = torch.cat((stacked, seq.narrow(0, i * cutting_stride, audio_feat_samples).unsqueeze(0)))
    return stacked.to(device)


def broadcast_elements(batch, repeat_no):
    total_tensors = []
    for i in range(0, batch.size()[0]):
        total_tensors += [torch.stack(repeat_no * [batch[i]])]
    return torch.stack(total_tensors)


def audio_preproc(path, conversion_dict, audio_rate):
    fs = None
    info = mediainfo(path)
    fs = int(info['sample_rate'])
    audio = np.array(AudioSegment.from_file(path, info['format_name']).set_channels(1).get_array_of_samples())

    if info['sample_fmt'] in conversion_dict:
        audio = audio.astype(conversion_dict[info['sample_fmt']])
    else:
        if max(audio) > np.iinfo(np.int16).max:
            audio = audio.astype(np.int32)
        else:
            audio = audio.astype(np.int16)
    if fs is None:
        raise AttributeError("Audio provided without specifying the rate. Specify rate or use audio file!")
    if audio.ndim > 1 and audio.shape[1] > 1:
        audio = audio[:, 0]
    max_value = np.iinfo(audio.dtype).max
    if fs != audio_rate:
        seq_length = audio.shape[0]
        speech = torch.from_numpy(
            signal.resample(audio, int(seq_length * audio_rate / float(fs))) / float(max_value)).float()
        speech = speech.view(-1, 1)
    else:
        audio = torch.from_numpy(audio / float(max_value)).float()
        speech = audio.view(-1, 1)
    return speech


def image_preproc(path, fa, stablePntsIDs, mean_face, img_size, device):
    frm = Image.open(path)
    frm.thumbnail((400, 400), Image.BICUBIC)
    frame = np.array(frm)
    frame = preprocess_img(frame, fa, stablePntsIDs, mean_face, img_size)
    Image.fromarray(frame)
    img_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((img_size[0], img_size[1])),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    frame = img_transform(frame).to(device)
    return frame


def prime_factors(number):
    factor = 2
    factors = []
    while factor * factor <= number:
        if number % factor:
            factor += 1
        else:
            number //= factor
            factors.append(int(factor))
    if number > 1:
        factors.append(int(number))
    return factors


def calculate_padding(kernel_size, stride=1, in_size=0):
    out_size = ceil(float(in_size) / float(stride))
    return int((out_size - 1) * stride + kernel_size - in_size)


def calculate_output_size(in_size, kernel_size, stride, padding):
    return int((in_size + padding - kernel_size) / stride) + 1


def is_power2(num):
    return num != 0 and ((num & (num - 1)) == 0)