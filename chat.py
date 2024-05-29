#!/usr/bin/env python3

# Real-time speech recognition from a microphone with sherpa-onnx Python API
# with endpoint detection.
#
# Please refer to
# https://k2-fsa.github.io/sherpa/onnx/pretrained_models/index.html
# to download pre-trained models

import argparse
import numpy as np
import logging
import queue
import threading
import time
import json
import sys
from pathlib import Path

try:
    import sounddevice as sd
except ImportError:
    print("Please install sounddevice first. You can use")
    print()
    print("  pip install sounddevice")
    print()
    print("to install it")
    sys.exit(-1)

import sherpa_onnx


def assert_file_exists(filename: str):
    assert Path(filename).is_file(), (
        f"{filename} does not exist!\n"
        "Please refer to "
        "https://k2-fsa.github.io/sherpa/onnx/pretrained_models/index.html to download it"
    )

def read_file(filename):
    content = ""
    f = open(filename, "r")
    content = f.read()
    f.close()
    return content

def get_chat_config():
    chat_config = json.loads(read_file("chat_config.json"))
    return chat_config

def create_recognizer(chat_config):
    assert_file_exists(chat_config["asr_encoder"])
    assert_file_exists(chat_config["asr_decoder"])
    assert_file_exists(chat_config["asr_joiner"])
    assert_file_exists(chat_config["asr_tokens"])
    # Please replace the model files if needed.
    # See https://k2-fsa.github.io/sherpa/onnx/pretrained_models/index.html
    # for download links.
    recognizer = sherpa_onnx.OnlineRecognizer.from_transducer(
        tokens=chat_config["asr_tokens"],
        encoder=chat_config["asr_encoder"],
        decoder=chat_config["asr_decoder"],
        joiner=chat_config["asr_joiner"],
        num_threads=2,
        sample_rate=16000,
        feature_dim=80,
        enable_endpoint_detection=True,
        rule1_min_trailing_silence=2.4,
        rule2_min_trailing_silence=1.2,
        rule3_min_utterance_length=300,  # it essentially disables this rule
        decoding_method=chat_config["decoding_method"],
        provider=chat_config["provider"],
        hotwords_file=chat_config["hotwords-file"],
        hotwords_score=1.5,
        blank_penalty=0.0,
    )
    return recognizer

# TTS
# buffer saves audio samples to be played
buffer = queue.Queue()

# started is set to True once generated_audio_callback is called.
started = False

# stopped is set to True once all the text has been processed
stopped = False

# killed is set to True once ctrl + C is pressed
killed = False

# Note: When started is True, and stopped is True, and buffer is empty,
# we will exit the program since all audio samples have been played.

sample_rate = None

event = threading.Event()

speaking = False

first_message_time = None

def generated_audio_callback(samples: np.ndarray, progress: float):
    """This function is called whenever max_num_sentences sentences
    have been processed.

    Note that it is passed to C++ and is invoked in C++.

    Args:
      samples:
        A 1-D np.float32 array containing audio samples
    """
    global first_message_time
    if first_message_time is None:
        first_message_time = time.time()

    buffer.put(samples)
    global started

    if started is False:
        logging.info("Start playing ...")
    started = True


# see https://python-sounddevice.readthedocs.io/en/0.4.6/api/streams.html#sounddevice.OutputStream
def play_audio_callback(
    outdata: np.ndarray, frames: int, time, status: sd.CallbackFlags
):
    global speaking

    speaking = True

    if killed or (started and buffer.empty() and stopped):
        speaking = False
        event.set()

    # outdata is of shape (frames, num_channels)
    if buffer.empty():
        outdata.fill(0)
        speaking = False
        return

    n = 0
    while n < frames and not buffer.empty():
        remaining = frames - n
        k = buffer.queue[0].shape[0]

        if remaining <= k:
            outdata[n:, 0] = buffer.queue[0][:remaining]
            buffer.queue[0] = buffer.queue[0][remaining:]
            n = frames
            if buffer.queue[0].shape[0] == 0:
                buffer.get()

            break

        outdata[n : n + k, 0] = buffer.get()
        n += k

    if n < frames:
        outdata[n:, 0] = 0


# Please see
# https://python-sounddevice.readthedocs.io/en/0.4.6/usage.html#device-selection
# for how to select a device
def play_audio():
    if False:
        # This if branch can be safely removed. It is here to show you how to
        # change the default output device in case you need that.
        devices = sd.query_devices()
        print(devices)

        # sd.default.device[1] is the output device, if you want to
        # select a different device, say, 3, as the output device, please
        # use self.default.device[1] = 3

        default_output_device_idx = sd.default.device[1]
        print(
            f'Use default output device: {devices[default_output_device_idx]["name"]}'
        )

    with sd.OutputStream(
        channels=1,
        callback=play_audio_callback,
        dtype="float32",
        samplerate=sample_rate,
        blocksize=1024,
    ):
        event.wait()

    logging.info("Exiting ...")

def create_tts(chat_config):
    tts_config = sherpa_onnx.OfflineTtsConfig(
        model=sherpa_onnx.OfflineTtsModelConfig(
            vits=sherpa_onnx.OfflineTtsVitsModelConfig(
                model=chat_config["vits-model"],
                lexicon=chat_config["vits-lexicon"],
                data_dir=chat_config["vits-data-dir"],
                dict_dir=chat_config["vits-dict-dir"],
                tokens=chat_config["vits-tokens"],
            ),
            provider=chat_config["provider"],
            debug=False,
            num_threads=2,
        ),
        rule_fsts=chat_config["tts-rule-fsts"],
        max_num_sentences=1,
    )

    if not tts_config.validate():
        raise ValueError("Please check your config")

    logging.info("Loading model ...")
    tts = sherpa_onnx.OfflineTts(tts_config)
    logging.info("Loading model done.")

    global sample_rate
    sample_rate = tts.sample_rate

    return tts

# LLM
import requests
import os
import time
WENXIN_APIKEY = "" if os.environ.get('WENXIN_APIKEY') is None else os.environ.get('WENXIN_APIKEY')
WENXIN_SECRET = "" if os.environ.get('WENXIN_SECRET') is None else os.environ.get('WENXIN_SECRET')

token_time = {}

def get_access_token():
    """
    使用 AK，SK 生成鉴权签名（Access Token）
    :return: access_token，或是None(如果错误)
    """
    url = "https://aip.baidubce.com/oauth/2.0/token"
    params = {"grant_type": "client_credentials", "client_id": WENXIN_APIKEY, "client_secret": WENXIN_SECRET}
    return str(requests.post(url, params=params).json().get("access_token"))

def wenxin_token():
    global token_time
    if len(token_time.keys()) == 0:
        token_refresh = get_access_token()
        token_time = {
            token_refresh: time.time()
        }
        return token_refresh
    else:
        token = list(token_time.keys())[0]
        #20小时轮换一次
        if (time.time() - token_time[token]) > 20*3600:
            token_refresh = get_access_token()
            token_time = {
                token_refresh: time.time()
            }
            return token_refresh
        else:
            return token

def chat_wenxin(system_message, messages, model_name="ernie-bot", temperature=1e-10, retry_count=3):
    models = {
        "ernie-bot-4": "completions_pro",
        "ernie-bot": "completions",
        "ernie-bot-turbo": "completions",
        "ernie-bot-8k": "completions",
    }

    if model_name in models.keys():
        url = "https://aip.baidubce.com/rpc/2.0/ai_custom/v1/wenxinworkshop/chat/{}?access_token={}".format(models[model_name], wenxin_token())

    payload = json.dumps({
        "system": system_message,
        "messages": messages,
        "temperature": temperature
    })
    headers = {
        'Content-Type': 'application/json'
    }

    response = requests.request("POST", url, headers=headers, data=payload)

    respjson = json.loads(response.text)

    if "result" not in respjson.keys():
        print(response.text)
        if retry_count > 0:
            time.sleep(1)
            return chat_wenxin(system_message, messages, model_name, temperature, retry_count-1)
        else:
            return "百度接口QPS限制达到"

    return respjson['result']

def main():
    chat_config = get_chat_config()

    devices = sd.query_devices()
    if len(devices) == 0:
        print("No microphone devices found")
        sys.exit(0)

    print(devices)
    default_input_device_idx = sd.default.device[0]
    print(f'Use default device: {devices[default_input_device_idx]["name"]}')

    recognizer = create_recognizer(chat_config)
    print("Started! Please speak")

    # The model is using 16 kHz, we use 48 kHz here to demonstrate that
    # sherpa-onnx will do resampling inside.
    sample_rate = 48000
    samples_per_read = int(0.1 * sample_rate)  # 0.1 second = 100 ms

    stream = recognizer.create_stream()

    #TTS
    tts = create_tts(chat_config)
    play_back_thread = threading.Thread(target=play_audio)
    play_back_thread.start()

    last_result = ""
    segment_id = 0
    messages = []
    system_message = "你是一个人工智能助手，善于回答各类问题，说话风格简练"
    with sd.InputStream(channels=1, dtype="float32", samplerate=sample_rate) as s:
        while True:
            samples, _ = s.read(samples_per_read)  # a blocking read
            samples = samples.reshape(-1)
            if speaking:
                continue
            stream.accept_waveform(sample_rate, samples)
            while recognizer.is_ready(stream):
                recognizer.decode_stream(stream)

            is_endpoint = recognizer.is_endpoint(stream)

            result = recognizer.get_result(stream)

            if result and (last_result != result):
                last_result = result
                print("\r{}:{}".format(segment_id, result), end="", flush=True)
            if is_endpoint:
                if result:
                    print("\r{}:{}".format(segment_id, result), flush=True)

                    messages.append({
                        "role": "user",
                        "content": result
                    })

                    resp = chat_wenxin(system_message, messages, model_name="ernie-bot-8k")
                    print("AI:{}".format(resp), flush=True)
                    messages.append({
                        "role": "assistant",
                        "content": resp
                    })

                    audio = tts.generate(
                        resp,
                        sid=3,
                        speed=1.5,
                        callback=generated_audio_callback,
                    )
                    segment_id += 1
                recognizer.reset(stream)


if __name__ == "__main__":

    try:
        main()
    except KeyboardInterrupt:
        print("\nCaught Ctrl + C. Exiting")
