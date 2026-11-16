# Copyright (c) 2024 Alibaba Inc (authors: Xiang Lyu)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
import sys
import argparse
import logging
logging.getLogger('matplotlib').setLevel(logging.WARNING)
from fastapi import FastAPI, UploadFile, Form, File
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from starlette.background import BackgroundTask
import uvicorn
import numpy as np
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append('{}/../../..'.format(ROOT_DIR))
sys.path.append('{}/../../../third_party/Matcha-TTS'.format(ROOT_DIR))
from cosyvoice.cli.cosyvoice import AutoModel
from cosyvoice.utils.file_utils import load_wav
import tempfile
import shutil


def save_upload(prompt_wav):
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    try:
        shutil.copyfileobj(prompt_wav.file, tmp)
    except Exception:
        tmp.close()
        try:
            os.remove(tmp.name)
        except OSError:
            pass
        raise
    tmp.close()
    return tmp.name

app = FastAPI()
# set cross region allowance
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"])

cosyvoice = None


def _model():
    if cosyvoice is None:
        raise RuntimeError('模型未加载：请用 python server.py --model_dir ... 启动')
    return cosyvoice


def generate_data(model_output):
    for i in model_output:
        tts_audio = (i['tts_speech'].numpy() * (2 ** 15)).astype(np.int16).tobytes()
        yield tts_audio


def _remove_quietly(path):
    try:
        os.remove(path)
    except OSError:
        pass


def stream_and_cleanup(model_output, path):
    try:
        yield from generate_data(model_output)
    finally:
        _remove_quietly(path)


@app.get("/health")
async def health():
    m = _model()
    return {"ok": True, "sample_rate": m.sample_rate, "fp16": bool(m.fp16)}


@app.get("/inference_sft")
@app.post("/inference_sft")
async def inference_sft(tts_text: str = Form(), spk_id: str = Form(), speed: float = Form(1.0)):
    model_output = _model().inference_sft(tts_text, spk_id, speed=speed)
    return StreamingResponse(generate_data(model_output))


@app.get("/inference_zero_shot")
@app.post("/inference_zero_shot")
async def inference_zero_shot(tts_text: str = Form(), prompt_text: str = Form(),
                              prompt_wav: UploadFile = File(), speed: float = Form(1.0)):
    prompt_path = save_upload(prompt_wav)
    model_output = _model().inference_zero_shot(tts_text, prompt_text, prompt_path, speed=speed)
    return StreamingResponse(stream_and_cleanup(model_output, prompt_path),
                            background=BackgroundTask(_remove_quietly, prompt_path))


@app.get("/inference_cross_lingual")
@app.post("/inference_cross_lingual")
async def inference_cross_lingual(tts_text: str = Form(), prompt_wav: UploadFile = File(),
                                  speed: float = Form(1.0)):
    prompt_path = save_upload(prompt_wav)
    model_output = _model().inference_cross_lingual(tts_text, prompt_path, speed=speed)
    return StreamingResponse(stream_and_cleanup(model_output, prompt_path),
                            background=BackgroundTask(_remove_quietly, prompt_path))


@app.get("/inference_instruct")
@app.post("/inference_instruct")
async def inference_instruct(tts_text: str = Form(), spk_id: str = Form(),
                             instruct_text: str = Form(), speed: float = Form(1.0)):
    model_output = _model().inference_instruct(tts_text, spk_id, instruct_text, speed=speed)
    return StreamingResponse(generate_data(model_output))


@app.get("/inference_instruct2")
@app.post("/inference_instruct2")
async def inference_instruct2(tts_text: str = Form(), instruct_text: str = Form(),
                              prompt_wav: UploadFile = File(), speed: float = Form(1.0)):
    prompt_path = save_upload(prompt_wav)
    model_output = _model().inference_instruct2(tts_text, instruct_text, prompt_path, speed=speed)
    return StreamingResponse(stream_and_cleanup(model_output, prompt_path),
                            background=BackgroundTask(_remove_quietly, prompt_path))


@app.get("/add_zero_shot_spk")
@app.post("/add_zero_shot_spk")
async def add_zero_shot_spk(spk_id: str = Form(), prompt_text: str = Form(), prompt_wav: UploadFile = File()):
    prompt_path = save_upload(prompt_wav)
    try:
        ok = _model().add_zero_shot_spk(prompt_text, prompt_path, spk_id)
    finally:
        try:
            os.remove(prompt_path)
        except OSError:
            pass
    return {"ok": bool(ok), "spk_id": spk_id, "spks": _model().list_available_spks()}


@app.get("/inference_zero_shot_spk")
@app.post("/inference_zero_shot_spk")
async def inference_zero_shot_spk(tts_text: str = Form(), spk_id: str = Form(), speed: float = Form(1.0)):
    m = _model()
    if spk_id not in m.frontend.spk2info:
        return JSONResponse({"ok": False, "error": "unknown spk_id"}, status_code=404)
    model_output = m.inference_zero_shot(tts_text, '', '', zero_shot_spk_id=spk_id, speed=speed)
    return StreamingResponse(generate_data(model_output))

# ------------------------------------------------------------
# split-prompt：情绪/音色解耦（参考 IndexTTS2 思路）
#   LLM 端用情绪参考（prompt_text + 语音 token）决定韵律/情绪，
#   flow 端用音色参考（speech token / fbank / 说话人向量）决定音色。
#   emo_prompt_text 必须是情绪参考音频的转录（CosyVoice3 需带 <|endofprompt|> 前缀）。
# ------------------------------------------------------------
_FLOW_KEYS = ('flow_prompt_speech_token', 'flow_prompt_speech_token_len',
              'prompt_speech_feat', 'prompt_speech_feat_len', 'flow_embedding')


def _split_prompt_gen(m, tts_text, emo_prompt_text, emo_path, spk_path, speed):
    emo_prompt_text = m.frontend.text_normalize(emo_prompt_text, split=False, text_frontend=True)
    spk_part = None
    for seg in m.frontend.text_normalize(tts_text, split=True, text_frontend=True):
        model_input = m.frontend.frontend_zero_shot(seg, emo_prompt_text, emo_path, m.sample_rate, '')
        if spk_part is None:
            spk_input = m.frontend.frontend_zero_shot(seg, emo_prompt_text, spk_path, m.sample_rate, '')
            spk_part = {k: spk_input[k] for k in _FLOW_KEYS}
        model_input.update(spk_part)
        for out in m.model.tts(**model_input, stream=False, speed=speed):
            yield out


def _stream_and_cleanup2(model_output, p1, p2):
    try:
        yield from generate_data(model_output)
    finally:
        _remove_quietly(p1)
        _remove_quietly(p2)


def _remove_two(p1, p2):
    _remove_quietly(p1)
    _remove_quietly(p2)


@app.post("/inference_split_prompt")
async def inference_split_prompt(tts_text: str = Form(), emo_prompt_text: str = Form(),
                                 emo_prompt_wav: UploadFile = File(), spk_prompt_wav: UploadFile = File(),
                                 speed: float = Form(1.0)):
    m = _model()
    emo_path = save_upload(emo_prompt_wav)
    try:
        spk_path = save_upload(spk_prompt_wav)
    except Exception:
        _remove_quietly(emo_path)
        raise
    gen = _split_prompt_gen(m, tts_text, emo_prompt_text, emo_path, spk_path, speed)
    return StreamingResponse(_stream_and_cleanup2(gen, emo_path, spk_path),
                             background=BackgroundTask(_remove_two, emo_path, spk_path))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--port',
                        type=int,
                        default=50000)
    parser.add_argument('--model_dir',
                        type=str,
                        default='iic/CosyVoice2-0.5B',
                        help='local path or modelscope repo id')
    parser.add_argument('--fp16',
                        action='store_true',
                        help='以 fp16 加载模型（仅 CUDA 下生效）')
    args = parser.parse_args()
    cosyvoice = AutoModel(model_dir=args.model_dir, fp16=args.fp16)
    uvicorn.run(app, host=os.environ.get('COSYVOICE_HOST', '127.0.0.1'), port=args.port)
