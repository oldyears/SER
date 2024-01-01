import subprocess
import os
import shutil
import argparse

HAPPY = "very happy"
SAD = "very sad"
ANGRY = "very angry"
SURPRISED = "very surprised"
NEUTRAL = "neutral"
FEARFUL = "very fearful"
DISGUSTED = "very disgusted"
CALM = "very calm"

youngwoman="92"
youngman="9017"
smallgirl="12787"
Middlemale="6097"
speakers = {"youngwoman":"92","youngman":"9017","smallgirl":"12787","Middlemale":"6097"}


def run_speech_emotion_recognition(audio_path):
    # Navigate to Speech-Emotion-Recognition directory and run prediction with the audio file path
    os.chdir('Speech-Emotion-Recognition')
    subprocess.run(['python', 'predict.py', 
                    '--config', 'configs/cnn1d.yaml',
                    '--file_path',audio_path])
    os.chdir('..')


def run_paddle_speech_recognition(audio_file):
    # This function needs to be run in the PaddleSpeech directory
    # os.chdir('PaddleSpeech')
    from PaddleSpeech.paddlespeech.cli.asr.infer import ASRExecutor
    asr = ASRExecutor()
    result = asr(audio_file=audio_file)
    print("您的识别结果是"+result)
    # os.chdir('..')
    return result

def get_phonemes_from_file(file_path):
    with open(file_path, 'r') as file:
        phonemes = file.read().strip()
    return phonemes

def run_emotivoice_synthesis(recognized_text, speaker_id="8051", emotion="very angry"):
    os.chdir('EmotiVoice')
    
    # 运行前端处理脚本生成音素
    with open('data/my_text.txt', 'w') as file:
        file.write(recognized_text)
    subprocess.run(['python', 'frontend.py', 'data/my_text.txt'], stdout=open('data/my_text_for_tts.txt', 'w'))
    
    # 读取音素数据
    phonemes = get_phonemes_from_file('data/my_text_for_tts.txt')

    # 格式化 EmotiVoice 输入
    formatted_input = f"{speaker_id}|{emotion}|{phonemes}|{recognized_text}"
    with open('data/formatted_input.txt', 'w') as file:
        file.write(formatted_input)

    # 运行 EmotiVoice 推理
    subprocess.run([
        'python', 'inference_am_vocoder_joint.py',
        '--logdir', 'prompt_tts_open_source_joint',
        '--config_folder', 'config/joint',
        '--checkpoint', 'g_00140000',
        '--test_file', 'data/formatted_input.txt'
    ])
    os.chdir('..')
    synthesized_audio_path = '/mnt/workspace/EmotiVoice/outputs/prompt_tts_open_source_joint/test_audio/audio/g_00140000/1.wav'
    destination_path = '/mnt/workspace/output'
    shutil.copy(synthesized_audio_path, destination_path)
    print("success")

if __name__ == '__main__':
    # 创建一个参数解析器
    parser = argparse.ArgumentParser(description='Process a file path.')
    # 添加参数
    parser.add_argument('--file_path', type=str, help='the path to the file to process')
    parser.add_argument('--Emotion', type=str)
    parser.add_argument('--speaker', type=str)
    # 解析参数
    args = parser.parse_args()

    audio_path=args.file_path
    run_speech_emotion_recognition(audio_path)
    speech_content = run_paddle_speech_recognition(audio_path)
    run_emotivoice_synthesis(speech_content,speaker_id= speakers[args.speaker],emotion=args.Emotion)
