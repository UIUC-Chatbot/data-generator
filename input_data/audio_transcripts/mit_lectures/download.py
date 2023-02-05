# from __future__ import unicode_literals
# import youtube_dl

# ydl = youtube_dl.YoutubeDL({
#     'format': 'bestaudio/best',
#     'postprocessors': [{

#         'key': 'FFmpegExtractAudio',
#         'preferredquality': '192',
#         'preferredcodec': 'mp3'
#     }]
# })

# with ydl:
#     result = ydl.extract_info('https://www.youtube.com/playlist?list=PL9B24A6A9D5754E70')

import os
import whisper
import json

transcripts = []
directory = '/Users/joshuamin/Desktop/data-generator/input_data/audio_transcripts/mit_lectures/mit_lecture_audios'


for filename in os.listdir(directory):
    model = whisper.load_model("medium")
    result = model.transcribe("/Users/joshuamin/Desktop/data-generator/input_data/audio_transcripts/mit_lectures/mit_lecture_audios/" + filename)
    transcripts.append({filename:result['text']})
    print(filename, 'done')
print("All Done!")


with open('mit_lecture_transcripts.json', 'w', encoding='utf-8') as f:
    json.dump(transcripts, f, ensure_ascii=False, indent=4)

    