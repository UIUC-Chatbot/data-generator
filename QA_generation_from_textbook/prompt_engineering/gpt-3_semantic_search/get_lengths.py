import json
top_quality = json.load(open('1_top_quality.json', 'r'))
decent_enough_to_keep = json.load(open('2_decent_enough_to_keep.json', 'r'))
to_delete = json.load(open('3_to_delete.json', 'r'))
print(len(top_quality))
print(len(decent_enough_to_keep))
print(len(to_delete))