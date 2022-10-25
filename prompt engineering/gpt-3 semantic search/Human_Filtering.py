import os
import json
import pprint
import readchar
import signal

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

class Human_Filtering():
  def __init__(self, json_file_path: str = './GPT-3_semantic_search.json'):
    # Items to keep    
    self.top_quality = json.load(open('1_top_quality.json', 'r'))
    self.decent_enough_to_keep = json.load(open('2_decent_enough_to_keep.json', 'r'))
    self.to_delete = json.load(open('3_to_delete.json', 'r'))
    self.invalid_items = json.load(open('4_invalid_items.json', 'r'))
    # File to evaluate     
    self.QA_pairs = json.load(open(json_file_path))
    
    self.last_completed_index = 0
    
    # capture ctrl-c, and save json to disk.
    signal.signal(signal.SIGINT, self.handler)
    
    start_index = int(input(f"{bcolors.BOLD}Enter the index to start from (or 0 to start fresh).{bcolors.ENDC}\n(This would have been displayed the last time you ran this program)\n{bcolors.BOLD}Start index: {bcolors.ENDC}"))
    self.evaluate_all_enteries(start_index)

  def evaluate_all_enteries(self, start_index):
    for item in self.QA_pairs:
      # skip items that have already been evaluated
      if self.QA_pairs.index(item) < start_index:
        continue
      
      pprint.pprint(item, width=120, sort_dicts=False)
      keep_val = input('Should we keep this?\n1 = top quality, 2 = decent enough to keep, 3 = to delete\n')
      try:
        keep_val = int(keep_val)
      except ValueError:
        print(f'{bcolors.WARNING}Invalid input. Please enter a number next time.{bcolors.ENDC}')
        self.invalid_items.append(item)
        continue
      
      if keep_val == 1:
        self.top_quality.append(item)
      elif keep_val == 2:
        self.decent_enough_to_keep.append(item)
      elif keep_val == 3:
        self.to_delete.append(item)
      else:
        print("invalid input")
        self.invalid_items.append(item)
      self.last_completed_index = self.QA_pairs.index(item)
      print("Last completed index: ", self.last_completed_index)
      # break

  def save_json_to_disk(self):
    # write json top_quality to file 
    with open('1_top_quality.json', 'w') as f:
      json.dump(self.top_quality, f, indent=2)
    # write json decent_enough_to_keep to file 
    with open('2_decent_enough_to_keep.json', 'w') as f:
      json.dump(self.decent_enough_to_keep, f, indent=2)
    # write json to_delete to file
    with open('3_to_delete.json', 'w') as f:
      json.dump(self.to_delete, f, indent=2)
    # write json invalid_items to file
    with open('4_invalid_items.json', 'w') as f:
      json.dump(self.invalid_items, f, indent=2)
      
  def handler(self, signum, frame):
      # msg = "Ctrl-c was pressed. Do you really want to exit? y/n "
      # print(msg, end="", flush=True)
      # res = readchar.readchar()
      print("")
      self.save_json_to_disk()
      print("Last completed index is: ", self.last_completed_index)
      print("☝️ Use this to pick up where you left off last time.")
      exit(1)


# run evalution!
_ = Human_Filtering()