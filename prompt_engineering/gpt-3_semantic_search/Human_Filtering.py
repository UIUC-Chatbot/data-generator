import os
import json
import pprint
import readchar
import signal
import time

class bcolors:
  """Class for printing in color"""
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
  def __init__(self, json_file_path: str = 'GPT-3_semantic_search.json'):
    """
    Simply change the json_file_path to the path of the json file you want to evaluate (like GPT-3_semantic_search.json)
    Progress evaluating it is saved between runs. 
    """
    # todo(wishlist) add command line selection of json file to evaluate
    
    # check if input is valid
    try:
      self.json_file_path = json_file_path
      self.QA_pairs = json.load(open(json_file_path))
    except FileNotFoundError:
      print(f'{bcolors.FAIL}File not found. Please check the class variable `json_file_path`.{bcolors.ENDC}')
      exit(1)
    
    # if files don't exist, we start from scratch.
    if not os.path.exists('1_top_quality.json'):
      """ start evalution from scratch. """
      print(f"{bcolors.WARNING}Warning: Starting evaluation from scratch!{bcolors.WARNING} (because 1_top_quality.json doesn't exist.)")
      
      # results
      self.top_quality = []
      self.decent_enough_to_keep = []
      self.to_delete = [] 
      self.invalid_items = []
      
      # utils 
      self.last_completed_index = int(self.progress_json)
      self.progress_file = f'progress_human_filtering_{self.json_file_path}.json'
    else:
      """ Continue from where we left off last time """
      # Items to keep    
      self.top_quality = json.load(open('1_top_quality.json', 'r'))
      self.decent_enough_to_keep = json.load(open('2_decent_enough_to_keep.json', 'r'))
      self.to_delete = json.load(open('3_to_delete.json', 'r'))
      self.invalid_items = json.load(open('4_invalid_items.json', 'r'))
      # Save progress between runs
      self.progress_file = f'progress_human_filtering.json'
      try:
        self.progress_json = json.load(open(self.progress_file, 'r'))
        self.last_completed_index = int(self.progress_json)
      except FileNotFoundError:
        print(f'{bcolors.WARNING}Warning: progress file not found. Starting from scratch. If this is wrong, don\'t save this run.{bcolors.ENDC}')
        self.last_completed_index = 0
    
      print(f"{bcolors.BOLD}Resuming evalution (starting at index {self.last_completed_index}).{bcolors.ENDC}")
      time.sleep(1)
    
    """Universal commands"""
    # capture ctrl-c, and save json to disk.
    signal.signal(signal.SIGINT, self.handler)
    self.evaluate_all_enteries(self.last_completed_index)
    # end __init__

  def evaluate_all_enteries(self, start_index: int):
    for item in self.QA_pairs:
      # skip items that have already been evaluated
      if self.QA_pairs.index(item) < start_index:
        continue
      
      # todo: add facy printing here. try: except: with bold chars in specific json fields. 
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
    # save progress between runs     
    with open(self.progress_file, 'w') as f:
      json.dump(self.last_completed_index, f, indent=2)
      
  def handler(self, signum, frame):
    msg = "Ctrl-c was pressed. Do you want to save your work? (Only save if you made good filtering decisions) y/n "
    print(msg, end="\n", flush=True)
    res = readchar.readchar()
    if res == 'y':
      print("")
      self.save_json_to_disk()
      print("Last completed index is: ", self.last_completed_index, end="\n", flush=True)
      print(f"{bcolors.OKGREEN}Your progress was saved! Exiting...{bcolors.ENDC}", end="\r", flush=True)
      exit(1)
    else:
      print("Not saving. Exiting...", end="\n", flush=True)
      print("Last completed index is: ", self.last_completed_index, end="\n", flush=True)
      print("☝️ Use this to pick up where you left off last time (if you want).", flush=True)
      exit(1)
        
# run evalution!
_ = Human_Filtering()