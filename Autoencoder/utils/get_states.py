import requests
import os

def get_state(state_path,save_path):
    save_path=save_path+"/model_state"
    os.makedirs(save_path,exist_ok = True)
    r = requests.get(state_path, allow_redirects=True)
    print("requesting...")
    with open(save_path+"/model.pt", 'wb')as f:
        f.write(r.content)