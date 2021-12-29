import json
with open("./fileReturnJson",'r') as load_f:
    load_dict = json.load(load_f)
    print(load_dict)
    print(load_dict["threadList"])
    print(load_dict["dataLists"])
    print(load_dict["dataLists"][0])