import  os


# directory root
root_project = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# data
dir_data = os.path.join(root_project,"data")
dir_data_processed = os.path.join(dir_data,"processed")



# class
categories = ["head","helmet","person"]
num_worker = 2
