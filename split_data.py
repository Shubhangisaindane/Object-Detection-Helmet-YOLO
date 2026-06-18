import os

root_dir = os.path.dirname(__file__) # root dir

images = os.path.join(root_dir,"images")
labels = os.path.join(root_dir,"labels")

images_train = os.path.join(images, "train")
images_val = os.path.join(images, "val")

labels_train = os.path.join(labels, "train")
labels_val = os.path.join(labels, "val")

current_batch = os.listdir(images_train)
N = len(current_batch)

val_size = int(N*0.25)
val_batch = current_batch[:val_size]
val_batch = [x.replace(".png","") for x in val_batch]

#Cut, copy paste from train -> val
try:
    for f in val_batch:
        source = os.path.join(images_train, f+".png")
        destination = os.path.join(images_val, f+".png")

        os.rename(source, destination) # cut-copy paste
        print(f"Images transfered")
except Exception as err:
    print("While handling images")
    print(str(err))

try:
    for f in val_batch:
        source = os.path.join(labels_train, f+".txt")
        destination = os.path.join(labels_val, f+".txt")

        os.rename(source, destination)
        print(f"Labels transfered")
except Exception as err:
    print("While handling labels")
    print(str(err))