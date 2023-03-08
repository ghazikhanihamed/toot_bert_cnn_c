import os
from settings import settings

# We change the name of the file in the representations folder that contains "finetuned". If the name of the file ends with "ionchannels" then we add "_iontransporters.h5" other wise we add ".h5

#for f in os.listdir(settings.REPRESENTATIONS_PATH):
#    if os.path.isfile(os.path.join(settings.REPRESENTATIONS_PATH, f)) and "finetuned" in f:
#        if f.endswith("membraneproteins.h5"):
#            os.rename(settings.REPRESENTATIONS_PATH + f, settings.REPRESENTATIONS_PATH + f[:-3] + "_imbalanced.h5")
#        elif f.endswith("iontransporters.h5"):
#            os.rename(settings.REPRESENTATIONS_PATH + f, settings.REPRESENTATIONS_PATH + f[:-3] + "_imbalanced.h5")

# We filter those files that contain "finetuned" if the first three parts of the name split by "_" are the same as the last three parts of the name split by "_" withouth the extension ".h5" + those files that contain "frozen"
files = [f for f in os.listdir(settings.REPRESENTATIONS_PATH) if os.path.isfile(os.path.join(settings.REPRESENTATIONS_PATH, f)) and "finetuned" in f and "_".join(f.split("_")[:3]) == "_".join(f.split("_")[-3:]).split(".")[0] or "frozen" in f]

# We copy the files to the folder "representations_filtered"
for f in files:
    os.system("cp " + settings.REPRESENTATIONS_PATH + f + " " + settings.REPRESENTATIONS_PATH + "representations_filtered/")
