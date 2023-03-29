import os
from settings import settings

# We remove all the files that ends with "__full.csv" in the results folder
for f in os.listdir(settings.RESULTS_PATH):
    if os.path.isfile(os.path.join(settings.RESULTS_PATH, f)) and f.endswith("__full.csv"):
        os.remove(settings.RESULTS_PATH + f)



# # We filter those files that contain "finetuned" and "full" and end with ".h5"
# for f in os.listdir(settings.REPRESENTATIONS_FILTERED_PATH + "backup/"):
#     if os.path.isfile(os.path.join(settings.REPRESENTATIONS_FILTERED_PATH + "backup/", f)) and "finetuned" in f and f.endswith(".h5") and "full" in f:
#         # If the file starts with "ionchannels_iontransporters" then we keep those files that parts 7 and 8 are "ionchannels" and "iontransporters" split by "_" and the rest cut in the backup folder
#         if f.startswith("ionchannels_iontransporters"):
#             if f.split("_")[7] == "ionchannels" and f.split("_")[8] == "iontransporters.h5":
#                 # We copy the file from the backup folder to the filtered folder
#                 os.rename(settings.REPRESENTATIONS_FILTERED_PATH + "backup/" + f, settings.REPRESENTATIONS_FILTERED_PATH + f)
#             else:
#                 continue
#         # If the file starts with "ionchannels_membraneproteins_balanced" then we keep those files that parts 9 and 10 and 11 are "ionchannels" and "membraneproteins" and "balanced" split by "_" and the rest cut in the backup folder
#         elif f.startswith("ionchannels_membraneproteins_balanced"):
#             if f.split("_")[9] == "ionchannels" and f.split("_")[10] == "membraneproteins":
#                 if f.split("_")[11] == "balanced.h5":
#                     # We copy the file from the backup folder to the filtered folder
#                     os.rename(settings.REPRESENTATIONS_FILTERED_PATH + "backup/" + f, settings.REPRESENTATIONS_FILTERED_PATH + f)
#                 else:
#                     continue
#             else:
#                 continue
#         # If the file starts with "ionchannels_membraneproteins_imbalanced" then we keep those files that parts 8 and 9 and 10 are "ionchannels" and "membraneproteins" and "imbalanced" split by "_" and the rest cut in the backup folder
#         elif f.startswith("ionchannels_membraneproteins_imbalanced"):
#             if f.split("_")[8] == "ionchannels" and f.split("_")[9] == "membraneproteins":
#                 if f.split("_")[10] == "imbalanced.h5":
#                     # We copy the file from the backup folder to the filtered folder
#                     os.rename(settings.REPRESENTATIONS_FILTERED_PATH + "backup/" + f, settings.REPRESENTATIONS_FILTERED_PATH + f)
#                 else:
#                     continue
#             else:
#                 continue
#         # If the file starts with "iontransporters_membraneproteins_balanced" then we keep those files that parts 9 and 10 and 11 are "iontransporters" and "membraneproteins" and "balanced" split by "_" and the rest cut in the backup folder
#         elif f.startswith("iontransporters_membraneproteins_balanced"):
#             if f.split("_")[9] == "iontransporters" and f.split("_")[10] == "membraneproteins":
#                 if f.split("_")[11] == "balanced.h5":
#                     # We copy the file from the backup folder to the filtered folder
#                     os.rename(settings.REPRESENTATIONS_FILTERED_PATH + "backup/" + f, settings.REPRESENTATIONS_FILTERED_PATH + f)
#                 else:
#                     continue
#             else:
#                 continue
#         # If the file starts with "iontransporters_membraneproteins_imbalanced" then we keep those files that parts 8 and 9 and 10 are "iontransporters" and "membraneproteins" and "imbalanced" split by "_" and the rest cut in the backup folder
#         elif f.startswith("iontransporters_membraneproteins_imbalanced"):
#             if f.split("_")[8] == "iontransporters" and f.split("_")[9] == "membraneproteins":
#                 if f.split("_")[10] == "imbalanced.h5":
#                     # We copy the file from the backup folder to the filtered folder
#                     os.rename(settings.REPRESENTATIONS_FILTERED_PATH + "backup/" + f, settings.REPRESENTATIONS_FILTERED_PATH + f)
#                 else:
#                     continue
#             else:
#                 continue



# # We change the name of the files that contain "frozen" and end with ".h5_full.h5" to end with ".h5"
# for f in os.listdir(settings.REPRESENTATIONS_FILTERED_PATH):
#     if os.path.isfile(os.path.join(settings.REPRESENTATIONS_FILTERED_PATH, f)) and "frozen" in f and f.endswith(".h5_full.h5"):
#         os.rename(settings.REPRESENTATIONS_FILTERED_PATH + f, settings.REPRESENTATIONS_FILTERED_PATH + f[:-11] + ".h5")

# # We change the name of the files that contain "finetuned" and end with "full" to file - "full" + "ionchannels_iontransporters.h5"
# for f in os.listdir(settings.REPRESENTATIONS_FILTERED_PATH):
#     if os.path.isfile(os.path.join(settings.REPRESENTATIONS_FILTERED_PATH, f)) and "finetuned" in f and f.endswith("full"):
#         os.rename(settings.REPRESENTATIONS_FILTERED_PATH + f, settings.REPRESENTATIONS_FILTERED_PATH + f[:-4] + "ionchannels_iontransporters.h5")

# # We change the name of the files that contain "finetuned" and end with "full_ionchannels_membraneproteins" to file - "full" + "ionchannels_membraneproteins_balanced.h5"
# for f in os.listdir(settings.REPRESENTATIONS_FILTERED_PATH):
#     if os.path.isfile(os.path.join(settings.REPRESENTATIONS_FILTERED_PATH, f)) and "finetuned" in f and f.endswith("full_ionchannels_membraneproteins"):
#         os.rename(settings.REPRESENTATIONS_FILTERED_PATH + f, settings.REPRESENTATIONS_FILTERED_PATH + f[:-33] + "ionchannels_membraneproteins_balanced.h5")

# # We change the name of the files that contain "finetuned" and end with "full_ionchannels" to file - "full" + "ionchannels_membraneproteins_imbalanced.h5"
# for f in os.listdir(settings.REPRESENTATIONS_FILTERED_PATH):
#     if os.path.isfile(os.path.join(settings.REPRESENTATIONS_FILTERED_PATH, f)) and "finetuned" in f and f.endswith("full_ionchannels"):
#         os.rename(settings.REPRESENTATIONS_FILTERED_PATH + f, settings.REPRESENTATIONS_FILTERED_PATH + f[:-16] + "ionchannels_membraneproteins_imbalanced.h5")

# # We change the name of the files that contain "finetuned" and end with "full_iontransporters_membraneproteins" to file - "full" + "iontransporters_membraneproteins_balanced.h5"
# for f in os.listdir(settings.REPRESENTATIONS_FILTERED_PATH):
#     if os.path.isfile(os.path.join(settings.REPRESENTATIONS_FILTERED_PATH, f)) and "finetuned" in f and f.endswith("full_iontransporters_membraneproteins"):
#         os.rename(settings.REPRESENTATIONS_FILTERED_PATH + f, settings.REPRESENTATIONS_FILTERED_PATH + f[:-37] + "iontransporters_membraneproteins_balanced.h5")

# # We change the name of the files that contain "finetuned" and end with "full_iontransporters" to file - "full" + "iontransporters_membraneproteins_imbalanced.h5"
# for f in os.listdir(settings.REPRESENTATIONS_FILTERED_PATH):
#     if os.path.isfile(os.path.join(settings.REPRESENTATIONS_FILTERED_PATH, f)) and "finetuned" in f and f.endswith("full_iontransporters"):
#         os.rename(settings.REPRESENTATIONS_FILTERED_PATH + f, settings.REPRESENTATIONS_FILTERED_PATH + f[:-20] + "iontransporters_membraneproteins_imbalanced.h5")


# We filter those files that contain "finetuned" and "full" and delete those files that first part of the name split by "_" is different from the 





# We change the name of the file in the representations folder that contains "finetuned". If the name of the file ends with "ionchannels" then we add "_iontransporters.h5" other wise we add ".h5

#for f in os.listdir(settings.REPRESENTATIONS_PATH):
#    if os.path.isfile(os.path.join(settings.REPRESENTATIONS_PATH, f)) and "finetuned" in f:
#        if f.endswith("membraneproteins.h5"):
#            os.rename(settings.REPRESENTATIONS_PATH + f, settings.REPRESENTATIONS_PATH + f[:-3] + "_imbalanced.h5")
#        elif f.endswith("iontransporters.h5"):
#            os.rename(settings.REPRESENTATIONS_PATH + f, settings.REPRESENTATIONS_PATH + f[:-3] + "_imbalanced.h5")

# # We filter those files that contain "finetuned" if the first three parts of the name split by "_" are the same as the last three parts of the name split by "_" withouth the extension ".h5" + those files that contain "frozen"
# files = [f for f in os.listdir(settings.REPRESENTATIONS_PATH) if os.path.isfile(os.path.join(settings.REPRESENTATIONS_PATH, f)) and "finetuned" in f and "_".join(f.split("_")[:3]) == "_".join(f.split("_")[-3:]).split(".")[0] or "frozen" in f]

# # We copy the files to the folder "representations_filtered"
# for f in files:
#     os.system("cp " + settings.REPRESENTATIONS_PATH + f + " " + settings.REPRESENTATIONS_PATH + "representations_filtered/")

# We filter those files that contain "finetuned" where first and sixth part is "ionchannels" and the second and the seventh part is "iontransporters" and eighth part is "imbalanced"
# files = [f for f in os.listdir(settings.REPRESENTATIONS_PATH) if os.path.isfile(os.path.join(settings.REPRESENTATIONS_PATH, f)) and "finetuned" in f and f.split("_")[0] == "ionchannels" and f.split("_")[6] == "ionchannels" and f.split("_")[1] == "iontransporters" and f.split("_")[7] == "iontransporters" and f.split("_")[8] == "imbalanced.h5"]

# We copy the files to the folder "representations_filtered"
# for f in files:
#     os.system("cp " + settings.REPRESENTATIONS_PATH + f + " " + settings.REPRESENTATIONS_PATH + "representations_filtered/")
    