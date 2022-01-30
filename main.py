import method
import data_loader
import os
import numpy as np

# Hyperparameters (fixed across all dataset)
HIDDEN_DIM = 256
BATCH_SIZE = 131072
EPOCHS = 10000
EARLYSTOPPING_PATIENCE = 30
FIRST_OMEGA_0 = 3000
VERBOSE = True  # if True, you can check current status of operation.
EVALUATE = True  # # if False, you can check training time without evaluation (in terms of Best F1-score)

"""
Since size of datasets we use in our experiments exceeds the available max limit by CMT, 
we show the simplified implementation example on a subset (only one entity among 28 entities) of SMD dataset.
In order to check clear reproducibility for the result on every dataset from our proposed method INRAD, please refer to 'README.txt' file.
"""

# DATASET can be one of "SMD", "MSL", "SMAP", "SWaT", "WADI"

DATASET = "SMD"  # Before downloading all dataset by refering README.txt, choice is restriced by SMD only.

# Set VARIANT to True if INRAD-c
VARIANT = False

# Set location of main.py as PATH (Assuming that /data folder is also included in the same directory)

PATH = os.getcwd() + "\\data/\\"

dataset = data_loader.dataset_choice(DATASET, path=PATH)
inrad = method.INRAD(dataset, variant=VARIANT, evaluate=EVALUATE, verbose=VERBOSE)
print("Dataset: {}, Variant: {}".format(DATASET, VARIANT))


inrad.Representation_Learning(
    hidden_dim=HIDDEN_DIM,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    earlystopping_patience=EARLYSTOPPING_PATIENCE,
    first_omega_0=FIRST_OMEGA_0,
)

average_p, average_r, average_f1 = inrad.evaluation(verbose=False)
print("Precision: {}, Recall {}, F1-score: {}".format(average_p, average_r, average_f1))
if VARIANT == False:
    print(
        "average training time per epoch over entities: ",
        np.mean(inrad.train_time_per_epoch),
    )
