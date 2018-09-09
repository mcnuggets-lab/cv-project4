import pandas as pd
from matplotlib import pyplot as plt

STRIDE = 32
log_data = pd.read_csv("./logs/fcn{}_training.log".format(STRIDE))

plt.plot(log_data["epoch"], log_data["loss"], label="Training Loss")
plt.plot(log_data["epoch"], log_data["val_loss"], label="Testing Loss")
plt.title("FCN-{}s training curve".format(STRIDE))
plt.xlabel("epoch")
plt.ylabel("loss")
plt.ylim(ymin=0)
plt.legend()
plt.savefig("./plots/fcn{}_training.png".format(STRIDE))
plt.show()


