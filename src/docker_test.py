import imageio
import mlflow
from matplotlib import pyplot as plt


print("cool print")
plt.imshow(imageio.imread("figures/brtseg_workflow.png"))
plt.show()

with open("output_txt.txt", 'w') as output_txt:
    output_txt.write("cool write\n")

mlflow.log_artifact("output_txt.txt")
