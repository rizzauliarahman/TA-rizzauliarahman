import random
import os
import pickle

home = os.path.dirname(os.getcwd())

img_names = []
races = []

for root, dirs, files in os.walk(home + "\\Dataset"):
    for d in dirs:
        filenames = [(home + "\\Dataset\\" + d + "\\" + f) for f in os.listdir(home + "\\Dataset\\" + d) if
                     os.path.isfile(home + "\\Dataset\\" + d + "\\" + f)]

        for f in filenames:
            ran = random.randint(1, len(filenames))
            if ran <= 2000:
                img_names.append(f)
                races.append(d)

svfile_races = open(home + '\\Attributes\\races.dat', mode='wb')
svfile_imgs = open(home + '\\Attributes\\imgs.dat', mode='wb')

pickle.dump(races, svfile_races)
pickle.dump(img_names, svfile_imgs)

svfile_races.close()
svfile_imgs.close()
