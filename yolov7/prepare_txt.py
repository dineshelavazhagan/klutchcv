import os


path = "runs/detect/all_train/train/images/"


# train_file = "data/train_basker_ball.txt"

# file_l = open(train_file, "w")

# for i in os.listdir(path):
#     print(path + i)
#     file_l.write(path + i)
#     file_l.write("\n")
# file_l.close()

path = "runs/detect/all_train/valid/images/"



# train_file = "data/valid_basker_ball.txt"

# file_l = open(train_file, "w")

# for i in os.listdir(path):
#     print(path + i)
#     file_l.write(path + i)
#     file_l.write("\n")
# file_l.close()

path = "runs/detect/all_train/test/images/"


train_file = "data/test_basker_ball.txt"

file_l = open(train_file, "w")

for i in os.listdir(path):
    print(path + i)
    file_l.write(path + i)
    file_l.write("\n")
file_l.close()
