import csv
from naive_bayes import *


def read_files(file_name_x, file_name_y):
    file_x = csv.reader(file_name_x, delimiter=",", quotechar='\n')
    file_y = csv.reader(file_name_y, delimiter=",", quotechar='\n')
    features = []
    label = []

    next(file_x, None)
    next(file_y, None)

    while True:
        x = next(file_x, None)
        y = next(file_y, None)

        if not x or not y:
            break

        features.append(list(map(lambda x: int(x), x[1::])))
        label.append(int(y[1]))

    return features, label


if __name__ == "__main__":
    features, labels = read_files(open("trainNaive.csv"), open("trainNaiveLabels.csv"))
    PF1L0, PF0L0, PF1L1, PF0L1, prior0, prior1 = train_model(features, labels)

    user_input = (0, 1, 1, 0, 1, 0, 0, 0, 0, 1)   # csv data can be fed instead of just one tuple in place of user_input

    label_0_prob, label_1_prob = naive_bayes(user_input, PF0L0=PF0L0, PF0L1=PF0L1, PF1L0=PF1L0, PF1L1=PF1L1)

    print(f"\n_________________________________________________________________________________________________________\n"
          f"Given {user_input}, the probability (P(label|feature)) of Label=1 is {label_1_prob} and of Label=0\n"
          f"is {label_0_prob}. The probability of '{'Label=0' if label_0_prob > label_1_prob else 'Label=1'}' "
          f"occurring is more likely since its probability is larger than the \nother.\n"
          f"_________________________________________________________________________________________________________\n")
