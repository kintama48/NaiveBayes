def train_model(x, y):  # calculate prob of y given x, P(y|x) (y=label1, x=[feat1...feat10]
    count0, prior0 = calculate_priori(y, 0)
    count1, prior1 = calculate_priori(y, 1)

    PF1L0, PF0L0, PF1L1, PF0L1 = prob_A_given_B(x, y, count0, count1)

    return PF1L0, PF0L0, PF1L1, PF0L1, prior0, prior1


def calculate_priori(y: list, filter_by: int):
    count = len(list(filter(lambda i: i == filter_by, y)))
    return count, count / len(y)


def prob_A_given_B(x: list, y: list, prior_count_0: int, prior_count_1: int):  # lists of P(A|B0) & P(A|B1)

    prob_features_0_label_0 = []
    prob_features_1_label_0 = []

    prob_features_0_label_1 = []
    prob_features_1_label_1 = []

    # number of 0s and 1s when label is 0
    num_0_label_0 = 0
    num_1_label_0 = 0

    # number of 0s and 1s when label is 1
    num_0_label_1 = 0
    num_1_label_1 = 0

    for column in range(len(x[0])):
        for row in range(len(x)):
            #        when label is 1
            if y[row]:
                if not x[row][column]:  # when feat is 0
                    num_0_label_1 += 1
                else:                   # when feat is 1
                    num_1_label_1 += 1

            #       when label is 0
            else:
                if not x[row][column]:  # when feat is 0
                    num_0_label_0 += 1
                else:                   # when feat is 1
                    num_1_label_0 += 1

        prob_features_0_label_1.append(num_0_label_1 / prior_count_1)
        prob_features_1_label_1.append(num_1_label_1 / prior_count_1)

        prob_features_0_label_0.append(num_0_label_0 / prior_count_0)
        prob_features_1_label_0.append(num_1_label_0 / prior_count_0)

        num_0_label_0 = 0
        num_1_label_0 = 0
        num_0_label_1 = 0
        num_1_label_1 = 0

    return prob_features_1_label_0, prob_features_0_label_0, prob_features_1_label_1, prob_features_0_label_1


def naive_bayes(data, **kwargs):
    label_0_prob = 1
    label_1_prob = 1

    for i in range(len(data)):
        if data[i] == 0:            # Prob when Feature=0 for both states of Label
            label_0_prob *= kwargs["PF0L0"][i]
            label_1_prob *= kwargs["PF0L1"][i]
        else:                       # Prob when Feature=1 for both states of Label
            label_0_prob *= kwargs["PF1L0"][i]
            label_1_prob *= kwargs["PF1L1"][i]

    return label_0_prob, label_1_prob
