from scipy.linalg import svd, norm
from sklearn.metrics import accuracy_score, classification_report
import h5py
import matplotlib.pyplot as plt
import numpy as np
import sys


# Load in the dataset
with h5py.File('Data/usps.h5', 'r') as f:
    train = f.get('train')
    test = f.get('test')
    X_train = np.array(train.get('data'))
    y_train = np.array(train.get('target'))
    X_test = np.array(test.get('data'))
    y_test = np.array(test.get('target'))


''' Output the matrix shapes of the traning and test data. '''
def print_train_test_shapes():
    print('X_train shape:', X_train.shape) #(7291, 256)
    print('y_train shape:', y_train.shape) #(7291, 1)
    print('X_test shape:', X_test.shape) #(2007, 256)
    print('y_test shape:', y_test.shape) #(2007, 1)


''' Test outputting the first 10 digit images. '''
def show_first_10_Img():
    for i in range(0,10):
        img = X_test[i].reshape(16,16)
        plt.imshow(img, cmap='Greys')
        plt.show()


''' Test outputting the first 10 digit images. '''
def show_first_Digit_Img():
    for i in range(0,10):
        for j in range(y_train.size):
            if y_train[j] == i:
                subplot = plt.subplot(2,5,i+1)
                img = np.array(X_train[j])
                plt.imshow(img.reshape(16,16),cmap='Greys')
                plt.title('Digit: ' + str(i))
                subplot.axes.get_xaxis().set_visible(False)
                subplot.axes.get_yaxis().set_visible(False)
                plt.xticks([])
                plt.yticks([])
                break
    plt.tight_layout()
    plt.show()


''' Graphs the digit distribution for the training
    and testing dataset. '''
def graph_digit_sample_distribution():
    train_digit_distribution = np.zeros(10)
    test_digit_distribution = np.zeros(10)
    x = range(0, 10)
    for digit in range(y_train.size):
        train_digit_distribution[y_train[digit]] += 1
    plt.figure(figsize=(5,5))
    plt.bar(x, train_digit_distribution, color='royalblue', width = 0.8)
    plt.title('Digit Distribution for Training Dataset')
    plt.xlabel('Digit')
    plt.ylabel('Number')
    plt.xticks(x)
    plt.show()
    for digit in range(y_test.size):
        test_digit_distribution[y_test[digit]] += 1
    plt.figure(figsize=(5,5))
    plt.bar(x, test_digit_distribution, color='royalblue', width = 0.8)
    plt.title('Digit Distribution for Testing Dataset')
    plt.xlabel('Digit')
    plt.ylabel('Number')
    plt.xticks(x)
    plt.show()


''' Split X_train into dictionary key-value pairs, where the key
    is the digit class and the value is a numpy array of all the
    image arrays for that key. '''
def split_data():
    digit_dict = {}
    for img in range(y_train.size):
        digit = str(y_train[img])
        if digit in digit_dict.keys():
            digit_dict.update({digit: np.vstack((digit_dict[digit], np.array(X_train)[img]))})
        else:
            digit_dict[digit] = np.array(X_train)[img]
    return digit_dict


''' Output split digit dictionary data to 
    a file to use for visualization purposes. '''
def output_split_data(digit_dict):
    f = open('Visuals/Output/split.txt', 'w+')
    print(digit_dict, file = f)


''' Output image matrix array data for all digits to 
    a file to use for visualization purposes. '''
def output_digit_data():
    for digit in range(10):
        f = open('Visuals/Output/' + str(digit) + '.txt', 'w+')
        for img in range(y_train.size):
            if y_train[img] == digit:
                print('Picture' + str(img) + ':', file = f)
                print(X_train[img], file = f)
                print('\n\n', file = f)


''' Print out the shape of each individual digit
    Perform SVD on each digit using the now split data. '''
def digit_svd(digit_dict):
    svd_dict = {}
    for digit in range(10):
        (u,s,v_t) = svd(np.transpose(digit_dict[str(digit)]), full_matrices=False)
        svd_dict[str(digit)] = (u,s,v_t)
    return svd_dict


''' Output svd numpy array data for each digit class to a file
    to use for visualization purposes. '''
def output_svd_data(svd_dict):
    f = open('Visuals/Output/svd.txt', 'w+')
    for digit in range(10):
        print('Digit' + str(digit) + ':', file = f)
        print(svd_dict[str(digit)], file = f)
        print('\n\n', file = f)


''' Computes the N rank 1 matrix for each digit
    and outputs the machine learned image matrix. '''
def rank1(svd_dict, n):
    rank1 = {}
    for digit in range(10):
        intermediate_rank1 = None
        for rank in range(n):
            if intermediate_rank1 is None:
                intermediate_rank1 = svd_dict[str(digit)][0][:,rank] * svd_dict[str(digit)][1][rank] * svd_dict[str(digit)][2][:,rank]
            else:
                intermediate_rank1 = intermediate_rank1 + (svd_dict[str(digit)][0][:,rank] * svd_dict[str(digit)][1][rank] * svd_dict[str(digit)][2][:,rank])
        rank1[str(digit)] = intermediate_rank1
    return rank1


''' Outputs the N rank1 approximation images. '''
def rank1_graph(rank1):
    for digit in range(10):  
        subplot = plt.subplot(1,10,digit+1)
        img = rank1[str(digit)]
        plt.imshow(img.reshape(16,16),cmap='Greys')
        subplot.axes.get_xaxis().set_visible(False)
        subplot.axes.get_yaxis().set_visible(False)
        plt.xticks([])
        plt.yticks([])
    plt.tight_layout()
    plt.show()


''' Output a specific digit ordered by top 10 clearest images
    based on a user input number. '''
def digit_clearest_images(svd_dict):
    plt.figure(figsize=(5,5))
    while True:
        try:
            num = input('Please enter a number 0-9: ')
            if str(num) in [str(num) for num in range(10)]:
                break;
            else:
                print('Error: please enter a number 0-9.')
        except ValueError:
            print('Error: please enter a number 0-9.')
    for i in range(10):
        plt.subplot(2,5,i+1)
        plt.imshow(svd_dict[num][0][:,i].reshape(16,16),cmap='binary')
    plt.show()


''' Output singular value graphs for each number to show the range of
    the singular value. The beginning ones are the highest because sigma
    is order by largest to smallest. '''
def singular_value_graphs(svd_dict):
    plt.figure(figsize=(15,6))
    for i in range(10):
        plt.subplot(1,10,i+1)
        plt.plot(svd_dict[str(i)][1], color='royalblue', marker='o')
        plt.title(f'Digit {i} Singular Values')
        plt.xlabel('Count')
        plt.ylabel('Singular Values')
        plt.xscale('linear')
        plt.yscale('log')
    plt.tight_layout()
    plt.show()


''' Output singular value graphs for the first 5 singular values
    for each number. The beginning ones are the highest because sigma
    is order by largest to smallest. '''
def first_N_singular_value_graphs(svd_dict, N):
    plt.figure(figsize=(15,6))
    x = range(N)
    for i in range(10):
        plt.subplot(2,5,i+1)
        plt.plot(x, svd_dict[str(i)][1][0:N], color='royalblue', marker='o')
        plt.title(f'Digit {i}')
        plt.xlabel('Count')
        plt.ylabel('Singular Values')
        plt.xscale('linear')
        plt.yscale('linear')
    plt.tight_layout()
    plt.show()


''' Uses Least Squares equation to predict the digit of a test image
    using a specified singular value. '''
def singular_value_least_squares_predit(svd_dict, singular_value):
    predictions = []
    I = np.identity(X_test.shape[1])
    for img in range(y_test.size):
        least_squares = []
        for digit in range(10):
            u = svd_dict[str(digit)][0][:,0:singular_value]
            val = norm(np.dot(I - np.dot(u, np.transpose(u)), X_test[img])) / norm(X_test[img])
            least_squares.append(val)
        predictions.append(np.argmin(least_squares))
    return predictions


''' Uses Least Squares equation to predict the digit of a test image
    using a specified range of singular values. Outputs the accuracy score
    for each singular value. '''
def singular_value_range_least_squares_predit(svd_dict, min_value, max_value):
    scores = []
    I = np.identity(X_test.shape[1])
    for singular_value in range(min_value, max_value+1):
        predictions = singular_value_least_squares_predit(svd_dict, singular_value)
        accuracy = accuracy_score(y_test, predictions)
        scores.append(accuracy)
    print('Number of Singular Values', ' Accuracy Score')
    for s in range(len(scores)):
        print(s+min_value, '\t\t\t', round(scores[s], 4))
    return scores


''' Output number of singular values vs accuracy percentage
    graph using a specified range of singular values. '''
def accuracy_graph(svd_dict, min_value, max_value):
    scores = singular_value_range_least_squares_predit(svd_dict, min_value, max_value)
    plt.figure(figsize=(12,6))
    x = range(min_value, max_value+1)
    plt.plot(x, scores, color='royalblue', marker='o')
    plt.title('Number of Singular Values vs Accuracy Percentage')
    plt.xlabel('Number of Singular Values')
    plt.ylabel('Accuracy Percentage')
    plt.xscale('linear')
    plt.yscale('linear')
    plt.xticks(x)
    plt.show()


''' Outputs the accuracy of each digit using
    the 12th singular value. '''
def individual_digit_accuracy(svd_dict):
    digit_amount = np.zeros(10)
    digit_correct = np.zeros(10)
    predictions = singular_value_least_squares_predit(svd_dict, 12)
    for digit in range(y_test.size):
        digit_amount[y_test[digit]] += 1
        if y_test[digit] == predictions[digit]:
            digit_correct[y_test[digit]] += 1
    digit_scores = []
    for digit in range(10):
        score = digit_correct[digit] / digit_amount[digit]
        digit_scores.append(score)
    plt.figure(figsize=(6,6))
    x = range(10)
    y = np.arange(0.84,1,0.02)
    plt.bar(x, digit_scores, color='royalblue', width = 0.8)
    plt.title('Accuracy Percentage of Each Digit')
    plt.xlabel('Digit')
    plt.ylabel('Accuracy Percentage')
    plt.xscale('linear')
    plt.yscale('linear')
    plt.ylim([0.84, 1])
    plt.xticks(x)
    plt.yticks(y)
    plt.show()
    print(digit_scores)


''' Output the classification report (precision, recall, 
    f1-score, support) after calculating the least square 
    for each digit. '''
def print_classification_report(svd_dict, singular_value):
    predictions = singular_value_least_squares_predit(svd_dict, singular_value)
    print(classification_report(y_test, predictions))


''' Outputs all misclassified images for a specific singular value. '''
def misclassified_images(svd_dict, singular_value):
    predictions = singular_value_least_squares_predit(svd_dict, singular_value)
    misclassified = np.where(y_test != predictions)
    list = [*range(0,misclassified[0].size,50)]
    for i in list:
        if i == list[-1]:
            remainder = misclassified[0].size - list[-1]
            num = remainder if remainder != 0 else 50
        else:
            num = 50
        for m in range(num):
            misclassified_id = misclassified[0][m+(1*i)]
            subplot = plt.subplot(5,10,m+1)
            img = np.array(X_test[misclassified_id])
            plt.imshow(img.reshape(16,16),cmap='Greys')
            plt.title('Correct:' + str(y_test[misclassified_id]) + '\n' + 'Predicted:' + str(predictions[misclassified_id]), fontsize = 5)
            subplot.axes.get_xaxis().set_visible(False)
            subplot.axes.get_yaxis().set_visible(False)
            plt.xticks([])
            plt.yticks([])
        plt.tight_layout()
        plt.show()
    
    
def main():
    np.set_printoptions(threshold=sys.maxsize)
    # print_train_test_shapes()
    # show_first_10_Img()
    # show_first_Digit_Img()
    # graph_digit_sample_distribution()
    digit_dict = split_data()
    # output_split_data(digit_dict)
    # output_digit_data()
    # svd_dict = digit_svd(digit_dict)
    # output_svd_data(svd_dict)
    # rank1_dict = rank1(svd_dict, 12)
    # rank1_graph(rank1_dict)
    # digit_clearest_images(svd_dict)
    # singular_value_graphs(svd_dict)
    # first_N_singular_value_graphs(svd_dict, 20)
    # singular_value_least_squares_predit(svd_dict, 12)
    # singular_value_range_least_squares_predit(svd_dict, 1, 15)
    # accuracy_graph(svd_dict, 1, 20)
    # individual_digit_accuracy(svd_dict)
    # print_classification_report(svd_dict, 12)
    # misclassified_images(svd_dict, 12)


if __name__ == '__main__': main()