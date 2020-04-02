import numpy as np
from word_processing import *
from w2v_utils import *

def neutralize(word, g, word_to_vec_map):
    """
    Removes the bias of "word" by projecting it on the space orthogonal to the bias axis.
    This function ensures that gender neutral words are zero in the gender subspace.

    Arguments:
        word -- string indicating the word to debias
        g -- numpy-array of shape (50,), corresponding to the bias axis (such as gender)
        word_to_vec_map -- dictionary mapping words to their corresponding vectors.

    Returns:
        e_debiased -- neutralized word vector representation of the input "word"
    """

    ### START CODE HERE ###
    # Select word vector representation of "word". Use word_to_vec_map. (≈ 1 line)
    e = word_to_vec_map[word]

    # Compute e_biascomponent using the formula given above. (≈ 1 line)
    e_biascomponent = np.dot(e, g) / np.sum(np.dot(g, g)) * g
    #     e_biascomponent = np.sqrt(np.sum(np.dot(e,e))) * cosine_similarity(e, g) * g/np.sqrt(np.sum(np.dot(g,g)))
    # Neutralize e by subtracting e_biascomponent from it
    # e_debiased should be equal to its orthogonal projection. (≈ 1 line)
    e_debiased = e - e_biascomponent
    ### END CODE HERE ###

    return e_debiased


def equalize(pair, bias_axis, word_to_vec_map):
    """
    Debias gender specific words by following the equalize method described in the figure above.

    Arguments:
    pair -- pair of strings of gender specific words to debias, e.g. ("actress", "actor")
    bias_axis -- numpy-array of shape (50,), vector corresponding to the bias axis, e.g. gender
    word_to_vec_map -- dictionary mapping words to their corresponding vectors

    Returns
    e_1 -- word vector corresponding to the first word
    e_2 -- word vector corresponding to the second word
    """

    ### START CODE HERE ###
    # Step 1: Select word vector representation of "word". Use word_to_vec_map. (≈ 2 lines)
    w1, w2 = pair
    e_w1, e_w2 = (word_to_vec_map[w1], word_to_vec_map[w2])

    # Step 2: Compute the mean of e_w1 and e_w2 (≈ 1 line)
    mu = (e_w1 + e_w2) / 2

    # Step 3: Compute the projections of mu over the bias axis and the orthogonal axis (≈ 2 lines)
    mu_B = np.dot(mu, bias_axis) / np.sum(np.dot(bias_axis, bias_axis)) * bias_axis
    mu_orth = mu - mu_B

    # Step 4: Use equations (7) and (8) to compute e_w1B and e_w2B (≈2 lines)
    e_w1B = np.dot(e_w1, bias_axis) / np.sum(np.dot(bias_axis, bias_axis)) * bias_axis
    e_w2B = np.dot(e_w2, bias_axis) / np.sum(np.dot(bias_axis, bias_axis)) * bias_axis

    # Step 5: Adjust the Bias part of e_w1B and e_w2B using the formulas (9) and (10) given above (≈2 lines)
    corrected_e_w1B = np.sqrt(np.abs(1 - np.sum(np.dot(mu_orth, mu_orth)))) * (e_w1B - mu_B) / np.sqrt(
        np.sum(np.dot(e_w1 - mu_orth - mu_B, e_w1 - mu_orth - mu_B)))
    corrected_e_w2B = np.sqrt(np.abs(1 - np.sum(np.dot(mu_orth, mu_orth)))) * (e_w2B - mu_B) / np.sqrt(
        np.sum(np.dot(e_w2 - mu_orth - mu_B, e_w2 - mu_orth - mu_B)))

    # Step 6: Debias by equalizing e1 and e2 to the sum of their corrected projections (≈2 lines)
    e1 = corrected_e_w1B + mu_orth
    e2 = corrected_e_w2B + mu_orth

    ### END CODE HERE ###

    return e1, e2

if __name__ == '__main__':
    words, word_to_vec_map = read_glove_vecs('GloveWord/glove.6B.50d.txt')
    g = word_to_vec_map['woman'] - word_to_vec_map['man']

    print('List of names and their similarities with constructed vector:')

    # girls and boys name
    name_list = ['john', 'marie', 'sophie', 'ronaldo', 'priya', 'rahul', 'danielle', 'reza', 'katy', 'yasmin']

    for w in name_list:
        print(w, cosine_similarity(word_to_vec_map[w], g))

    e1, e2 = equalize(('doctor', 'nurse'), g, word_to_vec_map)
    print(cosine_similarity(e1, g), cosine_similarity(e2, g))