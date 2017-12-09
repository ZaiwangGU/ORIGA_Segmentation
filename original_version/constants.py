import os


def get_dir(directory):
    """
    Creates the givens directory if it does not exits.

    :param directory: The path to the directory.
    :return: The path to the directory.
    """

    if not os.path.exists(directory):
        os.makedirs(directory)
    return directory


# Saver
saver = get_dir('Images')
hardexudates_saver = get_dir(os.path.join(saver, 'hardexudates'))
hemorrhages_saver = get_dir(os.path.join(saver, 'hemorrhages'))
redsmalldots_saver = get_dir(os.path.join(saver, 'redsmalldots'))
softexudates_saver = get_dir(os.path.join(saver, 'softexudates'))
train_images_saver = get_dir(os.path.join(saver, 'train_images'))
test_images_saver = get_dir(os.path.join(saver, 'test_images'))
