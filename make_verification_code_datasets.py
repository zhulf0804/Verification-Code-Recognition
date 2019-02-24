from captcha.image import ImageCaptcha
import numpy as np
from PIL import Image
import random
import sys

number = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']


def random_captcha_text(char_set=number, captcha_size=4):
    # verification code list
    captcha_text = []
    for i in range(captcha_size):
        c = random.choice(char_set)
        # add into verification code list
        captcha_text.append(c)
    return captcha_text


# generate the verification code from char
def gen_captcha_text_image():
    image = ImageCaptcha()
    captcha_text = random_captcha_text()
    captcha_text = ''.join(captcha_text)
    captcha = image.generate(captcha_text)

    image.write(captcha_text, 'captcha/images/' + captcha_text + '.jpg')  # writes to the file

# the number is less than 10000 for the repeated names
num = 10000

if __name__ == '__main__':
    for i in range(num):
        gen_captcha_text_image()
        sys.stdout.write('\r>> Creating image %d/%d' % (i + 1, num))
        sys.stdout.flush()
    sys.stdout.write('\n')
    sys.stdout.flush()
