#! /usr/bin/env python3
"""
This program provides the ability to convert an image file to ANSI compatible
output for display in terminal sessions that support ANSI code.

#
This is a derivative of the improvements made by Takumi Sueda (puhitaku) on
image-to-ansi.py at https://gist.github.com/puhitaku/7eaf6142fa5a42425f55 which
provided significant (7x) speed improvements in turn is a derivative of the
work of Kevin Lange (klange) which was originally built on from Micah Elliott
(MicahElliott) a huge thank you to all these folks for the great foundational
work that I was able to slightly modify.

This version updates a variable to allow for a string variable (filename) to
be passed at run time from cli to the program. This simplifies execution by
allowing the user to run: python image-to-ansi.py <filename>

# Major Improvement List #
- direct passing of file variable to script
- basic error checking for existance of file
- basic error checking for single variable
- addition of help section explaining syntax
- general cleanup
- slight update and rework of the header format

-torrycrass
"""

__author__ = "Micah Elliott http://MicahElliott.com; Kevin Lange <k@dakko.us>; Takumi Sueda <puhitaku@gmail.com>; Torry Crass <tc.github@outlook.com>"
__copyright__ = "Copyright (C) 2011 Micah Elliott.  All rights reserved."
__credits__ = ["Micah Elliott", "Kevin Lange", "Takumi Sueda", "Torry Crass"]

__license__ = "WTFPL http://sam.zoy.org/wtfpl/"
__version__ = "0.2"
__maintainer__ = "Torry Crass"
__email__ = "tc.github@outlook.com"
__status__ = "Development"

import sys
import os.path

CLUT = [
    ('000000', '00'),
    ('800000', '01'),
    ('008000', '02'),
    ('808000', '03'),
    ('000080', '04'),
    ('800080', '05'),
    ('008080', '06'),
    ('c0c0c0', '07'),
    ('808080', '08'),
    ('ff0000', '09'),
    ('00ff00', '10'),
    ('ffff00', '11'),
    ('0000ff', '12'),
    ('ff00ff', '13'),
    ('00ffff', '14'),
    ('ffffff', '15'),
    ('000000', '16'),
    ('00005f', '17'),
    ('000087', '18'),
    ('0000af', '19'),
    ('0000d7', '20'),
    ('0000ff', '21'),
    ('005f00', '22'),
    ('005f5f', '23'),
    ('005f87', '24'),
    ('005faf', '25'),
    ('005fd7', '26'),
    ('005fff', '27'),
    ('008700', '28'),
    ('00875f', '29'),
    ('008787', '30'),
    ('0087af', '31'),
    ('0087d7', '32'),
    ('0087ff', '33'),
    ('00af00', '34'),
    ('00af5f', '35'),
    ('00af87', '36'),
    ('00afaf', '37'),
    ('00afd7', '38'),
    ('00afff', '39'),
    ('00d700', '40'),
    ('00d75f', '41'),
    ('00d787', '42'),
    ('00d7af', '43'),
    ('00d7d7', '44'),
    ('00d7ff', '45'),
    ('00ff00', '46'),
    ('00ff5f', '47'),
    ('00ff87', '48'),
    ('00ffaf', '49'),
    ('00ffd7', '50'),
    ('00ffff', '51'),
    ('5f0000', '52'),
    ('5f005f', '53'),
    ('5f0087', '54'),
    ('5f00af', '55'),
    ('5f00d7', '56'),
    ('5f00ff', '57'),
    ('5f5f00', '58'),
    ('5f5f5f', '59'),
    ('5f5f87', '60'),
    ('5f5faf', '61'),
    ('5f5fd7', '62'),
    ('5f5fff', '63'),
    ('5f8700', '64'),
    ('5f875f', '65'),
    ('5f8787', '66'),
    ('5f87af', '67'),
    ('5f87d7', '68'),
    ('5f87ff', '69'),
    ('5faf00', '70'),
    ('5faf5f', '71'),
    ('5faf87', '72'),
    ('5fafaf', '73'),
    ('5fafd7', '74'),
    ('5fafff', '75'),
    ('5fd700', '76'),
    ('5fd75f', '77'),
    ('5fd787', '78'),
    ('5fd7af', '79'),
    ('5fd7d7', '80'),
    ('5fd7ff', '81'),
    ('5fff00', '82'),
    ('5fff5f', '83'),
    ('5fff87', '84'),
    ('5fffaf', '85'),
    ('5fffd7', '86'),
    ('5fffff', '87'),
    ('870000', '88'),
    ('87005f', '89'),
    ('870087', '90'),
    ('8700af', '91'),
    ('8700d7', '92'),
    ('8700ff', '93'),
    ('875f00', '94'),
    ('875f5f', '95'),
    ('875f87', '96'),
    ('875faf', '97'),
    ('875fd7', '98'),
    ('875fff', '99'),
    ('878700', '100'),
    ('87875f', '101'),
    ('878787', '102'),
    ('8787af', '103'),
    ('8787d7', '104'),
    ('8787ff', '105'),
    ('87af00', '106'),
    ('87af5f', '107'),
    ('87af87', '108'),
    ('87afaf', '109'),
    ('87afd7', '110'),
    ('87afff', '111'),
    ('87d700', '112'),
    ('87d75f', '113'),
    ('87d787', '114'),
    ('87d7af', '115'),
    ('87d7d7', '116'),
    ('87d7ff', '117'),
    ('87ff00', '118'),
    ('87ff5f', '119'),
    ('87ff87', '120'),
    ('87ffaf', '121'),
    ('87ffd7', '122'),
    ('87ffff', '123'),
    ('af0000', '124'),
    ('af005f', '125'),
    ('af0087', '126'),
    ('af00af', '127'),
    ('af00d7', '128'),
    ('af00ff', '129'),
    ('af5f00', '130'),
    ('af5f5f', '131'),
    ('af5f87', '132'),
    ('af5faf', '133'),
    ('af5fd7', '134'),
    ('af5fff', '135'),
    ('af8700', '136'),
    ('af875f', '137'),
    ('af8787', '138'),
    ('af87af', '139'),
    ('af87d7', '140'),
    ('af87ff', '141'),
    ('afaf00', '142'),
    ('afaf5f', '143'),
    ('afaf87', '144'),
    ('afafaf', '145'),
    ('afafd7', '146'),
    ('afafff', '147'),
    ('afd700', '148'),
    ('afd75f', '149'),
    ('afd787', '150'),
    ('afd7af', '151'),
    ('afd7d7', '152'),
    ('afd7ff', '153'),
    ('afff00', '154'),
    ('afff5f', '155'),
    ('afff87', '156'),
    ('afffaf', '157'),
    ('afffd7', '158'),
    ('afffff', '159'),
    ('d70000', '160'),
    ('d7005f', '161'),
    ('d70087', '162'),
    ('d700af', '163'),
    ('d700d7', '164'),
    ('d700ff', '165'),
    ('d75f00', '166'),
    ('d75f5f', '167'),
    ('d75f87', '168'),
    ('d75faf', '169'),
    ('d75fd7', '170'),
    ('d75fff', '171'),
    ('d78700', '172'),
    ('d7875f', '173'),
    ('d78787', '174'),
    ('d787af', '175'),
    ('d787d7', '176'),
    ('d787ff', '177'),
    ('d7af00', '178'),
    ('d7af5f', '179'),
    ('d7af87', '180'),
    ('d7afaf', '181'),
    ('d7afd7', '182'),
    ('d7afff', '183'),
    ('d7d700', '184'),
    ('d7d75f', '185'),
    ('d7d787', '186'),
    ('d7d7af', '187'),
    ('d7d7d7', '188'),
    ('d7d7ff', '189'),
    ('d7ff00', '190'),
    ('d7ff5f', '191'),
    ('d7ff87', '192'),
    ('d7ffaf', '193'),
    ('d7ffd7', '194'),
    ('d7ffff', '195'),
    ('ff0000', '196'),
    ('ff005f', '197'),
    ('ff0087', '198'),
    ('ff00af', '199'),
    ('ff00d7', '200'),
    ('ff00ff', '201'),
    ('ff5f00', '202'),
    ('ff5f5f', '203'),
    ('ff5f87', '204'),
    ('ff5faf', '205'),
    ('ff5fd7', '206'),
    ('ff5fff', '207'),
    ('ff8700', '208'),
    ('ff875f', '209'),
    ('ff8787', '210'),
    ('ff87af', '211'),
    ('ff87d7', '212'),
    ('ff87ff', '213'),
    ('ffaf00', '214'),
    ('ffaf5f', '215'),
    ('ffaf87', '216'),
    ('ffafaf', '217'),
    ('ffafd7', '218'),
    ('ffafff', '219'),
    ('ffd700', '220'),
    ('ffd75f', '221'),
    ('ffd787', '222'),
    ('ffd7af', '223'),
    ('ffd7d7', '224'),
    ('ffd7ff', '225'),
    ('ffff00', '226'),
    ('ffff5f', '227'),
    ('ffff87', '228'),
    ('ffffaf', '229'),
    ('ffffd7', '230'),
    ('ffffff', '231'),
    ('080808', '232'),
    ('121212', '233'),
    ('1c1c1c', '234'),
    ('262626', '235'),
    ('303030', '236'),
    ('3a3a3a', '237'),
    ('444444', '238'),
    ('4e4e4e', '239'),
    ('585858', '240'),
    ('626262', '241'),
    ('6c6c6c', '242'),
    ('767676', '243'),
    ('808080', '244'),
    ('8a8a8a', '245'),
    ('949494', '246'),
    ('9e9e9e', '247'),
    ('a8a8a8', '248'),
    ('b2b2b2', '249'),
    ('bcbcbc', '250'),
    ('c6c6c6', '251'),
    ('d0d0d0', '252'),
    ('dadada', '253'),
    ('e4e4e4', '254'),
    ('eeeeee', '255'),
]


def print_help():

    print("")
    print(75 * "=")
    print(30 * "=", "IMAGE to ANSI", 30 * "=")
    print(75 * "=")

    print("\nCredits: ", __credits__[0], "\n\t", __credits__[1], "\n\t",
          __credits__[2], "\n\t", __credits__[3])
    print("License: ", __license__)
    print("Version: ", __version__)
    print("Maintainer: ", __maintainer__, " ", __email__)
    print("Status: ", __status__)

    print("\nThis program allows you to convert a graphic file (preferably png) into\n" \
          "ANSI code that will display a rendition of the image in a console that\n" \
          "supports ANSI color codes.\n")

    print("You need to have python, python-image, python-pillow. You can either\n" \
          "install these with your package manager (apt/yum) or install python-pip\n" \
          "and install the necessary modules from there.\n")

    print("See the github repository for more information (if available).\n")

    print("\n### USAGE ###\n")
    print("Standard:\n" \
        "\tpython image-to-ansi.py <inputfile>\n")

    print(75 * "-")

    print("Source: https://github.com/torrycrass/image-to-ansi")

    print(75 * "-")
    print("")


def _create_incs_lut():
    incs = [(0x00, 0x5f), (0x5f, 0x87), (0x87, 0xaf), (0xaf, 0xd7),
            (0xd7, 0xff)]
    res = []
    for part in range(256):
        for s, b in incs:
            if s <= part <= b:
                if abs(s - part) < abs(b - part):
                    res.append('%02x' % s)
                else:
                    res.append('%02x' % b)
                break
    return res


RGB2SHORT_DICT = dict(CLUT)
INCS_LUT = _create_incs_lut()


def lut(part):
    return INCS_LUT[part]


def rgb2short_fast(r, g, b):
    return RGB2SHORT_DICT['%s%s%s' % (lut(r), lut(g), lut(b))]


if __name__ == '__main__':

    #Check that the user passed a single variable to the script
    if len(sys.argv) != 2:
        print("")
        print(30 * ">", "ERROR", 30 * "<")
        print("\nError: You must pass a single option to the program.")
        print("use --help or -h for syntax assistance.\n")
        print(67 * "^")
        print("")

    #Handle a help request
    elif str(sys.argv[1]) == "--help" or str(sys.argv[1]) == "-h":
        print_help()

    #Check that the variable provided is a valid file name on the system
    elif os.path.isfile(sys.argv[1]) is not True:
        print("")
        print(30 * ">", "ERROR", 30 * "<")
        print("\nError: This program requires an input file, the value provided\n" \
            "is not a file. Please use a valid file name.\n")
        print(67 * "^")
        print("")

    #If all other criteria are not met, process the file
    else:
        #Ensure the variable type is a string
        myoption = str(sys.argv[1])

        #Process the image file
        from PIL import Image
        im = Image.open(myoption)
        x = im.size[0]
        im = list(im.getdata())
        s = []
        for i, p in enumerate(im):
            short = rgb2short_fast(p[0], p[1], p[2])
            if len(p) > 3 and p[3] == 0:
                s.append("\033[0m  ")
            else:
                s.append("\033[48;5;%sm  " % short)

            if (i + 1) % x == 0:
                s.append("\033[0m\n")
        s.append("\n")
        sys.stdout.write(''.join(s))
