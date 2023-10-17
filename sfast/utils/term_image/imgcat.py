from __future__ import print_function

import base64
import os
import sys


# tmux requires unrecognized OSC sequences to be wrapped with DCS tmux;
# <sequence> ST, and for all ESCs in <sequence> to be replaced with ESC ESC. It
# only accepts ESC backslash for ST.
def print_osc(terminal):
    if terminal.startswith('screen') or terminal.startswith('tmux'):
        print_partial("\033Ptmux;\033\033]")
    else:
        print_partial("\033]")


# More of the tmux workaround described above.
def print_st(terminal):
    if terminal.startswith('screen') or terminal.startswith('tmux'):
        print_partial("\a\033\\")
    else:
        print_partial("\a")


# print_image filename inline base64contents print_filename
#   filename: Filename to convey to client
#   inline: 0 or 1
#   base64contents: Base64-encoded contents
#   print_filename: If non-empty, print the filename
#                   before outputting the image
def print_image(image_file_name=None, data=None, width=None, height=None):
    terminal = os.environ.get('TERM', '')
    print_osc(terminal)
    print_partial('1337;File=')
    args = []
    if image_file_name:
        b64_file_name = base64.b64encode(image_file_name.encode('ascii')).decode('ascii')
        args.append('name=' + b64_file_name)
        with open(image_file_name, "rb") as image_file:
            b64_data = base64.b64encode(image_file.read()).decode('ascii')
    elif data:
        b64_data = base64.b64encode(data).decode('ascii')
    else:
        raise ValueError("Expected image_file_name or data")

    args.append('size=' + str(len(b64_data)))
    if width is not None:
        args.append('width=' + str(width))
    if height is not None:
        args.append('height=' + str(height))
    args.append("inline=1")

    print_partial(';'.join(args))
    print_partial(":")
    print_partial(b64_data)
    print_st(terminal)


def show_help():
    print("Usage: imgcat filename ...")
    print("   or: cat filename | python imgcat.py -")
    exit()


def print_partial(msg):
    print(msg, end='')


def _read_binary_stdin():
    # see https://stackoverflow.com/a/38939320/474819 for other platform notes
    PY3 = sys.version_info >= (3, 0)
    if PY3:
        source = sys.stdin.buffer
    else:
        # Python 2 on Windows opens sys.stdin in text mode, and
        # binary data that read from it becomes corrupted on \r\n
        if sys.platform == "win32":
            # set sys.stdin to binary mode
            import msvcrt
            msvcrt.setmode(sys.stdin.fileno(), os.O_BINARY)
        source = sys.stdin

    return source.read()


def main():
    filename = None
    data = None

    if len(sys.argv) != 2:
        show_help()

    if sys.argv[1] != '-':
        filename = sys.argv[1]
        print_image(image_file_name=filename)
    else:
        data = _read_binary_stdin()
        print_image(data=data)
    if not filename and not data:
        show_help()


if __name__ == '__main__':
    main()
