from os import popen

def _copy_to_clipboard(x):
   """copies a string to the X11 clipboard."""
   popen('xsel -b','wb').write(x)

def C(*args):
    """copies a specified input line to the clipboard (default=last)."""
    the_list = []
    for a in args:
        if isinstance(a, list):
            the_list += a
        else:
            the_list += [a]
    to_copy = ''
    for a in the_list:
       to_copy += In[a]
    _copy_to_clipboard(to_copy)


