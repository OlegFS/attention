"""
====================================
Set of tools to assist with testing.
====================================
"""

def needs_attention(func):
    """Decorator for incomplete tests."""
    def wrapper(*args, **kwargs):
        print ' '
        print 'WARNING: the test <'+func.func_name+'> needs to be completed'
        return func(*args, **kwargs)
    return wrapper


def skip(func):
    """Decorator for skipping tests."""
    def replacement_func(*args, **kwargs):
        pass
    def wrapper(*args, **kwargs):
        print ' '
        print 'WARNING: the test <'+func.func_name+'> is being skipped'
        return replacement_func
    return wrapper

