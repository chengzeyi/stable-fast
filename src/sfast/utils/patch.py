def patch_module(m, filter_func, patch_func, stack=None, inplace=False):
    if stack is None:
        stack = [(None, m)]
        if filter_func(stack):
            if inplace:
                patch_func(m)
            else:
                m = patch_func(m)
    for name, child in m.named_children():
        stack.append((name, child))
        if filter_func(stack):
            if inplace:
                patch_func(child)
            else:
                setattr(m, name, patch_func(child))
        else:
            patch_module(child, filter_func, patch_func, stack, inplace)
        stack.pop()
    return m
