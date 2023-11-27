def patch_module(m, filter_func, patch_func, stack=None):
    if stack is None:
        stack = []
    for name, child in m.named_children():
        stack.append((name, child))
        if filter_func(stack):
            setattr(m, name, patch_func(child))
        else:
            patch_module(child, filter_func, patch_func, stack)
        stack.pop()
