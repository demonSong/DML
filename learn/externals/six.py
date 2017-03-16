


# 这是个什么模式，比较神奇
def with_metaclass(meta, *bases):
    return meta("NewBase",bases,{})