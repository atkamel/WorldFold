# headless stub: the real mjviser is only used by the interactive viewer
def __getattr__(name):
    raise RuntimeError("mjviser stub: viewer not available headless (%s)" % name)
