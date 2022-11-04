def get_save_time():
    import time
    return time.strftime('%Y_%m_%d_%H_%M_%S', time.localtime(time.time()))