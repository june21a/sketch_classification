import pickle


# verbose 구현을 위한 print decorator
def printer_dec(verbose):
    def printer(*args):
        if verbose:
            for s in args:
                print(s, end=" ")
    return printer


# 경로에 pkl로 객체 저장
def dump_params_dict(file_path, params):
    with open(file_path, 'wb') as fw:
        pickle.dump(params, fw)


# 경로로부터 pkl 받아오기
def load_params_dict(file_path):
    with open(file_path, 'rb') as fr:
        loaded = pickle.load(fr)
    return loaded