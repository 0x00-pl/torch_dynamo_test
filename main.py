import os
from torch_dynamo_test.collect_op_on_modles import collect_op
from tests.test_model_pool import test_huggingface_model


def main():
    test_huggingface_model()


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    os.environ['https_proxy'] = 'http://127.0.0.1:7890'
    main()
