import os

from tests.test_dynamo_extract_node import test_dynamo_extract_node


def main():
    test_dynamo_extract_node()


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    os.environ['https_proxy'] = 'http://127.0.0.1:7890'
    main()
