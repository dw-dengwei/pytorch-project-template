

from http.client import PRECONDITION_FAILED


class Prints:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

    @staticmethod
    def _colored_print(color, msg, width=80, *args,  **kwargs):
        if 'process' in kwargs.keys():
            process = ''
            for item in kwargs['process']:
                try:
                    process += ' ' + item['Name'] + \
                               ' [' + str(item['Cur']).rjust(len(str(item['Tot'])), '0') + \
                               '/' + str(item['Tot']) + ']'
                except KeyError:
                    f = str(
                        [{
                            'Name': 'Name1',
                            'Cur': 1,
                            'Tot': 100,
                        }, {
                            'Name': 'Name2',
                            'Cur': 1,
                            'Tot': 100,
                        }]
                    )
                    Prints.FAIL('please follow the format: '
                                f'process = {f}')
                    raise KeyError
            process = process[1:]
            msg += ' ' * (width - len(msg) - len(process)) + process
            kwargs.pop('process')
        print(color + msg + Prints.ENDC, *args, **kwargs)

    @staticmethod
    def warning(msg, *args, **kwargs):
        Prints._colored_print(Prints.WARNING, msg, *args, **kwargs)

    @staticmethod
    def info(msg, *args, **kwargs):
        Prints._colored_print('', msg, *args, **kwargs)

    @staticmethod
    def ok(msg, *args, **kwargs):
        Prints._colored_print(Prints.OKGREEN, msg, *args, **kwargs)
    
    @staticmethod
    def train(msg, *args, **kwargs):
        Prints._colored_print(Prints.OKCYAN, msg, *args, **kwargs)

    @staticmethod
    def evaluate(msg, *args, **kwargs):
        Prints._colored_print(Prints.FAIL, msg, *args, **kwargs)

    @staticmethod
    def heading(msg, *args, **kwargs):
        Prints._colored_print(Prints.BOLD, msg, *args, **kwargs)