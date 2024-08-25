from colorama import Fore, Style

def printLog(log):
    print(f'{Fore.GREEN}{log}{Style.RESET_ALL}')

def printError(error):
    print(f'{Fore.RED}{error}{Style.RESET_ALL}')

def printInfo(info):
    print(f'{Fore.BLUE}{info}{Style.RESET_ALL}')

def printHeader(currency):
    printLog('\n=============================================================')
    printLog(f'||                          {currency}                          ||')
    printLog('=============================================================')