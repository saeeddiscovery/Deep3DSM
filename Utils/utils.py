import os
def myPrint(text, path):
    if not os.path.exists(path+'/reports/'):
        os.mkdir(path+'/reports/')
    print(text)
    print(text, file=open(path+'/reports/output.txt', 'a'))