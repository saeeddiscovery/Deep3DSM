import os
def myPrint(text, path):
    if not os.path.exists(path+'/reports/'):
        os.mkdir(path+'/reports/')
    print(text)
    print(text, file=open(path+'/reports/output.txt', 'a'))
    
def visualizeDataset(dataset, plotSize=[4,4]):
    import matplotlib.pyplot as plt
<<<<<<< HEAD
    plt.figure()
=======
>>>>>>> 7042cc93aa7242c35e17cb21a164ee5f5c3a4ea0
    for num in range(len(dataset)):
        plt.subplot(plotSize[0],plotSize[1],num+1)
        centerSlice = int(dataset.shape[1]/2)
        plt.imshow(dataset[num, :, centerSlice, :, 0], cmap='gray')
        plt.axis('off')
    plt.suptitle('Center Coronal Slice\nfrom each training image')