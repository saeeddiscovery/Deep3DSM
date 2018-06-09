import os
def myPrint(text, path):
    if not os.path.exists(path+'/reports/'):
        os.mkdir(path+'/reports/')
    print(text)
    print(text, file=open(path+'/reports/output.txt', 'a'))
    
def visualizeDataset(dataset, plotSize=[4,4]):
    import matplotlib.pyplot as plt
    plt.figure()
    for num in range(len(dataset)):
        plt.subplot(plotSize[0],plotSize[1],num+1)
        centerSlice = int(dataset.shape[1]/2)
        plt.imshow(dataset[num, :, centerSlice, :, 0], cmap='gray')
        plt.axis('off')
    plt.suptitle('Center Coronal Slice\nfrom each training image')