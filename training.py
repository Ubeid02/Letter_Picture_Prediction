import numpy as np
import os
import matplotlib.pyplot as plt

#print option untuk print full array
np.set_printoptions(threshold=np.inf)

#lokasi data test
path = ('E:/perkuliahan/semester 5/Kendali cerdas/deep learning/image clasification/data/train/sampel/')
listGambar = os.listdir(path)
#print("Jumlah Gambar Training : ", len(listGambar))

#X Training
#Membaca dan mengubah gambar test ke grey
def rgb2gray(rgb):
    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    return gray

#Mengubah nilai gambar menjadi array gambar, jika nilai array  > 1 = 1 dan jika nilai array <= 0 = 0, untuk efisiensi komputasi
gambarBin = []

for img_name in listGambar:
    img_path = os.path.join(path, img_name)
    img = plt.imread(img_path)  # Read the image using matplotlib.pyplot.imread
    img_gray = rgb2gray(img)  # Convert the image to grayscale
    img_bin = np.where(img_gray > 0, 1, 0)  # Convert to binary values

    gambarBin.append(img_bin)

gambarBin = np.array(gambarBin)

#mengubah menjadi flat/satu array (64)
gambarFlat = np.array([x.flatten() for x in gambarBin])
x = gambarFlat
#print(gambarFlat[0])

#Y Training
#Membaca dan mengubah huruf gambar dari nama gambar sebagai target
namaGambar = list()
for n_gbr in listGambar :
    nama = n_gbr[0]
    namaGambar.append(nama)
namaGambar = np.array(namaGambar)
#print(namaGambar)

# Membuat kamus untuk mapping huruf ke indeks numerik
kelas_unik = np.unique(namaGambar)
kelas_to_index = {kelas: idx for idx, kelas in enumerate(kelas_unik)}

# Mengonversi nama kelas menjadi indeks numerik
labels_numerik = np.array([kelas_to_index[kelas] for kelas in namaGambar])

# Mengonversi indeks numerik menjadi one-hot encoding
num_kelas = len(kelas_unik)
y = np.eye(num_kelas)[labels_numerik]
#print(y[10])

#konfigurasi nilai X dan Y
inputx = x
target = y
#print("Jumlah input X : ", inputx.shape)
#print("Jumlah target Y : ", target.shape)

#konfigurasi jumlah node input layer, hidden layer, dan output laye
nodeInput = inputx.shape[1]
print("Jumlah node input : ", nodeInput)
nodeHidden = int(input("Masukkan node Hidden Layer yang diinginkan : "))
nodeOutput = target.shape[1]
print("Jumlah node Output : ", nodeOutput)

#fungsi aktivasi sigmoid
def sigmoid(x):
    return 1/(1 + np.exp(-x))

#inisialisasi pembobot awal
w1 = np.random.uniform(low = -1, high = 1, size = (nodeInput, nodeHidden))
w2 = np.random.uniform(low = -1, high = 1, size = (nodeHidden, nodeOutput))
#print("Jumlah nilai pembobot awal W1 : ", w1)
#print("Jumlah nilai pembobot awal W2 : ", w2)

#Menampung nilai MSE dan akurasi
errValue = []
accValue = []

#konfigurasi learning rate dan epoch
lr = 0.5
print("Learning Rate : ", lr)
epoch = int(input("Masukkan perulangan yang diinginkan : "))

#perulangan epoch
for b in range (epoch):
    mse = 0
    newTarget = np.zeros(target.shape)

    #perulangan maju dan mundur tiap index input x
    for idx, inp in enumerate(inputx):
        #proses maju
        #hitung output hidden tiap index input x - fungsi aktifasi sigmoid
        h = np.matmul(inputx[idx], w1)
        h = sigmoid(h)

        #hitung output y tiap input x - fungsi aktifasi sigmoid
        out = np.matmul(h, w2)
        out = sigmoid(out)

        #hitung eror output tiap index inpu x
        e = target[idx]-out

        #hitung mse tiap index input x
        mse = mse+(np.sum((e**2))/e.shape)

        #prediksi target baru
        newTarget[idx] = out.round()

        #proses mundur
        #hitung eror hidden tiap index x
        eh = np.matmul(e, w2.T)

        #update nilai pembobot w1 dan w2
        w2 = w2+(lr*((e*out*(1-out))*h[np.newaxis].T))
        w1 = w1+(lr*((eh*h*(1-h))*inputx[idx][np.newaxis].T))

    #hitung rata-rata mse dari seluruh index
    mse = (mse/inputx.shape[0])

    #hitung akurasi keberhasilan dari output
    d = np.absolute(target-newTarget)
    acc = 1-np.average([np.max(m) for m in d])

    #menyimpan nilai mse dan akurasi ke list
    accValue.append(acc)
    errValue.append(mse)

    #tampilkan hasil perhitungan
    print("Epoch : ", b, "/", epoch, "|| MSE : ", mse, "Accuracy : ", acc)

#menampilkan grafik MSE
grafik1 = plt.figure(1)
plt.title("Grafik tingkat error")
plt.xlabel("iterasi")
plt.ylabel("error")
plt.plot(errValue)
plt.show()

