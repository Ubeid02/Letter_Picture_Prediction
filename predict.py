import numpy as np
import os
import matplotlib.pyplot as plt
from tkinter import *
from tkinter import filedialog
from PIL import Image, ImageTk

# Print option untuk print full array
np.set_printoptions(threshold=np.inf)

# Lokasi data train
path = ('E:/perkuliahan/semester 5/Kendali cerdas/deep learning/image clasification/data/train/sampel/')
listGambar = os.listdir(path)

# Fungsi untuk mengubah gambar ke grayscale
def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])

# Mengubah nilai gambar menjadi array gambar
gambarBin = []

for img_name in listGambar:
    img_path = os.path.join(path, img_name)
    img = plt.imread(img_path)
    img_gray = rgb2gray(img)
    img_bin = np.where(img_gray > 0, 1, 0)
    gambarBin.append(img_bin)

gambarBin = np.array(gambarBin)

# Mengubah menjadi flat/satu array (64)
gambarFlat = np.array([x.flatten() for x in gambarBin])
x = gambarFlat

# Y Training
namaGambar = [n_gbr[0] for n_gbr in listGambar]
namaGambar = np.array(namaGambar)

# Membuat kamus untuk mapping huruf ke indeks numerik
kelas_unik = np.unique(namaGambar)
kelas_to_index = {kelas: idx for idx, kelas in enumerate(kelas_unik)}
index_to_kelas = {idx: kelas for kelas, idx in kelas_to_index.items()}

# Mengonversi nama kelas menjadi indeks numerik
labels_numerik = np.array([kelas_to_index[kelas] for kelas in namaGambar])

# Mengonversi indeks numerik menjadi one-hot encoding
num_kelas = len(kelas_unik)
y = np.eye(num_kelas)[labels_numerik]

# Konfigurasi nilai X dan Y
inputx = x
target = y

# Konfigurasi jumlah node input layer, hidden layer, dan output layer
nodeInput = inputx.shape[1]  # Seharusnya 64 untuk gambar 8x8
nodeHidden = 100  # Anda bisa menyesuaikan ini
nodeOutput = target.shape[1]

# Fungsi aktivasi sigmoid
def sigmoid(x):
    return 1/(1 + np.exp(-x))

# Inisialisasi pembobot awal
w1 = np.random.uniform(low=-1, high=1, size=(nodeInput, nodeHidden))
w2 = np.random.uniform(low=-1, high=1, size=(nodeHidden, nodeOutput))

# Konfigurasi learning rate dan epoch
lr = 0.5
epoch = 1000  # Anda bisa menyesuaikan ini

# Fungsi untuk melatih model
def train_model():
    global w1, w2
    for _ in range(epoch):
        for idx, inp in enumerate(inputx):
            # Proses maju
            h = sigmoid(np.matmul(inputx[idx], w1))
            out = sigmoid(np.matmul(h, w2))

            # Proses mundur
            e = target[idx] - out
            eh = np.matmul(e, w2.T)

            # Update nilai pembobot w1 dan w2
            w2 += lr * ((e * out * (1 - out)) * h[np.newaxis].T)
            w1 += lr * ((eh * h * (1 - h)) * inputx[idx][np.newaxis].T)

# Melatih model
train_model()

# Fungsi untuk melakukan prediksi
def predict(image):
    img_gray = rgb2gray(image)
    img_bin = np.where(img_gray > 0, 1, 0)
    img_flat = img_bin.flatten()

    h = sigmoid(np.matmul(img_flat, w1))
    out = sigmoid(np.matmul(h, w2))

    predicted_class = np.argmax(out)
    confidence = np.max(out)

    return index_to_kelas[predicted_class], confidence

# Fungsi untuk memilih gambar
def choose_image():
    file_path = filedialog.askopenfilename()
    if file_path:
        image = Image.open(file_path).convert('RGB')
        image = image.resize((8, 8))  # Resize ke 8x8
        photo = ImageTk.PhotoImage(image.resize((200, 200)))  # Resize untuk tampilan
        image_label.config(image=photo)
        image_label.image = photo

        # Melakukan prediksi
        img_array = np.array(image)
        predicted_letter, confidence = predict(img_array)

        result_label.config(text=f"Prediksi: Huruf {predicted_letter}")
        accuracy_label.config(text=f"Akurasi: {confidence:.2%}")

# Membuat GUI
root = Tk()
root.title("Klasifikasi Huruf")

# Mendapatkan ukuran layar
screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()

# Menentukan ukuran jendela (setengah dari ukuran layar)
window_width = screen_width // 2
window_height = screen_height // 2

# Menentukan posisi jendela (di tengah layar)
position_top = int(screen_height/2 - window_height/2)
position_right = int(screen_width/2 - window_width/2)

# Mengatur geometri jendela
root.geometry(f"{window_width}x{window_height}+{position_right}+{position_top}")

choose_button = Button(root, text="Pilih Gambar", command=choose_image)
choose_button.pack(pady=20)

image_label = Label(root)
image_label.pack()

result_label = Label(root, text="Prediksi: ")
result_label.pack(pady=10)

accuracy_label = Label(root, text="Akurasi: ")
accuracy_label.pack()

root.mainloop()