# 1. Core Training System

* **datasets** = kode untuk membaca dataset dan menyiapkannya menjadi tensor yang bisa diproses model.

  * `dataset_loader.py` → mendefinisikan class Dataset yang membaca file **T1 image, T2 image, dan mask**, lalu mengubahnya menjadi tensor PyTorch dan mengembalikannya dalam format `(t1, t2, mask)` untuk setiap sampel.

  * `transform.py` → berisi operasi preprocessing dan augmentation dataset seperti **normalization, random flip, rotation, cropping**, dan transform lain agar model lebih robust terhadap variasi data.

---

* **models** = mendefinisikan arsitektur model yang akan dilatih.

  * `swin_earlyfusion.py` → mendefinisikan model berbasis **Swin Transformer** yang menerima input **T1 dan T2**, melakukan **early fusion (concatenate channel)**, memprosesnya melalui **encoder transformer**, lalu menghasilkan **peta perubahan (change map)** melalui decoder.

---

* **loss** = mendefinisikan fungsi loss yang digunakan untuk menghitung kesalahan prediksi model.

  * `bce_dice_loss.py` → menggabungkan dua loss yaitu

    * **Binary Cross Entropy (BCE)** → mengukur perbedaan probabilitas prediksi dengan label ground truth
    * **Dice Loss** → mengukur overlap antara prediksi dan mask ground truth, biasanya digunakan pada segmentation agar model lebih sensitif terhadap bentuk objek.

---

* **train.py** = orchestrator pipeline yang mengatur seluruh proses training.

  * `DataLoader` → mengambil dataset dari `dataset_loader.py`, membuat batch data, dan mengirim batch tersebut ke GPU.

  * `model` → menerima input `(t1, t2)`, melakukan **forward pass** melalui encoder dan decoder untuk menghasilkan prediksi perubahan.

  * `loss` → menghitung error antara **prediksi model** dan **ground truth mask**.

  * `optimizer` → memperbarui parameter model berdasarkan gradient yang dihitung dari loss.

  * `training loop` → menjalankan proses training berulang selama beberapa **epoch** sampai model konvergen.

---

# 2. Evaluation System

* **evaluation** = sistem untuk mengevaluasi performa model setelah training.

  * `metrics.py` → berisi fungsi untuk menghitung metrik evaluasi seperti

    * **IoU (Intersection over Union)**
    * **F1 Score**
    * **Precision dan Recall**
    * **Overall Accuracy**
    * **Kappa coefficient**

  * `evaluate.py` → menjalankan proses evaluasi dengan cara **memuat model yang telah dilatih**, menjalankan inference pada dataset validation/test, lalu menghitung metrik menggunakan fungsi dari `metrics.py`.

---

# 3. Experiment Management

* **configs** = tempat menyimpan konfigurasi eksperimen agar parameter training tidak hardcoded di kode.

  * `train_config.py` → berisi parameter penting seperti

    * learning rate
    * batch size
    * jumlah epoch
    * jumlah channel input
    * path dataset
    * optimizer yang digunakan

  Tujuannya agar eksperimen bisa diubah dengan mudah tanpa mengubah kode utama.

---

# 4. Utilities System

* **utils** = kumpulan fungsi pendukung yang digunakan oleh berbagai bagian pipeline.

  * `seed.py` → mengatur random seed untuk memastikan eksperimen dapat direproduksi dengan hasil yang konsisten.
  * `logger.py` → mencatat proses training seperti loss dan metrik ke dalam file log atau console.
  * `visualize.py` → menampilkan hasil prediksi model seperti **T1 image, T2 image, prediction map, dan ground truth mask** untuk analisis visual.
---

# 5. Logging System

* **logs** = folder untuk menyimpan log eksperimen.

  Log biasanya berisi informasi seperti:

  * training loss per epoch
  * validation loss
  * metrik evaluasi
  * waktu training

  Log ini dapat digunakan untuk membuat **training curve** atau membandingkan eksperimen.

---

# 6. Checkpoint System

* **checkpoints** = tempat menyimpan model yang telah dilatih.

  Contoh file:

```
best_model.pth
last_model.pth
```

Fungsinya:

* menyimpan model terbaik selama training
* memungkinkan training dilanjutkan jika proses training terhenti.

---

# 7. Inference System

* **inference** = sistem untuk menggunakan model yang telah dilatih pada data baru.

  * `predict.py` → memuat model yang telah dilatih (`best_model.pth`), menerima input **T1 dan T2 image**, lalu menghasilkan **peta perubahan (change detection map)** yang bisa disimpan atau divisualisasikan.

# Gambaran Besar Pipeline Sistem

Jika semua komponen digabung, pipeline penelitian secara keseluruhan menjadi:

```
Dataset Preprocessing
        ↓
datasets (dataset_loader)
        ↓
DataLoader
        ↓
Model 
        ↓
Loss Function
        ↓
Backpropagation
        ↓
Optimizer Update
        ↓
Training Loop
        ↓
Checkpoint Saving
        ↓
Evaluation Metrics
        ↓
Inference / Visualization

| Model                | Forward   |
| -------------------- | --------- |
| UNetEarlyFusion      | `(t1,t2)` |
| SwinEarlyFusionFPN   | `(t1,t2)` |
| SegFormerEarlyFusion | `(t1,t2)` |
| BIT                  | `(t1,t2)` |
| STANet               | `(t1,t2)` |
| GSWIN_TAC            | `(t1,t2)` |

