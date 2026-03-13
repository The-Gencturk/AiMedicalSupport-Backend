import os
import sys
import traceback
import time
import shutil
import numpy as np
import threading
import tkinter as tk
from tkinter import messagebox, Label, Button, Checkbutton, BooleanVar, filedialog
from PIL import Image, ImageTk
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.optimizers import Adam

# TensorFlow loglarını ve OneDNN uyarılarını sustur
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# --- DÜZELTME: EXE İÇİN SAHTE KONSOL (NullWriter) ---
# EXE penceresiz çalışırken print komutları programı çökertmesin diye.
class NullWriter:
    def write(self, text): pass
    def flush(self): pass

if getattr(sys, 'frozen', False):
    sys.stdout = NullWriter()
    sys.stderr = NullWriter()

# --- DOSYA YOLU AYARLARI ---
if getattr(sys, 'frozen', False):
    BASE_DIR = os.path.dirname(sys.executable)
else:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, 'beyin_bt_modeli.h5')
WATCH_FOLDER = os.path.join(BASE_DIR, "Sectra_Export")
RETRAIN_FOLDER = os.path.join(BASE_DIR, "Hatali_Veriler")

if not os.path.exists(WATCH_FOLDER): os.makedirs(WATCH_FOLDER)
if not os.path.exists(RETRAIN_FOLDER): os.makedirs(RETRAIN_FOLDER)

# --- YARDIMCI FONKSİYONLAR ---
def find_last_conv_layer(model):
    for layer in reversed(model.layers):
        if 'conv' in layer.name.lower(): return layer.name
    for layer in reversed(model.layers):
        try:
            if hasattr(layer, 'output_shape'):
                shape = layer.output_shape
                if isinstance(shape, list): shape = shape[0]
                if shape and len(shape) == 4: return layer.name
            elif hasattr(layer, 'output'):
                shape = layer.output.shape
                if shape and len(shape) == 4: return layer.name
        except: continue
    return None

def get_gradcam_heatmap(model, img_array, last_conv_layer_name, pred_index=None):
    try:
        def veriyi_soy(veri):
            while isinstance(veri, (list, tuple)):
                if len(veri) > 0: veri = veri[0]
                else: return None
            return veri

        grad_model = Model(
            inputs=model.inputs,
            outputs=[model.get_layer(last_conv_layer_name).output, model.output]
        )

        with tf.GradientTape() as tape:
            ham_cikti = grad_model(img_array)
            if isinstance(ham_cikti, (list, tuple)) and len(ham_cikti) == 2:
                conv_outputs, preds = ham_cikti
            else:
                conv_outputs, preds = ham_cikti[0], ham_cikti[1]

            preds = veriyi_soy(preds)
            conv_outputs = veriyi_soy(conv_outputs)

            if preds is None or conv_outputs is None: return None
            if pred_index is None: pred_index = tf.argmax(preds[0])
            class_channel = preds[:, pred_index]

        grads = tape.gradient(class_channel, conv_outputs)
        grads = veriyi_soy(grads)
        if grads is None: return None

        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        if len(conv_outputs.shape) == 4: conv_outputs = conv_outputs[0]

        heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)
        heatmap = tf.maximum(heatmap, 0)
        max_val = tf.math.reduce_max(heatmap)
        
        if max_val == 0: return heatmap.numpy()
        return (heatmap / max_val).numpy()
    except: return None

def overlay_heatmap(heatmap, original_img, alpha=0.4):
    try:
        heatmap = cv2.resize(heatmap, (original_img.shape[1], original_img.shape[0]))
        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        if original_img.max() <= 1.0: original_img_uint8 = np.uint8(255 * original_img)
        else: original_img_uint8 = np.uint8(original_img)
        return cv2.addWeighted(heatmap, alpha, original_img_uint8, 1 - alpha, 0)
    except: return original_img

def read_image_safe(path):
    try:
        with open(path, "rb") as f:
            bytes_data = bytearray(f.read())
            np_arr = np.asarray(bytes_data, dtype=np.uint8)
            img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        return img
    except: return None

# --- EĞİTİM FONKSİYONU (DÜZELTİLDİ: verbose=0 ve Hata Loglama) ---
def train_on_single_image(model, img_path, actual_label):
    try:
        img = read_image_safe(img_path)
        if img is None: return False
        
        img = cv2.resize(img, (224, 224)) / 255.0
        img = np.reshape(img, (1, 224, 224, 3))
        label = np.array([[actual_label]])
        
        model.compile(optimizer=Adam(learning_rate=0.00001), 
                      loss='binary_crossentropy', 
                      metrics=['accuracy'])
        
        # verbose=0 ÇOK ÖNEMLİ (EXE'nin çökmemesi için)
        model.fit(img, label, epochs=1, verbose=0)
        
        # Modeli geçici dosyaya kaydet
        temp_path = os.path.join(BASE_DIR, "gecici_model.h5")
        if os.path.exists(temp_path): os.remove(temp_path)
        model.save(temp_path)
        
        # Asıl modelin üzerine yaz
        if os.path.exists(MODEL_PATH):
            try: os.remove(MODEL_PATH)
            except: pass # Silinemezse bile move overwrite edebilir
            
        shutil.move(temp_path, MODEL_PATH)
        return True
        
    except Exception as e:
        # Hata olursa log dosyasına yaz
        log_path = os.path.join(BASE_DIR, "HATA_LOGU.txt")
        with open(log_path, "w", encoding="utf-8") as f:
            f.write(f"Zaman: {time.ctime()}\n")
            f.write(f"Hata: {str(e)}\n")
            f.write(traceback.format_exc())
        return False

# --- ARAYÜZ SINIFI ---
class BeyinAsistaniApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Radyoloji Asistanı v7.5 (EXE FIX)")
        self.root.geometry("650x950")
        self.root.configure(bg="#1e272e")
        
        self.model = None
        self.last_conv_layer = None
        self.is_running = True
        self.show_heatmap = BooleanVar(value=True)
        self.current_image_path = None

        Label(root, text="AKTİF ÖĞRENME SİSTEMİ", font=("Segoe UI", 20, "bold"), bg="#1e272e", fg="#e84393").pack(pady=10)
        self.status = Label(root, text="Sistem Başlatılıyor...", bg="#485460", fg="white", width=70, height=2)
        self.status.pack(pady=5)
        
        frame_top = tk.Frame(root, bg="#1e272e")
        frame_top.pack(pady=5)
        Button(frame_top, text="📂 KLASÖRÜ AÇ", command=self.open_folder_dialog, bg="#0984e3", fg="white").pack(side=tk.LEFT, padx=5)
        Checkbutton(frame_top, text="Isı Haritası", variable=self.show_heatmap, bg="#1e272e", fg="#00d2d3", selectcolor="#1e272e").pack(side=tk.LEFT, padx=5)

        self.img_lbl = Label(root, bg="black", width=400, height=400)
        self.img_lbl.pack(pady=10)
        self.res_lbl = Label(root, text="Resim Bekleniyor...", font=("Arial", 18, "bold"), bg="#1e272e", fg="white")
        self.res_lbl.pack(pady=10)
        
        # Eğitim Butonları
        Label(root, text="Model yanıldıysa DOĞRUSUNU öğretin:", bg="#1e272e", fg="#ff9f43").pack()
        btn_frm = tk.Frame(root, bg="#1e272e")
        btn_frm.pack(pady=10)
        Button(btn_frm, text="Aslında NORMAL", command=lambda: self.teach_model(0), bg="#1dd1a1", fg="black", width=15).pack(side=tk.LEFT, padx=10)
        Button(btn_frm, text="Aslında KANAMA", command=lambda: self.teach_model(1), bg="#ff6b6b", fg="white", width=15).pack(side=tk.LEFT, padx=10)

        # Otomatik Başlatma
        self.root.after(1000, self.load_ai_model)

    def load_ai_model(self):
        if not os.path.exists(MODEL_PATH):
            self.status.config(text="Model Dosyası Bulunamadı!", bg="red")
            return
        
        self.status.config(text="Model Yükleniyor... Lütfen Bekleyiniz")
        self.root.update()
        
        try:
            self.model = load_model(MODEL_PATH)
            self.last_conv_layer = find_last_conv_layer(self.model)
            msg = "SİSTEM HAZIR"
            self.status.config(text=msg, bg="#20bf6b")
            # Arka plan izlemeyi başlat
            threading.Thread(target=self.watch_folder, daemon=True).start()
        except Exception as e:
            self.status.config(text=f"Model Yükleme Hatası: {str(e)}", bg="red")

    def open_folder_dialog(self):
        try:
            path = filedialog.askopenfilename(title="Görüntü Seç", filetypes=[("Resimler", "*.jpg *.png *.jpeg *.bmp")])
            if path:
                self.predict_image(path)
        except Exception as e:
            messagebox.showerror("Hata", str(e))

    def predict_image(self, img_path):
        try:
            self.current_image_path = img_path
            orig = read_image_safe(img_path)
            if orig is None: return
            
            img = cv2.resize(orig, (224, 224))
            img_norm = img / 255.0
            img_in = np.reshape(img_norm, (1, 224, 224, 3))
            
            pred = self.model.predict(img_in, verbose=0)[0][0]
            
            disp_img = orig.copy()
            if self.show_heatmap.get() and self.last_conv_layer:
                hm = get_gradcam_heatmap(self.model, img_in, self.last_conv_layer)
                if hm is not None:
                    disp_img = overlay_heatmap(hm, img_norm)
                    disp_img = np.uint8(disp_img)
            
            disp_img = cv2.resize(disp_img, (400, 400))
            disp_img = cv2.cvtColor(disp_img, cv2.COLOR_BGR2RGB)
            im_pil = Image.fromarray(disp_img)
            imgtk = ImageTk.PhotoImage(image=im_pil)
            self.img_lbl.imgtk = imgtk
            self.img_lbl.configure(image=imgtk)

            if pred > 0.5: 
                self.res_lbl.config(text=f"🚨 KANAMA (%{int(pred*100)})", fg="#ff6b6b")
            else:
                self.res_lbl.config(text=f"✅ NORMAL (%{int((1-pred)*100)})", fg="#1dd1a1")
            
            self.status.config(text=f"İncelendi: {os.path.basename(img_path)}")
        except Exception as e:
            print(e)

    def teach_model(self, true_label):
        if not self.current_image_path: return
        answ = messagebox.askyesno("Eğitim", "Bu görüntüyü modele öğretmek istiyor musunuz?")
        if not answ: return
        
        self.status.config(text="🧠 EĞİTİLİYOR... Lütfen Bekleyiniz...", bg="#e1b12c")
        self.root.update()
        
        success = train_on_single_image(self.model, self.current_image_path, true_label)
        
        if success:
            self.status.config(text="✅ EĞİTİM BAŞARILI! Model Güncellendi.", bg="#20bf6b")
            self.root.update()
            time.sleep(1)
            # Eğitilen veriyi yedekle
            tgt = os.path.join(RETRAIN_FOLDER, f"EGITILDI_{true_label}_" + os.path.basename(self.current_image_path))
            try: shutil.copy(self.current_image_path, tgt)
            except: pass
            # Tekrar tahmin et (sonucu görmek için)
            self.predict_image(self.current_image_path)
        else:
            self.status.config(text="❌ Eğitim Hatası! Log dosyasına bakın.", bg="red")
            messagebox.showerror("Hata", "Eğitim sırasında hata oluştu. Programın yanındaki 'HATA_LOGU.txt' dosyasını kontrol edin.")

    def watch_folder(self):
        processed = set()
        while self.is_running:
            try:
                if not os.path.exists(WATCH_FOLDER): continue
                files = set(os.listdir(WATCH_FOLDER))
                new_files = files - processed
                for f in new_files:
                    if f.lower().endswith(('.jpg', '.png', '.jpeg')):
                        time.sleep(1)
                        full_path = os.path.join(WATCH_FOLDER, f)
                        self.root.after(0, lambda p=full_path: self.predict_image(p))
                        processed.add(f)
                time.sleep(2)
            except: pass

if __name__ == "__main__":
    root = tk.Tk()
    app = BeyinAsistaniApp(root)
    root.mainloop()