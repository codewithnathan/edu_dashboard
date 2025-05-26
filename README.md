# Proyek Akhir: Menyelesaikan Permasalahan Institusi Pendidikan

## Business Understanding

Sejak didirikan pada tahun 2000, Jaya Jaya Institut telah menjadi salah satu institusi pendidikan tinggi yang menghasilkan banyak lulusan berkualitas. Meski demikian, data menunjukkan adanya sejumlah siswa yang belum menyelesaikan pendidikannya atau mengalami putus studi (dropout) yang dapat menjadi masalah besar pada institusi tersebut.

### Permasalahan Bisnis

Jumlah dropout yang tinggi menyebabkan:

- Rendahnya tingkat kelulusan (graduation rate)
- Evaluasi akreditasi yang menurun
- Dampak reputasi bagi institusi tersebut

### Cakupan Proyek

- **Import & Eksplorasi Data**: Memahami struktur dan pola dari dataset `data.csv`
- **Data Preprocessing**: Pengecekan karakteristik data dan penggabungan fitur
- **Modeling dengan KMeans**: Segmentasi student ke dalam beberapa klaster
- **Evaluasi & Visualisasi**: Analisis hasil clustering melalui visualisasi grafis dan tabel
- **Insight & Rekomendasi**: Memberikan saran berbasis data dan dashboard kepada pihak institusi

### Persiapan

Sumber data: `data.csv`

#### Setup Environment

```
cd edu
python3 -m venv myenv
source myenv/bin/activate
pip3 install -r requirements.txt
```

#### Run steamlit app

```
streamlit run app.py
```

#### Metabase Credentials

```
Email address: root@mail.com
Password: root123
```

## Business Dashboard

Dashboard analitik dibangun menggunakan **Metabase**, berisi 7 visualisasi utama:

1. **Dropout Rate per Student Segment**: Mengidentifikasi segmen mahasiswa dengan tingkat dropout tertinggi.
2. **Dropout Rate by Average Grade**: Mengetahui hubungan antara performa akademik (rata-rata nilai) dengan kemungkinan dropout.
3. **Dropout Rate by Financial Stress**: Menganalisis pengaruh tekanan keuangan terhadap risiko mahasiswa berhenti kuliah.
4. **Dropout Rate by First Choice Admission**: Melihat apakah mahasiswa yang tidak masuk pilihan pertama lebih cenderung dropout.
5. **Dropout Rate by Parents’ Education Level:** Menilai apakah latar belakang pendidikan orang tua memengaruhi dropout mahasiswa.
6. **Overall Dropout Rate**: Menampilkan total persentase mahasiswa yang dropout sebagai indikator kinerja utama
7. **Comparison of Averages Between Active vs Dropout Students**: Membandingkan rata-rata nilai, approval rate, stres keuangan, dan latar belakang pendidikan orang tua berdasarkan status mahasiswa (dropout vs aktif).

## Menjalankan Sistem Machine Learning

1. Masuk ke dalam folder utama
   ```
   cd edu
   ```
2. Setup python
   ```
   python3 -m venv myenv
   source myenv/bin/activate
   pip3 install -r requirements.txt
   ```
3. Jalankan streamlit
   ```
   streamlit run app.py
   ```
   atau akses url publik dashboard streamlit di https://studentdashboard-74ysaeveiafjfnlnwrbaoq.streamlit.app

## Conclusion

Berdasarkan analisis dan clustering yang dilakukan, ditemukan beberapa temuan utama:

- **Segmen Tertentu Rentan Dropout**  
  Mahasiswa dari segmen tertentu (misalnya dengan kombinasi akademik rendah dan stres tinggi) menunjukkan tingkat dropout yang lebih tinggi dibandingkan segmen lainnya. Hal ini menekankan pentingnya pendekatan berbasis segmentasi risiko.
- **Nilai Rata-Rata (Avg Grade) Berkorelasi dengan Dropout**  
  Mahasiswa dengan nilai rata-rata rendah lebih sering mengalami dropout. Ini menunjukkan bahwa performa akademik merupakan indikator penting untuk deteksi dini risiko putus studi.
- **Stres Finansial Memiliki Dampak Signifikan**  
  Mahasiswa dengan tekanan keuangan tinggi (financial stress) memiliki kecenderungan lebih besar untuk dropout. Hal ini menunjukkan perlunya kampus menyediakan dukungan keuangan atau beasiswa tambahan.
- **Bukan Pilihan Pertama Berpotensi Menyebabkan Dropout**  
  Mahasiswa yang tidak diterima di jurusan pilihan pertama memiliki tingkat dropout yang lebih tinggi, kemungkinan karena kurangnya minat atau motivasi terhadap jurusan yang mereka ambil.
- **Latar Belakang Pendidikan Orang Tua Berpengaruh**  
  Mahasiswa dengan orang tua berpendidikan rendah cenderung lebih rentan terhadap dropout, kemungkinan karena kurangnya dukungan akademik dari lingkungan keluarga.
- **Dropout Rate Total Perlu Diwaspadai**  
  Jika total dropout rate melebihi 10–20%, maka angka ini tergolong tinggi dan perlu menjadi perhatian institusi pendidikan untuk mengambil langkah-langkah perbaikan.
- **Perbedaan Signifikan Antara Mahasiswa Aktif dan Dropout**  
  Mahasiswa yang dropout memiliki rata-rata nilai lebih rendah, tingkat approval lebih kecil, tingkat stres keuangan lebih tinggi, serta orang tua dengan tingkat pendidikan yang lebih rendah. Ini menunjukkan bahwa faktor penyebab dropout bersifat multifaktor.

### Rekomendasi Action Items

- Membangun sistem peringatan dini (early warning system) dengan indikator nilai akademik, stres keuangan, dan pilihan jurusan.
- Menyediakan program dukungan finansial dan psikologis yang lebih menyeluruh.
- Meningkatkan layanan pendampingan akademik dan motivasional, terutama bagi mahasiswa dari segmen yang berisiko tinggi.
