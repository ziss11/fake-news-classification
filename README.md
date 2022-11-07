# Submission 1: Fake News Classification
Nama: Abdul Azis

Username dicoding: zizz1181

| | Deskripsi |
| ----------- | ----------- |
| Dataset | [Fake and real news dataset](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset) |
| Masalah | Zaman sekarang sering sekali berita yang disebarluaskan bukan merupakan berita yang benar-benar terjadi direalitanya, hal ini menimbulkan opini negatif dan masyarakat yang sering menerima berita palsu akan bingung dalam membedakan berita faktual dan berita palsu (hoax) |
| Solusi machine learning | Maka dari itu dibutuhkan sebuah sistem machine learning yang dapat mendeteksi kepalsuan dari sebuah berita |
| Metode pengolahan | Metode pengolahan data yang digunakan pada proyek ini berupa tokenisasi fitur input (text dari sebuah berita) yang awalnya berupa text diubah menjadi susunan angka yang merepresentasikan text tersebut agar dapat dengan mudah dimengerti oleh model |
| Arsitektur model | Model yang dibangun menggunakan layer TextVectorization sebagai layer yang akan memproses input string kedalam bentuk susunan angka, kemudian layer Embedding yang bertugas untuk mempelajari kedekatan atau kemiripan dari sebuah kata yang berguna untuk mengetahui apakah kata tersebut merupakan kata negatif atau kata positif. Lalu terdapat 2 hidden layer dan 1 output layer. |
| Metrik evaluasi | Metric yang digunakan pada model yaitu BinaryAccuracy, TruePositive, FalsePositive, TrueNegative, FalseNegative untuk mengevaluasi performa model dalam menentukan klasifikasi|
| Performa model | Model yang dibuat menghasilkan performa yang cukup baik dalam memberikan prediksi untuk text berita yang diinputkan, dan dari pelatihan yang dilakukan model menghasilkan binary_accuracy dan val_binary_accuracy di sekitar 98% |