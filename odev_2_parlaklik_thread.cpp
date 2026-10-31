// Ödev - Soru 2: Görüntüyü 4 parçaya bölüp 4 thread ile farklı parlaklık değerlerine ayarlama
//                ve tek thread ile 4 thread çalışma süresini karşılaştırma.
//
// Görüntü 4 yatay banda (satır grubuna) bölünür. Her banda farklı bir parlaklık farkı eklenir:
//   1. parça: -60   2. parça: -20   3. parça: +20   4. parça: +60
// Sonuç 0..255 aralığına sınırlanır.
//
// Kullanım: ./parlaklik_thread <goruntu_dosyasi> [cikti_dizini] [tekrar_sayisi]
//   tekrar_sayisi: süre ölçümü için işlemin kaç kez tekrarlanacağı (varsayılan 200)
//
// Derleme (süre ölçümü için -O2 şart):
//   g++ -O2 odev_2_parlaklik_thread.cpp -o parlaklik_thread.exe -std=c++17 -pthread \
//       $(pkg-config --cflags --libs opencv5)

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>

#include <array>
#include <chrono>
#include <filesystem>
#include <iostream>
#include <string>
#include <thread>
#include <vector>

namespace fs = std::filesystem;

static const int NUM_PARTS = 4;

// src'nin [rowStart, rowEnd) satırlarına delta ekleyip dst'ye yazar (0..255'e sınırlar).
static void adjustRows(const cv::Mat& src, cv::Mat& dst, int rowStart, int rowEnd, int delta) {
    for (int y = rowStart; y < rowEnd; ++y) {
        const uchar* s = src.ptr<uchar>(y);
        uchar* d = dst.ptr<uchar>(y);
        for (int x = 0; x < src.cols; ++x) {
            int v = s[x] + delta;
            d[x] = static_cast<uchar>(v < 0 ? 0 : (v > 255 ? 255 : v));
        }
    }
}

// Tek thread: 4 parçayı sırayla ana thread işler.
static void runSingleThread(const cv::Mat& src, cv::Mat& dst,
                            const std::array<int, NUM_PARTS + 1>& bounds,
                            const std::array<int, NUM_PARTS>& deltas) {
    for (int i = 0; i < NUM_PARTS; ++i)
        adjustRows(src, dst, bounds[i], bounds[i + 1], deltas[i]);
}

// 4 thread: her parça ayrı bir thread'de işlenir.
static void runMultiThread(const cv::Mat& src, cv::Mat& dst,
                           const std::array<int, NUM_PARTS + 1>& bounds,
                           const std::array<int, NUM_PARTS>& deltas) {
    std::vector<std::thread> workers;
    workers.reserve(NUM_PARTS);
    for (int i = 0; i < NUM_PARTS; ++i) {
        workers.emplace_back(adjustRows, std::cref(src), std::ref(dst),
                             bounds[i], bounds[i + 1], deltas[i]);
    }
    for (auto& t : workers) t.join();
}

// fn'yi repeat kez çalıştırır, tek çalıştırmanın ortalama süresini milisaniye olarak döndürür.
template <typename Fn>
static double averageMs(Fn fn, int repeat) {
    fn();  // ısınma turu (ilk çalıştırma ölçüme katılmaz)
    auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < repeat; ++i) fn();
    auto t1 = std::chrono::steady_clock::now();
    return std::chrono::duration<double, std::milli>(t1 - t0).count() / repeat;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Kullanim: " << argv[0] << " <goruntu_dosyasi> [cikti_dizini] [tekrar_sayisi]\n";
        return 1;
    }

    const std::string inPath = argv[1];
    const fs::path outDir = (argc >= 3) ? fs::path(argv[2]) : fs::path("cikti_parlaklik");
    const int repeat = (argc >= 4) ? std::max(1, std::stoi(argv[3])) : 200;
    fs::create_directories(outDir);

    cv::Mat src = cv::imread(inPath, cv::IMREAD_GRAYSCALE);
    if (src.empty()) {
        std::cerr << "Hata: goruntu okunamadi: " << inPath << "\n";
        return 1;
    }

    // 4 yatay bant: satır sınırları
    const int h = src.rows;
    const std::array<int, NUM_PARTS + 1> bounds = {0, h / 4, h / 2, (3 * h) / 4, h};
    const std::array<int, NUM_PARTS> deltas = {-60, -20, 20, 60};

    cv::Mat dstSingle(src.size(), CV_8UC1);
    cv::Mat dstMulti(src.size(), CV_8UC1);

    // Önce her iki yöntemin sonucunu bir kez üret ve aynı olduklarını doğrula
    runSingleThread(src, dstSingle, bounds, deltas);
    runMultiThread(src, dstMulti, bounds, deltas);
    const bool same = cv::norm(dstSingle, dstMulti, cv::NORM_INF) == 0;

    // Süre ölçümü
    const double msSingle = averageMs([&] { runSingleThread(src, dstSingle, bounds, deltas); }, repeat);
    const double msMulti = averageMs([&] { runMultiThread(src, dstMulti, bounds, deltas); }, repeat);

    std::cout << "Goruntu boyutu     : " << src.cols << "x" << src.rows << " (gri seviye)\n";
    std::cout << "Parlaklik farklari : " << deltas[0] << ", " << deltas[1] << ", "
              << deltas[2] << ", " << deltas[3] << " (4 parca icin)\n";
    std::cout << "Islemci thread sayisi (bilgisayar): " << std::thread::hardware_concurrency() << "\n";
    std::cout << "Tekrar sayisi      : " << repeat << " (sureler ortalamadir)\n\n";
    std::cout << "Sonuclar ayni mi   : " << (same ? "EVET" : "HAYIR") << "\n";
    std::cout << "1 thread  ortalama : " << msSingle << " ms\n";
    std::cout << "4 thread  ortalama : " << msMulti << " ms\n";
    if (msMulti > 0) {
        std::cout << "Hizlanma (1 thread / 4 thread): " << (msSingle / msMulti) << "x\n";
    }

    cv::imwrite((outDir / "1_gri_orijinal.png").string(), src);
    cv::imwrite((outDir / "2_parlaklik_1thread.png").string(), dstSingle);
    cv::imwrite((outDir / "3_parlaklik_4thread.png").string(), dstMulti);
    std::cout << "\nKaydedilenler (" << outDir.string() << "): orijinal, 1 thread sonucu, 4 thread sonucu\n";
    return 0;
}
