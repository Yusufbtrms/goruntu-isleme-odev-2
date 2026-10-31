// Ödev - Soru 3: Görüntüyü 4 parçaya bölüp her parçanın histogramını 4 thread ile paralel
//                hesaplama, parça histogramlarını birleştirip ana histogramı elde etme ve
//                tek thread ile 4 thread çalışma süresini karşılaştırma.
//
// Fikir:
//   - Her thread kendi parçası için KENDİ histogram dizisini doldurur (thread'ler arasında
//     ortak veri yazımı olmaz, kilit gerekmez).
//   - Thread'ler bitince 4 parça histogramı bin bin toplanır = ana histogram.
//
// Kullanım: ./histogram_thread <goruntu_dosyasi> [cikti_dizini] [tekrar_sayisi]
//
// Derleme (süre ölçümü için -O2 şart):
//   g++ -O2 odev_3_histogram_thread.cpp -o histogram_thread.exe -std=c++17 -pthread \
//       $(pkg-config --cflags --libs opencv5)

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>

#include <algorithm>
#include <array>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <thread>
#include <vector>

namespace fs = std::filesystem;

static const int NUM_PARTS = 4;
static const int BINS = 256;

using Hist = std::array<unsigned int, BINS>;

// Thread'lerin histogramları bellekte yan yana gelip birbirini yavaşlatmasın diye hizalı yapı.
struct alignas(64) PartHist {
    Hist h;
};

// src'nin [rowStart, rowEnd) satırlarının histogramını h'ye yazar.
static void histRows(const cv::Mat& src, int rowStart, int rowEnd, Hist& h) {
    h.fill(0);
    for (int y = rowStart; y < rowEnd; ++y) {
        const uchar* s = src.ptr<uchar>(y);
        for (int x = 0; x < src.cols; ++x) ++h[s[x]];
    }
}

// Tek thread: tüm görüntünün histogramı tek seferde.
static Hist histSingleThread(const cv::Mat& src) {
    Hist total;
    histRows(src, 0, src.rows, total);
    return total;
}

// 4 thread: her parça kendi histogramını çıkarır, sonra bin bin toplanır.
static Hist histMultiThread(const cv::Mat& src, const std::array<int, NUM_PARTS + 1>& bounds,
                            std::vector<PartHist>& parts) {
    std::vector<std::thread> workers;
    workers.reserve(NUM_PARTS);
    for (int i = 0; i < NUM_PARTS; ++i) {
        workers.emplace_back(histRows, std::cref(src), bounds[i], bounds[i + 1],
                             std::ref(parts[i].h));
    }
    for (auto& t : workers) t.join();

    Hist total;
    total.fill(0);
    for (int i = 0; i < NUM_PARTS; ++i)
        for (int b = 0; b < BINS; ++b) total[b] += parts[i].h[b];
    return total;
}

template <typename Fn>
static double averageMs(Fn fn, int repeat) {
    fn();  // ısınma turu
    auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < repeat; ++i) fn();
    auto t1 = std::chrono::steady_clock::now();
    return std::chrono::duration<double, std::milli>(t1 - t0).count() / repeat;
}

static unsigned long long sumHist(const Hist& h) {
    unsigned long long s = 0;
    for (unsigned int v : h) s += v;
    return s;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Kullanim: " << argv[0] << " <goruntu_dosyasi> [cikti_dizini] [tekrar_sayisi]\n";
        return 1;
    }

    const std::string inPath = argv[1];
    const fs::path outDir = (argc >= 3) ? fs::path(argv[2]) : fs::path("cikti_histogram");
    const int repeat = (argc >= 4) ? std::max(1, std::stoi(argv[3])) : 200;
    fs::create_directories(outDir);

    cv::Mat src = cv::imread(inPath, cv::IMREAD_GRAYSCALE);
    if (src.empty()) {
        std::cerr << "Hata: goruntu okunamadi: " << inPath << "\n";
        return 1;
    }

    const int h = src.rows;
    const std::array<int, NUM_PARTS + 1> bounds = {0, h / 4, h / 2, (3 * h) / 4, h};
    std::vector<PartHist> parts(NUM_PARTS);

    // Doğruluk kontrolü
    const Hist single = histSingleThread(src);
    const Hist multi = histMultiThread(src, bounds, parts);
    const bool same = (single == multi);
    const unsigned long long totalPixels = static_cast<unsigned long long>(src.rows) * src.cols;

    // Süre ölçümü
    const double msSingle = averageMs([&] { volatile auto r = histSingleThread(src)[0]; (void)r; }, repeat);
    const double msMulti = averageMs([&] { volatile auto r = histMultiThread(src, bounds, parts)[0]; (void)r; }, repeat);

    std::cout << "Goruntu boyutu : " << src.cols << "x" << src.rows << " (gri seviye), "
              << totalPixels << " piksel\n";
    std::cout << "Islemci thread sayisi (bilgisayar): " << std::thread::hardware_concurrency() << "\n";
    std::cout << "Tekrar sayisi  : " << repeat << " (sureler ortalamadir)\n\n";

    std::cout << "Parca histogramlarindaki piksel sayilari:\n";
    for (int i = 0; i < NUM_PARTS; ++i) {
        std::cout << "  Parca " << (i + 1) << " (satir " << bounds[i] << ".." << bounds[i + 1] - 1
                  << "): " << sumHist(parts[i].h) << " piksel\n";
    }
    std::cout << "  Birlestirilen ana histogram toplami: " << sumHist(multi)
              << " (goruntudeki piksel sayisi: " << totalPixels << ")\n\n";

    std::cout << "1 thread ve 4 thread histogramlari ayni mi: " << (same ? "EVET" : "HAYIR") << "\n";
    std::cout << "1 thread  ortalama : " << msSingle << " ms\n";
    std::cout << "4 thread  ortalama : " << msMulti << " ms\n";
    if (msMulti > 0) {
        std::cout << "Hizlanma (1 thread / 4 thread): " << (msSingle / msMulti) << "x\n";
    }

    // CSV: bin, parça1..parça4, toplam
    {
        std::ofstream csv(outDir / "histogram.csv");
        csv << "bin,parca1,parca2,parca3,parca4,toplam\n";
        for (int b = 0; b < BINS; ++b) {
            csv << b << "," << parts[0].h[b] << "," << parts[1].h[b] << "," << parts[2].h[b] << ","
                << parts[3].h[b] << "," << multi[b] << "\n";
        }
    }

    // Ana histogramı görüntü olarak çiz: 256 bin x 2 piksel genişlik, 300 piksel yükseklik
    {
        const int histH = 300, binW = 2;
        cv::Mat plot(histH, BINS * binW, CV_8UC1, cv::Scalar(255));
        const unsigned int maxCount = *std::max_element(multi.begin(), multi.end());
        for (int b = 0; b < BINS; ++b) {
            int barH = maxCount ? static_cast<int>(static_cast<double>(multi[b]) / maxCount * (histH - 10)) : 0;
            for (int x = b * binW; x < (b + 1) * binW; ++x)
                for (int y = histH - 1; y >= histH - 1 - barH; --y) plot.at<uchar>(y, x) = 0;
        }
        cv::imwrite((outDir / "histogram.png").string(), plot);
    }

    std::cout << "\nKaydedilenler (" << outDir.string() << "):\n"
              << "  histogram.csv : her parcanin ve toplam histogram degerleri\n"
              << "  histogram.png : ana histogram grafigi\n";
    return 0;
}
