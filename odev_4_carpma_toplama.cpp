// Ödev - Soru 4: Piksel değerlerini 0.75 ile çarpıp 20 ekleme (B = 0.75*p + 20) ve
//                sadece 0.75 ile çarpma (A = 0.75*p) sonuçlarını görsel ve sayısal karşılaştırma.
//
// Kullanım: ./carpma_toplama <goruntu_dosyasi> [cikti_dizini]
//
// Derleme:
//   g++ -O2 odev_4_carpma_toplama.cpp -o carpma_toplama.exe -std=c++17 \
//       $(pkg-config --cflags --libs opencv5)

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

namespace fs = std::filesystem;

struct Stats {
    double mean, stddev, minV, maxV;
};

static Stats computeStats(const cv::Mat& img) {
    cv::Scalar m, s;
    cv::meanStdDev(img, m, s);
    double mn = 0, mx = 0;
    cv::minMaxLoc(img, &mn, &mx);
    return {m[0], s[0], mn, mx};
}

static void printRow(const std::string& name, const Stats& st) {
    std::cout << std::left << std::setw(26) << name << std::right << std::fixed << std::setprecision(2)
              << std::setw(10) << st.mean << std::setw(12) << st.stddev << std::setw(8) << st.minV
              << std::setw(8) << st.maxV << "\n";
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Kullanim: " << argv[0] << " <goruntu_dosyasi> [cikti_dizini]\n";
        return 1;
    }

    const std::string inPath = argv[1];
    const fs::path outDir = (argc >= 3) ? fs::path(argv[2]) : fs::path("cikti_carpma_toplama");
    fs::create_directories(outDir);

    cv::Mat src = cv::imread(inPath, cv::IMREAD_GRAYSCALE);
    if (src.empty()) {
        std::cerr << "Hata: goruntu okunamadi: " << inPath << "\n";
        return 1;
    }

    cv::Mat A(src.size(), CV_8UC1);  // A = 0.75 * p
    cv::Mat B(src.size(), CV_8UC1);  // B = 0.75 * p + 20

    long long clipped = 0;  // B hesabında 0..255 dışına taşan piksel sayısı
    for (int y = 0; y < src.rows; ++y) {
        for (int x = 0; x < src.cols; ++x) {
            const int p = src.at<uchar>(y, x);
            const double a = 0.75 * p;
            const double b = 0.75 * p + 20.0;
            if (b < 0.0 || b > 255.0) ++clipped;
            A.at<uchar>(y, x) = cv::saturate_cast<uchar>(a);
            B.at<uchar>(y, x) = cv::saturate_cast<uchar>(b);
        }
    }

    // B - A farkının piksel bazında incelenmesi
    int minDiff = 255, maxDiff = -255;
    long long notTwenty = 0;
    double sumAbsDiff = 0;
    for (int y = 0; y < src.rows; ++y) {
        for (int x = 0; x < src.cols; ++x) {
            const int d = static_cast<int>(B.at<uchar>(y, x)) - static_cast<int>(A.at<uchar>(y, x));
            minDiff = std::min(minDiff, d);
            maxDiff = std::max(maxDiff, d);
            if (d != 20) ++notTwenty;
            sumAbsDiff += std::abs(d);
        }
    }
    const double totalPixels = static_cast<double>(src.rows) * src.cols;

    std::cout << "Goruntu boyutu: " << src.cols << "x" << src.rows << " (gri seviye)\n\n";
    std::cout << std::left << std::setw(26) << "Goruntu" << std::right << std::setw(10) << "Ortalama"
              << std::setw(12) << "Std.sapma" << std::setw(8) << "Min" << std::setw(8) << "Maks" << "\n";
    printRow("Orijinal", computeStats(src));
    printRow("A = 0.75*p", computeStats(A));
    printRow("B = 0.75*p + 20", computeStats(B));

    std::cout << "\nB ile A arasindaki fark (B - A), piksel piksel:\n";
    std::cout << "  En kucuk fark          : " << minDiff << "\n";
    std::cout << "  En buyuk fark          : " << maxDiff << "\n";
    std::cout << "  Ortalama mutlak fark   : " << std::fixed << std::setprecision(3)
              << sumAbsDiff / totalPixels << "\n";
    std::cout << "  Farki 20 olmayan piksel: " << notTwenty << "\n";
    std::cout << "  B'de 0..255 disina tasan (kirpilan) piksel: " << clipped << "\n";

    // Görsel karşılaştırma: yan yana (orijinal | A | B), her biri 480 piksel genişliğe küçültülür
    auto small = [](const cv::Mat& m) {
        cv::Mat r;
        const double s = 480.0 / m.cols;
        cv::resize(m, r, cv::Size(), s, s, cv::INTER_AREA);
        return r;
    };
    std::vector<cv::Mat> row = {small(src), small(A), small(B)};
    cv::Mat montage;
    cv::hconcat(row, montage);

    cv::imwrite((outDir / "1_orijinal.png").string(), src);
    cv::imwrite((outDir / "2_sadece_0.75_carpma.png").string(), A);
    cv::imwrite((outDir / "3_0.75_carpma_ve_20_ekleme.png").string(), B);
    cv::imwrite((outDir / "4_yan_yana_orijinal_A_B.png").string(), montage);

    std::cout << "\nKaydedilenler (" << outDir.string() << "):\n"
              << "  1_orijinal.png\n"
              << "  2_sadece_0.75_carpma.png\n"
              << "  3_0.75_carpma_ve_20_ekleme.png\n"
              << "  4_yan_yana_orijinal_A_B.png  (soldan saga: orijinal | A | B)\n";
    return 0;
}
