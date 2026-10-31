// Ödev - Soru 1: Tek kanallı (gri seviye) görüntüde 8-bit -> 6-bit parlaklık nicemleme.
//
// 8-bit  : 256 parlaklık seviyesi (0..255)
// 6-bit  :  64 parlaklık seviyesi (0..63)
// Her 4 ardışık 8-bit değer tek bir 6-bit değere iner: q = p / 4  (yani p >> 2)
//
// Kullanım: ./nicemleme <goruntu_dosyasi> [cikti_dizini]

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>

#include <cmath>
#include <filesystem>
#include <iostream>
#include <string>

namespace fs = std::filesystem;

// Görüntüde kaç farklı parlaklık değeri olduğunu sayar.
static int countLevels(const cv::Mat& img) {
    bool seen[256] = {false};
    for (int y = 0; y < img.rows; ++y)
        for (int x = 0; x < img.cols; ++x)
            seen[img.at<uchar>(y, x)] = true;
    int count = 0;
    for (bool s : seen) count += s ? 1 : 0;
    return count;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Kullanim: " << argv[0] << " <goruntu_dosyasi> [cikti_dizini]\n";
        return 1;
    }

    const std::string inPath = argv[1];
    const fs::path outDir = (argc >= 3) ? fs::path(argv[2]) : fs::path("cikti_nicemleme");
    fs::create_directories(outDir);

    // Görüntüyü tek kanallı (gri seviye) oku
    cv::Mat gray = cv::imread(inPath, cv::IMREAD_GRAYSCALE);
    if (gray.empty()) {
        std::cerr << "Hata: goruntu okunamadi: " << inPath << "\n";
        return 1;
    }

    cv::Mat q6(gray.size(), CV_8UC1);      // gerçek 6-bit değerler (0..63)
    cv::Mat gorsel(gray.size(), CV_8UC1);  // gözle karşılaştırmak için 0..255'e geri ölçeklenmiş

    for (int y = 0; y < gray.rows; ++y) {
        for (int x = 0; x < gray.cols; ++x) {
            uchar p = gray.at<uchar>(y, x);
            uchar q = static_cast<uchar>(p >> 2);  // 8-bit -> 6-bit
            q6.at<uchar>(y, x) = q;
            // 0..63 aralığını görüntülemek için 0..255'e yay
            gorsel.at<uchar>(y, x) = cv::saturate_cast<uchar>(q * 255.0 / 63.0);
        }
    }

    double qMin = 0, qMax = 0;
    cv::minMaxLoc(q6, &qMin, &qMax);

    std::cout << "Goruntu boyutu            : " << gray.cols << "x" << gray.rows << "\n";
    std::cout << "Orijinal farkli seviye    : " << countLevels(gray) << " (en fazla 256)\n";
    std::cout << "Nicemlenmis farkli seviye : " << countLevels(q6) << " (en fazla 64)\n";
    std::cout << "6-bit deger araligi       : " << qMin << " .. " << qMax << "\n";
    std::cout << "Ort. mutlak fark (gorsel vs orijinal): "
              << cv::mean(cv::abs(gray - gorsel))[0] << "\n";

    // PNG kayıpsızdır, seviyeler bozulmaz (JPG kullanma)
    cv::imwrite((outDir / "1_gri_orijinal.png").string(), gray);
    cv::imwrite((outDir / "2_6bit_ham_0_63.png").string(), q6);
    cv::imwrite((outDir / "3_6bit_gorsel_0_255.png").string(), gorsel);

    std::cout << "\nKaydedilenler (" << outDir.string() << "):\n"
              << "  1_gri_orijinal.png       : 8-bit gri orijinal\n"
              << "  2_6bit_ham_0_63.png      : gercek 6-bit degerler (cok koyu gorunur, normal)\n"
              << "  3_6bit_gorsel_0_255.png  : 6-bit sonucun gorunur hali\n";
    return 0;
}
