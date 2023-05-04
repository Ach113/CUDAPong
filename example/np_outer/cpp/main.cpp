#include <iostream>
#include <vector>


#include <iostream>

double** outer(double* a, double* b, int m, int n) {
    double** result = new double*[m];
    for (int i = 0; i < m; i++) {
        result[i] = new double[n];
        for (int j = 0; j < n; j++) {
            result[i][j] = a[i] * b[j];
        }
    }
    return result;
}

int main() {
    // double a[] = {6, 2};
    // double b[] = {2, 5};

    double a[] = {1, 2, 3};
    double b[] = {4, 5, 6};

    // double a[] = {1.0, 2.0, 3.0};
    // double b[] = {4.0, 5.0, 6.0};
    // int m = 2;
    // int n = 2;
    int m = sizeof(a) / sizeof(double);
    int n = sizeof(b) / sizeof(double);
    double** result = outer(a, b, m, n);
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            std::cout << result[i][j] << " ";
        }
        std::cout << std::endl;
    }
    return 0;
}



// std::vector<std::vector<double>> outer(const std::vector<double>& a, const std::vector<double>& b) {
//     int m = a.size();
//     int n = b.size();
//     std::vector<std::vector<double>> result(m, std::vector<double>(n));
//     for (int i = 0; i < m; i++) {
//         for (int j = 0; j < n; j++) {
//             result[i][j] = a[i] * b[j];
//         }
//     }
//     return result;
// }

// int main() {
//     std::vector<double> a = {6, 2};
//     std::vector<double> b = {2, 5};
//     std::vector<std::vector<double>> result = outer(a, b);
//     for (int i = 0; i < a.size(); i++) {
//         for (int j = 0; j < b.size(); j++) {
//             std::cout << result[i][j] << " ";
//         }
//         std::cout << std::endl;
//     }
//     return 0;
// }