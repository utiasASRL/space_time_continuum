#include <gtest/gtest.h>
#include <assert.h>
#include <cmath>
#include <algorithm>
#include <iostream>
#include <filesystem>
#include <fstream>
#include <sstream>

#include <Eigen/Eigenvalues>
#include <unsupported/Eigen/MatrixFunctions>
#include <lgmath.hpp>

#include <spacetime/estimator/types/States.hpp>
#include <spacetime/estimator/utilities/functions.hpp>

bool CSVReader::burnLine()
{
    std::string row;
    std::getline(m_stream, row, '\n');
    return true;
}

void CSVReader::close()
{
    m_stream.close();
}

Eigen::SparseMatrix<double> getFullRankProjection(const Eigen::MatrixX<double> &A)
{
    assert(A.rows() == A.cols());

    std::vector<Eigen::Triplet<double>> P_tripletList;
    int offset = 0;

    for (int i = 0; i < A.rows(); i++)
    {
        // Extract non-zero rows of Lk via projection matrix P
        if (A.row(i).any())
        {
            P_tripletList.emplace_back(i, offset, 1.0);
            offset++;
        }
    }
    Eigen::SparseMatrix<double> P(A.rows(), offset);
    P.setFromTriplets(P_tripletList.begin(), P_tripletList.end());

    return P;
}

// namespace lgmath::se3
// {
//     Eigen::Matrix<double, 6, 6> getdJdt(const Eigen::Matrix<double, 18, 1> &gamma, bool dt, int num_terms)
//     {
//         assert(num_terms >= 1);

//         Eigen::Matrix<double, 6, 6> dJdt = ZERO(6), dx, x;
//         x = curlyhat(gamma.block<6, 1>(0, 0));
//         if (dt)
//             dx = curlyhat(gamma.block<6, 1>(12, 0));
//         else
//             dx = curlyhat(gamma.block<6, 1>(6, 0));

//         for (int n = 0; n < num_terms; ++n)
//         {
//             double factor = 1.0 / std::tgamma(n + 3); // std::tgamma(n + 3) is equivalent to (n + 2)!
//             Eigen::Matrix<double, 6, 6> term = ZERO(6);
//             for (int i = 0; i <= n; ++i)
//             {
//                 Eigen::Matrix<double, 6, 6> _term = EYE(6);
//                 for (int j = 0; j < i; ++j)
//                 {
//                     _term = x * _term;
//                 }
//                 _term = dx * _term;
//                 for (int j = i + 1; j <= n; ++j)
//                 {
//                     _term = x * _term;
//                 }
//                 term += _term;
//             }
//             dJdt += factor * term;
//         }

//         return dJdt;
//     }
// }