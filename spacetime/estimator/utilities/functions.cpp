#include <assert.h>
#include <cmath>
#include <algorithm>
#include <iostream>
#include <filesystem>
#include <fstream>
#include <sstream>

#include <Eigen/Eigenvalues>
#include <unsupported/Eigen/MatrixFunctions>

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
