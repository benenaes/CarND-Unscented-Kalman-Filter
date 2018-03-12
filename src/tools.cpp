#include <iostream>
#include "tools.h"

using Eigen::VectorXd;
using Eigen::MatrixXd;
using std::vector;

Tools::Tools() {}

Tools::~Tools() {}

VectorXd Tools::CalculateRMSE(
	const vector<VectorXd> &estimations,
    const vector<VectorXd> &ground_truth) 
{
	VectorXd rmse(4);
	rmse << 0, 0, 0, 0;

	if (estimations.size() == 0)
	{
		cout << "the estimation vector size should not be zero";
		return rmse;
	}

	if (estimations.size() != ground_truth.size())
	{
		cout << "the estimation vector size should equal ground truth vector size";
		return rmse;
	}

	//accumulate squared residuals
	for (unsigned int i = 0; i < estimations.size(); ++i)
	{
		VectorXd term = (estimations[i] - ground_truth[i]);
		term = term.array() * term.array();
		rmse += term;
	}

	rmse /= estimations.size();
	rmse = rmse.array().sqrt();

	return rmse;
}