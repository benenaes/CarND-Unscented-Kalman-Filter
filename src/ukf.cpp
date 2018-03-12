#include "ukf.h"
#include "Eigen/Dense"
#include <iostream>

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

/**
 * Initializes Unscented Kalman filter
 * This is scaffolding, do not modify
 */
UKF::UKF() :
	is_initialized_(false),
	previous_timestamp_(0),
	n_x_(5),
	n_aug_(7),
	lambda_(3 - n_aug_),
	nis_(0.5),
	radar_nis_count_(0),
	radar_nis_5_percentile_count_(0),
	radar_nis_95_percentile_count_(0),
	lidar_nis_count_(0),
	lidar_nis_5_percentile_count_(0),
	lidar_nis_95_percentile_count_(0)
{
	// if this is false, laser measurements will be ignored (except during init)
	use_laser_ = true;

	// if this is false, radar measurements will be ignored (except during init)
	use_radar_ = true;

	// initial state vector
	x_ = VectorXd(n_x_);

	// initial covariance matrix
	P_ = MatrixXd::Identity(n_x_, n_x_);

	// Process noise standard deviation longitudinal acceleration in m/s^2
	std_a_ = 1.6;

	// Process noise standard deviation yaw acceleration in rad/s^2
	std_yawdd_ = 0.6;
  
	//DO NOT MODIFY measurement noise values below these are provided by the sensor manufacturer.
	// Laser measurement noise standard deviation position1 in m
	std_laspx_ = 0.15;

	// Laser measurement noise standard deviation position2 in m
	std_laspy_ = 0.15;

	// Radar measurement noise and process matrix
	R_laser_ = MatrixXd(2, 2);
	H_laser_ = MatrixXd(2, 5);
	H_laser_ <<
		1, 0, 0, 0, 0,
		0, 1, 0, 0, 0;
	//measurement covariance matrix - laser
	R_laser_ << std_laspx_ * std_laspx_, 0,
		0, std_laspy_ * std_laspy_;

	// Radar measurement noise standard deviation radius in m
	std_radr_ = 0.3;

	// Radar measurement noise standard deviation angle in rad
	std_radphi_ = 0.03;

	// Radar measurement noise standard deviation radius change in m/s
	std_radrd_ = 0.3;
	//DO NOT MODIFY measurement noise values above these are provided by the sensor manufacturer.

	//measurement covariance matrix - radar
	R_radar_ = MatrixXd(3, 3);
	R_radar_ << std_radr_ * std_radr_, 0, 0,
		0, std_radphi_ * std_radphi_, 0,
		0, 0, std_radrd_ * std_radrd_;

	Xsig_pred_ = MatrixXd(n_x_, 2 * n_aug_ + 1);

	//create vector for weights
	weights_ = VectorXd(2 * n_aug_ + 1);

	//set weights
	weights_(0) = lambda_ / (lambda_ + n_aug_);
	for (unsigned int i = 1; i < 2 * n_aug_ + 1; ++i)
	{
		weights_(i) = 1 / (2 * (lambda_ + n_aug_));
	}
}

UKF::~UKF() {}

MatrixXd UKF::AugmentedSigmaPoints()
{
	//create augmented mean vector
	VectorXd x_aug = VectorXd(n_aug_);

	//create augmented state covariance
	MatrixXd P_aug = MatrixXd(n_aug_, n_aug_);

	//create sigma point matrix
	MatrixXd Xsig_aug = MatrixXd(n_aug_, 2 * n_aug_ + 1);

	//create augmented mean state
	x_aug.head(n_x_) = x_;
	x_aug(n_x_) = 0;
	x_aug(n_x_ + 1) = 0;

	//create augmented covariance matrix
	P_aug.fill(0.0);
	P_aug.topLeftCorner(n_x_, n_x_) = P_;
	P_aug(n_x_, n_x_) = std_a_ * std_a_;
	P_aug(n_x_ + 1, n_x_ + 1) = std_yawdd_ * std_yawdd_;

	//create square root matrix
	MatrixXd L = P_aug.llt().matrixL();

	//create augmented sigma points
	Xsig_aug.col(0) = x_aug;
	for (int i = 0; i < n_aug_; ++i)
	{
		Xsig_aug.col(i + 1) = x_aug + sqrt(lambda_ + n_aug_) * L.col(i);
		Xsig_aug.col(i + 1 + n_aug_) = x_aug - sqrt(lambda_ + n_aug_) * L.col(i);
	}

	// cout << Xsig_aug << endl;

	return Xsig_aug;
}

void UKF::SigmaPointPrediction(const MatrixXd& Xsig_aug, const double delta_t)
{
	//predict sigma points
	for (int i = 0; i < 2 * n_aug_ + 1; ++i)
	{
		//extract values for better readability
		double p_x = Xsig_aug(0, i);
		double p_y = Xsig_aug(1, i);
		double v = Xsig_aug(2, i);
		double yaw = Xsig_aug(3, i);
		double yawd = Xsig_aug(4, i);
		double nu_a = Xsig_aug(5, i);
		double nu_yawdd = Xsig_aug(6, i);

		//predicted state values
		double px_p, py_p;

		//avoid division by zero
		if (fabs(yawd) > 0.001) 
		{
			px_p = p_x + v / yawd * (sin(yaw + yawd * delta_t) - sin(yaw));
			py_p = p_y + v / yawd * (cos(yaw) - cos(yaw + yawd * delta_t));
		}
		else 
		{
			px_p = p_x + v * delta_t * cos(yaw);
			py_p = p_y + v * delta_t * sin(yaw);
		}

		double v_p = v;
		double yaw_p = yaw + yawd * delta_t;
		double yawd_p = yawd;

		//add noise
		px_p = px_p + 0.5 * nu_a * delta_t * delta_t * cos(yaw);
		py_p = py_p + 0.5 * nu_a * delta_t * delta_t * sin(yaw);
		v_p = v_p + nu_a * delta_t;

		yaw_p = yaw_p + 0.5 * nu_yawdd * delta_t * delta_t;
		yawd_p = yawd_p + nu_yawdd * delta_t;

		//write predicted sigma point into right column
		Xsig_pred_(0, i) = px_p;
		Xsig_pred_(1, i) = py_p;
		Xsig_pred_(2, i) = v_p;
		Xsig_pred_(3, i) = yaw_p;
		Xsig_pred_(4, i) = yawd_p;
	}

	//print result
	// std::cout << "Xsig_pred = " << std::endl << Xsig_pred_ << std::endl;
}

void UKF::PredictMeanAndCovariance() 
{
	// predict state mean
	x_ = Xsig_pred_ * weights_;

	// predict state covariance matrix
	MatrixXd temp = Xsig_pred_.colwise() - x_;

	for (int i = 0; i < 2 * n_aug_ + 1; i++) 
	{
		//angle normalization
		while (temp(3,i)> M_PI) temp(3,i) -= 2.*M_PI;
		while (temp(3,i)<-M_PI) temp(3,i) += 2.*M_PI;
	}

	const MatrixXd temp2 = temp.transpose();
	const MatrixXd temp3 = temp2.array().colwise() * weights_.array();
	P_ = temp * temp3;

	//print result
	// std::cout << "Predicted state" << std::endl;
	// std::cout << x_ << std::endl;
	// std::cout << "Predicted covariance matrix" << std::endl;
	// std::cout << P_ << std::endl;
}

/**
 * @param {MeasurementPackage} meas_package The latest measurement data of
 * either radar or laser.
 */
void UKF::ProcessMeasurement(const MeasurementPackage & meas_package) 
{
	if (!is_initialized_)
	{
		// first measurement

		if (meas_package.sensor_type_ == MeasurementPackage::RADAR)
		{
			/**
			Convert radar from polar to cartesian coordinates and initialize state.
			*/
			const double px = cos(meas_package.raw_measurements_[1]) * meas_package.raw_measurements_[0];
			const double py = sin(meas_package.raw_measurements_[1]) * meas_package.raw_measurements_[0];
			// For initialization, I assume the velocity of the detected object is along the axis 
			// defined by phi (measurement_pack.raw_measurements_[1])
			// This resulted in better performance than initializing with zeroes on dataset 2 (where the 1st measurement is RADAR).
			const double v = 0;
			const double psi = meas_package.raw_measurements_[2];
			const double psi_dot = 0;
			x_ << px, py, v, psi, psi_dot;
		}
		else if (meas_package.sensor_type_ == MeasurementPackage::LASER)
		{
			x_ << meas_package.raw_measurements_[0], meas_package.raw_measurements_[1], 0, 0, 0;
		}

		previous_timestamp_ = meas_package.timestamp_;

		// done initializing, no need to predict or update
		is_initialized_ = true;
		cout << "Initialized !" << endl;
		return;
	}

	/*****************************************************************************
	*  Prediction
	****************************************************************************/

	// dt - expressed in seconds
	float dt = (meas_package.timestamp_ - previous_timestamp_) / 1000000.0;
	previous_timestamp_ = meas_package.timestamp_;

	Prediction(dt);

	/*****************************************************************************
	*  Update
	****************************************************************************/

	if (meas_package.sensor_type_ == MeasurementPackage::RADAR && use_radar_)
	{
		UpdateRadar(meas_package);
	}
	else if (meas_package.sensor_type_ == MeasurementPackage::LASER && use_laser_)
	{
		// Laser updates
		UpdateLidar(meas_package);
	}

	// print the output
	// cout << "x_ = " << x_ << endl;
	// cout << "P_ = " << P_ << endl;
}

/**
 * Predicts sigma points, the state, and the state covariance matrix.
 * @param {double} delta_t the change in time (in seconds) between the last
 * measurement and this one.
 */
void UKF::Prediction(double delta_t) 
{
	const MatrixXd & SigmaPoints = AugmentedSigmaPoints();
	SigmaPointPrediction(SigmaPoints, delta_t);
	PredictMeanAndCovariance();
}

/**
 * Updates the state and the state covariance matrix using a laser measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateLidar(const MeasurementPackage & meas_package) 
{
	const VectorXd z_pred = H_laser_ * x_;
	const VectorXd ErrorMatrix = meas_package.raw_measurements_ - z_pred;

	const MatrixXd Ht = H_laser_.transpose();
	const MatrixXd S = H_laser_ * P_ * Ht + R_laser_;
	const MatrixXd Si = S.inverse();
	const MatrixXd K = P_ * Ht * Si;

	//new estimate
	x_ = x_ + (K * ErrorMatrix);
	long x_size = x_.size();
	const MatrixXd I = MatrixXd::Identity(x_size, x_size);
	P_ = (I - K * H_laser_) * P_;

	// NIS
	nis_ = ErrorMatrix.transpose() * Si * ErrorMatrix;
	cout << "LIDAR nis: " << nis_ << endl;

	if (nis_ < 0.103)
	{
		++lidar_nis_5_percentile_count_;
	}
	else if (nis_ > 5.991)
	{
		++lidar_nis_95_percentile_count_;
	}
	++lidar_nis_count_;

	std::cout << "Fraction of lidar NIS below 0.103: " << lidar_nis_5_percentile_count_ / (double)lidar_nis_count_ << endl;
	std::cout << "Fraction of lidar NIS above 5.991: " << lidar_nis_95_percentile_count_ / (double)lidar_nis_count_ << endl;
}

/**
 * Updates the state and the state covariance matrix using a radar measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateRadar(const MeasurementPackage & meas_package)
{
	const unsigned int n_z = 3;

	//create matrix for sigma points in measurement space
	MatrixXd Zsig = MatrixXd(n_z, 2 * n_aug_ + 1);

	//mean predicted measurement
	VectorXd z_pred = VectorXd(n_z);

	//measurement covariance matrix S
	MatrixXd S = MatrixXd(n_z, n_z);

	//transform sigma points into measurement space
	for (unsigned int i = 0; i < 2 * n_aug_ + 1; ++i)
	{
		const double px = Xsig_pred_(0, i);
		const double py = Xsig_pred_(1, i);
		const double v = Xsig_pred_(2, i);
		const double rho = Xsig_pred_(3, i);
		Zsig(0, i) = sqrt(px * px + py * py);
		if (fabs(px) < 0.0001)
		{
			Zsig(1, i) = atan2(py, 0.0001);
		}
		else
		{
			Zsig(1, i) = atan2(py, px);
		}
		Zsig(2, i) = (px * cos(rho) + py * sin(rho)) * v;
		if (fabs(Zsig(0, i)) < 0.0001)
		{
			Zsig(2, i) /= 0.0001;
		}
		else
		{
			Zsig(2, i) /= Zsig(0, i);
		}
	}

	//calculate mean predicted measurement
	z_pred = Zsig * weights_;

	//calculate innovation covariance matrix S
	MatrixXd OffsetMatrix = Zsig.colwise() - z_pred;

	for (int i = 0; i < 2 * n_aug_ + 1; ++i) 
	{
		//angle normalization
		while (OffsetMatrix(1, i) > M_PI) OffsetMatrix(1, i) -= 2.*M_PI;
		while (OffsetMatrix(1, i) < -M_PI) OffsetMatrix(1, i) += 2.*M_PI;
	}

	const MatrixXd OffsetMatrixTranspose = OffsetMatrix.transpose();
	const MatrixXd S_temp = OffsetMatrixTranspose.array().colwise() * weights_.array();
	S = OffsetMatrix * S_temp;
	const MatrixXd Si = S.inverse();
	S += R_radar_;

	//calculate cross correlation matrix
	MatrixXd temp = Xsig_pred_.colwise() - x_;

	for (int i = 0; i < 2 * n_aug_ + 1; ++i)
	{
		//angle normalization
		while (temp(3, i) > M_PI) temp(3, i) -= 2.*M_PI;
		while (temp(3, i) < -M_PI) temp(3, i) += 2.*M_PI;
	}

	const MatrixXd temp2 = OffsetMatrix.transpose().array().colwise() * weights_.array();

	MatrixXd Tc = MatrixXd(n_x_, n_z);
	Tc = temp * temp2;

	//calculate Kalman gain K;
	const MatrixXd K = Tc * S.inverse();

	//update state mean and covariance matrix
	VectorXd z = meas_package.raw_measurements_;
	const VectorXd ErrorMatrix = z - z_pred;

	x_ = x_ + K * ErrorMatrix;
	P_ = P_ - K * S * K.transpose();

	// NIS
	nis_ = ErrorMatrix.transpose() * (Si * ErrorMatrix);

	if (nis_ < 0.352)
	{
		++radar_nis_5_percentile_count_;
	}
	else if (nis_ > 7.815)
	{
		++radar_nis_95_percentile_count_;
	}
	++radar_nis_count_;

	std::cout << "RADAR nis: " << nis_ << endl;
	std::cout << "Fraction of radar NIS below 0.352: " << radar_nis_5_percentile_count_ / (double) radar_nis_count_ << endl;
	std::cout << "Fraction of radar NIS above 7.815: " << radar_nis_95_percentile_count_ / (double) radar_nis_count_ << endl;
}
