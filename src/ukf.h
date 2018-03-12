#ifndef UKF_H
#define UKF_H

#include "measurement_package.h"
#include "Eigen/Dense"
#include <vector>
#include <string>
#include <fstream>

class UKF {
public:

  ///* initially set to false, set to true in first call of ProcessMeasurement
  bool is_initialized_;

  ///* if this is false, laser measurements will be ignored (except for init)
  bool use_laser_;

  ///* if this is false, radar measurements will be ignored (except for init)
  bool use_radar_;

  ///* state vector: [pos1 pos2 vel_abs yaw_angle yaw_rate] in SI units and rad
  Eigen::VectorXd x_;

  ///* state covariance matrix
  Eigen::MatrixXd P_;

  ///* predicted sigma points matrix
  Eigen::MatrixXd Xsig_pred_;

  Eigen::MatrixXd R_laser_;
  Eigen::MatrixXd H_laser_;
  Eigen::MatrixXd R_radar_;

  // previous timestamp
  long long previous_timestamp_;

  ///* Process noise standard deviation longitudinal acceleration in m/s^2
  double std_a_;

  ///* Process noise standard deviation yaw acceleration in rad/s^2
  double std_yawdd_;

  ///* Laser measurement noise standard deviation position1 in m
  double std_laspx_;

  ///* Laser measurement noise standard deviation position2 in m
  double std_laspy_;

  ///* Radar measurement noise standard deviation radius in m
  double std_radr_;

  ///* Radar measurement noise standard deviation angle in rad
  double std_radphi_;

  ///* Radar measurement noise standard deviation radius change in m/s
  double std_radrd_ ;

  ///* Weights of sigma points
  Eigen::VectorXd weights_;

  ///* State dimension
  const int n_x_;

  ///* Augmented state dimension
  const int n_aug_;

  ///* Sigma point spreading parameter
  const double lambda_;

  ///* NIS
  double nis_;

  unsigned int radar_nis_count_;
  unsigned int radar_nis_5_percentile_count_;
  unsigned int radar_nis_95_percentile_count_;
  unsigned int lidar_nis_count_;
  unsigned int lidar_nis_5_percentile_count_;
  unsigned int lidar_nis_95_percentile_count_;

  /**
   * Constructor
   */
  UKF();

  /**
   * Destructor
   */
  virtual ~UKF();

  /**
   * ProcessMeasurement
   * @param meas_package The latest measurement data of either radar or laser
   */
  void ProcessMeasurement(const MeasurementPackage & meas_package);

  /**
   * Prediction Predicts sigma points, the state, and the state covariance
   * matrix
   * @param delta_t Time between k and k+1 in s
   */
  void Prediction(double delta_t);

  /**
   * Updates the state and the state covariance matrix using a laser measurement
   * @param meas_package The measurement at k+1
   */
  void UpdateLidar(const MeasurementPackage & meas_package);

  /**
   * Updates the state and the state covariance matrix using a radar measurement
   * @param meas_package The measurement at k+1
   */
  void UpdateRadar(const MeasurementPackage & meas_package);

private:
	Eigen::MatrixXd AugmentedSigmaPoints();

	void SigmaPointPrediction(const Eigen::MatrixXd& Xsig_aug, const double delta_t);

	void PredictMeanAndCovariance();
};

#endif /* UKF_H */
