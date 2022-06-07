#pragma once
#include "NonRigidreg.h"
#include "tools/AndersonAcceleration.h"
#include "median.h"

typedef Eigen::Triplet<Scalar> Triplet;

NonRigidreg::NonRigidreg() {
};

NonRigidreg::~NonRigidreg()
{
}

void NonRigidreg::Initialize()
{
    ldlt_ = new Eigen::SimplicialCholesky<SparseMatrix>;
    nonrigid_init();
    // Initialize BFGS paras
    iter_ = 0;
    col_idx_ = 0;

    weight_d_.setOnes();
    weight_s_.setOnes();
    Scalar sample_radius;

    Timer timer;
    Timer::EventID time1, time2;
    time1 = timer.get_time();
    sample_radius = src_sample_nodes.sampleAndconstuct(*src_mesh_, pars_.uni_sample_radio, svr::nodeSampler::X_AXIS);
    time2 = timer.get_time();

    // DEBUG: output node graph
    if(pars_.print_each_step_info)
    {
        std::string out_node = pars_.out_each_step_info;// + "_init.obj";
        src_sample_nodes.print_nodes(*src_mesh_, out_node);//init sample nodes
    }

    num_sample_nodes = src_sample_nodes.nodeSize();
    pars_.num_sample_nodes = num_sample_nodes;

    all_s_.resize(4 * num_sample_nodes * 3, pars_.lbfgs_m); all_s_.setZero();
    all_t_.resize(4 * num_sample_nodes * 3, pars_.lbfgs_m); all_t_.setZero();

    Smat_X_.resize(4 * num_sample_nodes, 3); Smat_X_.setZero();
    Weight_PV_.resize(n_src_vertex_, 4 * num_sample_nodes);
    Smat_P_.resize(n_src_vertex_, 3);

    Smat_R_.resize(3 * num_sample_nodes, 3); Smat_R_.setZero();
    Smat_L_.resize(4 * num_sample_nodes, 4 * num_sample_nodes);
    Smat_J_.resize(4 * num_sample_nodes, 3 * num_sample_nodes);

    std::vector<Triplet> coeffv(4 * num_sample_nodes);
    std::vector<Triplet> coeffL(3 * num_sample_nodes);
    std::vector<Triplet> coeffJ(3 * num_sample_nodes);
    for (int i = 0; i < num_sample_nodes; i++)
    {
        // Smat_X_
        Smat_X_.block(4 * i, 0, 3, 3) = MatrixXX::Identity(3, 3);

        // Smat_R_
        Smat_R_.block(3 * i, 0, 3, 3) = MatrixXX::Identity(3, 3);

        // Smat_L_
        coeffL[3 * i] = Triplet(4 * i, 4 * i, 1.0);
        coeffL[3 * i + 1] = Triplet(4 * i + 1, 4 * i + 1, 1.0);
        coeffL[3 * i + 2] = Triplet(4 * i + 2, 4 * i + 2, 1.0);

        // Smat_J_
        coeffJ[3 * i] = Triplet(4 * i, 3 * i, 1.0);
        coeffJ[3 * i + 1] = Triplet(4 * i + 1, 3 * i + 1, 1.0);
        coeffJ[3 * i + 2] = Triplet(4 * i + 2, 3 * i + 2, 1.0);

    }
    Smat_L_.setFromTriplets(coeffL.begin(), coeffL.end());
    Smat_J_.setFromTriplets(coeffJ.begin(), coeffJ.end());
    direction_.resize(4*num_sample_nodes, 3);

    // update coefficient matrices
    src_sample_nodes.initWeight(Weight_PV_, Smat_P_, Smat_B_, Smat_D_, Sweight_s_);

    //**using landmark points as an energy term**//
/*    if(pars_.use_landmark && pars_.landmark_src.size() > 0)
   {
       size_t n_landmarks = pars_.landmark_src.size();
       Sub_PV_.resize(n_landmarks, 4*num_sample_nodes);
       Sub_UP_.resize(n_landmarks, 3);
       if(pars_.landmark_tar.size() != n_landmarks)
       {
           std::cout << "Error: The source and target points do not match!" << std::endl;
           exit(1);
       }
       for(size_t i = 0; i < n_landmarks; i++)
       {
           size_t src_idx = pars_.landmark_src[i];
           size_t tar_idx = pars_.landmark_tar[i];
           VectorX row_val = Weight_PV_.row(src_idx);
           for(int j = 0; j < 4*num_sample_nodes; j++) {
               if(fabs(row_val[j])>1e-12) {
                   Sub_PV_.insert(i, j) =  row_val[j];
               }
           }
           Sub_UP_.row(i) = tar_points_.col(tar_idx).transpose() - Smat_P_.row(src_idx);
       }
       pars_.gamma = pars_.gamma * n_landmarks / n_src_vertex_;
   }*/

    pars_.alpha = pars_.alpha * Smat_B_.rows()/ n_src_vertex_;
    pars_.beta = pars_.beta *  num_sample_nodes / n_src_vertex_;
}


Scalar NonRigidreg::DoNonRigid()
{
    Scalar data_err, smooth_err, orth_err;

    MatrixXX curV =  MatrixXX::Zero(n_src_vertex_, 3);
    MatrixXX prevV = MatrixXX::Zero(n_src_vertex_, 3);

    Weight_PV0 = Weight_PV_;
    Smat_B0 = Smat_B_;
    Smat_D0 = Smat_D_;
    // welsch_sweight
    VectorX welsch_weight_s = VectorX::Ones(Sweight_s_.rows(), 1);
    bool run_once = true;

    Timer time;
    Timer::EventID begin_time, run_time;
    pars_.each_energys.clear();
    pars_.each_gt_mean_errs.clear();
    pars_.each_gt_max_errs.clear();
    pars_.each_times.clear();
    pars_.each_iters.clear();
    pars_.each_term_energy.clear();

    // Data term parameters
    Scalar nu1 = pars_.Data_initk * pars_.Data_nu;
    Scalar average_len = CalcEdgelength(src_mesh_, 1);
    Scalar end_nu1 = pars_.Data_endk * average_len;
    Scalar nu2 = pars_.Smooth_nu * average_len;
    // Smooth term parameters
    w_data = optimize_data = 1.0;
    w_reg = optimize_reg = pars_.alpha;
    w_rot = optimize_rot = pars_.beta;

    if(pars_.data_use_welsch)
    {
        w_data = optimize_data *(2.0*nu1*nu1);
    }
    if(pars_.smooth_use_welsch)
    {
        w_reg = optimize_reg * (2*nu2*nu2);
    }

    std::cout <<"wdata = " <<w_data << " wreg = " << w_reg << " wrot = " << w_rot
             << " nu1 = " << nu1 << " " << nu2 << " end nu1 = " << end_nu1 << std::endl;

    // print initial run results
    mat_U0_.setZero();
#pragma omp parallel for
    for (size_t i = 0; i < correspondence_pairs_.size(); i++)
    {
        mat_U0_.col(correspondence_pairs_[i].src_idx) = correspondence_pairs_[i].position;
    }
    pars_.each_energys.push_back(0.0);
    pars_.each_gt_max_errs.push_back(pars_.init_gt_max_errs);
    pars_.each_gt_mean_errs.push_back(pars_.init_gt_mean_errs);
    pars_.each_iters.push_back(0);
    pars_.each_times.push_back(pars_.non_rigid_init_time);
    pars_.each_term_energy.push_back(Vector3(0,0,0));

    Scalar gt_err;

    bool dynamic_stop = false;
    int accumulate_iter = 0;
    Scalar ignore_time = 0.0;
    Scalar energy, prev_energy = 1e12;
    int  reset_num = 0;

    AndersonAcceleration Acc;
    Acc.init(pars_.anderson_m, num_sample_nodes*12, Smat_X_.data());
    MatrixXX prev_X = Smat_X_;
    curV = Weight_PV0 * Smat_X_ + Smat_P_;

    begin_time = time.get_time();
    while (!dynamic_stop)
    {
        int out_iter = 0;
        while(out_iter < pars_.max_outer_iters)
        {
            // according correspondence_pairs to update mat_U0_;
            mat_U0_.setZero();
            corres_pair_ids_.setZero();
            #pragma omp parallel for
            for (size_t i = 0; i < correspondence_pairs_.size(); i++)
            {
                mat_U0_.col(correspondence_pairs_[i].src_idx) = correspondence_pairs_[i].position;
                corres_pair_ids_[correspondence_pairs_[i].src_idx] = 1;
            }
            update_R();
            energy = welsch_error(nu1, nu2, weight_d_, welsch_weight_s);

            if(pars_.use_AA && energy > prev_energy)
            {
                Smat_X_ = prev_X;
                curV = Weight_PV0 * Smat_X_ + Smat_P_;
                gt_err = SetMeshPoints(src_mesh_, curV);
                FindClosestPoints(correspondence_pairs_);
                Acc.replace(Smat_X_.data());
                prev_energy = 1e12;
                continue;
            }

            prev_energy = energy;
            // update weight
            if(pars_.data_use_welsch)
            {
                welsch_weight(weight_d_, nu1);
            }
            else
                weight_d_.setOnes();

            if(pars_.smooth_use_welsch)
            {
                welsch_weight(welsch_weight_s, nu2);
            }
            else
                welsch_weight_s.setOnes();

            // int welsch_iter;
            int total_inner_iters = 0;

            if(run_once == true && pars_.use_landmark == true)
            {
                weight_d_.setOnes();
                weight_s_.setOnes();
            }


            // update V,U and D
            weight_d_ = weight_d_.cwiseSqrt().cwiseProduct(corres_pair_ids_);
            Weight_PV_ = weight_d_.asDiagonal() * Weight_PV0;
            welsch_weight_s = welsch_weight_s.cwiseSqrt().cwiseProduct(Sweight_s_);
            Smat_B_ = welsch_weight_s.asDiagonal() * Smat_B0;

            Smat_D_ = welsch_weight_s.asDiagonal() * Smat_D0;
            Smat_UP_ = weight_d_.asDiagonal() * (mat_U0_.transpose() - Smat_P_);

            // construct matrix A0 and pre-decompose
            mat_A0_ = 2*(optimize_data * Weight_PV_.adjoint() * Weight_PV_
                    + optimize_reg * Smat_B_.adjoint() * Smat_B_
                    + optimize_rot * Smat_L_);

            // auxiliary variable 4m * 3
            mat_VU_ = 2*optimize_data * (Weight_PV_).transpose() * Smat_UP_
                    + 2*optimize_reg * (Smat_B_).transpose() * Smat_D_;

//            if(pars_.use_landmark)
//            {
//                mat_A0_ += 2*pars_.gamma * Sub_PV_.transpose() * Sub_PV_;
//                mat_VU_ += 2*pars_.gamma * Sub_PV_.transpose() * Sub_UP_;
//            }

            if(run_once)
            {
                ldlt_->analyzePattern(mat_A0_);
                run_once = false;
            }
            ldlt_->factorize(mat_A0_);

            run_time = time.get_time();
            double eps_time = time.elapsed_time(begin_time, run_time);
            pars_.each_times.push_back(eps_time-ignore_time);

            if(!pars_.use_lbfgs)
            {
                MatrixXX b = (2*optimize_rot * Smat_J_ * Smat_R_ + mat_VU_);
#pragma omp parallel for
                for (int col_id = 0; col_id < 3; col_id++)
                {
                    Smat_X_.col(col_id) = ldlt_->solve(b.col(col_id));
                }
                total_inner_iters += 1;
            }
            else
            {
                total_inner_iters += QNSolver(data_err, smooth_err, orth_err);
            }

            if(pars_.use_AA)
            {
                prev_X = Smat_X_;
                Smat_X_ = Eigen::Map<const MatrixXX>(Acc.compute(Smat_X_.data()).data(), 4*num_sample_nodes, 3);
            }

            curV = Weight_PV0 * Smat_X_ + Smat_P_;
            gt_err = SetMeshPoints(src_mesh_, curV);

            pars_.each_gt_mean_errs.push_back(gt_err);
            pars_.each_gt_max_errs.push_back(0);

            pars_.each_energys.push_back(energy);

            pars_.each_iters.push_back(total_inner_iters);
            pars_.each_term_energy.push_back(Vector3(data_err, smooth_err, orth_err));

            if(pars_.print_each_step_info)
            {
                std::string out_obj = pars_.out_each_step_info + "/iter_obj_" + std::to_string(accumulate_iter) + ".ply";

                OpenMesh::IO::write_mesh( *src_mesh_, out_obj.c_str() );
                std::cout << "iter = " << out_iter << " time = " << eps_time
                          << "  inner iter = " << total_inner_iters << std::endl;

                std::string out_weight_d = pars_.out_each_step_info + "/iter_w_data" + std::to_string(accumulate_iter) + ".txt";
                std::ofstream out_w(out_weight_d);
                out_w << weight_d_ << std::endl;
                out_w.close();
            }

            // Find clost points
            FindClosestPoints(correspondence_pairs_);
            accumulate_iter++;
//            SimplePruning(correspondence_pairs_, pars_.use_distance_reject, pars_.use_normal_reject);

            // stop condition should be revised
            if((curV - prevV).rowwise().norm().maxCoeff() < pars_.stop)
            {
                break;
            }
            prevV = curV;
            out_iter++;
        }

        if(fabs(nu1-end_nu1)<1e-8 || !pars_.use_Dynamic_nu || !pars_.data_use_welsch)
            dynamic_stop = true;
        nu1 = (0.5*nu1> end_nu1)?0.5*nu1:end_nu1;
        nu2 *= 0.5;

        if(pars_.data_use_welsch)
        {
            w_data = optimize_data *(2.0*nu1*nu1);
        }
        if(pars_.smooth_use_welsch)
        {
            w_reg = optimize_reg * (2*nu2*nu2);
        }

        reset_num++;

        if(pars_.use_AA)
            Acc.reset(Smat_X_.data());

    }
    return 0;
}


Scalar NonRigidreg::DoNonRigid_2()
{
    Scalar data_err, smooth_err, orth_err;

    MatrixXX curV =  MatrixXX::Zero(n_src_vertex_, 3);
    MatrixXX prevV = MatrixXX::Zero(n_src_vertex_, 3);

    Weight_PV0 = Weight_PV_;
    Smat_B0 = Smat_B_;
    Smat_D0 = Smat_D_;
    // welsch_sweight
    VectorX welsch_weight_s = VectorX::Ones(Sweight_s_.rows(), 1);
    bool run_once = true;

    Timer time;
    Timer::EventID begin_time, run_time;
    pars_.each_energys.clear();
    pars_.each_gt_mean_errs.clear();
    pars_.each_gt_max_errs.clear();
    pars_.each_times.clear();
    pars_.each_iters.clear();
    pars_.each_term_energy.clear();

    // Data term parameters
    Scalar nu1 = pars_.Data_initk * pars_.Data_nu;
    Scalar average_len = CalcEdgelength(src_mesh_, 1);
    Scalar end_nu1 = pars_.Data_endk * average_len;
    Scalar nu2 = pars_.Smooth_nu * average_len;

    while(nu1 > end_nu1)
    {
        nu1 = (0.5*nu1> end_nu1)?0.5*nu1:end_nu1;
        nu2 *= 0.5;
    }

    // Smooth term parameters
    w_data = optimize_data = 1.0;
    w_reg = optimize_reg = pars_.alpha;
    w_rot = optimize_rot = pars_.beta;

    if(pars_.data_use_welsch)
    {
        w_data = optimize_data *(2.0*nu1*nu1);
    }
    if(pars_.smooth_use_welsch)
    {
        w_reg = optimize_reg * (2*nu2*nu2);
    }

    std::cout <<"wdata = " <<w_data << " wreg = " << w_reg << " wrot = " << w_rot
             << " nu1 = " << nu1 << " " << nu2 << " end nu1 = " << end_nu1 << std::endl;

    // print initial run results
    mat_U0_.setZero();
#pragma omp parallel for
    for (size_t i = 0; i < correspondence_pairs_.size(); i++)
    {
        mat_U0_.col(correspondence_pairs_[i].src_idx) = correspondence_pairs_[i].position;
    }
    pars_.each_energys.push_back(0.0);
    pars_.each_gt_max_errs.push_back(pars_.init_gt_max_errs);
    pars_.each_gt_mean_errs.push_back(pars_.init_gt_mean_errs);
    pars_.each_iters.push_back(0);
    pars_.each_times.push_back(pars_.non_rigid_init_time);
    pars_.each_term_energy.push_back(Vector3(0,0,0));

    Scalar gt_err;

    bool dynamic_stop = false;
    int accumulate_iter = 0;
    Scalar ignore_time = 0.0;
    Scalar energy, prev_energy = 1e12;
    int  reset_num = 0;

    AndersonAcceleration Acc;
    Acc.init(pars_.anderson_m, num_sample_nodes*12, Smat_X_.data());
    MatrixXX prev_X = Smat_X_;
    curV = Weight_PV0 * Smat_X_ + Smat_P_;

    begin_time = time.get_time();
    while (!dynamic_stop)
    {
        int out_iter = 0;
        while(out_iter < pars_.max_outer_iters)
        {
            // according correspondence_pairs to update mat_U0_;
            mat_U0_.setZero();
            corres_pair_ids_.setZero();
#pragma omp parallel for
            for (size_t i = 0; i < correspondence_pairs_.size(); i++)
            {
                mat_U0_.col(correspondence_pairs_[i].src_idx) = correspondence_pairs_[i].position;
                corres_pair_ids_[correspondence_pairs_[i].src_idx] = 1;
            }
            update_R();

            energy = welsch_error(nu1, nu2, weight_d_, welsch_weight_s);

            if(pars_.use_AA && energy > prev_energy)
            {
                Smat_X_ = prev_X;
                curV = Weight_PV0 * Smat_X_ + Smat_P_;
                gt_err = SetMeshPoints(src_mesh_, curV);
                FindClosestPoints(correspondence_pairs_);
                Acc.replace(Smat_X_.data());
                prev_energy = 1e12;
                continue;
            }

            prev_energy = energy;
            // update weight
            if(pars_.data_use_welsch)
            {
                welsch_weight(weight_d_, nu1);
            }
            else
                weight_d_.setOnes();

            if(pars_.smooth_use_welsch)
            {
                welsch_weight(welsch_weight_s, nu2);
            }
            else
                welsch_weight_s.setOnes();

            // int welsch_iter;
            int total_inner_iters = 0;

            if(run_once == true && pars_.use_landmark == true)
            {
                weight_d_.setOnes();
                weight_s_.setOnes();
            }


            // update V,U and D
            weight_d_ = weight_d_.cwiseSqrt().cwiseProduct(corres_pair_ids_);
            Weight_PV_ = weight_d_.asDiagonal() * Weight_PV0;
            welsch_weight_s = welsch_weight_s.cwiseSqrt().cwiseProduct(Sweight_s_);
//            welsch_weight_s.setOnes();
            Smat_B_ = welsch_weight_s.asDiagonal() * Smat_B0;

            Smat_D_ = welsch_weight_s.asDiagonal() * Smat_D0;
            Smat_UP_ = weight_d_.asDiagonal() * (mat_U0_.transpose() - Smat_P_);

            // construct matrix A0 and pre-decompose
            mat_A0_ = 2*(optimize_data * Weight_PV_.adjoint() * Weight_PV_
                    + optimize_reg * Smat_B_.adjoint() * Smat_B_
                    + optimize_rot * Smat_L_);

            // auxiliary variable 4m * 3
            mat_VU_ = 2*optimize_data * (Weight_PV_).transpose() * Smat_UP_
                    + 2*optimize_reg * (Smat_B_).transpose() * Smat_D_;

            if(pars_.use_landmark)
            {
                mat_A0_ += 2*pars_.gamma * Sub_PV_.transpose() * Sub_PV_;
                mat_VU_ += 2*pars_.gamma * Sub_PV_.transpose() * Sub_UP_;
            }

            if(run_once)
            {
                ldlt_->analyzePattern(mat_A0_);
                run_once = false;
            }
            ldlt_->factorize(mat_A0_);

            run_time = time.get_time();
            double eps_time = time.elapsed_time(begin_time, run_time);
            pars_.each_times.push_back(eps_time-ignore_time);

            if(!pars_.use_lbfgs)
            {
                MatrixXX b = (2*optimize_rot * Smat_J_ * Smat_R_ + mat_VU_);
#pragma omp parallel for
                for (int col_id = 0; col_id < 3; col_id++)
                {
                    Smat_X_.col(col_id) = ldlt_->solve(b.col(col_id));
                }
                total_inner_iters += 1;
            }
            else
            {
                total_inner_iters += QNSolver(data_err, smooth_err, orth_err);
            }

            if(pars_.use_AA)
            {
                prev_X = Smat_X_;
                Smat_X_ = Eigen::Map<const MatrixXX>(Acc.compute(Smat_X_.data()).data(), 4*num_sample_nodes, 3);
            }

            curV = Weight_PV0 * Smat_X_ + Smat_P_;
            gt_err = SetMeshPoints(src_mesh_, curV);

            pars_.each_gt_mean_errs.push_back(gt_err);
            pars_.each_gt_max_errs.push_back(0);

            pars_.each_energys.push_back(energy);

            pars_.each_iters.push_back(total_inner_iters);
            pars_.each_term_energy.push_back(Vector3(data_err, smooth_err, orth_err));

            if(pars_.print_each_step_info)
            {
                std::string out_obj = pars_.out_each_step_info + "/iter_obj_" + std::to_string(accumulate_iter) + ".ply";

                OpenMesh::IO::write_mesh( *src_mesh_, out_obj.c_str() );
                std::cout << "iter = " << out_iter << " time = " << eps_time
                          << "  inner iter = " << total_inner_iters << std::endl;

                std::string out_weight_d = pars_.out_each_step_info + "/iter_w_data" + std::to_string(accumulate_iter) + ".txt";
                std::ofstream out_w(out_weight_d);
                out_w << weight_d_ << std::endl;
                out_w.close();
            }

            // Find clost points
            FindClosestPoints(correspondence_pairs_);
            accumulate_iter++;
//            SimplePruning(correspondence_pairs_, pars_.use_distance_reject, pars_.use_normal_reject);

            // stop condition
            if((curV - prevV).rowwise().norm().maxCoeff() < pars_.stop)
            {
                break;
            }
            prevV = curV;
            out_iter++;
        }

        if(fabs(nu1-end_nu1)<1e-8 || !pars_.use_Dynamic_nu || !pars_.data_use_welsch)
            dynamic_stop = true;
        nu1 = (0.5*nu1> end_nu1)?0.5*nu1:end_nu1;
        nu2 *= 0.5;
        reset_num++;

        if(pars_.use_AA)
            Acc.reset(Smat_X_.data());

    }
    return 0;
}



Scalar NonRigidreg::DoNonRigid_l2r()
{
    Scalar data_err, smooth_err, orth_err;

    MatrixXX curV =  MatrixXX::Zero(n_src_vertex_, 3);
    MatrixXX prevV = MatrixXX::Zero(n_src_vertex_, 3);

    Weight_PV0 = Weight_PV_;
    Smat_B0 = Smat_B_;
    Smat_D0 = Smat_D_;
    // welsch_sweight
    VectorX welsch_weight_s = VectorX::Ones(Sweight_s_.rows(), 1);
    bool run_once = true;

    Timer time;
    Timer::EventID begin_time, run_time;
    pars_.each_energys.clear();
    pars_.each_gt_mean_errs.clear();
    pars_.each_gt_max_errs.clear();
    pars_.each_times.clear();
    pars_.each_iters.clear();
    pars_.each_term_energy.clear();

    // Data term parameters
    Scalar nu1 = pars_.Data_initk * pars_.Data_nu;
    Scalar average_len = CalcEdgelength(src_mesh_, 1);
    Scalar end_nu1 = pars_.Data_endk * average_len;
    Scalar nu2 = pars_.Smooth_nu * average_len;

    pars_.distance_threshold = 3.0*nu1;
    pars_.normal_threshold = 3.0*nu2;
    // Smooth term parameters
    w_data = optimize_data = 1.0;
    w_reg = optimize_reg = pars_.alpha;
    w_rot = optimize_rot = pars_.beta;

    if(pars_.data_use_welsch)
    {
        w_data = optimize_data *(2.0*nu1*nu1);
    }
    if(pars_.smooth_use_welsch)
    {
        w_reg = optimize_reg * (2*nu2*nu2);
    }

    std::cout <<"wdata = " <<w_data << " wreg = " << w_reg << " wrot = " << w_rot
             << " nu1 = " << nu1 << " " << nu2 << " end nu1 = " << end_nu1 << std::endl;

    // print initial run results
    mat_U0_.setZero();
#pragma omp parallel for
    for (size_t i = 0; i < correspondence_pairs_.size(); i++)
    {
        mat_U0_.col(correspondence_pairs_[i].src_idx) = correspondence_pairs_[i].position;
    }
    pars_.each_energys.push_back(0.0);
    pars_.each_gt_max_errs.push_back(pars_.init_gt_max_errs);
    pars_.each_gt_mean_errs.push_back(pars_.init_gt_mean_errs);
    pars_.each_iters.push_back(0);
    pars_.each_times.push_back(pars_.non_rigid_init_time);
    pars_.each_term_energy.push_back(Vector3(0,0,0));

    Scalar gt_err;

    bool dynamic_stop = false;
    int accumulate_iter = 0;
    Scalar ignore_time = 0.0;
    Scalar energy, prev_energy = 1e12;
    int  reset_num = 0;

    AndersonAcceleration Acc;
    Acc.init(pars_.anderson_m, num_sample_nodes*12, Smat_X_.data());
    MatrixXX prev_X = Smat_X_;
    curV = Weight_PV0 * Smat_X_ + Smat_P_;

    begin_time = time.get_time();
    while (!dynamic_stop)
    {
        int out_iter = 0;
        while(out_iter < pars_.max_outer_iters)
        {
            // according correspondence_pairs to update mat_U0_;
            mat_U0_.setZero();
            corres_pair_ids_.setZero();
#pragma omp parallel for
            for (size_t i = 0; i < correspondence_pairs_.size(); i++)
            {
                mat_U0_.col(correspondence_pairs_[i].src_idx) = correspondence_pairs_[i].position;
                corres_pair_ids_[correspondence_pairs_[i].src_idx] = 1;
            }
            update_R();

            energy = welsch_error(nu1, nu2, weight_d_, welsch_weight_s);

            if(pars_.use_AA && energy > prev_energy)
            {
                Smat_X_ = prev_X;
                curV = Weight_PV0 * Smat_X_ + Smat_P_;
                gt_err = SetMeshPoints(src_mesh_, curV);
                FindClosestPoints(correspondence_pairs_);
                Acc.replace(Smat_X_.data());
                prev_energy = 1e12;
                continue;
            }
            prev_energy = energy;
            // update weight
            if(pars_.data_use_welsch)
            {
                welsch_weight(weight_d_, nu1);
            }
            else
                weight_d_.setOnes();

            if(pars_.smooth_use_welsch)
            {
                welsch_weight(welsch_weight_s, nu2);
            }
            else
                welsch_weight_s.setOnes();

            // int welsch_iter;
            int total_inner_iters = 0;

            if(run_once == true && pars_.use_landmark == true)
            {
                weight_d_.setOnes();
                weight_s_.setOnes();
            }


            // update V,U and D
            weight_d_ = weight_d_.cwiseSqrt().cwiseProduct(corres_pair_ids_);
            Weight_PV_ = weight_d_.asDiagonal() * Weight_PV0;
            welsch_weight_s = welsch_weight_s.cwiseSqrt().cwiseProduct(Sweight_s_);
            Smat_B_ = welsch_weight_s.asDiagonal() * Smat_B0;

            Smat_D_ = welsch_weight_s.asDiagonal() * Smat_D0;
            Smat_UP_ = weight_d_.asDiagonal() * (mat_U0_.transpose() - Smat_P_);

            // construct matrix A0 and pre-decompose
            mat_A0_ = 2*(optimize_data * Weight_PV_.adjoint() * Weight_PV_
                    + optimize_reg * Smat_B_.adjoint() * Smat_B_
                    + optimize_rot * Smat_L_);

            // auxiliary variable 4m * 3
            mat_VU_ = 2*optimize_data * (Weight_PV_).transpose() * Smat_UP_
                    + 2*optimize_reg * (Smat_B_).transpose() * Smat_D_;

//            if(pars_.use_landmark)
//            {
//                mat_A0_ += 2*pars_.gamma * Sub_PV_.transpose() * Sub_PV_;
//                mat_VU_ += 2*pars_.gamma * Sub_PV_.transpose() * Sub_UP_;
//            }

            if(run_once)
            {
                ldlt_->analyzePattern(mat_A0_);
                run_once = false;
            }
            ldlt_->factorize(mat_A0_);

            run_time = time.get_time();
            double eps_time = time.elapsed_time(begin_time, run_time);
            pars_.each_times.push_back(eps_time-ignore_time);

            if(!pars_.use_lbfgs)
            {
                MatrixXX b = (2*optimize_rot * Smat_J_ * Smat_R_ + mat_VU_);
#pragma omp parallel for
                for (int col_id = 0; col_id < 3; col_id++)
                {
                    Smat_X_.col(col_id) = ldlt_->solve(b.col(col_id));
                }
                total_inner_iters += 1;
            }
            else
            {
                total_inner_iters += QNSolver(data_err, smooth_err, orth_err);
            }

            if(pars_.use_AA)
            {
                prev_X = Smat_X_;
                Smat_X_ = Eigen::Map<const MatrixXX>(Acc.compute(Smat_X_.data()).data(), 4*num_sample_nodes, 3);
            }

            curV = Weight_PV0 * Smat_X_ + Smat_P_;
            gt_err = SetMeshPoints(src_mesh_, curV);

            pars_.each_gt_mean_errs.push_back(gt_err);
            pars_.each_gt_max_errs.push_back(0);

            pars_.each_energys.push_back(energy);

            pars_.each_iters.push_back(total_inner_iters);
            pars_.each_term_energy.push_back(Vector3(data_err, smooth_err, orth_err));

            if(pars_.print_each_step_info)
            {
                std::string out_obj = pars_.out_each_step_info + "/iter_obj_" + std::to_string(accumulate_iter) + ".ply";

                OpenMesh::IO::write_mesh( *src_mesh_, out_obj.c_str() );
                std::cout << "iter = " << out_iter << " time = " << eps_time
                          << "  inner iter = " << total_inner_iters << std::endl;

                std::string out_weight_d = pars_.out_each_step_info + "/iter_w_data" + std::to_string(accumulate_iter) + ".txt";
                std::ofstream out_w(out_weight_d);
                out_w << weight_d_ << std::endl;
                out_w.close();
            }

            // Find clost points
            FindClosestPoints(correspondence_pairs_);
            accumulate_iter++;
            SimplePruning(correspondence_pairs_, pars_.use_distance_reject, pars_.use_normal_reject);

            // stop condition 
            if((curV - prevV).rowwise().norm().maxCoeff() < pars_.stop)
            {
                break;
            }
            prevV = curV;
            out_iter++;
        }

        if(fabs(nu1-end_nu1)<1e-8 || !pars_.use_Dynamic_nu || !pars_.data_use_welsch)
            dynamic_stop = true;
        nu1 = (0.5*nu1> end_nu1)?0.5*nu1:end_nu1;
        nu2 *= 0.5;

        if(pars_.data_use_welsch)
        {
            w_data = optimize_data *(2.0*nu1*nu1);
        }
        if(pars_.smooth_use_welsch)
        {
            w_reg = optimize_reg * (2*nu2*nu2);
        }

        reset_num++;

        if(pars_.use_AA)
            Acc.reset(Smat_X_.data());

    }
    return 0;
}

Scalar NonRigidreg::sample_energy(Scalar& data_err, Scalar& smooth_err, Scalar& orth_err)
{
    data_err =  ( Weight_PV_ * Smat_X_  - Smat_UP_).squaredNorm();
    smooth_err = ( (Smat_B_ * Smat_X_ - Smat_D_)).squaredNorm();
    VectorX orth_errs(num_sample_nodes);
#pragma omp parallel for
    for (int i = 0; i < num_sample_nodes; i++)
    {
        Eigen::JacobiSVD<MatrixXX> svd(Smat_X_.block(4 * i, 0, 3, 3), Eigen::ComputeThinU | Eigen::ComputeThinV);

        if (svd.matrixU().determinant()*svd.matrixV().determinant() < 0.0) {
            Vector3 S = Vector3::Ones(); S(2) = -1.0;
            Smat_R_.block(3 * i, 0, 3, 3) = svd.matrixU()*S.asDiagonal()*svd.matrixV().transpose();
        }
        else {
            Smat_R_.block(3 * i, 0, 3, 3) = svd.matrixU()*svd.matrixV().transpose();
        }
        orth_errs[i] = (Smat_X_.block(4 * i, 0, 3, 3) - Smat_R_.block(3 * i, 0, 3, 3)).squaredNorm();
    }
    orth_err = orth_errs.sum();
    Scalar total_err = optimize_data * data_err + optimize_reg * smooth_err + optimize_rot * orth_err;
    if(pars_.use_landmark)
        total_err += pars_.gamma * ( Sub_PV_ * Smat_X_  - Sub_UP_).squaredNorm();
    return total_err;
}

void NonRigidreg::update_R()
{
#pragma omp parallel for
    for (int i = 0; i < num_sample_nodes; i++)
    {
        Eigen::JacobiSVD<MatrixXX> svd(Smat_X_.block(4 * i, 0, 3, 3), Eigen::ComputeThinU | Eigen::ComputeThinV);
        if (svd.matrixU().determinant()*svd.matrixV().determinant() < 0.0) {
            Vector3 S = Vector3::Ones(); S(2) = -1.0;
            Smat_R_.block(3 * i, 0, 3, 3) = svd.matrixU()*S.asDiagonal()*svd.matrixV().transpose();
        }
        else {
            Smat_R_.block(3 * i, 0, 3, 3) = svd.matrixU()*svd.matrixV().transpose();
        }
    }
}

void NonRigidreg::sample_gradient()
{
    grad_X_ = mat_A0_ * Smat_X_ - (mat_VU_ + 2*optimize_rot * Smat_J_ * Smat_R_);
}

// col vector
void NonRigidreg::LBFGS(int iter, MatrixXX & dir) const
{
    VectorX rho(pars_.lbfgs_m);
    VectorX kersi(pars_.lbfgs_m);
    MatrixXX q(4 * num_sample_nodes, 3);
    MatrixXX temp(4 * num_sample_nodes, 3);
    int k = iter;
    q.setZero();
    dir = q;
    q = -grad_X_;
    int m_k = std::min(k, pars_.lbfgs_m);
    for (int i = k - 1; i > k - m_k - 1; i--)
    {
        int col = (pars_.lbfgs_m + col_idx_ - (k - 1 - i)) % pars_.lbfgs_m;
        rho(k - 1 - i) = all_t_.col(col).transpose().dot(all_s_.col(col));
        Scalar lbfgs_err_scalar = Eigen::Map<VectorX>(q.data(), q.size()).dot(all_s_.col(col));
        kersi(k - 1 - i) = lbfgs_err_scalar / rho(k - 1 - i);
        Eigen::Map<VectorX>(q.data(), q.size()) -= kersi(k - 1 - i) * all_t_.col(col);
    }
#pragma omp parallel for
    for(int cid = 0; cid < 3; cid++)
    {
        dir.col(cid) = ldlt_->solve(q.col(cid));
    }

    for (int i = k - m_k; i < k; i++)
    {
        int col = (pars_.lbfgs_m + col_idx_ - (k - 1 - i)) % pars_.lbfgs_m;
        Scalar lbfgs_err_scalar = all_t_.col(col).dot(Eigen::Map<VectorX>(dir.data(), dir.size()));
        Scalar eta = kersi(k - 1 - i) - lbfgs_err_scalar / rho(k - 1 - i);
        Eigen::Map<VectorX>(dir.data(), dir.size()) += all_s_.col(col) * eta;
    }

    rho.resize(0);
    kersi.resize(0);
    q.resize(0, 0);
    temp.resize(0, 0);
    return;
}


int NonRigidreg::QNSolver(Scalar& data_err, Scalar& smooth_err, Scalar& orth_err)
{
    MatrixXX prev_X;
    int count_linesearch = 0;

    int iter;
    for (iter = 0; iter < pars_.max_inner_iters; iter++)
    {
        //EvalGradient();
        sample_gradient();

        // update decent direction
        if (iter == 0)
        {
            MatrixXX temp = -grad_X_;
#pragma omp parallel for
            for(int cid = 0; cid < 3; cid++)
            {
                direction_.col(cid) = ldlt_->solve(temp.col(cid));
            }

            col_idx_ = 0;
            all_s_.col(col_idx_) = -Eigen::Map<VectorX>(Smat_X_.data(), 4 * num_sample_nodes * 3);
            all_t_.col(col_idx_) = -Eigen::Map<VectorX>(grad_X_.data(), 4 * num_sample_nodes * 3);
        }
        else
        {
            all_s_.col(col_idx_) += Eigen::Map<VectorX>(Smat_X_.data(), 4 * num_sample_nodes * 3);
            all_t_.col(col_idx_) += Eigen::Map<VectorX>(grad_X_.data(), 4 * num_sample_nodes * 3);

            // Get descent direction
            LBFGS(iter, direction_);
            col_idx_ = (col_idx_ + 1) % pars_.lbfgs_m;
            all_s_.col(col_idx_) = -Eigen::Map<VectorX>(Smat_X_.data(), 4 * num_sample_nodes * 3);
            all_t_.col(col_idx_) = -Eigen::Map<VectorX>(grad_X_.data(), 4 * num_sample_nodes * 3);
        }

        Scalar alpha = 2.0;
        prev_X = Smat_X_;
        Scalar new_err = sample_energy(data_err, smooth_err, orth_err);
        Scalar prev_err = new_err;
        Scalar gamma = 0.3;
        Scalar x = (grad_X_.transpose() * direction_).trace();
        do
        {
            alpha /= 2.0;
            Smat_X_ = prev_X + alpha * direction_;
            new_err = sample_energy(data_err, smooth_err, orth_err);
            count_linesearch++;
        } while (new_err > prev_err + gamma * alpha * x);
        if (fabs(new_err - prev_err) < pars_.stop_inner)
        {
            break;
        }
        iter_++;
    }
    return iter;
}

Scalar NonRigidreg::welsch_error(Scalar nu1, Scalar nu2, VectorX & data_errs, VectorX & reg_errs)
{
    data_errs.resize(n_src_vertex_);
    for(size_t i = 0; i < correspondence_pairs_.size(); i++)
    {
        Scalar err = (src_mesh_->point(src_mesh_->vertex_handle(correspondence_pairs_[i].src_idx))
                - tar_mesh_->point(tar_mesh_->vertex_handle(correspondence_pairs_[i].tar_idx))).norm();
        data_errs[i] = err*err;
    }
    Scalar data_err;
    if(pars_.data_use_welsch)
        data_err = welsch_energy(data_errs, nu1);
    else
        data_err = data_errs.sum();
    reg_errs = ((Smat_B0 * Smat_X_ - Smat_D0)).rowwise().squaredNorm();
    Scalar smooth_err;
    if(pars_.smooth_use_welsch)
        smooth_err = welsch_energy(reg_errs, nu2);
    else
        smooth_err = reg_errs.sum();

    VectorX orth_errs(num_sample_nodes);
#pragma omp parallel for
    for (int i = 0; i < num_sample_nodes; i++)
    {
        orth_errs[i] = (Smat_X_.block(4 * i, 0, 3, 3) - Smat_R_.block(3 * i, 0, 3, 3)).squaredNorm();
    }
    return w_data * data_err + w_reg * smooth_err + w_rot * orth_errs.sum();
}

Scalar NonRigidreg::welsch_energy(VectorX& r, Scalar p) {
    VectorX energy(r.rows());
#pragma omp parallel for
    for (int i = 0; i<r.rows(); ++i) {
        energy[i] = 1.0 - std::exp(-r(i) / (2 * p*p));
    }
    return energy.sum();
}

void NonRigidreg::welsch_weight(VectorX& r, Scalar p) {
#pragma omp parallel for
    for (int i = 0; i<r.rows(); ++i) {
        r(i) = std::exp(-r(i)/**r(i)*/ / (2 * p*p));
    }
}

Scalar NonRigidreg::SetMeshPoints(Mesh* mesh, const MatrixXX & target)
{
    VectorX gt_errs(n_src_vertex_);
#pragma omp parallel for
    for (int i = 0; i < n_src_vertex_; i++)
    {
        MatrixXX tar_p = target.block(i, 0, 1, 3);
        Vec3 p(tar_p(0, 0), tar_p(0, 1), tar_p(0, 2));
        mesh->set_point(mesh->vertex_handle(i), p);
        if(pars_.calc_gt_err)
            gt_errs[i] = (tar_p.transpose() - tar_points_.col(i)).squaredNorm();
    }
    if(pars_.calc_gt_err)
        return gt_errs.sum()/n_src_vertex_;
    else
        return -1.0;
}
