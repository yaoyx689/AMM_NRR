#include "tools/io_mesh.h"
#include "tools/OmpHelper.h"
#include "NonRigidreg.h"


int main(int argc, char **argv)
{
    Mesh src_mesh;
    Mesh tar_mesh;
    std::string src_file;
    std::string tar_file;
    std::string out_file, outpath;
    std::string landmark_file;
    RegParas paras;
    Scalar input_alpha = 100;
    Scalar input_beta = 10;
    Scalar input_nu_d = 1;
    Scalar input_nu_r = 3;
    Scalar input_radius = 5;

    paras.use_distance_reject = false;
    paras.distance_threshold = 0.05;
    paras.use_normal_reject = false;
    paras.normal_threshold = M_PI/3;


    enum METHOD{l2, fixmax, fixmin, dy, qns, l2_r} method = dy;
    if(argc==4)
    {
        src_file = argv[1];
        tar_file = argv[2];
        outpath = argv[3];  
    }
    else if(argc==5)
    {
        src_file = argv[1];
        tar_file = argv[2];
        outpath = argv[3];
        landmark_file = argv[4];
        paras.use_landmark = true;
    }
    else if(argc==7)
    {
        src_file = argv[1];
        tar_file = argv[2];
        outpath = argv[3];
        input_radius = std::stod(argv[4]);
        input_alpha = std::stod(argv[5]);
        input_beta = std::stod(argv[6]);
    }
    else
    {
        std::cout << "Usage: <srcFile> <tarFile> <outPath>\n    or <srcFile> <tarFile> <outPath> <landmarkFile>" << std::endl;
        std::cout << "    or <srcFile> <tarFile> <outPath> <radius> <alpha> <beta>" << std::endl;
        exit(0);
    }

    paras.stop = 1e-5;
    paras.stop_inner = 1e-3;
    paras.max_outer_iters = 100;
    paras.use_AA = true;
    paras.use_lbfgs = false;
    paras.use_Dynamic_nu = false;
    paras.max_inner_iters = 1;

    switch (method) {
    case l2:
    {
        paras.data_use_welsch = false;
        paras.smooth_use_welsch = false;
        std::cout << "USE L2-norm" << std::endl;
        break;
    }
    case fixmax:
    {
        paras.data_use_welsch = true;
        paras.smooth_use_welsch = true;
        std::cout << "USE Welsch fix max" << std::endl;
        break;
    }
    case fixmin:
    {
        paras.data_use_welsch = true;
        paras.smooth_use_welsch = true;
        std::cout << "USE Welsch fix min" << std::endl;
        break;
    }
    case dy:
    {
        paras.data_use_welsch = true;
        paras.smooth_use_welsch = true;
        paras.use_Dynamic_nu = true;
        std::cout << "USE Welsch dynamic" << std::endl;
        break;
    }
    case qns:
    {
        paras.use_AA = false;
        paras.use_lbfgs = true;
        paras.use_Dynamic_nu = true;
        paras.max_inner_iters = 20;
        std::cout << "USE Welsch qns" << std::endl;
        break;
    }
    case l2_r:
    {
        paras.data_use_welsch = false;
        paras.smooth_use_welsch = false;
        paras.use_AA = false;
        paras.use_distance_reject = true;
        paras.use_normal_reject = false;
        std::cout << "USE L2-norm with rejection" << std::endl;
        break;
    }
    }
    paras.use_AA = false;

    std::cout << "data use welsc = " << paras.data_use_welsch
              << "\nreg_use welsch = " << paras.smooth_use_welsch
              << "\nuse_AA = " << paras.use_AA
              << "\nuse_lbfgs = " << paras.use_lbfgs
              << "\nuse_dynamic = " << paras.use_Dynamic_nu << std::endl;

    // Setting paras
    paras.alpha = input_alpha;
    paras.beta = input_beta;
    paras.gamma = 1e8;

    paras.rigid_iters = 0;
    paras.anderson_m = 5;

    paras.Data_initk = input_nu_d;
    paras.Smooth_nu = input_nu_r;

    paras.calc_gt_err = true;
    paras.uni_sample_radio = input_radius;

    std::cout << "uni_sample_radio = " << paras.uni_sample_radio << std::endl;
    paras.print_each_step_info = false;

    paras.out_gt_file = outpath + "res.txt";
    out_file = outpath + "res.obj";

    read_data(src_file, src_mesh);
    read_data(tar_file, tar_mesh);
    if(src_mesh.n_vertices()==0 || tar_mesh.n_vertices()==0)
        exit(0);

    if(src_mesh.n_vertices() != tar_mesh.n_vertices())
        paras.calc_gt_err = false;

    if(paras.use_landmark)
        read_landmark(landmark_file.c_str(), paras.landmark_src, paras.landmark_tar);
    double scale = mesh_scaling(src_mesh, tar_mesh);
    std::cout << "scale = " << scale << std::endl;

    NonRigidreg* reg;
    reg = new NonRigidreg;

    Timer time;
    std::cout << "\nrigid registration to initial..." << std::endl;
    Timer::EventID time1 = time.get_time();
    reg->rigid_init(src_mesh, tar_mesh, paras);
    reg->DoRigid();
    Timer::EventID time2 = time.get_time();
    std::cout << "rgid registration... " << std::endl;
    // non-rigid initialize
    std::cout << "non-rigid registration to initial..." << std::endl;
    Timer::EventID time3 = time.get_time();
    reg->Initialize();
    Timer::EventID time4 = time.get_time();
    reg->pars_.non_rigid_init_time = time.elapsed_time(time3, time4);
    std::cout << "non-rigid registration... " << std::endl;

    if(method == fixmin)
    {
        reg->DoNonRigid_2();
    }
    else if(method == l2_r)
        reg->DoNonRigid_l2r();
    else
        reg->DoNonRigid();

    Timer::EventID time5 = time.get_time();

    std::cout << "Registration done!\nrigid_init time : "
              << time.elapsed_time(time1, time2) << " s \trigid-reg run time = " << time.elapsed_time(time2, time3)
              << " s \nnon-rigid init time = "
              << time.elapsed_time(time3, time4) << " s \tnon-rigid run time = "
              << time.elapsed_time(time4, time5) << " s\n" << std::endl;
    write_data(out_file.c_str(), src_mesh, scale);

    std::ofstream out(outpath + "_err.txt");
    for(size_t i = 0; i < reg->pars_.each_times.size(); i++)
    {
        out << reg->pars_.each_iters[i] << " " << reg->pars_.each_times[i] << " "
            << reg->pars_.each_energys[i] << " " << reg->pars_.each_gt_mean_errs[i] << std::endl;
    }
    out.close();

    std::cout << "#node number = " << reg->pars_.num_sample_nodes << std::endl;
    std::cout<< "write result to " << out_file << std::endl;
    delete reg;

    return 0;
}
