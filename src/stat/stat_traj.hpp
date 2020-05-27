//
// Created by Andy Wang
//

#ifndef SFERES2_STAT_TRAJ_HPP
#define SFERES2_STAT_TRAJ_HPP

#include <sferes/stat/stat.hpp>

namespace sferes {
    namespace stat {

        SFERES_STAT(Trajectories, Stat)
        {
        public:
            typedef Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> matrix_t;

            template<typename EA>
            void refresh(EA &ea) 
            {
                if ((ea.gen() % Params::stat::save_trajectories == 0) || (ea.gen() == 1) ) 
                {
                   std::string prefix = "traj_" + add_leading_zeros(ea.gen());
                    _write_container(prefix, ea);
                }
            }

            template<typename EA>
            void _write_container(const std::string &prefix, const EA &ea) const {

                std::string fname = ea.res_dir() + "/" + prefix + std::string(".dat");
                std::cout << "writing..." << fname << std::endl;

                // retrieve all phenotypes and trajectories                
                matrix_t phen, traj;
                std::vector<int> is_traj;
                boost::fusion::at_c<0>(ea.fit_modifier()).get_phen(ea.pop(), phen);
                boost::fusion::at_c<0>(ea.fit_modifier()).get_trajectories(ea.pop(), traj, is_traj);
                
                // filter out the realised trajectories
                matrix_t filtered_traj;
                std::vector<bool> boundaries;
                Eigen::VectorXi is_trajectory;
                boost::fusion::at_c<0>(ea.fit_modifier()).get_network_loader()->vector_to_eigen(is_traj, is_trajectory);
                boost::fusion::at_c<0>(ea.fit_modifier()).get_network_loader()->filter_trajectories(traj, is_trajectory, filtered_traj, boundaries);
                
                // get the reconstruction
                matrix_t reconstruction;
                boost::fusion::at_c<0>(ea.fit_modifier()).get_reconstruction(phen, traj, is_traj, reconstruction);

                std::ofstream ofs(fname.c_str());
                ofs.precision(17);
                Eigen::IOFormat CommaInitFmt(Eigen::StreamPrecision, Eigen::DontAlignCols, ", ", ", ", "", "");

                // std::cout << "pop" << ea.pop().size() << std::endl;
                // std::cout << "traj" << filtered_traj.rows() << ",c " << filtered_traj.cols() << std::endl;
                // std::cout << "reconstruction" << reconstruction.rows() << ",c " << reconstruction.cols() << std::endl;

                // hack to make the do while loop below work
                boundaries.push_back(true);
            
                // there are more trajectories than reconstructions as there is only one recon per phen
                size_t traj_index = 0;
                ofs << "FORMAT: INDIV_INDEX, RECON/ACTUAL, DATA\n";
                for (int i{0}; i < reconstruction.rows(); ++i)
                {
                    ofs << i << ", RECON," <<  reconstruction.row(i).format(CommaInitFmt) << std::endl;
                    do
                    {
                        ofs << i << ", ACTUAL," <<  filtered_traj.row(traj_index).format(CommaInitFmt) << std::endl;
                        ++traj_index;
                    }
                    while (!boundaries[traj_index]);
                }
            }


        };

    }
}


#endif
