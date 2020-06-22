//
// Created by Andy Wang
//

#ifndef SFERES2_STAT_AE_LOSSES_HPP
#define SFERES2_STAT_AE_LOSSES_HPP

#include <sferes/stat/stat.hpp>

namespace sferes {
    namespace stat {

        SFERES_STAT(Losses, Stat)
        {
        public:
            typedef Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> matrix_t;

            template<typename EA>
            void refresh(EA &ea) 
            {
                std::string prefix = "ae_loss";
                _write_losses(prefix, ea);
            }
            

            template<typename EA>
            void _write_losses(const std::string &prefix, const EA &ea) {

                std::string fname = ea.res_dir() + "/" + prefix + std::string(".dat");
                std::cout << "writing... " << fname << std::endl;

                // retrieve all phenotypes and trajectories                
                matrix_t gen, img;
                boost::fusion::at_c<0>(ea.fit_modifier()).get_geno(ea.pop(), gen);
                boost::fusion::at_c<0>(ea.fit_modifier()).get_image(ea.pop(), img);
                
                matrix_t descriptors, recon_loss, recon_loss_unred, reconstruction, L2_loss, KL_loss, decoder_var;
                boost::fusion::at_c<0>(ea.fit_modifier()).get_stats(gen, img, descriptors, reconstruction, recon_loss, recon_loss_unred, 
                                                                    L2_loss, KL_loss, decoder_var);

                std::ofstream ofs(fname.c_str(), std::ofstream::app);
                ofs.precision(17);
                double recon = recon_loss.mean();

                #ifdef VAE
                // these three are unreduced, need row wise sum and then mean
                double L2 = L2_loss.rowwise().sum().mean();
                double KL = KL_loss.rowwise().sum().mean();
                double var = decoder_var.rowwise().sum().mean();

                // retrieve images without any interference from random observations
                matrix_t undisturbed_images(ea.pop().size(), Params::nov::discretisation * Params::nov::discretisation);
                for (size_t i{0}; i < ea.pop().size(); ++i)
                {undisturbed_images.row(i) = ea.pop()[i]->fit().get_undisturbed_image();}
                
                double L2_undisturbed = (undisturbed_images - reconstruction).array().square().rowwise().sum().mean();

                ofs << ea.gen() << ", " << recon << ", " << KL << ", " << var << ", " << L2 << ", " << L2_undisturbed;
                #else

                #ifdef AURORA
                ofs << ea.gen() << ", " << recon;
                #else // AE

                matrix_t undisturbed_images(ea.pop().size(), Params::nov::discretisation * Params::nov::discretisation);
                for (size_t i{0}; i < ea.pop().size(); ++i)
                {undisturbed_images.row(i) = ea.pop()[i]->fit().get_undisturbed_image();}
                
                double L2_undisturbed = (undisturbed_images - reconstruction).array().square().rowwise().sum().mean();

                ofs << ea.gen() << ", " << recon << ", " << L2_undisturbed;
                #endif 

                #endif

                if (boost::fusion::at_c<0>(ea.fit_modifier()).is_train_gen())
                {
                    #ifndef AURORA
                    ofs << ", " << boost::fusion::at_c<0>(ea.fit_modifier()).get_random_extension_ratio() << ", " << boost::fusion::at_c<0>(ea.fit_modifier()).get_network_loader()->get_epochs_trained() << "/" << Params::ae::nb_epochs << ", IS_TRAIN";
                    #else
                    ofs << ", " << boost::fusion::at_c<0>(ea.fit_modifier()).get_network_loader()->get_epochs_trained() << "/" << Params::ae::nb_epochs << ", IS_TRAIN";
                    #endif
                }
                ofs << "\n";
            }
        };

    }
}


#endif
