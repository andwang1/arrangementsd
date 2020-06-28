#ifndef __NETWORK__LOADER__HPP__
#define __NETWORK__LOADER__HPP__

#include <sferes/misc/rand.hpp>

#include <chrono>
#include <iomanip>
#include <tuple>

#ifdef VAE
#include "autoencoder/autoencoder_VAE.hpp"
#include "autoencoder/encoder_VAE.hpp"
#include "autoencoder/decoder_VAE.hpp"
#else
#include "autoencoder/autoencoder_AE.hpp"
#ifdef AURORA
#include "autoencoder/encoder_AE_AURORA.hpp"
#else
#include "autoencoder/encoder_AE.hpp"
#endif
#include "autoencoder/decoder_AE.hpp"
#endif

template <typename TParams, typename Exact = stc::Itself>
class AbstractLoader : public stc::Any<Exact> {
public:
    typedef Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> MatrixXf_rm;

    explicit AbstractLoader(std::size_t latent_size, torch::nn::AnyModule auto_encoder_module) :
            m_global_step(0),
            m_auto_encoder_module(std::move(auto_encoder_module)),
            m_adam_optimiser(torch::optim::Adam(m_auto_encoder_module.ptr()->parameters(),
                                                torch::optim::AdamOptions(TParams::ae::learning_rate)
                                                        .betas(std::make_tuple(0.9, 0.999)))),
            m_device(torch::cuda::is_available() ? torch::kCUDA : torch::kCPU)
    {
        if (torch::cuda::is_available()) 
        {
            const char* cuda_visible_devices = std::getenv("CUDA_VISIBLE_DEVICES");
            std::string str_cuda_visible_devices = (cuda_visible_devices == NULL) ? std::string("") : std::string(cuda_visible_devices);
            if (not str_cuda_visible_devices.empty()) 
            {
                int index_device_to_use = std::stoi(str_cuda_visible_devices);
                m_device = torch::Device(m_device.type(), index_device_to_use);
                std::cout << "Torch -> Using CUDA ; index device: " << index_device_to_use << std::endl;
            } 
            else 
            {
                std::cout << "Torch -> Using CUDA ; no specified index device " << std::endl;
            }

            // std::cout << "Torch -> Using CUDA" << std::endl;
        } 
        else {std::cout << "Torch -> Using CPU" << std::endl;}

        this->m_auto_encoder_module.ptr()->to(this->m_device);
    }

    void eval(const MatrixXf_rm &gen,
              const MatrixXf_rm &img,
              MatrixXf_rm &descriptors,
              MatrixXf_rm &reconstructed_data,
              MatrixXf_rm &recon_loss,
              MatrixXf_rm &recon_loss_unred,
              MatrixXf_rm &L2_loss,
              MatrixXf_rm &KL_loss,
              MatrixXf_rm &decoder_var,
              bool is_train_set = false) {
        stc::exact(this)->eval(gen, img, descriptors, reconstructed_data, recon_loss, recon_loss_unred, 
                               L2_loss, KL_loss, decoder_var, is_train_set);
    }
    
    void prepare_batches(std::vector<std::tuple<torch::Tensor, torch::Tensor>> &batches, 
                        const MatrixXf_rm &gen, const MatrixXf_rm &img) const {
        stc::exact(this)->prepare_batches(batches, gen, img);
    }

    size_t split_dataset(const MatrixXf_rm &gen_d, const MatrixXf_rm &img_d,
                       MatrixXf_rm &train_gen, MatrixXf_rm &valid_gen, 
                       MatrixXf_rm &train_img, MatrixXf_rm &valid_img) 
    {
        size_t l_train_gen, l_valid_gen;
        
        if (gen_d.rows() > 500) 
        {
            l_train_gen = floor(gen_d.rows() * TParams::ae::CV_fraction);
            l_valid_gen = gen_d.rows() - l_train_gen;
        } 
        else 
        {
            l_train_gen = gen_d.rows();
            l_valid_gen = gen_d.rows();
        }
        assert(l_train_gen != 0 && l_valid_gen != 0);

        train_gen = gen_d.topRows(l_train_gen);
        valid_gen = gen_d.bottomRows(l_valid_gen);

        train_img = img_d.topRows(l_train_gen);
        valid_img = img_d.bottomRows(l_valid_gen);

        return l_train_gen;
    }

    float training(const MatrixXf_rm &gen_d, const MatrixXf_rm &img_d, bool full_train = false, int generation = 1000) 
    {
        return stc::exact(this)->training(gen_d, img_d, full_train, generation);
    }

    float get_avg_recon_loss(const MatrixXf_rm &gen, const MatrixXf_rm &img, bool is_train_set = false) {
        MatrixXf_rm descriptors, reconst, recon_loss, recon_loss_unred, L2_loss, KL_loss, decoder_var;
        eval(gen, img, descriptors, reconst, recon_loss, recon_loss_unred, L2_loss, KL_loss, decoder_var, is_train_set);
        return recon_loss.mean();
    }

    torch::nn::AnyModule get_auto_encoder() {
        return this->m_auto_encoder_module;
    }

    torch::nn::AnyModule& auto_encoder() {
        return this->m_auto_encoder_module;
    }

    int32_t m_global_step;


protected:
    torch::nn::AnyModule m_auto_encoder_module;
    torch::optim::Adam m_adam_optimiser;
    torch::Device m_device;
    double _log_2_pi;


    void get_torch_tensor_from_eigen_matrix(const MatrixXf_rm &M, torch::Tensor &T) const {

        T = torch::rand({M.rows(), M.cols()});
        float *data = T.data_ptr<float>();
        memcpy(data, M.data(), M.cols() * M.rows() * sizeof(float));
    }

    void get_eigen_matrix_from_torch_tensor(const torch::Tensor &T, MatrixXf_rm &M) const {
        if (T.dim() == 0) {
            M = MatrixXf_rm(1, 1); //scalar
            float *data = T.data_ptr<float>();
            M = Eigen::Map<MatrixXf_rm>(data, 1, 1);
        } else {
            size_t total_size_individual_tensor = 1;
            for (size_t dim = 1; dim < T.dim(); ++dim) {
                total_size_individual_tensor *= T.size(dim);
            }
            M = MatrixXf_rm(T.size(0), total_size_individual_tensor);
            float *data = T.data_ptr<float>();
            M = Eigen::Map<MatrixXf_rm>(data, T.size(0), total_size_individual_tensor);
        }
    }

    void get_tuple_from_eigen_matrices(const MatrixXf_rm &M1, const MatrixXf_rm &M2,
                                        torch::Tensor &T1, torch::Tensor &T2, 
                                        std::tuple<torch::Tensor, torch::Tensor> &tuple) const {

        T1 = torch::rand({M1.rows(), M1.cols()});
        T2 = torch::rand({M2.rows(), M2.cols()});
        get_torch_tensor_from_eigen_matrix(M1, T1);
        get_torch_tensor_from_eigen_matrix(M2, T2);
        tuple = std::make_tuple(T1, T2);
    }
};

template <typename TParams, typename Exact = stc::Itself>
class NetworkLoaderAutoEncoder : public AbstractLoader<TParams, typename stc::FindExact<NetworkLoaderAutoEncoder<TParams, Exact>, Exact>::ret> {
public:
    typedef AbstractLoader<TParams, typename stc::FindExact<NetworkLoaderAutoEncoder<TParams, Exact>, Exact>::ret> TParentLoader;

    explicit NetworkLoaderAutoEncoder() :
            TParentLoader(TParams::qd::behav_dim,
                          #ifdef AURORA
                          torch::nn::AnyModule(AutoEncoder(TParams::nov::discretisation * TParams::nov::discretisation, TParams::ae::aurora_en_dim1, TParams::ae::aurora_en_dim2, TParams::ae::aurora_en_dim3, TParams::qd::behav_dim, 
                                                           TParams::ae::de_hid_dim1, TParams::ae::de_hid_dim2, TParams::ae::de_hid_dim3, TParams::nov::discretisation * TParams::nov::discretisation))),
                          #else
                          torch::nn::AnyModule(AutoEncoder(TParams::qd::gen_dim, TParams::ae::en_hid_dim1, TParams::ae::en_hid_dim2, TParams::ae::aurora_en_dim3, TParams::qd::behav_dim, 
                                                           TParams::ae::de_hid_dim1, TParams::ae::de_hid_dim2, TParams::ae::de_hid_dim3, TParams::nov::discretisation * TParams::nov::discretisation))),
                          #endif
            _log_2_pi(log(2 * M_PI)),
            _huber_delta_exp(std::pow(0.5, 3)),
            _epochs_trained(0) {}

    typedef Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> MatrixXf_rm;

    void prepare_batches(std::vector<std::tuple<torch::Tensor, torch::Tensor>> &batches, 
                        const MatrixXf_rm &gen, const MatrixXf_rm &img) const 
    {
        if (gen.rows() <= TParams::ae::batch_size) 
            {batches = std::vector<std::tuple<torch::Tensor, torch::Tensor>>(1);} 
        else 
            {batches = std::vector<std::tuple<torch::Tensor, torch::Tensor>>(floor(gen.rows() / (TParams::ae::batch_size)));}

        // in loop do filtering before passing to make tuple
        if (batches.size() == 1) 
        {
            torch::Tensor T1, T2;
            this->get_tuple_from_eigen_matrices(gen, img, T1, T2, batches[0]);
        } 
        else 
        {
            for (size_t ind = 0; ind < batches.size(); ++ind) 
            {
                torch::Tensor T1, T2;
                this->get_tuple_from_eigen_matrices(gen.middleRows(ind * TParams::ae::batch_size, TParams::ae::batch_size),
                                                    img.middleRows(ind * TParams::ae::batch_size, TParams::ae::batch_size),
                                                    T1, T2, batches[ind]);
            }
        }
    }

    void vector_to_eigen(std::vector<int> &is_trajectories, Eigen::VectorXi &is_traj) const
        {is_traj = Eigen::Map<Eigen::VectorXi> (is_trajectories.data(), is_trajectories.size());}

    int get_epochs_trained() const
    {return _epochs_trained;}

    float training(const MatrixXf_rm &gen_d, const MatrixXf_rm &img_d, bool full_train = false, int generation = 1000) 
    {
        AutoEncoder auto_encoder = std::static_pointer_cast<AutoEncoderImpl>(this->m_auto_encoder_module.ptr());
        auto_encoder->train();
        std::cout << "Total Size Dataset incl. readditions: " << gen_d.rows() << std::endl;
        MatrixXf_rm train_gen, valid_gen, train_img, valid_img;
        size_t l_train_img = this->split_dataset(gen_d, img_d, train_gen, valid_gen, train_img, valid_img);
        
        float init_tr_recon_loss = this->get_avg_recon_loss(train_gen, train_img);
        float init_vl_recon_loss = this->get_avg_recon_loss(valid_gen, valid_img);

        std::cout << "INIT recon train loss: " << init_tr_recon_loss << "   valid recon loss: " << init_vl_recon_loss << std::endl;
        std::cout << "Training: Total num of images " << img_d.rows() << std::endl;
        bool _continue = true;
        Eigen::VectorXd previous_avg = Eigen::VectorXd::Ones(TParams::ae::running_mean_num_epochs) * 50;

        int epoch(0);

        // initialise training variables
        torch::Tensor encoder_mu, encoder_logvar, decoder_logvar;

        while (_continue && (epoch < TParams::ae::nb_epochs)) {
            std::vector<std::tuple<torch::Tensor, torch::Tensor>> batches;
            prepare_batches(batches, train_gen, train_img);

            for (auto &tup : batches) {
                // Get the names below with the inspect_graph.py script applied on the generated graph_text.pb file.
                this->m_auto_encoder_module.ptr()->zero_grad();
                
                // tup[1] is the image tensor
                torch::Tensor img = std::get<1>(tup).to(this->m_device);

                // tup[0] is the phenotype
                #ifdef AURORA
                torch::Tensor reconstruction_tensor = auto_encoder->forward_(img, encoder_mu, encoder_logvar, decoder_logvar, TParams::ae::sigmoid);
                #else
                torch::Tensor reconstruction_tensor = auto_encoder->forward_(std::get<0>(tup).to(this->m_device), encoder_mu, encoder_logvar, decoder_logvar, TParams::ae::sigmoid);
                #endif
                torch::Tensor loss_tensor = torch::empty(1, torch::device(this->m_device));

                if (TParams::ae::full_loss)
                {
                    if (TParams::ae::loss_function == TParams::ae::loss::L2)
                    {loss_tensor = torch::sum(torch::pow(img - reconstruction_tensor, 2) / (2 * torch::exp(decoder_logvar)) + 0.5 * (decoder_logvar + _log_2_pi), {1}).mean();}
                    else if (TParams::ae::loss_function == TParams::ae::loss::L1)
                    {loss_tensor = torch::sum(torch::abs(img - reconstruction_tensor) / (2 * torch::exp(decoder_logvar)) + 0.5 * (decoder_logvar + _log_2_pi), {1}).mean();}
                    else if (TParams::ae::loss_function == TParams::ae::loss::BCE)
                    {loss_tensor = torch::sum(torch::binary_cross_entropy(reconstruction_tensor, img, {} , 0) / (2 * torch::exp(decoder_logvar)) + 0.5 * (decoder_logvar + _log_2_pi), {1}).mean();}
                    else if (TParams::ae::loss_function == TParams::ae::loss::Huber)
                    {
                        torch::Tensor mask = torch::abs(img - reconstruction_tensor).le(0.5);
                        torch::Tensor flipped_mask = mask.logical_not();
                        loss_tensor = 0.5 * torch::sum(torch::pow((img - reconstruction_tensor).index(mask), 2) / (2 * torch::exp(decoder_logvar.index(mask))) + 0.5 * (decoder_logvar.index(mask) + _log_2_pi));
                        loss_tensor += torch::sum(0.5 * (torch::abs((img - reconstruction_tensor).index(flipped_mask)) / (2 * torch::exp(decoder_logvar.index(flipped_mask))) + 0.5 * (decoder_logvar.index(flipped_mask) + _log_2_pi)) - _huber_delta_exp);
                        loss_tensor /= img.size(0);
                    }
                }
                else
                {
                    if (TParams::ae::loss_function == TParams::ae::loss::L2)
                    {loss_tensor = torch::sum(torch::pow(img - reconstruction_tensor, 2), {1}).mean();}
                    else if (TParams::ae::loss_function == TParams::ae::loss::L1)
                    {loss_tensor = torch::sum(torch::abs(img - reconstruction_tensor), {1}).mean();}
                    else if (TParams::ae::loss_function == TParams::ae::loss::BCE)
                    {loss_tensor = torch::sum(torch::binary_cross_entropy(reconstruction_tensor, img, {}, 0), {1}).mean();}
                    else if (TParams::ae::loss_function == TParams::ae::loss::Huber)
                    {
                        torch::Tensor mask = torch::abs(img - reconstruction_tensor).le(0.5);
                        loss_tensor = 0.5 * torch::sum(torch::pow((img - reconstruction_tensor).index(mask), 2));
                        loss_tensor += torch::sum(0.5 * torch::abs((img - reconstruction_tensor).index(mask.logical_not())) - _huber_delta_exp);
                        loss_tensor /= img.size(0);
                    }
                }

                #ifdef VAE
                loss_tensor += -0.5 * TParams::ae::beta * torch::sum(1 + encoder_logvar - torch::pow(encoder_mu, 2) - torch::exp(encoder_logvar), {1}).mean();
                #endif

                loss_tensor.backward();
                
                this->m_adam_optimiser.step();
                ++epoch;
            }

            this->m_global_step++;

            // early stopping
            if (!full_train) {
                float current_avg = this->get_avg_recon_loss(valid_gen, valid_img);
                for (size_t t = 1; t < previous_avg.size(); t++)
                    previous_avg[t - 1] = previous_avg[t];

                previous_avg[previous_avg.size() - 1] = current_avg;

                // if the running average on the val set is increasing and train loss is higher than at the beginning
                if ((previous_avg.array() - previous_avg[0]).mean() > 0 && epoch > TParams::ae::min_num_epochs &&
                    this->get_avg_recon_loss(train_gen, train_img) < init_tr_recon_loss)
                        {_continue = false;}
            }

            float recon_loss_t = this->get_avg_recon_loss(train_gen, train_img);
            float recon_loss_v = this->get_avg_recon_loss(valid_gen, valid_img);

            std::cout.precision(5);
            std::cout << "training dataset: " << train_gen.rows() << "  valid dataset: " << valid_gen.rows() << " - ";
            std::cout << std::setw(5) << epoch << "/" << std::setw(5) << TParams::ae::nb_epochs;
            std::cout << " recon loss (t): " << std::setw(8) << recon_loss_t;
            std::cout << " (v): " << std::setw(8) << recon_loss_v;
            std::cout << std::flush << '\r';
        }

        float full_dataset_recon_loss = this->get_avg_recon_loss(gen_d, img_d);
        std::cout << "Final full dataset recon loss: " << full_dataset_recon_loss << '\n';

        _epochs_trained = epoch + 1;

        return full_dataset_recon_loss;
    }

    void eval(const MatrixXf_rm &gen,
              const MatrixXf_rm &img,
              MatrixXf_rm &descriptors,
              MatrixXf_rm &reconstructed_data,
              MatrixXf_rm &recon_loss,
              MatrixXf_rm &recon_loss_unred,
              MatrixXf_rm &L2_loss,
              MatrixXf_rm &KL_loss,
              MatrixXf_rm &decoder_var,
              bool is_train_set = false) 
    {
        torch::NoGradGuard no_grad;
        AutoEncoder auto_encoder = std::static_pointer_cast<AutoEncoderImpl>(this->m_auto_encoder_module.ptr());
        auto_encoder->eval();

        torch::Tensor gen_tensor, img_tensor;
        this->get_torch_tensor_from_eigen_matrix(gen, gen_tensor);
        this->get_torch_tensor_from_eigen_matrix(img, img_tensor);
	    img_tensor = img_tensor.to(this->m_device);

        torch::Tensor encoder_mu, encoder_logvar, decoder_logvar, descriptors_tensor;
        #ifdef AURORA
        torch::Tensor reconstruction_tensor = auto_encoder->forward_get_latent(img_tensor, encoder_mu, encoder_logvar, decoder_logvar, descriptors_tensor, TParams::ae::sigmoid);
        #else
        torch::Tensor reconstruction_tensor = auto_encoder->forward_get_latent(gen_tensor.to(this->m_device), encoder_mu, encoder_logvar, decoder_logvar, descriptors_tensor, TParams::ae::sigmoid);
        #endif
        torch::Tensor recon_loss_unreduced = torch::empty({gen.rows(), TParams::nov::discretisation * TParams::nov::discretisation}, torch::device(this->m_device));

        if (TParams::ae::full_loss)
        {
            if (TParams::ae::loss_function == TParams::ae::loss::L2)
            {recon_loss_unreduced = torch::pow(img_tensor - reconstruction_tensor, 2) / (2 * torch::exp(decoder_logvar)) + 0.5 * (decoder_logvar + _log_2_pi);}
            else if (TParams::ae::loss_function == TParams::ae::loss::L1)
            {recon_loss_unreduced = torch::abs(img_tensor - reconstruction_tensor) / (2 * torch::exp(decoder_logvar)) + 0.5 * (decoder_logvar + _log_2_pi);}
            else if (TParams::ae::loss_function == TParams::ae::loss::BCE)
            {recon_loss_unreduced = torch::binary_cross_entropy(reconstruction_tensor, img_tensor, {}, 0) / (2 * torch::exp(decoder_logvar)) + 0.5 * (decoder_logvar + _log_2_pi);}
            else if (TParams::ae::loss_function == TParams::ae::loss::Huber)
            {
                for (int i{0}; i < img_tensor.size(0); ++i)
                {
                    torch::Tensor mask = torch::abs(img_tensor[i] - reconstruction_tensor[i]).le(0.5);
                    torch::Tensor flipped_mask = mask.logical_not();
                    long num_le = mask.sum().item<long>();

                    // rows and columns passed in
                    recon_loss_unreduced.index_put_({torch::ones(1, torch::dtype(torch::kLong)) * i, 
                                                    torch::arange(num_le, torch::dtype(torch::kLong))}, 
                                                    0.5 * (torch::pow((img_tensor[i] - reconstruction_tensor[i]).index(mask), 2) / (2 * torch::exp(decoder_logvar[i].index(mask))) + 0.5 * (decoder_logvar[i].index(mask) + _log_2_pi)), false);

                    recon_loss_unreduced.index_put_({torch::ones(1, torch::dtype(torch::kLong)) * i, 
                                                    torch::arange(num_le, static_cast<long>(TParams::nov::discretisation * TParams::nov::discretisation), torch::dtype(torch::kLong))},
                                                    0.5 * (torch::abs((img_tensor[i] - reconstruction_tensor[i]).index(flipped_mask)) / (2 * torch::exp(decoder_logvar[i].index(flipped_mask))) + 0.5 * (decoder_logvar[i].index(flipped_mask) + _log_2_pi)) - _huber_delta_exp, false);
                }
            }
        }
        else
        {
            if (TParams::ae::loss_function == TParams::ae::loss::L2)
            {recon_loss_unreduced = torch::pow(img_tensor - reconstruction_tensor, 2);}
            else if (TParams::ae::loss_function == TParams::ae::loss::L1)
            {recon_loss_unreduced = torch::abs(img_tensor - reconstruction_tensor);}
            else if (TParams::ae::loss_function == TParams::ae::loss::BCE)
            {recon_loss_unreduced = torch::binary_cross_entropy(reconstruction_tensor, img_tensor, {}, 0);}
            else if (TParams::ae::loss_function == TParams::ae::loss::Huber)
            {
                for (int i{0}; i < img_tensor.size(0); ++i)
                {
                    torch::Tensor mask = torch::abs(img_tensor[i] - reconstruction_tensor[i]).le(0.5);
                    recon_loss_unreduced.index_put_({torch::ones(1, torch::dtype(torch::kLong)) * i, 
                                                    torch::arange(mask.sum().item<long>(), torch::dtype(torch::kLong))}, 
                                                    0.5 * torch::pow((img_tensor[i] - reconstruction_tensor[i]).index(mask), 2), false);

                    recon_loss_unreduced.index_put_({torch::ones(1, torch::dtype(torch::kLong)) * i, 
                                                    torch::arange(mask.sum().item<long>(), static_cast<long>(TParams::nov::discretisation * TParams::nov::discretisation), torch::dtype(torch::kLong))},
                                                    0.5 * torch::abs((img_tensor[i] - reconstruction_tensor[i]).index(mask.logical_not())) - _huber_delta_exp, false);
                }
            }
        }
        torch::Tensor L2 = torch::pow(img_tensor - reconstruction_tensor, 2);
        torch::Tensor reconstruction_loss = torch::sum(recon_loss_unreduced, {1});
        // KL divergence
        #ifdef VAE
        torch::Tensor KL = -0.5 * TParams::ae::beta * (1 + encoder_logvar - torch::pow(encoder_mu, 2) - torch::exp(encoder_logvar));
        reconstruction_loss += torch::sum(KL, {1});
        #endif


        this->get_eigen_matrix_from_torch_tensor(descriptors_tensor.cpu(), descriptors);
        this->get_eigen_matrix_from_torch_tensor(reconstruction_tensor.cpu(), reconstructed_data);
        this->get_eigen_matrix_from_torch_tensor(reconstruction_loss.cpu(), recon_loss);
        this->get_eigen_matrix_from_torch_tensor(recon_loss_unreduced.cpu(), recon_loss_unred);
        this->get_eigen_matrix_from_torch_tensor(L2.cpu(), L2_loss);

        #ifdef VAE
        this->get_eigen_matrix_from_torch_tensor(torch::exp(decoder_logvar).cpu(), decoder_var);
        this->get_eigen_matrix_from_torch_tensor(KL.cpu(), KL_loss);
        #endif
    }

    float _log_2_pi;
    float _huber_delta_exp;
    int _epochs_trained;
};

#endif
